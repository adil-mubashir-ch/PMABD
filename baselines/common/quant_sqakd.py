"""
EWGS-style learned-range quantizers, as used by SQAKD (and therefore DAQAKD).

PROVENANCE
    Ported from the authors' released code:
      baselines/third_party/SQAKD/CIFAR/models/custom_modules.py
    (Zhao & Zhao, "Self-Supervised Quantization-Aware Knowledge Distillation",
     AISTATS 2024; their quantizer is in turn EWGS, Lee et al. CVPR 2021.)

    The discretizers below are line-for-line the authors'. What changed:
      * `args`-style construction replaced with explicit bit-width arguments,
        so a module can be built without their global argparse namespace.
      * The tensorboard `save_dict` instrumentation was dropped (it only
        logged u_mean/u_std and has no effect on the forward or backward).
      * Added QLinear, which their CIFAR code does not need but MobileNetV2's
        classifier does.

WHY NOT USE THEIR TinyImageNet PATH DIRECTLY
    SQAKD/TinyImageNet/main.py depends on NVIDIA DALI (Linux-only) and on the
    bundled MQBench torch.fx graph-mode quantizer written for torch 1.10. We
    run Windows + torch 2.8. Their CIFAR path implements the same quantizer in
    plain PyTorch, so that is what is ported here. The *method* is unchanged;
    only the plumbing is.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["QConv2d", "QLinear", "quantize_model", "init_quant_ranges"]


# ─────────────────────────────────────────────────────────────────────────────
# Discretizers (verbatim from the authors, minus logging hooks)
# ─────────────────────────────────────────────────────────────────────────────
class STE_discretizer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x_in, num_levels):
        x = x_in * (num_levels - 1)
        x = torch.round(x)
        return x / (num_levels - 1)

    @staticmethod
    def backward(ctx, g):
        return g, None


class EWGS_discretizer(torch.autograd.Function):
    """Element-wise gradient scaling. x_in must already lie in [0, 1]."""

    @staticmethod
    def forward(ctx, x_in, num_levels, scaling_factor):
        x = x_in * (num_levels - 1)
        x = torch.round(x)
        x_out = x / (num_levels - 1)
        ctx._scaling_factor = scaling_factor
        ctx.save_for_backward(x_in - x_out)
        return x_out

    @staticmethod
    def backward(ctx, g):
        diff = ctx.saved_tensors[0]
        delta = ctx._scaling_factor
        u = g * delta * torch.sign(g)
        return g + u * diff, None, None


# ─────────────────────────────────────────────────────────────────────────────
# Quantized layers
# ─────────────────────────────────────────────────────────────────────────────
class _QuantMixin:
    """Shared learned-range machinery for QConv2d / QLinear.

    Ranges (lW, uW, lA, uA) are nn.Parameters learned by gradient descent, and
    are lazily initialised from the first batch that flows through the layer
    (`init` buffer), exactly as the authors do.
    """

    def _setup_quant(self, w_bits, a_bits, ewgs=True,
                     bkwd_scaling_factor_w=0.0, bkwd_scaling_factor_a=0.0,
                     per_channel_w=False, n_out=1, is_depthwise=False):
        self.w_bits = w_bits
        self.a_bits = a_bits
        self.quan_weight = w_bits is not None and w_bits < 32
        self.quan_act = a_bits is not None and a_bits < 32
        self.ewgs = ewgs

        # PER-CHANNEL WEIGHT QUANTIZATION
        #
        # One (lW, uW) pair per output channel instead of one per tensor. At 8
        # or 4 bits this is a modest refinement; at 2 bits it is the difference
        # between working and not, because per-tensor gives FOUR levels shared
        # across an entire layer, and channels whose weights are small relative
        # to the layer's spread all collapse onto the same level.
        #
        # It exists here to remove a confound rather than to improve a number.
        # The PMABD ladder quantizes weights per channel (pipeline.py:459,
        # `per_channel_weights=(not is_depthwise)`) while this port was
        # per-tensor throughout, so a W2A2 comparison between them would have
        # measured the quantizer as much as the method. Depthwise convolutions
        # are excluded here for exactly the same reason they are there: one
        # output channel sees one input channel, so a per-channel range is
        # estimated from very few weights and is noisier than the shared one.
        #
        # NOTE this is faithful to PMABD, not to SQAKD, whose own TinyImageNet
        # config sets `per_channel: False`. Enabling it therefore makes the
        # baselines STRONGER than their published configuration -- deliberately,
        # so a collapse cannot be blamed on our port. Say so in the paper.
        self.per_channel_w = bool(per_channel_w) and not is_depthwise

        if self.quan_weight:
            self.weight_levels = 2 ** w_bits
            shape = (n_out,) if self.per_channel_w else ()
            self.uW = nn.Parameter(torch.zeros(shape))
            self.lW = nn.Parameter(torch.zeros(shape))
            self.register_buffer("bkwd_scaling_factorW",
                                 torch.tensor(float(bkwd_scaling_factor_w)))
        if self.quan_act:
            self.act_levels = 2 ** a_bits
            self.uA = nn.Parameter(torch.tensor(0.0))
            self.lA = nn.Parameter(torch.tensor(0.0))
            self.register_buffer("bkwd_scaling_factorA",
                                 torch.tensor(float(bkwd_scaling_factor_a)))

        self.register_buffer("init", torch.tensor([0], dtype=torch.long))
        self.output_scale = nn.Parameter(torch.tensor(1.0))

    def _w_view(self, t):
        """Reshape a per-channel range so it broadcasts over the weight tensor.

        Conv2d weights are [out, in/groups, kH, kW] and Linear are [out, in],
        so the channel axis is 0 in both cases and the rest are singleton.
        """
        if not self.per_channel_w:
            return t
        return t.view(-1, *([1] * (self.weight.dim() - 1)))

    def _discretize(self, x, levels, scale):
        if self.ewgs:
            return EWGS_discretizer.apply(x, levels, scale)
        return STE_discretizer.apply(x, levels)

    @torch.no_grad()
    def _initialise_ranges(self, x):
        """Authors' `initialize()`, ported exactly. Do not "simplify" this.

        Three details matter and all three were got wrong on the first pass,
        which cost four invalid runs:

        * Weights: +/- 3 sigma.
        * Activations: `std / sqrt(1 - 2/pi) * 3`, NOT `std * 3`. Post-ReLU
          activations are half-normal, whose observed std is sqrt(1 - 2/pi)
          ~= 0.603 times the underlying normal's sigma. Dropping the
          correction makes the range 1.66x too tight, so most activations
          saturate at the clamp and stop passing gradient.
        * output_scale is calibrated to the ratio of FP32 to quantized output
          magnitude, so the quantized layer *matches the FP32 layer's scale at
          initialisation*. This is what keeps the BatchNorm running statistics
          inherited from the FP32 checkpoint valid. Leaving it at 1.0 puts
          every conv output at the wrong scale: training still works because
          BatchNorm uses batch statistics in train mode, but evaluation reads
          the stale running statistics and sits at chance until the learning
          rate anneals to zero and the stats finally catch up.
        """
        Qweight = self.weight
        Qact = x

        if self.quan_weight:
            if self.per_channel_w:
                # Each output channel gets its own 3-sigma range, computed from
                # that channel's own weights.
                std = self.weight.detach().reshape(
                    self.weight.shape[0], -1).std(dim=1)
                self.uW.data.copy_(std * 3.0)
                self.lW.data.copy_(-std * 3.0)
            else:
                self.uW.data.fill_(self.weight.std() * 3.0)
                self.lW.data.fill_(-self.weight.std() * 3.0)
            Qweight = self._quant_weight()

        if self.quan_act:
            self.uA.data.fill_(x.std() / math.sqrt(1 - 2 / math.pi) * 3.0)
            self.lA.data.fill_(x.min())
            Qact = self._quant_act(x)

        # With dequantizing fake-quantization the quantized output is already
        # on the FP32 scale, so this ratio is ~1 by construction rather than
        # the large correction the authors' non-dequantizing version needs.
        # It is still computed (cheaply) so any residual bias is absorbed, and
        # clamped so a degenerate layer cannot inject a wild scale.
        Qout = self._op(Qact, Qweight)
        out = self._op(x, self.weight)
        denom = Qout.abs().mean()
        if torch.isfinite(denom) and denom > 0:
            ratio = float(out.abs().mean() / denom)
            self.output_scale.data.fill_(min(max(ratio, 0.5), 2.0))
        else:
            self.output_scale.data.fill_(1.0)

        self.init.fill_(1)

    # ── fake quantization: quantize, then DEQUANTIZE to the original scale ──
    #
    # DEVIATION FROM THE AUTHORS' CIFAR CODE, and why it is the right call.
    #
    # Their CIFAR QConv leaves weights in [-1,1] and activations in [0,1] and
    # corrects the resulting magnitude error with one learned scalar,
    # output_scale. That works only because every quantized conv in their
    # ResNet/VGG models is fed by a ReLU, so lA = 0 and the activation map is
    # purely multiplicative — a scalar can undo it exactly.
    #
    # MobileNetV2 breaks that. Each inverted residual's *expand* conv is fed by
    # the previous block's linear bottleneck, which is signed (measured
    # lA = -60.08 at features.2.conv.0.0). The map (x - lA)/span is then
    # AFFINE, and its constant term is not a scalar multiple of anything — no
    # output_scale can remove it. Measured relative output error against FP32
    # at initialisation, per layer: > 1.0 on 16 of 51 layers even at W8A8 while
    # using all 256 levels, i.e. the error is structural rather than a
    # bit-depth effect. BatchNorm recalibration masks it at 8 bits by absorbing
    # a near-constant offset; at 3 bits the coarseness compounds and the
    # student collapses to constant output.
    #
    # Dequantizing restores the original scale directly, so the conv output
    # matches FP32 up to genuine quantization error, and output_scale is no
    # longer load-bearing. This is standard fake quantization, and it is what
    # the authors' *TinyImageNet* path actually does — that path uses MQBench
    # (`prepare_by_platform` with DoReFa/PACT/LSQ FakeQuantize), not this CIFAR
    # module. So this moves the port closer to the code that produced their
    # published MobileNetV2 numbers, not further away.
    def _quant_weight(self):
        if not self.quan_weight:
            return self.weight
        lo = self._w_view(self.lW)
        hi = self._w_view(self.uW)
        span = (hi - lo).clamp(min=1e-8)
        w = ((self.weight - lo) / span).clamp(min=0, max=1)
        w = self._discretize(w, self.weight_levels, self.bkwd_scaling_factorW)
        return w * span + lo

    def _quant_act(self, x):
        if not self.quan_act:
            return x
        span = (self.uA - self.lA).clamp(min=1e-8)
        a = ((x - self.lA) / span).clamp(min=0, max=1)
        a = self._discretize(a, self.act_levels, self.bkwd_scaling_factorA)
        return a * span + self.lA


class QConv2d(nn.Conv2d, _QuantMixin):
    def __init__(self, in_channels, out_channels, kernel_size, w_bits=32,
                 a_bits=32, stride=1, padding=0, dilation=1, groups=1,
                 bias=True, ewgs=True, bkwd_scaling_factor_w=0.0,
                 bkwd_scaling_factor_a=0.0, per_channel_w=False):
        nn.Conv2d.__init__(self, in_channels, out_channels, kernel_size,
                           stride, padding, dilation, groups, bias)
        # Depthwise: one output channel per input channel. Excluded from
        # per-channel ranges, matching pipeline.py:459.
        depthwise = groups > 1 and groups == in_channels == out_channels
        self._setup_quant(w_bits, a_bits, ewgs,
                          bkwd_scaling_factor_w, bkwd_scaling_factor_a,
                          per_channel_w=per_channel_w, n_out=out_channels,
                          is_depthwise=depthwise)

    def _op(self, a, w):
        return F.conv2d(a, w, self.bias, self.stride, self.padding,
                        self.dilation, self.groups)

    def forward(self, x):
        if self.init == 0 and self.training:
            self._initialise_ranges(x)
        w = self._quant_weight()
        a = self._quant_act(x)
        return self._op(a, w) * torch.abs(self.output_scale)


class QLinear(nn.Linear, _QuantMixin):
    def __init__(self, in_features, out_features, w_bits=32, a_bits=32,
                 bias=True, ewgs=True, bkwd_scaling_factor_w=0.0,
                 bkwd_scaling_factor_a=0.0, per_channel_w=False):
        nn.Linear.__init__(self, in_features, out_features, bias)
        self._setup_quant(w_bits, a_bits, ewgs,
                          bkwd_scaling_factor_w, bkwd_scaling_factor_a,
                          per_channel_w=per_channel_w, n_out=out_features)

    def _op(self, a, w):
        return F.linear(a, w, self.bias)

    def forward(self, x):
        if self.init == 0 and self.training:
            self._initialise_ranges(x)
        w = self._quant_weight()
        a = self._quant_act(x)
        return self._op(a, w) * torch.abs(self.output_scale)


# ─────────────────────────────────────────────────────────────────────────────
# Model surgery
# ─────────────────────────────────────────────────────────────────────────────
def _copy_conv(src: nn.Conv2d, w_bits, a_bits, **kw) -> QConv2d:
    dst = QConv2d(src.in_channels, src.out_channels, src.kernel_size,
                  w_bits=w_bits, a_bits=a_bits, stride=src.stride,
                  padding=src.padding, dilation=src.dilation,
                  groups=src.groups, bias=src.bias is not None, **kw)
    dst.weight.data.copy_(src.weight.data)
    if src.bias is not None:
        dst.bias.data.copy_(src.bias.data)
    return dst


def _copy_linear(src: nn.Linear, w_bits, a_bits, **kw) -> QLinear:
    dst = QLinear(src.in_features, src.out_features, w_bits=w_bits,
                  a_bits=a_bits, bias=src.bias is not None, **kw)
    dst.weight.data.copy_(src.weight.data)
    if src.bias is not None:
        dst.bias.data.copy_(src.bias.data)
    return dst


def quantize_model(model: nn.Module, w_bits: int, a_bits: int,
                   first_last_bits=None, ewgs: bool = True,
                   per_channel_w: bool = False, **kw) -> nn.Module:
    """Replace every Conv2d/Linear in `model` with its quantized counterpart.

    `first_last_bits` controls the first conv and the final classifier — the
    single biggest fairness knob in this whole comparison:

        None      -> leave them FP32.  This is SQAKD's and DAQAKD's own
                     convention, and is what you must use to reproduce their
                     published numbers.
        (w, a)    -> quantize them to (w, a).  Pass (8, 8) to match the PMABD
                     ladder, which holds first/last at W8A8 and is therefore
                     strictly harder than the published setting.

    `per_channel_w=True` gives each output channel its own learned weight
    range, matching the PMABD ladder (pipeline.py:459). Off by default so that
    already-recorded runs keep their meaning; see _setup_quant for why it
    matters most at 2 bits.

    Weights are copied across, so pass a model already loaded with the FP32
    initialisation.
    """
    convs, linears = [], []
    for name, mod in model.named_modules():
        for cname, child in list(mod.named_children()):
            full = f"{name}.{cname}" if name else cname
            if isinstance(child, nn.Conv2d):
                convs.append((mod, cname, full, child))
            elif isinstance(child, nn.Linear):
                linears.append((mod, cname, full, child))

    if not convs:
        raise ValueError("No Conv2d found — is this really a CNN?")

    first_conv_id = id(convs[0][3])
    last_linear_id = id(linears[-1][3]) if linears else None

    for parent, cname, _full, child in convs:
        if id(child) == first_conv_id:
            if first_last_bits is None:
                continue  # keep FP32
            wb, ab = first_last_bits
        else:
            wb, ab = w_bits, a_bits
        setattr(parent, cname, _copy_conv(child, wb, ab, ewgs=ewgs,
                                          per_channel_w=per_channel_w, **kw))

    for parent, cname, _full, child in linears:
        if id(child) == last_linear_id:
            if first_last_bits is None:
                continue
            wb, ab = first_last_bits
        else:
            wb, ab = w_bits, a_bits
        setattr(parent, cname, _copy_linear(child, wb, ab, ewgs=ewgs,
                                            per_channel_w=per_channel_w, **kw))

    return model


@torch.no_grad()
def init_quant_ranges(model: nn.Module, loader, device, batches: int = 1):
    """Run a few batches in train mode so every layer's lazy range init fires.

    Without this the first real training batch initialises ranges *and* takes a
    gradient step on them in the same iteration, which is where most "loss is
    NaN at step 1" reports come from.
    """
    was_training = model.training
    model.train()
    for i, (x, _y) in enumerate(loader):
        model(x.to(device, non_blocking=True))
        if i + 1 >= batches:
            break
    model.train(was_training)
    return model


def quant_range_parameters(model: nn.Module):
    """The learned clipping bounds — the authors give these their own optimizer."""
    names = ("uW", "lW", "uA", "lA", "output_scale")
    return [p for n, p in model.named_parameters()
            if n.rsplit(".", 1)[-1] in names]


def weight_parameters(model: nn.Module):
    names = ("uW", "lW", "uA", "lA", "output_scale")
    return [p for n, p in model.named_parameters()
            if n.rsplit(".", 1)[-1] not in names]
