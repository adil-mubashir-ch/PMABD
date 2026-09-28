"""
Shared-weight switchable-precision layers, used by Any-Precision DNNs and by
InstantNet's Cascade Distillation Training.

PROVENANCE
    Ported from the authors' released code:
      baselines/third_party/Any-Precision-DNNs/models/quan_ops.py
      baselines/third_party/InstantNet/quantize.py
    Both are DoReFa-style quantizers over one shared weight tensor, with a
    *separate BatchNorm per bit-width* (SwitchBatchNorm). That per-bit BN is
    not a detail: without it a single shared-weight network cannot serve
    several precisions at once, because each precision induces a different
    activation distribution.

DIFFERENCE FROM THE PMABD LADDER — worth stating in the paper
    These methods train ONE set of weights that must serve every precision
    simultaneously. PMABD trains a separate model per rung and lets finished
    rungs teach later ones. So "GPU hours to reach W3A3" means something
    different for each family: here a single run yields W8A8/W4A4/W3A3
    together, which is exactly why they are cheap and why the per-precision
    cost must be reported as (total run) / (precisions served) alongside the
    raw total. `engine.py` logs both.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["SwitchBatchNorm2d", "SwitchConv2d", "SwitchLinear",
           "switchable_mobilenetv2", "set_bits"]


# ─────────────────────────────────────────────────────────────────────────────
# DoReFa quantization (authors' `qfn` / weight_quantize_fn / activation_quantize_fn)
# ─────────────────────────────────────────────────────────────────────────────
class _qfn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, k):
        n = float(2 ** k - 1)
        return torch.round(x * n) / n

    @staticmethod
    def backward(ctx, g):
        return g, None


def _quantize(x, k):
    return _qfn.apply(x, k)


def quantize_weight(w, bits):
    """DoReFa weight quantization via tanh normalisation. bits=32 -> identity."""
    if bits == 32:
        return w
    if bits == 1:  # BWN-style sign with per-tensor scale
        e = w.abs().mean()
        return (torch.sign(w / e) * e - w).detach() + w
    t = torch.tanh(w)
    t = t / (2 * t.abs().max()) + 0.5
    return 2 * _quantize(t, bits) - 1


def quantize_act(x, bits, lo=None, hi=None):
    """Uniform activation quantization over a learned range [lo, hi].

    WHY THIS DEVIATES FROM THE AUTHORS' CODE — state this in the paper.
        Both source repos quantize activations as `_quantize(clamp(x, 0, 1))`,
        i.e. they assume activations already live in [0, 1]. That holds for
        their models: Any-Precision ships resnet20q/resnet50q whose
        nonlinearity is a clamp/tanh chosen to land in [0, 1] (their README
        says so explicitly), and InstantNet's search space is built the same
        way.

        Stock MobileNetV2 does not satisfy that assumption twice over. Its
        ReLU6 outputs live in [0, 6], so a hard clamp to [0, 1] discards
        five-sixths of the range and returns exactly zero gradient wherever
        x > 1. And each inverted-residual block's *expand* conv is fed by the
        previous block's linear projection, which is signed and unbounded, so
        clamping at 0 destroys the negative half outright.

        Measured effect of the verbatim version: every parameter in
        features.0-features.5 received exactly zero gradient in all of the 3-,
        4- and 8-bit branches while the 32-bit branch trained normally. The
        early layers simply did not learn.

        The fix keeps the quantizer uniform and keeps the bit-width semantics
        identical; it only learns where the range sits, per layer and per
        bit-width, initialised from the first batch. This is the same
        treatment the SQAKD/DAQAKD rows already get from the EWGS quantizer,
        so the comparison stays even across methods rather than handicapping
        the two switchable ones with a mismatched activation range.
    """
    if bits == 32:
        return x
    if lo is None or hi is None:
        return _quantize(torch.clamp(x, 0, 1), bits)
    span = (hi - lo).clamp(min=1e-5)
    xn = ((x - lo) / span).clamp(0, 1)
    return _quantize(xn, bits) * span + lo


# ─────────────────────────────────────────────────────────────────────────────
# Switchable layers
# ─────────────────────────────────────────────────────────────────────────────
class _BitState:
    """Module-tree-wide current bit-width, set by `set_bits(model, b)`.

    The authors thread the bit-width through every forward() signature. That
    means rewriting torchvision's MobileNetV2 forward. Holding it as state on
    the module instead lets us reuse the stock torchvision graph untouched,
    which removes a whole class of transcription error from the comparison.
    """


class SwitchBatchNorm2d(nn.Module):
    """One BatchNorm per bit-width, sharing nothing. Authors' SwitchBatchNorm2d."""

    def __init__(self, num_features, bit_list):
        super().__init__()
        self.bit_list = list(bit_list)
        self.bn_dict = nn.ModuleDict(
            {str(b): nn.BatchNorm2d(num_features) for b in self.bit_list})
        self.abit = self.bit_list[-1]

    def forward(self, x):
        return self.bn_dict[str(self.abit)](x)


class SwitchConv2d(nn.Conv2d):
    """Shared weights, DoReFa weight quantization, learned per-bit act range."""

    def __init__(self, *args, bit_list, quantize_input=True, **kw):
        super().__init__(*args, **kw)
        self.bit_list = list(bit_list)
        self.wbit = self.bit_list[-1]
        self.abit = self.bit_list[-1]
        self.quantize_input = quantize_input

        # One activation range per bit-width, mirroring SwitchBatchNorm's logic:
        # different precisions induce different activation distributions, so a
        # single shared range would be wrong for most of them.
        qbits = [b for b in self.bit_list if b != 32]
        self.act_lo = nn.ParameterDict(
            {str(b): nn.Parameter(torch.tensor(0.0)) for b in qbits})
        self.act_hi = nn.ParameterDict(
            {str(b): nn.Parameter(torch.tensor(6.0)) for b in qbits})
        self.register_buffer("act_init",
                             torch.zeros(len(qbits), dtype=torch.long))
        self._qbit_index = {b: i for i, b in enumerate(qbits)}

    @torch.no_grad()
    def _maybe_init_range(self, x):
        i = self._qbit_index[self.abit]
        if self.act_init[i] == 1:
            return
        key = str(self.abit)
        # Percentile-free min/max is too sensitive to a single outlier; the
        # mean +/- 3 sigma window is what the EWGS port uses, so both quantizer
        # families start from the same rule.
        lo = torch.minimum(x.min(), torch.zeros((), device=x.device))
        hi = x.mean() + 3.0 * x.std()
        self.act_lo[key].data.fill_(float(lo))
        self.act_hi[key].data.fill_(float(torch.maximum(hi, lo + 1e-3)))
        self.act_init[i] = 1

    def forward(self, x):
        if self.quantize_input and self.abit != 32:
            if self.training:
                self._maybe_init_range(x)
            key = str(self.abit)
            x = quantize_act(x, self.abit, self.act_lo[key], self.act_hi[key])
        w = quantize_weight(self.weight, self.wbit)
        return F.conv2d(x, w, self.bias, self.stride, self.padding,
                        self.dilation, self.groups)


class SwitchLinear(nn.Linear):
    def __init__(self, *args, bit_list, **kw):
        super().__init__(*args, **kw)
        self.bit_list = list(bit_list)
        self.wbit = self.bit_list[-1]
        self.abit = self.bit_list[-1]

    def forward(self, x):
        w = quantize_weight(self.weight, self.wbit)
        return F.linear(x, w, self.bias)


def set_bits(model: nn.Module, bits: int):
    """Switch the whole network to `bits`. Must be called before every forward."""
    for m in model.modules():
        if isinstance(m, (SwitchConv2d, SwitchLinear)):
            m.wbit = bits
            m.abit = bits
        elif isinstance(m, SwitchBatchNorm2d):
            m.abit = bits
    return model


# ─────────────────────────────────────────────────────────────────────────────
# MobileNetV2 surgery
# ─────────────────────────────────────────────────────────────────────────────
def _swap_conv(src: nn.Conv2d, bit_list, quantize_input=True) -> SwitchConv2d:
    dst = SwitchConv2d(src.in_channels, src.out_channels, src.kernel_size,
                       stride=src.stride, padding=src.padding,
                       dilation=src.dilation, groups=src.groups,
                       bias=src.bias is not None, bit_list=bit_list,
                       quantize_input=quantize_input)
    dst.weight.data.copy_(src.weight.data)
    if src.bias is not None:
        dst.bias.data.copy_(src.bias.data)
    return dst


def _swap_bn(src: nn.BatchNorm2d, bit_list) -> SwitchBatchNorm2d:
    dst = SwitchBatchNorm2d(src.num_features, bit_list)
    for b in bit_list:  # every branch starts from the FP32 statistics
        bn = dst.bn_dict[str(b)]
        bn.weight.data.copy_(src.weight.data)
        bn.bias.data.copy_(src.bias.data)
        bn.running_mean.data.copy_(src.running_mean.data)
        bn.running_var.data.copy_(src.running_var.data)
    return dst


def _swap_linear(src: nn.Linear, bit_list) -> SwitchLinear:
    dst = SwitchLinear(src.in_features, src.out_features,
                       bias=src.bias is not None, bit_list=bit_list)
    dst.weight.data.copy_(src.weight.data)
    if src.bias is not None:
        dst.bias.data.copy_(src.bias.data)
    return dst


def switchable_mobilenetv2(model: nn.Module, bit_list, keep_first_last_fp=True):
    """In-place conversion of a torchvision MobileNetV2 to switchable precision.

    `bit_list` must be ascending (e.g. [3, 4, 8, 32]); the highest entry is the
    full-precision branch that supervises the rest.

    `keep_first_last_fp` follows both source repos, which never quantize the
    stem convolution or the classifier. Set False to match the PMABD ladder's
    harder W8A8 first/last convention.
    """
    bit_list = sorted(bit_list)

    convs, bns, linears = [], [], []
    for name, mod in model.named_modules():
        for cname, child in list(mod.named_children()):
            if isinstance(child, nn.Conv2d):
                convs.append((mod, cname, child))
            elif isinstance(child, nn.BatchNorm2d):
                bns.append((mod, cname, child))
            elif isinstance(child, nn.Linear):
                linears.append((mod, cname, child))

    first_conv_id = id(convs[0][2])
    last_linear_id = id(linears[-1][2]) if linears else None

    for parent, cname, child in convs:
        if keep_first_last_fp and id(child) == first_conv_id:
            continue
        # The stem's consumer sees raw normalised pixels, which are not in
        # [0,1]; every other conv is fed by a ReLU6 output, which is.
        setattr(parent, cname, _swap_conv(child, bit_list, quantize_input=True))

    for parent, cname, child in bns:
        setattr(parent, cname, _swap_bn(child, bit_list))

    for parent, cname, child in linears:
        if keep_first_last_fp and id(child) == last_linear_id:
            continue
        setattr(parent, cname, _swap_linear(child, bit_list))

    model.bit_list = bit_list
    set_bits(model, bit_list[-1])
    return model


def switchable_parameters_by_bn(model: nn.Module):
    """Split params into (shared weights, per-bit BN params).

    Any-Precision applies weight decay to the shared weights only; the BN
    branches are excluded. `optimizer.py` in their repo does the same.
    """
    aux_ids = set()
    for m in model.modules():
        if isinstance(m, SwitchBatchNorm2d):
            aux_ids.update(id(p) for p in m.parameters())
        elif isinstance(m, SwitchConv2d):
            # Learned activation ranges belong with the BN params: they are
            # scale parameters, and weight-decaying them would drag the
            # quantization range toward zero.
            aux_ids.update(id(p) for p in m.act_lo.parameters())
            aux_ids.update(id(p) for p in m.act_hi.parameters())
    shared = [p for p in model.parameters() if id(p) not in aux_ids]
    aux = [p for p in model.parameters() if id(p) in aux_ids]
    return shared, aux
