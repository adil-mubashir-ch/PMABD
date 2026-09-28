"""
Any-Precision DNNs (AAAI 2021) and InstantNet Cascade Distillation Training
(DAC 2021), on MobileNetV2/TinyImageNet.

Both train ONE shared-weight network that serves every bit-width in `bit_list`
at once, with a separate BatchNorm per precision. One run therefore produces
W8A8, W4A4 and W3A3 together — which is why they are cheap, and why their
GPU-hour figure must be reported both raw and divided by precisions served.

WHY INSTANTNET IS THE ONE TO WATCH
    InstantNet's CDT makes every bit-width distil from *all higher* bit-widths.
    That is structurally the same idea as the PMABD ladder, where each finished
    rung joins the teacher pool for later rungs. The differences that remain
    are worth being precise about in the paper, because they are the
    contribution:
      * CDT shares one weight tensor across precisions; PMABD trains a separate
        model per rung, so a rung can specialise instead of compromising.
      * CDT distils with MSE on logits; PMABD uses entropy-weighted KL over a
        pool that also contains different architectures (the ResNet teachers).
      * CDT's precisions are symmetric (b bits for both W and A); the ladder
        passes through asymmetric rungs (W8A4, W4A3) that CDT cannot express.
    None of those is self-evidently decisive. That is what the experiment is
    for.

    The NAS half of InstantNet is not used here. We run CDT on a fixed
    MobileNetV2, which is what their own CDT ablation does.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..common.quant_switchable import (
    switchable_mobilenetv2, set_bits, switchable_parameters_by_bn,
)
from ..common.setup import build_fp32_mobilenetv2

__all__ = ["AnyPrecision", "InstantNetCDT"]


class _Switchable:
    """Shared plumbing for the two shared-weight methods."""

    def __init__(self, cfg, sched, precision, w_bits, a_bits, device="cuda",
                 fp32_source="kd", first_last_bits=None, train_loader=None):
        if w_bits != a_bits:
            raise ValueError(
                f"{self.name} trains one bit-width for weights and activations "
                f"together, so it cannot express {precision}. Asymmetric rungs "
                "(W8A4, W4A3) are outside this method's family — report them as "
                "not applicable rather than fabricating a number."
            )
        self.device = device
        self.bitwidth = (w_bits, a_bits)
        self.precision = precision

        self.bit_list = sorted(sched.extra.get("bit_list", [3, 4, 8, 32]))
        if w_bits not in self.bit_list:
            self.bit_list = sorted(set(self.bit_list) | {w_bits})
        self.full_bits = self.bit_list[-1]

        model = build_fp32_mobilenetv2(cfg, fp32_source, device)
        self.model = switchable_mobilenetv2(
            model, self.bit_list,
            keep_first_last_fp=(first_last_bits is None),
        ).to(device)

        self.loss_weights = {"ce": 1.0, "kd": None, "T": None}

    def parameters(self):
        return self.model.parameters()

    def aux_parameters(self):
        # Per-bit BN params are excluded from weight decay, matching the
        # authors' optimizer.py; the engine gives aux params weight_decay=0.
        _shared, bn_params = switchable_parameters_by_bn(self.model)
        return bn_params

    def eval_targets(self):
        """Primary precision first, then every other symmetric rung this run serves.

        The engine evaluates all of them each epoch and stores them in
        stages.json, so one run fills several cells of the results table.
        """
        targets = [(self.precision, self.bitwidth[0])]
        for b in sorted(self.bit_list, reverse=True):
            if b == self.bitwidth[0]:
                continue
            label = "fp32" if b == 32 else f"w{b}a{b}"
            targets.append((label, b))
        return targets


class AnyPrecision(_Switchable):
    """Recursive supervision: each precision is taught by the next one up.

    Authors' train.py: the highest bit-width trains on hard labels, then for
    each lower bit-width in descending order the loss is a soft cross-entropy
    against the *previous* (one step higher) precision's softmax. The teacher
    is re-assigned as the loop descends, hence "recursive".
    """

    name = "any_precision"

    def train_batch(self, x, y):
        losses = {}
        total = 0.0

        # Highest precision: hard labels.
        set_bits(self.model, self.full_bits)
        logits_full = self.model(x)
        loss_full = F.cross_entropy(logits_full, y)
        loss_full.backward()
        total += loss_full.item()
        losses["loss_ce"] = loss_full.item()

        target_soft = F.softmax(logits_full.detach(), dim=1)
        primary_logits = logits_full if self.full_bits == self.bitwidth[0] else None

        kd_acc = 0.0
        for b in sorted(self.bit_list, reverse=True)[1:]:
            set_bits(self.model, b)
            logits = self.model(x)
            # Soft cross-entropy, the authors' CrossEntropyLossSoft.
            loss = -(target_soft * F.log_softmax(logits, dim=1)).sum(1).mean()
            loss.backward()
            total += loss.item()
            kd_acc += loss.item()
            target_soft = F.softmax(logits.detach(), dim=1)
            if b == self.bitwidth[0]:
                primary_logits = logits.detach()

        losses.update({"loss": total, "loss_kd": kd_acc,
                       "logits": primary_logits})
        return losses


class InstantNetCDT(_Switchable):
    """Cascade Distillation Training: every precision distils from all higher ones.

    Authors' train.py, bit_schedule='avg_loss', cascad=True:

        teacher_list = []
        for bits in bit_list[::-1]:            # high -> low
            logit = model(input, bits)
            loss  = CE(logit, target)
            for t in teacher_list:
                loss += distill_weight * MSE(logit, t)
            teacher_list.append(logit.detach())
            loss.backward()

    Note the CE term is kept at *every* precision (unlike Any-Precision, where
    only the top precision sees labels), and the distillation distance is MSE
    on raw logits rather than KL on softened ones.
    """

    name = "instantnet_cdt"

    def __init__(self, cfg, sched, *args, **kw):
        super().__init__(cfg, sched, *args, **kw)
        self.distill_weight = sched.extra.get("distill_weight", 1.0)
        self.loss_weights = {"ce": 1.0, "kd": self.distill_weight, "T": None}

    def train_batch(self, x, y):
        teacher_logits = []
        total, ce_acc, kd_acc = 0.0, 0.0, 0.0
        primary_logits = None

        for b in sorted(self.bit_list, reverse=True):
            set_bits(self.model, b)
            logits = self.model(x)
            loss_ce = F.cross_entropy(logits, y)
            loss = loss_ce
            for t in teacher_logits:
                loss = loss + self.distill_weight * F.mse_loss(logits, t)
            teacher_logits.append(logits.detach())

            loss.backward()
            total += loss.item()
            ce_acc += loss_ce.item()
            kd_acc += loss.item() - loss_ce.item()
            if b == self.bitwidth[0]:
                primary_logits = logits.detach()

        return {"loss": total, "loss_ce": ce_acc, "loss_kd": kd_acc,
                "logits": primary_logits}
