"""
SQAKD (AISTATS 2024) and DAQAKD (arXiv 2509.03850) on MobileNetV2/TinyImageNet.

SQAKD
    Quantization-aware training where the student's *only* supervision is the
    KL divergence to a frozen full-precision teacher of the same architecture.
    The published TinyImageNet run names encode the loss weights directly:
        "..._kd_gamma0_alpha100_..."  ->  L = 0*CE + 100*KL
    gamma=0 is the whole point — "self-supervised" here means no labels, not
    contrastive pretraining.

DAQAKD
    SQAKD plus a data-augmentation policy chosen to maximise "Contextual Mutual
    Information". No code was released, so this is a reimplementation; see the
    honesty note on DAQAKD below for exactly what is and is not reproduced.

Both initialise the student from the full-precision model and quantize with
the EWGS learned-range quantizer ported in common/quant_sqakd.py. Neither
quantizes the first conv or the final classifier — that is their convention,
and it is *easier* than the PMABD ladder, which holds both at W8A8. Pass
--first-last-bits 8,8 to remove that asymmetry.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..common.quant_sqakd import (
    quantize_model, init_quant_ranges, quant_range_parameters,
)
from ..common.setup import build_fp32_mobilenetv2, freeze

__all__ = ["SQAKD", "DAQAKD"]


class SQAKD:
    """L = kd_gamma * CE(student, y) + kd_alpha * T^2 * KL(student || teacher)."""

    name = "sqakd"

    # Subclasses that do not distil (see methods/qat_only.py) set this False so
    # the teacher is never built. It is a flag rather than an override because
    # the teacher is a second full MobileNetV2 held on the GPU for the whole
    # run and forwarded on every batch: skipping it is worth real time and
    # memory, and duplicating this __init__ to achieve that would let the two
    # copies drift apart, which is exactly what the control row must not do.
    builds_teacher = True

    def __init__(self, cfg, sched, precision, w_bits, a_bits, device="cuda",
                 fp32_source="kd", first_last_bits=None, train_loader=None):
        self.device = device
        self.bitwidth = (w_bits, a_bits)
        self.precision = precision

        # Teacher: the frozen FP32 MobileNetV2. Same weights the student starts
        # from, which is exactly what the authors do (--load_pretrain pointing
        # at the fp checkpoint, --teacher_arch the same architecture).
        self.teacher = (freeze(build_fp32_mobilenetv2(cfg, fp32_source, device))
                        if self.builds_teacher else None)

        student = build_fp32_mobilenetv2(cfg, fp32_source, device)
        self.model = quantize_model(
            student, w_bits, a_bits, first_last_bits=first_last_bits,
            ewgs=sched.extra.get("ewgs", True),
            # Per-channel weight ranges match the PMABD ladder's quantizer.
            # Off unless asked for, so recorded runs keep their meaning.
            per_channel_w=sched.extra.get("per_channel_w", False))
        self.model.to(device)

        # Fire every layer's lazy range initialisation before the first real
        # optimizer step, otherwise step 1 both initialises and updates the
        # clipping bounds and the loss frequently blows up.
        if train_loader is not None:
            init_quant_ranges(self.model, train_loader, device, batches=2)

        self.gamma = sched.extra.get("kd_gamma", 0.0)   # CE weight
        self.alpha = sched.extra.get("kd_alpha", 100.0)  # KL weight
        self.T = sched.extra.get("kd_T", 4.0)
        self.loss_weights = {"ce": self.gamma, "kd": self.alpha, "T": self.T}

    # ── engine interface ────────────────────────────────────────────────
    def parameters(self):
        return self.model.parameters()

    def aux_parameters(self):
        return quant_range_parameters(self.model)

    def eval_targets(self):
        return [(self.precision, None)]

    def augment(self, x):
        """Identity for SQAKD; DAQAKD overrides."""
        return x

    def train_batch(self, x, y):
        x = self.augment(x)

        with torch.no_grad():
            t_logits = self.teacher(x)
        s_logits = self.model(x)

        loss_kd = F.kl_div(
            F.log_softmax(s_logits / self.T, dim=1),
            F.softmax(t_logits / self.T, dim=1),
            reduction="batchmean",
        ) * (self.T ** 2)

        if self.gamma:
            loss_ce = F.cross_entropy(s_logits, y)
        else:
            loss_ce = torch.zeros((), device=s_logits.device)

        loss = self.gamma * loss_ce + self.alpha * loss_kd
        loss.backward()

        return {"loss": loss.item(), "loss_ce": loss_ce.item(),
                "loss_kd": loss_kd.item(), "logits": s_logits.detach()}


class DAQAKD(SQAKD):
    """SQAKD + a stronger augmentation applied to the KD pair.

    WHAT IS REPRODUCED
        The training objective (identical to SQAKD) and the use of a strong
        RandAugment-style policy applied to the *shared* input, so teacher and
        student see the same augmented view and the KL target stays coherent.

    WHAT IS NOT REPRODUCED — say this in the paper
        DAQAKD's actual contribution is a *selection metric*: it scores
        candidate augmentation policies by Contextual Mutual Information (the
        information not directly tied to the label) subject to per-class
        predictions staying close to the ground truth on average, then trains
        with the winner. No code was released, and rerunning that search would
        be a project in itself. What runs here is SQAKD under a fixed strong
        policy (RandAugment n=2, m=9), which is the family the search selects
        from, not the searched-for optimum.

        So treat this row as "SQAKD + strong DA", a lower bound on DAQAKD,
        and label it that way in the table. If it already beats PMABD, the
        lower bound is enough to matter; if it does not, the gap to the real
        method is genuinely unknown and must not be claimed as a win.
    """

    name = "daqakd"

    def __init__(self, cfg, *args, **kw):
        super().__init__(cfg, *args, **kw)
        # torchvision's RandAugment works on uint8 tensors, but our
        # loader already normalises to float. Applying it in normalised space
        # would corrupt the colour ops, so the policy is applied to a
        # de-normalised copy and re-normalised afterwards. The statistics must
        # be the loader's own: ImageNet's on TinyImageNet, CIFAR's on CIFAR.
        from torchvision.transforms import RandAugment
        from ..common.setup import dataset_norm
        self._randaug = RandAugment(num_ops=2, magnitude=9)
        m, s = dataset_norm(cfg)
        mean = torch.tensor(m, device=self.device).view(1, 3, 1, 1)
        std = torch.tensor(s, device=self.device).view(1, 3, 1, 1)
        self._mean, self._std = mean, std

    def augment(self, x):
        with torch.no_grad():
            img = (x * self._std + self._mean).clamp(0, 1)
            img = (img * 255).to(torch.uint8)
            img = self._randaug(img)
            img = img.to(x.dtype) / 255.0
            return (img - self._mean) / self._std
