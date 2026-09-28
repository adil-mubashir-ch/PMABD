"""
QAT-only — the no-distillation control for the baseline table.

WHY THIS ROW EXISTS
    Every other row in this table distils from something: SQAKD and DAQAKD
    from a frozen FP32 MobileNetV2, CMT-KD from a set of quantized teachers,
    the PMABD ladder from a growing pool. Without a row that distils from
    nothing, the table cannot answer the first question a reviewer will ask —
    how much of the accuracy at W4A4 or W3A3 comes from the *distillation*, and
    how much from the EWGS learned-range quantizer plus the training schedule
    doing the work on their own?

    This is that control: plain PACT/DoReFa-style quantization-aware training,
    supervised only by the labels.

        L = 1.0 * CE(student, y)          (kd_alpha = 0, no KL term)

    Read the table as: (QAT-only -> SQAKD) is what a single FP32 teacher buys,
    and (SQAKD -> PMABD) is what the ladder buys on top of that.

WHAT IS HELD FIXED
    Everything except the loss. Same EWGS quantizer, same FP32 initialisation
    (M_fp32.pth), same splits, same optimizer/lr/wd/batch size/cosine schedule,
    same grad clip 5.0, same first/last-layer convention, same BN recalibration
    before every evaluation. Only `kd_alpha` changes, from 100 to 0. That is
    what makes this a controlled ablation of distillation rather than a
    comparison of two loosely related recipes; see schedules._qat_only_paper.

NO TEACHER IS BUILT
    `builds_teacher = False`, so the second MobileNetV2 is never constructed,
    never occupies GPU memory, and is never forwarded. Multiplying a teacher
    forward by a zero weight would produce identical gradients but pay the full
    cost, and this row should be the cheapest in the table, not merely the
    least useful per FLOP.

NOT A REIMPLEMENTATION OF A PAPER
    Nothing here is being reproduced, so nothing here is a deviation. SQAKD's
    own comparison tables call this configuration PACT/DoReFa. Label the row
    "QAT only (no KD)" and state that it runs SQAKD's optimisation recipe, so
    that the difference against the SQAKD row is attributable to the objective
    alone.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .sqakd import SQAKD

__all__ = ["QATOnly"]


class QATOnly(SQAKD):
    """SQAKD's machinery with the KD term switched off. L = CE(student, y)."""

    name = "qat_only"
    builds_teacher = False

    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        # schedules._qat_only_paper already sets these; assert rather than
        # assume, because a row that silently kept alpha=100 would look like a
        # working control while actually being a second SQAKD run.
        if self.alpha != 0.0:
            raise ValueError(
                f"{self.name} must run with kd_alpha=0 (got {self.alpha}). "
                "It is the no-distillation control; a non-zero KD weight "
                "would need a teacher, which is deliberately not built.")
        if not self.gamma:
            raise ValueError(
                f"{self.name} must run with a non-zero kd_gamma (got "
                f"{self.gamma}) — cross-entropy is its only supervision, so a "
                "zero weight would leave the loss identically zero.")
        self.loss_weights = {"ce": self.gamma, "kd": 0.0, "T": self.T}

    def train_batch(self, x, y):
        # No teacher forward. `augment` stays inherited so that a DA variant of
        # this control could be added later without touching the loop.
        x = self.augment(x)
        s_logits = self.model(x)

        loss_ce = F.cross_entropy(s_logits, y)
        loss = self.gamma * loss_ce
        loss.backward()

        # loss_kd is reported as a literal 0.0 rather than omitted, so that
        # epochs.csv keeps the same columns as every other method and the
        # aggregation scripts need no special case for this row.
        return {"loss": loss.item(), "loss_ce": loss_ce.item(),
                "loss_kd": 0.0, "logits": s_logits.detach()}
