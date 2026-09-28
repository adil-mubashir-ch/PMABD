"""
CMT-KD — Collaborative Multi-Teacher Knowledge Distillation for Learning Low
Bit-width Deep Neural Networks (Pham, Hoang & Do, WACV 2023), reimplemented
for MobileNetV2/TinyImageNet.

NO CODE WAS RELEASED. Everything here is transcribed from the paper (arXiv
2210.16103), section 3 and the implementation details in 4.1:

  * Teachers share the student's architecture but are quantized to *higher*
    bit-widths. For a 3-bit student we use {4, 6, 8}, per "teachers are
    quantized with higher different bit-widths".
  * Collaborative learning (eq. 1): at selected layers k, the teachers'
    activations are fused into shared knowledge
        F_k = sum_i softmax(pi_k)_i * Q(A_k_i, b_i)
    and F_k is then fed as the input to the next layers of every teacher. The
    importance factors pi are learned end-to-end at lr/10.
  * Feature distillation (eq. 4): attention loss (eq. 5, p=2) between F_k and
    the student's feature map at the same depth.
  * Mutual learning (eqs. 7-9): KDCL-MinLogit ensemble z of teacher and student
    logits, then L_KL^S = T^2 KL(p || p_S) and L_KL^T = T^2 KL(p || p_T).
  * Total (eq. 10): L = alpha*(CE_S + CE_T) + beta*(KL_S + KL_T) + gamma*L_feat
    with alpha=1, beta=0.5, gamma=100 for the attention loss.
  * Teachers and student are trained *simultaneously* — teachers are not
    frozen. This is why CMT-KD is expensive: every step is n+1 forward and
    backward passes.
  * First and last layers are not quantized.

FROZEN TEACHERS — THE LARGEST DEVIATION, MUST BE LABELLED IN THE TABLE
    `freeze_teachers=True` (the default) trains ONLY the student. This was
    adopted for compute reasons: training three teachers alongside the student
    cost ~36-48 min/epoch, against ~17 min for the whole PMABD ladder stage,
    and CMT-KD alone would have taken ~46 h of a ~100 h budget.

    What it removes is not incidental — it is the paper's headline contribution:
      * Mutual learning (L_KL^T, eq. 9) — the student's feedback to the
        teachers. Gone entirely; teachers no longer adapt to the student.
      * Collaborative learning between teachers, in the sense of teachers
        *training* on the fused shared knowledge. The fusion itself still
        happens and still feeds the next teacher layers, but the teachers'
        weights never move.
      * The teacher-side cross entropy L_CE^T, which has no gradient path.

    What survives: importance-weighted fusion at the selected layers (pi is
    still learned, via the feature-distillation loss), attention-based feature
    distillation from the fused teacher knowledge, and the KDCL-MinLogit
    ensemble target for the student.

    So this row is "multi-teacher KD from FIXED quantized teachers", which sits
    closest to the paper's own *Average teacher* / *CMT-KD (w/o ML)* ablations
    rather than to full CMT-KD. In their Table 1 (AlexNet/CIFAR-100) the
    without-mutual-learning ablation scored 70.9 against 72.1 for the full
    method, so expect this to understate CMT-KD by roughly a point.

    DO NOT label this row "CMT-KD" unqualified. Call it "CMT-KD (frozen
    teachers)" and state the omission, or a reviewer who knows the paper will
    read it as a misrepresentation of a competitor.

TWO FURTHER DELIBERATE DEVIATIONS, both to be stated in the paper
  1. The paper quantizes with HWGQ; we use the same EWGS learned-range
     quantizer as the SQAKD/DAQAKD rows. Holding the quantizer fixed across
     every baseline means the table varies the *distillation strategy*, which
     is the thing under study. A HWGQ-vs-EWGS difference would otherwise be
     silently attributed to CMT-KD's method.
  2. The paper evaluates AlexNet and ResNet-18 on CIFAR-100/ImageNet, never
     MobileNetV2 or TinyImageNet. Fusion points therefore had to be chosen for
     MobileNetV2 (see FUSION_BLOCKS); the paper's rule is "the last
     convolutional layer of each convolution block", which we map onto the ends
     of MobileNetV2's stride groups.
  These make the row a faithful-as-possible reimplementation, not a
  reproduction. Label it as such.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.utils.checkpoint
import torch.nn.functional as F

from ..common.quant_sqakd import quantize_model, quant_range_parameters
from ..common.setup import build_fp32_mobilenetv2

__all__ = ["CMTKD"]


# Ends of MobileNetV2's stride groups: c=32, c=64, c=96, c=160.
# features has 19 entries (0 stem, 1..17 inverted residuals, 18 final 1x1).
FUSION_BLOCKS = (6, 10, 13, 16)


def attention_map(feat: torch.Tensor) -> torch.Tensor:
    """Q = sum_j |A_j|^p over channels, p=2, then L2-normalised (Zagoruyko)."""
    a = feat.pow(2).mean(dim=1).flatten(1)
    return F.normalize(a, dim=1)


def attention_loss(f_teacher: torch.Tensor, f_student: torch.Tensor) -> torch.Tensor:
    if f_teacher.shape[-2:] != f_student.shape[-2:]:
        f_student = F.adaptive_avg_pool2d(f_student, f_teacher.shape[-2:])
    return (attention_map(f_teacher) - attention_map(f_student)).pow(2).mean()


def kdcl_min_logit(z_t: torch.Tensor, z_s: torch.Tensor,
                   y: torch.Tensor) -> torch.Tensor:
    """KDCL-MinLogit ensemble (eq. 7).

    Translate each network's logits so the target class sits at zero, then take
    the element-wise minimum. Translating first is what makes the minimum
    meaningful across networks with different logit scales.
    """
    idx = y.view(-1, 1)
    z_t_c = z_t - z_t.gather(1, idx)
    z_s_c = z_s - z_s.gather(1, idx)
    return torch.minimum(z_t_c, z_s_c)


class _SegmentedMobileNetV2(nn.Module):
    """MobileNetV2 that can be run one stride-group at a time.

    Needed because CMT-KD's collaborative learning injects the fused shared
    knowledge back into every teacher partway through the forward pass, which a
    single monolithic forward() cannot express.
    """

    def __init__(self, model: nn.Module, fusion_blocks=FUSION_BLOCKS):
        super().__init__()
        self.model = model
        bounds = [0, *[b + 1 for b in fusion_blocks], len(model.features)]
        self.segments = list(zip(bounds[:-1], bounds[1:]))

    def run_segment(self, h: torch.Tensor, i: int,
                    checkpoint: bool = False) -> torch.Tensor:
        start, end = self.segments[i]

        def fn(t):
            for j in range(start, end):
                t = self.model.features[j](t)
            return t

        # Gradient checkpointing: discard this segment's intermediate
        # activations and recompute them in the backward pass.
        #
        # WHY IT IS REQUIRED, not an optimisation. CMT-KD trains the student
        # and every teacher simultaneously, so a step holds one full activation
        # graph per model. At W8A8 `teacher_bits` is [32] -> 2 models, which
        # fits 8 GB and ran at 16.5 min/epoch. At W4A4/W3A3 there are three
        # teachers -> 4 models, and VRAM hit 7956/8192 MiB. Windows' WDDM
        # driver does not raise OOM at that point; it silently pages to system
        # RAM, so GPU utilisation collapsed to 23% and an iteration went from
        # 0.18 s to 2-3 minutes — an epoch would have taken ~11 days.
        #
        # Checkpointing the teachers (3 of the 4 graphs) removes most of that
        # memory for ~30% recompute. Batch size stays at 16 so the W4A4/W3A3
        # rows remain comparable with the W8A8 row already recorded.
        if checkpoint and self.training and torch.is_grad_enabled():
            return torch.utils.checkpoint.checkpoint(fn, h, use_reentrant=False)
        return fn(h)

    def head(self, h: torch.Tensor) -> torch.Tensor:
        h = F.adaptive_avg_pool2d(h, 1).flatten(1)
        return self.model.classifier(h)

    @property
    def n_segments(self) -> int:
        return len(self.segments)


class _SegmentedCifarResNet(nn.Module):
    """A CIFAR ResNet run one stage at a time.

    Covers both the chenyaofo CifarResNet (ResNet-20/32/44/56: layer1-3) and a
    torchvision ResNet with the CIFAR stem (ResNet-18: layer1-4, maxpool =
    Identity). The paper's fusion rule, "the last convolutional layer of each
    convolution block", maps onto the stage ends; the last stage feeds the
    head directly, so fusion happens after every stage but the last -- the
    same interface as _SegmentedMobileNetV2.
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self._layers = [n for n in ("layer1", "layer2", "layer3", "layer4")
                        if hasattr(model, n)]

    def _stage(self, i: int, t: torch.Tensor) -> torch.Tensor:
        m = self.model
        if i == 0:
            t = m.relu(m.bn1(m.conv1(t)))
            if hasattr(m, "maxpool"):
                t = m.maxpool(t)
        return getattr(m, self._layers[i])(t)

    def run_segment(self, h: torch.Tensor, i: int,
                    checkpoint: bool = False) -> torch.Tensor:
        # See _SegmentedMobileNetV2.run_segment for why checkpointing exists.
        if checkpoint and self.training and torch.is_grad_enabled():
            return torch.utils.checkpoint.checkpoint(
                lambda t: self._stage(i, t), h, use_reentrant=False)
        return self._stage(i, h)

    def head(self, h: torch.Tensor) -> torch.Tensor:
        return self.model.fc(self.model.avgpool(h).flatten(1))

    @property
    def n_segments(self) -> int:
        return len(self._layers)


def _segmented(model: nn.Module) -> nn.Module:
    return (_SegmentedMobileNetV2(model) if hasattr(model, "features")
            else _SegmentedCifarResNet(model))


class CMTKD:
    name = "cmtkd"

    def __init__(self, cfg, sched, precision, w_bits, a_bits, device="cuda",
                 fp32_source="kd", first_last_bits=None, train_loader=None):
        self.device = device
        self.bitwidth = (w_bits, a_bits)
        self.precision = precision

        teacher_bits = sched.extra.get("teacher_bits", [4, 6, 8])
        self.alpha = sched.extra.get("alpha", 1.0)
        self.beta = sched.extra.get("beta", 0.5)
        self.gamma = sched.extra.get("feat_gamma", 100.0)
        self.T = sched.extra.get("kd_T", 4.0)
        self.loss_weights = {"ce": self.alpha, "kd": self.beta, "T": self.T}

        # Student and teachers all start from the same FP32 model, quantized to
        # their own bit-width. Teachers are trainable — that is the method.
        pcw = sched.extra.get("per_channel_w", False)
        student = quantize_model(
            build_fp32_mobilenetv2(cfg, fp32_source, device),
            w_bits, a_bits, first_last_bits=first_last_bits,
            per_channel_w=pcw)
        self.student = _segmented(student).to(device)

        self.teachers = nn.ModuleList([
            _segmented(quantize_model(
                build_fp32_mobilenetv2(cfg, fp32_source, device),
                b, b, first_last_bits=first_last_bits, per_channel_w=pcw))
            for b in teacher_bits
        ]).to(device)
        self.teacher_bits = list(teacher_bits)
        self.freeze_teachers = sched.extra.get("freeze_teachers", True)
        # Only needed once there is more than one teacher; with a single
        # teacher (W8A8) the 2-model graph fits comfortably and checkpointing
        # would just cost recompute for nothing.
        # Number of LEADING teacher segments to checkpoint. Early segments
        # hold the largest activations (112x112, 56x56), so checkpointing just
        # those recovers most of the memory for a fraction of the recompute.
        # 0 disables it; see the measurements in run_segment's comment.
        # Measured on the RTX 3070 Ti (8 GiB), W4A4, bs16, 3 teachers:
        # W4A4, 3 models, bs32:
        #   segments  s/iter  peak alloc  min/epoch
        #        2     4.589    8.82 GiB      215    <- spills to system RAM
        #        3     1.142    7.53 GiB       54    <- still over the practical
        #                                               ceiling (~7 GiB)
        #        5     0.773    6.61 GiB       36    <- all segments, chosen
        # Counter-intuitively, checkpointing MORE is faster here: the recompute
        # is cheaper than paging to system RAM. Windows does not raise OOM when
        # VRAM runs out, it silently pages and costs 4-6x, so the safe side of
        # this trade is "checkpoint everything".
        # Only needed when teachers build activation graphs. Frozen teachers
        # run under no_grad, so there is nothing to checkpoint.
        self.checkpoint_segments = (
            0 if (self.freeze_teachers or len(self.teacher_bits) <= 1)
            else 10 ** 9)

        # Importance factors pi: one logit per teacher per fusion point,
        # softmaxed over teachers (eq. 1). Learned at lr/10 via aux_parameters.
        n_fusion = self.student.n_segments - 1
        self.pi = nn.Parameter(
            torch.zeros(n_fusion, len(teacher_bits), device=device))

        # `model` is what the engine checkpoints and evaluates: the student.
        self.model = student
        self._modules_for_params = nn.ModuleList([self.student, self.teachers])

        if train_loader is not None:
            from ..common.quant_sqakd import init_quant_ranges
            init_quant_ranges(self.student.model, train_loader, device, batches=2)
            for t in self.teachers:
                init_quant_ranges(t.model, train_loader, device, batches=2)

        # Freezing must come AFTER init_quant_ranges: the learned clipping
        # bounds initialise lazily on the first forward and only in train mode.
        if self.freeze_teachers:
            self.teachers.eval()
            for prm in self.teachers.parameters():
                prm.requires_grad_(False)

    # ── engine interface ────────────────────────────────────────────────
    def parameters(self):
        if self.freeze_teachers:
            yield from self.student.parameters()
        else:
            yield from self._modules_for_params.parameters()
        yield self.pi

    def aux_parameters(self):
        """pi and the learned quantization ranges — the paper trains pi at lr/10."""
        params = [self.pi]
        params += quant_range_parameters(self.student.model)
        if not self.freeze_teachers:
            for t in self.teachers:
                params += quant_range_parameters(t.model)
        return params

    def eval_targets(self):
        return [(self.precision, None)]

    # ── mid-stage resume ────────────────────────────────────────────────
    # CMT-KD trains the teachers and the importance factors alongside the
    # student, so the engine's default snapshot (`method.model` only) would
    # silently reset them on resume and the run would no longer be equivalent
    # to an uninterrupted one.
    def state_dict(self) -> dict:
        return {
            "model": self.model.state_dict(),
            "teachers": self.teachers.state_dict(),
            "pi": self.pi.detach().cpu(),
        }

    def load_state_dict(self, state: dict):
        self.model.load_state_dict(state["model"])
        self.teachers.load_state_dict(state["teachers"])
        with torch.no_grad():
            self.pi.copy_(state["pi"].to(self.pi.device))

    def train_batch(self, x, y):
        if not self.freeze_teachers and not self.teachers.training:
            # The engine only knows about `self.model` (the student), so in the
            # full method the teachers must be put in train mode here.
            self.teachers.train()
        n_seg = self.student.n_segments

        # ── forward the teachers with fusion between segments ────────────
        h_t = [x for _ in self.teachers]
        h_s = x
        feat_shared, feat_student = [], []

        for seg in range(n_seg):
            if self.freeze_teachers:
                # No graph is built through the teachers at all: this is where
                # nearly all the saving comes from, since their backward pass
                # was 2/3 of the step.
                with torch.no_grad():
                    h_t = [t.run_segment(h, seg)
                           for t, h in zip(self.teachers, h_t)]
            else:
                h_t = [t.run_segment(h, seg,
                                     checkpoint=seg < self.checkpoint_segments)
                       for t, h in zip(self.teachers, h_t)]
            h_s = self.student.run_segment(h_s, seg)

            if seg < n_seg - 1:
                # Shared knowledge F_k = sum_i softmax(pi_k)_i * A_k_i (eq. 1).
                # With frozen teachers the h_t tensors carry no grad, so `fused`
                # is differentiable w.r.t. pi alone — the importance factors
                # stay learnable (via the feature loss below) at negligible
                # cost, which keeps CMT-KD's importance-aware fusion intact.
                w = F.softmax(self.pi[seg], dim=0)
                fused = sum(w[i] * h_t[i] for i in range(len(h_t)))
                feat_shared.append(fused)
                feat_student.append(h_s)
                # ...which becomes the input to the next layers of every teacher.
                h_t = [fused.detach() if self.freeze_teachers else fused
                       for _ in h_t]

        if self.freeze_teachers:
            with torch.no_grad():
                z_t_each = [t.head(h) for t, h in zip(self.teachers, h_t)]
        else:
            z_t_each = [t.head(h) for t, h in zip(self.teachers, h_t)]
        z_s = self.student.head(h_s)
        # After the final fusion every teacher shares an input, so the natural
        # "combined teacher" logit is the mean over the teacher heads.
        z_t = torch.stack(z_t_each, 0).mean(0)

        # ── mutual learning via the KDCL-MinLogit ensemble ───────────────
        z = kdcl_min_logit(z_t.detach(), z_s.detach(), y)
        p = F.softmax(z / self.T, dim=1)
        loss_kl_s = F.kl_div(F.log_softmax(z_s / self.T, dim=1), p,
                             reduction="batchmean") * (self.T ** 2)
        # Mutual learning back into the teachers (L_KL^T) and the teacher-side
        # cross entropy (L_CE^T) only exist to TRAIN the teachers. With frozen
        # teachers they have no gradient path and are dropped.
        if self.freeze_teachers:
            loss_kl_t = torch.zeros((), device=z_s.device)
            loss_ce_t = torch.zeros((), device=z_s.device)
        else:
            loss_kl_t = F.kl_div(F.log_softmax(z_t / self.T, dim=1), p,
                                 reduction="batchmean") * (self.T ** 2)
            loss_ce_t = F.cross_entropy(z_t, y)

        loss_ce_s = F.cross_entropy(z_s, y)

        # ── intermediate feature distillation (attention loss) ───────────
        # The teacher side is detached only when the teachers are trainable;
        # with frozen teachers `fused` carries gradient for pi alone, so
        # detaching it would silently make the importance factors dead weights.
        loss_feat = sum(
            attention_loss(ft if self.freeze_teachers else ft.detach(), fs)
            for ft, fs in zip(feat_shared, feat_student)
        ) if feat_shared else torch.zeros((), device=z_s.device)

        loss = (self.alpha * (loss_ce_s + loss_ce_t)
                + self.beta * (loss_kl_s + loss_kl_t)
                + self.gamma * loss_feat)
        loss.backward()

        return {"loss": loss.item(),
                "loss_ce": (loss_ce_s + loss_ce_t).item(),
                "loss_kd": (loss_kl_s + loss_kl_t).item(),
                "loss_feat": float(loss_feat.detach()),
                "logits": z_s.detach()}
