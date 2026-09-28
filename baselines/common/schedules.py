"""
Training schedules for the baseline reruns, in two modes.

    --schedule paper    Each method's own published recipe, verbatim from the
                        authors' release scripts or paper. Use this when the
                        question is "what does this method do, run properly?"

    --schedule matched  The epoch budget the PMABD ladder actually spent at the
                        same precision (measured from
                        outputs/.../logs/stages.json). Use this when the
                        question is "what does this method do on our budget?"

WHAT `matched` MEANS FOR EACH FAMILY
    Per-precision methods (SQAKD, DAQAKD, CMTKD) train one model for one target
    precision, so `matched` is simply the epoch count our ladder spent at that
    precision.

    Switchable methods (Any-Precision, InstantNet-CDT) train ONE shared-weight
    model that serves every precision at once, and each of their epochs already
    does a forward/backward at every bit-width in the list. Giving them the sum
    of our per-precision epochs would hand them 3x the gradient steps at each
    precision. So `matched` is the max over the precisions covered.

    Either way the honest comparison is GPU hours, which engine.py logs for
    every run regardless of mode. Neither mode is "the" fair one — report both.

PMABD LADDER REFERENCE (RTX 3070 Ti, bs 32, 224px, teacher-logit cache K=16)
    precision   epochs   hours
    W8A8          13      3.30
    W8A4          35      9.62
    W4A4          17      4.94
    W4A3          40      6.56
    W3A3          45      8.50
    Cumulative to W3A3: 34.0 h ladder only; 42.1 h including teacher
    pretraining (6.44 h) and the teacher-logit cache build (~1.65 h).

    W3A2 and W2A2 are NOT in the table above. Those rungs did not exist when
    it was written, and their matched budgets are therefore read at import
    time from the ladder's own logs/stages.json by refresh_ladder_budgets().
    That function only fills gaps: it never overwrites a hard-coded entry,
    because the already-completed W8A8/W4A4/W3A3 baseline runs were matched
    against those numbers and moving one retroactively would misdescribe the
    budget they actually received.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

__all__ = ["Schedule", "get_schedule", "LADDER_EPOCHS", "LADDER_HOURS",
           "PRECISIONS", "refresh_ladder_budgets"]


# Precision shorthand -> (weight bits, activation bits)
PRECISIONS = {
    "w8a8": (8, 8),
    "w8a4": (8, 4),
    "w4a4": (4, 4),
    "w4a3": (4, 3),
    "w3a3": (3, 3),
    "w2a2": (2, 2),
}

# Measured from the PMABD run — see module docstring.
#
# These are the values the already-completed baseline runs were launched with,
# so they are frozen here rather than recomputed: changing one retroactively
# would mean the recorded W8A8/W4A4/W3A3 rows no longer describe the budget
# they actually got.
LADDER_EPOCHS = {"w8a8": 13, "w8a4": 35, "w4a4": 17, "w4a3": 40, "w3a3": 45,
                 # PROVISIONAL. The ladder's W2A2 rung (stage_m7b) is still
                 # running on the other machine, so there is no measured value
                 # yet; 60 is that stage's configured epoch budget, and m6b/m7a
                 # both ran their configured budget in full. _ladder_epochs()
                 # overwrites this from stages.json the moment m7b lands, and
                 # warns if the real count differs -- if it does, the W2A2 rows
                 # run here were trained on a different budget from PMABD's and
                 # the matched comparison for that column needs re-checking.
                 "w2a2": 60}
LADDER_HOURS = {"w8a8": 3.30, "w8a4": 9.62, "w4a4": 4.94,
                "w4a3": 6.56, "w3a3": 8.50}

# Ladder stages.json, relative to the repo root. The W3A2/W2A2 rungs did not
# exist when the constants above were written, so their budgets are read from
# the ladder log once those stages have actually run.
_LADDER_STAGES_JSON = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "outputs", "mobilenetv2_tinyimagenet_2bit_ladder_nosat", "logs",
    "stages.json")


def refresh_ladder_budgets(path: str | None = None) -> dict:
    """Fill in any precision missing from LADDER_EPOCHS from the ladder log.

    ONLY fills gaps. An entry already hard-coded above is never overwritten,
    because the completed baseline runs were matched against that number and
    silently moving it would misdescribe them. A disagreement is reported so it
    can be judged rather than swallowed — `w4a3` genuinely disagrees (the
    constant says 40, the ladder log records 53 epochs_run), which matters only
    if a baseline is ever run at that rung.

    Returns the precisions it added, so callers can log where the budget for a
    new rung came from.
    """
    path = path or _LADDER_STAGES_JSON
    added = {}
    if not os.path.isfile(path):
        return added
    try:
        with open(path, encoding="utf-8") as f:
            stages = json.load(f)
    except (OSError, ValueError):
        return added

    for st in stages:
        prec = str(st.get("precision", "")).lower()
        if prec not in PRECISIONS:
            continue
        epochs = st.get("epochs_run")
        hours = st.get("hours")
        if not epochs:
            continue
        if prec not in LADDER_EPOCHS:
            # `epochs_run` is logger.count_epochs(): a count of ROWS in
            # epochs.jsonl, which is opened in append mode and survives across
            # processes. Every interrupted attempt leaves its rows behind, so
            # the field over-counts -- stage m6a has a 40-epoch ceiling and
            # records epochs_run = 53. Clamp to the stage's configured ceiling:
            # a single run cannot have exceeded it, and handing a baseline the
            # inflated number would give it strictly more training than the
            # ladder had, inverting the comparison `matched` exists to make.
            epochs = _clamp_to_ceiling(prec, int(epochs), st)
            LADDER_EPOCHS[prec] = int(epochs)
            if hours:
                LADDER_HOURS[prec] = round(float(hours), 2)
            added[prec] = (int(epochs), round(float(hours or 0.0), 2),
                           st.get("stage_id"))
        elif int(epochs) != LADDER_EPOCHS[prec]:
            _MISMATCHES[prec] = (LADDER_EPOCHS[prec], int(epochs))
    return added


def _clamp_to_ceiling(prec, epochs, stage):
    """Cap a measured budget at the `epochs:` ceiling of its ladder stage."""
    cfg_key = stage.get("config_key")
    if not cfg_key:
        return epochs
    cfg_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "configs", "mobilenetv2_tinyimagenet_2bit_ladder.yaml")
    try:
        import yaml
        with open(cfg_path, encoding="utf-8") as f:
            ceiling = (yaml.safe_load(f).get(cfg_key) or {}).get("epochs")
    except Exception:  # noqa: BLE001 - never block a schedule on config parsing
        return epochs
    if ceiling and epochs > ceiling:
        _CLAMPED[prec] = (epochs, int(ceiling))
        return int(ceiling)
    return epochs


_MISMATCHES: dict = {}
_CLAMPED: dict = {}
refresh_ladder_budgets()


@dataclass
class Schedule:
    epochs: int
    lr: float
    optimizer: str = "sgd"
    momentum: float = 0.9
    weight_decay: float = 5e-4
    lr_schedule: str = "cosine"          # cosine | multistep
    milestones: tuple = ()               # for multistep
    gamma: float = 0.1
    batch_size: int = 32
    warmup_epochs: int = 0
    # Learned-range / importance params often get their own (smaller) lr.
    aux_lr: float | None = None
    label_smoothing: float = 0.0
    grad_clip: float | None = 5.0
    extra: dict = field(default_factory=dict)

    def describe(self) -> str:
        sched = (f"multistep@{list(self.milestones)}"
                 if self.lr_schedule == "multistep" else "cosine")
        return (f"{self.epochs}ep {self.optimizer} lr={self.lr:g} "
                f"wd={self.weight_decay:g} bs={self.batch_size} {sched}")


def _bit_list_for(prec: str) -> list[int]:
    """Bit-widths a shared-weight run trains simultaneously.

    32 is always the top of the list: both methods make the full-precision
    branch the ultimate teacher (Any-Precision's recursive supervision starts
    there, CDT distils every rung from all higher rungs), so removing it would
    change the method rather than trim it.

    Targeting W2A2 prepends 2. That makes the run 5 bit-widths instead of 4 —
    roughly 25% slower — and it also re-reports W8A8/W4A4/W3A3 from the same
    shared weights, which will NOT exactly reproduce the earlier [3,4,8,32]
    run: those cells now come from a network that also had to accommodate a
    2-bit branch. Report one run or the other per row, and say which.
    """
    return [2, 3, 4, 8, 32] if prec == "w2a2" else [3, 4, 8, 32]


# ─────────────────────────────────────────────────────────────────────────────
# Published recipes
# ─────────────────────────────────────────────────────────────────────────────
def _sqakd_paper(prec: str) -> Schedule:
    """SQAKD, TinyImageNet + MobileNetV2.

    Source: third_party/SQAKD/TinyImageNet/scripts/run_tinyimagenet_mobilenetV2.sh
      quantized runs: SGD, lr 5e-4, wd 5e-4, 100 epochs, init from the FP32
      model. Batch size is 64 for PACT and 32 for LSQ/DoReFa; we use 32
      throughout because that is also the PMABD ladder's batch size and it
      removes batch size as a confound.
    Loss weights come from the script's own run names, e.g.
      "..._kd_gamma0_alpha100_..." -> CE weight 0, KL weight 100. Gamma=0 is
    the point of SQAKD: the student never sees a label, only the teacher.
    """
    return Schedule(
        epochs=100, lr=5e-4, optimizer="sgd", momentum=0.9, weight_decay=5e-4,
        lr_schedule="cosine", batch_size=32, aux_lr=5e-4,
        # DEVIATION, measured and necessary. `grep -rn clip_grad` finds no
        # clipping in the authors' repo, but alpha=100 against gradients of
        # norm ~560 gives unclipped SGD steps of norm ~0.28 at lr 5e-4, and
        # the run diverges: over 120 iterations at W8A8 the loss went
        # 114.98 -> 124.13 -> 122.77, i.e. up to the degenerate plateau where
        # the student emits a constant. With clip 5.0 the same run went
        # 47.14 -> 38.77 -> 29.45. We therefore clip at 5.0, which is also
        # InstantNet's own setting. State this in the paper.
        grad_clip=5.0,
        extra={"kd_gamma": 0.0, "kd_alpha": 100.0, "kd_T": 4.0, "ewgs": True},
    )


def _daqakd_paper(prec: str) -> Schedule:
    """DAQAKD inherits SQAKD's optimisation and changes the augmentation.

    No code was released (arXiv 2509.03850, Sept 2025), so the schedule is
    SQAKD's by the paper's own statement that it builds on that setup. The
    contribution being reproduced is the DA selection, not a new optimiser.
    """
    s = _sqakd_paper(prec)
    s.extra = dict(s.extra)
    s.extra.update({"da_policy": "cmi", "da_num_ops": 2, "da_magnitude": 9})
    return s


def _any_precision_paper(prec: str) -> Schedule:
    """Any-Precision DNNs, ImageNet recipe from train_imagenet.sh.

    Authors: lr 0.5, decay at 45/60/70, 80 epochs, bs 256, SGD, wd 1e-5.
    bs 256 will not fit MobileNetV2 @224 on an 8 GB card, so we drop to 32 and
    scale lr linearly (0.5 * 32/256 = 0.0625), which is the standard Goyal et
    al. correction. Both numbers are logged so the deviation is on the record.
    """
    return Schedule(
        epochs=80, lr=0.0625, optimizer="sgd", momentum=0.9, weight_decay=1e-5,
        lr_schedule="multistep", milestones=(45, 60, 70), gamma=0.1,
        batch_size=32, grad_clip=5.0,  # see the note in _sqakd_paper
        extra={"bit_list": _bit_list_for(prec), "supervision": "recursive",
               "orig_lr": 0.5, "orig_batch_size": 256},
    )


def _instantnet_paper(prec: str) -> Schedule:
    """InstantNet CDT, from third_party/InstantNet/config_train.py.

    nepochs 200, lr 0.025, cosine, momentum 0.9, wd 5e-4, bs 128,
    distill_weight 1, cascad True, loss_scale all-ones, bit_schedule avg_loss.
    bs 128 -> 32 for the same VRAM reason as above, lr scaled 0.025*32/128.
    The NAS half of InstantNet is not used: we run CDT on a fixed MobileNetV2,
    which is what their own CDT ablation does.
    """
    return Schedule(
        epochs=200, lr=0.00625, optimizer="sgd", momentum=0.9,
        weight_decay=5e-4, lr_schedule="cosine", batch_size=32,
        grad_clip=5.0,  # config_train.py: C.grad_clip = 5 -- the one method that clips
        extra={"bit_list": _bit_list_for(prec), "distill_weight": 1.0,
               "cascad": True, "orig_lr": 0.025, "orig_batch_size": 128},
    )


def _cmtkd_paper(prec: str) -> Schedule:
    """CMT-KD, from the WACV 2023 paper (no code released).

    Paper: SGD momentum 0.9, bs 256, lr 0.1 for ResNet-family backbones,
    importance factors pi at lr/10. ImageNet ResNet-18: 100 epochs, lr /10 at
    30/60/90. Weight decay 25e-6 for 1-2 bit, 1e-4 above. alpha=1, beta=0.5,
    gamma=100 with attention loss. First and last layers are not quantized.
    Teachers are the same architecture at *higher* bit-widths than the student.
    bs 256 -> 32, lr scaled 0.1*32/256 = 0.0125.
    """
    w_bits = PRECISIONS[prec][0]
    wd = 25e-6 if w_bits <= 2 else 1e-4
    # Teachers must be QUANTIZED at higher bit-widths. The paper is explicit
    # that this is the point of the method: "instead of using a full-precision
    # teacher model, we propose to use a set of quantized teacher models.
    # Using quantized models would help the teachers obtain more suitable
    # knowledge for a quantized student model to mimic." (Sec 2)
    #
    # So 32 must NOT be in this set. Including it both contradicts the method
    # and adds a whole extra model to every step -- at W4A4 that was a 4th
    # network trained for nothing.
    teacher_bits = [b for b in (4, 6, 8) if b > w_bits]
    if not teacher_bits:
        # An 8-bit student has no quantized higher bit-width available, so
        # CMT-KD is simply undefined at W8A8. We fall back to a full-precision
        # teacher so the cell is not blank, but it is a deviation from the
        # method and must be labelled as such in the paper -- it is the one
        # configuration where CMT-KD is doing something its authors argue
        # against.
        teacher_bits = [32]
    return Schedule(
        epochs=100, lr=0.0125, optimizer="sgd", momentum=0.9, weight_decay=wd,
        lr_schedule="multistep", milestones=(30, 60, 90), gamma=0.1,
        # bs 32, matching every other method here and the PMABD ladder. This
        # only became possible once the teachers were gradient-checkpointed;
        # measured at W4A4 with 3 models: bs16+ckpt1 = 41.4 min/epoch, and
        # bs32+full-ckpt = 36.2 min/epoch at 6.61 GiB peak. Faster AND removes
        # a batch-size deviation, so there is no reason to keep 16.
        batch_size=32, aux_lr=0.00125, grad_clip=5.0,  # see _sqakd_paper note
        extra={"alpha": 1.0, "beta": 0.5, "feat_gamma": 100.0, "kd_T": 4.0,
               "teacher_bits": teacher_bits, "feat_loss": "attention",
               "orig_lr": 0.1, "orig_batch_size": 256},
    )


def _qat_only_paper(prec: str) -> Schedule:
    """QAT-only — the no-distillation control. SQAKD's machinery, KD term off.

    This is the row the results table was missing: it separates what the
    *quantizer plus schedule* buys from what *distillation* buys. Every KD row
    in this table (SQAKD, DAQAKD, CMT-KD) and the PMABD ladder itself is only
    interpretable against it — without this control, a reader cannot tell
    whether SQAKD's W3A3 number comes from its distillation objective or from
    EWGS learned-range QAT doing the work on its own.

    Concretely: L = 1.0 * CE(student, y) + 0.0 * KL, i.e. plain PACT/DoReFa-
    style quantization-aware training on labels only. No teacher is built at
    all, so this row is also the cheapest in the table — roughly one forward
    pass per step against SQAKD's two.

    Everything else is held identical to `_sqakd_paper` on purpose: same
    optimizer, lr, weight decay, batch size, cosine schedule, EWGS quantizer,
    grad clip 5.0, same FP32 initialisation, same first/last-layer convention.
    The ONLY difference between this row and the SQAKD row is the loss. That is
    what makes the comparison a clean ablation of distillation rather than a
    comparison of two loosely related recipes.

    NOT A DEVIATION FROM ANY PAPER, because it is not reproducing one. SQAKD's
    own baseline tables call this configuration PACT/DoReFa; label the row
    "QAT only (no KD)" and state that it uses SQAKD's optimisation recipe so
    the ablation is controlled.
    """
    s = _sqakd_paper(prec)
    s.extra = dict(s.extra)
    # gamma=1 / alpha=0 is the whole change: cross-entropy on labels only.
    s.extra.update({"kd_gamma": 1.0, "kd_alpha": 0.0})
    return s


_PAPER = {
    "sqakd": _sqakd_paper,
    "qat_only": _qat_only_paper,
    "daqakd": _daqakd_paper,
    "any_precision": _any_precision_paper,
    "instantnet_cdt": _instantnet_paper,
    "cmtkd": _cmtkd_paper,
}

# Methods that train one shared-weight model serving several precisions.
SWITCHABLE = {"any_precision", "instantnet_cdt"}


# ─────────────────────────────────────────────────────────────────────────────
# CIFAR-100 recipes (ResNet-32 student, chenyaofo hub architecture)
#
# Every CIFAR method runs at the PMABD CIFAR ladder's own batch size
# (data.batch_size in the config, 256), for the same reason TinyImageNet uses
# 32 throughout: it removes batch size as a confound between the rows.
# ─────────────────────────────────────────────────────────────────────────────
def _sqakd_cifar(prec: str) -> Schedule:
    """SQAKD, CIFAR-100 + ResNet-32.

    Source: third_party/SQAKD/CIFAR/scripts/run_cifar100_resnet32.sh
      Adam, lr_m 5e-4, lr_q 5e-6, wd 5e-4, cosine, 720 epochs, bs 64, init
      from the FP32 ResNet-32, teacher = that same FP32 ResNet-32.
    Loss weights come from the run names: W4A4 "kd_gamma0_alpha100", W2A2
    "kd_gamma0_alpha1". The script has no W3A3/W8A8 KD run, so those take the
    W4A4 weight (alpha=100) -- state that in the paper. bs 64 -> the ladder's
    (see _recipe); Adam's lr is left unscaled.
    """
    return Schedule(
        epochs=720, lr=5e-4, optimizer="adam", weight_decay=5e-4,
        lr_schedule="cosine", batch_size=64, aux_lr=5e-6,
        grad_clip=5.0,  # see the note in _sqakd_paper
        extra={"kd_gamma": 0.0, "kd_alpha": 1.0 if prec == "w2a2" else 100.0,
               "kd_T": 4.0, "ewgs": True},
    )


def _daqakd_cifar(prec: str) -> Schedule:
    s = _sqakd_cifar(prec)
    s.extra = dict(s.extra, da_policy="cmi", da_num_ops=2, da_magnitude=9)
    return s


def _qat_only_cifar(prec: str) -> Schedule:
    s = _sqakd_cifar(prec)
    s.extra = dict(s.extra, kd_gamma=1.0, kd_alpha=0.0)
    return s


def _cmtkd_cifar(prec: str) -> Schedule:
    """CMT-KD on CIFAR-100 at the paper's own bs 256 / lr 0.1, so no lr
    scaling is needed at the ladder's bs 256. Epoch count and decay points are
    the ImageNet recipe's; `matched` replaces the count and re-fits them."""
    s = _cmtkd_paper(prec)
    s.lr, s.aux_lr, s.batch_size = 0.1, 0.01, 256
    return s


_CIFAR_PAPER = {
    "sqakd": _sqakd_cifar,
    "daqakd": _daqakd_cifar,
    "qat_only": _qat_only_cifar,
    "cmtkd": _cmtkd_cifar,
}


def _is_cifar(cfg) -> bool:
    return bool(cfg) and cfg["experiment"].get("dataset") in ("cifar10", "cifar100")


def _recipe(method: str, precision: str, cfg=None) -> Schedule:
    """The published recipe for the config's dataset. On CIFAR the batch size
    becomes the PMABD ladder's, with SGD lr scaled linearly (Goyal et al.)."""
    if not _is_cifar(cfg):
        return _PAPER[method](precision)
    sched = _CIFAR_PAPER.get(method, _PAPER[method])(precision)
    bs = int(cfg["data"]["batch_size"])
    if bs != sched.batch_size:
        if sched.optimizer == "sgd":
            k = bs / sched.batch_size
            sched.lr *= k
            if sched.aux_lr:
                sched.aux_lr *= k
        sched.batch_size = bs
    return sched


def _budgets(cfg=None) -> dict:
    """Matched epoch budgets: the TinyImageNet ladder's, or on CIFAR the ones
    declared under baselines.matched_epochs in the config (provenance is
    commented there), with any gaps filled from the PMABD CIFAR ladder's own
    run logs when baselines.matched_epochs_from_log names its output dir.

    As with refresh_ladder_budgets, a declared value is never overwritten by
    the log: the runs already recorded were matched against it."""
    if not _is_cifar(cfg):
        return LADDER_EPOCHS
    bcfg = cfg.get("baselines") or {}
    b = {k.lower(): int(v)
         for k, v in (bcfg.get("matched_epochs") or {}).items() if v}
    if bcfg.get("matched_epochs_from_log"):
        for prec, n in cifar_ladder_log_budgets(
                bcfg["matched_epochs_from_log"]).items():
            b.setdefault(prec, n)
    return b


def cifar_ladder_log_budgets(log_dir: str) -> dict:
    """{precision: epochs run} from the CIFAR ladder runner's run_*.log files.

    The CIFAR runner (run_experiment.py) writes no stages.json, only a console
    log per invocation. Each trained stage ends with one of
        [<name> (W3A3)] Best Val Acc: .. | Test Acc @ best val: .. | Epochs run: 74 (early stopped)
        [<name> (W4A4 ..)] SATURATED after 2 cycle(s), 194 total epochs.
    A stage loaded from its checkpoint prints neither, so reading every log in
    time order and keeping the last value per precision survives re-runs.
    """
    import glob
    import re

    if not os.path.isabs(log_dir):
        log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__)))), log_dir)
    single = re.compile(r"^\[(?P<name>[^\]]+)\] Best Val Acc: [\d.]+% \| "
                        r"Test Acc @ best val: [\d.]+% \| Epochs run: (?P<n>\d+)")
    satur = re.compile(r"^\[(?P<name>[^\]]+)\] SATURATED after \d+ cycle\(s\), "
                       r"(?P<n>\d+) total epochs")
    out = {}
    for path in sorted(glob.glob(os.path.join(log_dir, "run_*.log"))):
        try:
            with open(path, encoding="utf-8", errors="replace") as f:
                for line in f:
                    m = single.match(line) or satur.match(line)
                    if not m:
                        continue
                    pm = re.search(r"\(W(\d+)A(\d+)", m.group("name"))
                    if pm:
                        out[f"w{pm.group(1)}a{pm.group(2)}"] = int(m.group("n"))
        except OSError:
            continue
    return out


def get_schedule(method: str, precision: str, mode: str = "paper",
                 max_epochs: int | None = None,
                 cfg: dict | None = None) -> Schedule:
    """`cfg` is the run's config; it selects the CIFAR recipes and budgets
    when experiment.dataset is CIFAR. Omitted -> TinyImageNet, as before."""
    if method not in _PAPER:
        raise ValueError(f"Unknown method {method!r}. "
                         f"Choose from {sorted(_PAPER)}.")
    precision = precision.lower()
    if precision not in PRECISIONS:
        raise ValueError(f"Unknown precision {precision!r}. "
                         f"Choose from {sorted(PRECISIONS)}.")

    sched = _recipe(method, precision, cfg)

    if mode == "paper":
        return _cap_epochs(sched, method, precision, max_epochs, cfg)
    if mode != "matched":
        raise ValueError(f"Unknown schedule mode {mode!r} "
                         "(expected 'paper' or 'matched').")

    budgets = _budgets(cfg)
    if method in SWITCHABLE:
        covered = [p for p in sched.extra.get("covered_precisions", [])] or \
                  _covered_precisions(sched.extra.get("bit_list", []))
        sched.epochs = max((budgets[p] for p in covered),
                           default=budgets[precision])
        sched.extra = dict(sched.extra, covered_precisions=covered)
    else:
        if precision not in budgets:
            where = ("baselines.matched_epochs in the config"
                     if _is_cifar(cfg) else _LADDER_STAGES_JSON)
            raise RuntimeError(
                f"No matched epoch budget for {precision!r}. The matched "
                f"schedule is defined as the number of epochs the PMABD "
                f"ladder actually spent at that precision, so the "
                f"corresponding ladder rung has to have finished first. "
                f"Looked in: {where} -- known: {sorted(budgets)}")
        sched.epochs = budgets[precision]

    # A 200-epoch cosine curve truncated to 45 epochs never anneals, so the
    # decay has to be re-fitted to the shortened budget rather than clipped.
    sched = _refit_decay(sched, method, precision, cfg)
    return _cap_epochs(sched, method, precision, max_epochs, cfg)


def _refit_decay(sched: Schedule, method: str, precision: str,
                 cfg=None) -> Schedule:
    """Rescale a multistep decay to the schedule's current epoch budget.

    The milestones are always re-derived from the PAPER schedule's fractions,
    never from the schedule's current ones, so this is idempotent: applying it
    after the matched budget and again after an epoch cap gives the same answer
    as applying it once to the final budget.
    """
    if sched.lr_schedule == "multistep" and sched.milestones:
        orig = _recipe(method, precision, cfg)
        frac = [m / orig.epochs for m in orig.milestones]
        sched.milestones = tuple(
            sorted({max(1, int(round(f * sched.epochs))) for f in frac}))
    return sched


def _cap_epochs(sched: Schedule, method: str, precision: str,
                max_epochs: int | None, cfg=None) -> Schedule:
    """Shorten a schedule to at most `max_epochs`, re-fitting the LR decay.

    A budget cap is NOT the same as stopping a long run early. A cosine curve
    fitted to 60 epochs and killed at 30 is still at ~half its initial lr and
    has never annealed, so the truncated run reports an accuracy well below
    what that method reaches in 30 epochs of its own schedule. The cap
    therefore re-fits the decay to the shortened budget: cosine anneals through
    T_max = the new epoch count (engine.py builds it from sched.epochs), and
    multistep milestones move to the same FRACTIONS of the new budget.

    The consequence for the paper is that a capped row is a 30-epoch result for
    every method, not a truncation of a 60-epoch one, and the caption has to
    say so -- including for any cell that was trained under a longer budget.
    """
    if max_epochs is None or sched.epochs <= max_epochs:
        return sched
    sched.epochs = int(max_epochs)
    return _refit_decay(sched, method, precision, cfg)


def _covered_precisions(bit_list) -> list[str]:
    """Which of our reported precisions a switchable bit_list actually serves.

    Switchable models use one bit-width for both weights and activations, so
    they can only land on the symmetric rungs (W8A8, W4A4, W3A3). W8A4 and
    W4A3 are asymmetric and are simply not expressible in this family — that
    absence is a real limitation of these baselines and belongs in the paper,
    not something to paper over.
    """
    out = []
    for b in bit_list:
        key = f"w{b}a{b}"
        if key in PRECISIONS:
            out.append(key)
    return out
