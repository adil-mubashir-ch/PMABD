#!/usr/bin/env python
"""
One command for the whole W2A2 / QAT-only programme.

    python run_w2a2_suite.py

Runs, in order, resumably, logging everything:

  TASK 1  PMABD ladder resumed from M6.pth -> M7a (W3A2) -> M7b (W2A2)
  TASK 2  SQAKD and DAQAKD at W2A2, matched schedule
  TASK 3  qat_only (no distillation control) at W8A8, W4A4, W3A3, W2A2

Task 2's matched epoch budget is the number of epochs Task 1's m7b stage
actually spent, which is why the tasks cannot be reordered: that number does
not exist until the ladder has finished. The driver reads it from
outputs/.../logs/stages.json between the tasks rather than being told it.

RESUMABILITY
    Nothing already recorded is recomputed. The ladder runner skips any stage
    present in its stages.json, run_baseline is skipped here when its stage_id
    is already in the method's stages.json, and an interrupted stage resumes
    mid-run from its own per-epoch snapshot. Re-running this script after a
    crash, a power cut, or a Ctrl-C picks up where it stopped.

WHAT IT WILL NOT DO
    It never re-runs m4..m6b, and never touches the recorded SQAKD / DAQAKD /
    CMT-KD results at W8A8 / W4A4 / W3A3.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime

REPO = os.path.dirname(os.path.abspath(__file__))
CONFIG = os.path.join("configs", "mobilenetv2_tinyimagenet_2bit_ladder.yaml")
LADDER_OUT = os.path.join("outputs", "mobilenetv2_tinyimagenet_2bit_ladder_nosat")
LADDER_STAGES = os.path.join(REPO, LADDER_OUT, "logs", "stages.json")
BASELINE_OUT = os.path.join(REPO, "baselines", "outputs")

# Where the TinyImageNet extraction may already live on this machine. The repo
# gitignores data/, so a fresh clone has no dataset; these are checked before
# falling back to the downloader inside pipeline_imagenet_patch.
DATA_SOURCE_CANDIDATES = [
    os.path.join(REPO, "..", "..", "0_imagenet_claude", "data", "imagenet"),
    os.path.join(REPO, "..", "..", "0_imagenet_claude", "data"),
]


# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────
class Tee:
    """Write to the console and to the suite log at the same time."""

    def __init__(self, path):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.fh = open(path, "a", encoding="utf-8", buffering=1)

    def write(self, text):
        sys.__stdout__.write(text)
        sys.__stdout__.flush()
        self.fh.write(text)

    def close(self):
        self.fh.close()


LOG = None


def say(msg=""):
    line = str(msg) + "\n"
    if LOG is not None:
        LOG.write(line)
    else:
        sys.__stdout__.write(line)


def banner(title):
    say()
    say("=" * 78)
    say(title)
    say("=" * 78)


# ─────────────────────────────────────────────────────────────────────────────
# Subprocess plumbing
# ─────────────────────────────────────────────────────────────────────────────
def data_flag(args, style):
    """The data_root override for a child, or nothing when ./data is correct.

    `style` is "dash" for the baselines' argparse (--data-root) and "under"
    for the ladder runner and teacher_cache, which use underscore flags.
    """
    root = getattr(args, "resolved_data_root", None)
    if not root:
        return []
    return [("--data_root" if style == "under" else "--data-root"), root]


def run(cmd, step_log, label):
    """Run a subprocess, streaming its output to the console and two files."""
    say()
    say("-" * 78)
    say(f"[{label}] {' '.join(cmd)}")
    say(f"[{label}] step log: {os.path.relpath(step_log, REPO)}")
    say("-" * 78)

    os.makedirs(os.path.dirname(step_log), exist_ok=True)
    t0 = time.time()
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    # Force UTF-8 on the child: several of these scripts print box-drawing and
    # arrow characters, and the Windows console default codepage raises
    # UnicodeEncodeError on them mid-run, which would kill a training job many
    # hours in for a reason that has nothing to do with the training.
    env["PYTHONIOENCODING"] = "utf-8"

    with open(step_log, "a", encoding="utf-8", buffering=1) as fh:
        fh.write(f"\n### {datetime.now().isoformat()}  {' '.join(cmd)}\n")
        proc = subprocess.Popen(
            cmd, cwd=REPO, env=env, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, encoding="utf-8",
            errors="replace", bufsize=1)
        for line in proc.stdout:
            sys.__stdout__.write(line)
            sys.__stdout__.flush()
            fh.write(line)
            if LOG is not None:
                LOG.fh.write(line)
        rc = proc.wait()

    dt = (time.time() - t0) / 3600.0
    say(f"[{label}] exit={rc}  wall={dt:.2f} h")
    return rc, dt


# ─────────────────────────────────────────────────────────────────────────────
# stages.json helpers
# ─────────────────────────────────────────────────────────────────────────────
def read_stages(path):
    if not os.path.isfile(path):
        return []
    try:
        with open(path, encoding="utf-8") as f:
            # NaN appears in the ladder's own log (the "teachers" pseudo-stage
            # has no val accuracy); json accepts it by default, but be explicit.
            return json.load(f)
    except (OSError, ValueError) as e:
        say(f"  ! could not read {path}: {e}")
        return []


def find_stage(path, stage_id):
    for s in read_stages(path):
        if s.get("stage_id") == stage_id:
            return s
    return None


def baseline_stages_path(method, schedule="matched"):
    return os.path.join(BASELINE_OUT, f"{method}_{schedule}", "logs",
                        "stages.json")


# ─────────────────────────────────────────────────────────────────────────────
# Preflight
# ─────────────────────────────────────────────────────────────────────────────
def inspect_root(root):
    """Is `root` a COMPLETE TinyImageNet-200 extraction? Returns (ok, detail).

    "Exists" is not the test. This repo's own ./data is stored in a synced
    folder, so it can be present but half-materialised — directories appearing
    one at a time while files are still arriving. A partial extraction does not
    fail fast: ImageFolder builds a class list from whatever directories happen
    to exist at that moment, so a run started too early trains on a silently
    wrong number of classes and every accuracy in this suite becomes
    uninterpretable. Counting is the only way to catch it before the GPU-hours
    are spent.
    """
    tin = os.path.join(root, "tiny-imagenet-200")
    if not os.path.isdir(tin):
        return False, "no tiny-imagenet-200/ here"

    train = os.path.join(tin, "train")
    if not os.path.isdir(train):
        return False, "no train/"
    n_train = sum(1 for e in os.scandir(train) if e.is_dir())

    # The loader prefers val_organised/<class>/ and falls back to val/.
    n_test, test_kind = 0, None
    org = os.path.join(tin, "val_organised")
    if os.path.isdir(org):
        n_test = sum(1 for e in os.scandir(org) if e.is_dir())
        test_kind = "val_organised class dirs"
    else:
        images = os.path.join(tin, "val", "images")
        if os.path.isdir(images):
            n_test = sum(1 for e in os.scandir(images) if e.is_file())
            test_kind = "val/images files"

    detail = f"train {n_train}/200 classes, {test_kind or 'no val'} {n_test}"
    if n_train != 200:
        return False, detail + "  <- incomplete"
    if test_kind == "val_organised class dirs" and n_test != 200:
        return False, detail + "  <- incomplete"
    if test_kind == "val/images files" and n_test < 10000:
        return False, detail + "  <- incomplete"
    if test_kind is None:
        return False, detail + "  <- no evaluation split"
    return True, detail


def resolve_data_root(explicit):
    """Pick a complete extraction, preferring the config's own ./data.

    Returns (root_for_children_or_None, ok). None means "the config's default
    is correct, pass no override" — which keeps the common case identical to
    how the ladder's earlier stages were run.
    """
    candidates = [explicit] if explicit else \
        [os.path.join(REPO, "data")] + DATA_SOURCE_CANDIDATES

    chosen, first = None, None
    for cand in candidates:
        cand = os.path.abspath(cand)
        ok, detail = inspect_root(cand)
        mark = "ok     " if ok else "PARTIAL"
        say(f"  data    : {mark} {cand}")
        say(f"            {detail}")
        if first is None:
            first = cand
        if ok and chosen is None:
            chosen = cand

    if chosen is None:
        say("  data    : no COMPLETE TinyImageNet-200 extraction found.")
        say("            If ./data is still syncing, wait for it to finish and "
            "re-run — this suite refuses to start on a partial dataset rather "
            "than train on the wrong number of classes.")
        say("            Otherwise pass --data-root <dir containing "
            "tiny-imagenet-200>.")
        return None, False

    default = os.path.abspath(os.path.join(REPO, "data"))
    if chosen == default:
        say("  data    : using ./data, the config default — no override passed "
            "to the child processes.")
        return None, True

    say(f"  data    : ./data is not usable; overriding data_root -> {chosen} "
        "for every child process. This is the same extraction, just at a "
        "different path, so the splits are unchanged: they are rebuilt from "
        "the same class-stratified seed either way.")
    return chosen, True


def preflight(args):
    banner("PREFLIGHT")
    ok = True

    say(f"  repo    : {REPO}")
    say(f"  python  : {sys.executable}")
    try:
        import torch
        say(f"  torch   : {torch.__version__}  cuda={torch.cuda.is_available()}")
        if torch.cuda.is_available():
            say(f"  gpu     : {torch.cuda.get_device_name(0)}")
        else:
            say("  gpu     : NONE — this suite is ~40+ GPU-hours, refusing to "
                "start on CPU. Activate the GPU environment.")
            ok = False
    except ImportError as e:
        say(f"  torch   : NOT IMPORTABLE ({e})")
        ok = False

    root, data_ok = resolve_data_root(args.data_root)
    args.resolved_data_root = root
    ok = data_ok and ok

    for name in ("M_fp32.pth", "M6.pth", "M1.pth", "M2.pth", "M3.pth"):
        p = os.path.join(REPO, LADDER_OUT, name)
        mark = "ok " if os.path.isfile(p) else "MISSING"
        say(f"  ckpt    : {mark} {name}")
        if not os.path.isfile(p):
            ok = False

    free = shutil.disk_usage(REPO).free / 1e9
    say(f"  disk    : {free:.1f} GB free")
    if args.teacher_cache_views and free < 4:
        say("            the teacher cache needs ~1.6 GB; this is tight.")

    return ok


# ─────────────────────────────────────────────────────────────────────────────
# Teacher logit cache
# ─────────────────────────────────────────────────────────────────────────────
def ensure_teacher_cache(args, manifest):
    """Build outputs/.../teacher_logits_K16.npy if it is wanted and missing.

    The choice matters to every GPU-hour number this suite reports, so it is
    logged either way rather than inferred later from timings.

    Arithmetic behind the default: Task 1 is m7a (45 epochs) + m7b (60), and
    the ladder's W3A3 rung ran 45 epochs in 8.51 h WITH the cache, i.e. about
    0.19 h/epoch. Without it the ladder is roughly 3x slower, so 105 epochs is
    ~20 h cached (plus ~1.65 h to build the cache) against ~60 h uncached. The
    cache pays for itself more than twenty times over here.

    It changes only the PMABD ladder. The baselines distil from their own
    MobileNetV2 teacher, which is not in this cache, so Tasks 2 and 3 are
    unaffected by this decision.
    """
    if not args.teacher_cache_views:
        say("  teacher cache: DISABLED by --no-teacher-cache. The ladder will "
            "run every ResNet teacher forward live; expect roughly 3x the "
            "GPU-hours per epoch in Task 1. Tasks 2 and 3 are unaffected — "
            "they never use this cache.")
        manifest["teacher_cache"] = {"used": False, "build_hours": 0.0}
        return True

    k = args.teacher_cache_views
    npy = os.path.join(REPO, LADDER_OUT, f"teacher_logits_K{k}.npy")
    js = os.path.join(REPO, LADDER_OUT, f"teacher_logits_K{k}.json")
    if os.path.isfile(npy) and os.path.isfile(js):
        gb = os.path.getsize(npy) / 1e9
        say(f"  teacher cache: present ({gb:.2f} GB, K={k}) — reused, not rebuilt.")
        manifest["teacher_cache"] = {"used": True, "views": k,
                                     "build_hours": 0.0, "prebuilt": True}
        return True

    say(f"  teacher cache: {os.path.relpath(npy, REPO)} is missing and will be "
        f"built now (~1.65 h, ~1.6 GB). This is Task 1 setup cost and is "
        f"reported separately from the ladder's own hours.")
    rc, hours = run(
        [sys.executable, "teacher_cache.py", "--config", CONFIG,
         "--views", str(k), "--num_workers", str(args.num_workers)]
        + data_flag(args, "under"),
        os.path.join(REPO, "logs", "w2a2_suite", "00_teacher_cache.log"),
        "teacher-cache")
    manifest["teacher_cache"] = {"used": rc == 0, "views": k,
                                 "build_hours": round(hours, 3),
                                 "prebuilt": False}
    if rc != 0:
        say("  teacher cache: BUILD FAILED. Re-run with --no-teacher-cache to "
            "proceed without it (about 3x slower in Task 1).")
    return rc == 0


# ─────────────────────────────────────────────────────────────────────────────
# Smoke tests
# ─────────────────────────────────────────────────────────────────────────────
# (methods, precision, min_acc_factor, eval_batches, hard_gate, why)
#
# WHERE THE GATE ACTUALLY LIVES, and why it is not at W2A2.
#
# All three methods share one quantizer, one FP32 initialisation and one set of
# splits; the only thing that differs between them is the loss, and the
# init-time evaluation happens BEFORE any gradient step. So at init they are
# the same network, and a broken quantizer is broken for all three at every
# precision. W8A8 is where that has power: quantization there is nearly
# lossless, so the eval must land near the FP32 model (70-75%) and anything at
# chance is unambiguous. That is the hard gate.
#
# W2A2 cannot carry a gate. Every method sits near chance (0.50%) there before
# training, by the nature of 2-bit quantization rather than by any defect, so
# the check cannot separate "broken" from "2 bits is destructive" -- measured
# on 96 images the three methods returned 1.04%, 1.56% and 0.00%, which is
# 1, 1 and 0 correct predictions, i.e. noise. It is run with far more eval
# batches and reported, but it does not block: failing the suite on it would
# be failing on a coin flip.
SMOKE_PLAN = [
    ("sqakd,daqakd,qat_only", "w8a8", 4.0, 12, True,
     "the real gate. Shared quantizer + the new qat_only code path. W8A8 is "
     "nearly lossless, so every method must land near the FP32 model "
     "(70-75%), not at chance"),
    ("sqakd,daqakd,qat_only", "w2a2", 1.5, 60, False,
     "diagnostic only, on ~1900 images instead of ~380. Near chance the "
     "measurement is noise-dominated, so this is reported and not gated"),
]


def smoke(args, manifest):
    banner("SMOKE TESTS  (before anything long)")
    say("Each evaluates in EVAL mode, after BN recalibration, BEFORE any "
        "gradient step. This is the check whose absence invalidated the first "
        "four runs.")
    results = []
    for methods, precision, factor, eval_batches, hard, why in SMOKE_PLAN:
        say()
        say(f"  {methods} @ {precision} "
            f"[{'GATE' if hard else 'diagnostic'}]: {why}")
        rc, _ = run(
            [sys.executable, "-m", "baselines.smoke_test",
             "--methods", methods, "--precision", precision,
             "--steps", str(args.smoke_steps),
             "--batch-size", str(args.smoke_batch),
             "--min-acc-factor", str(factor),
             "--eval-batches", str(eval_batches),
             "--config", CONFIG] + data_flag(args, "dash"),
            os.path.join(REPO, "logs", "w2a2_suite",
                         f"01_smoke_{precision}.log"),
            f"smoke-{precision}")
        results.append({"methods": methods, "precision": precision,
                        "min_acc_factor": factor,
                        "eval_batches": eval_batches,
                        "gate": hard, "rc": rc})
        if rc != 0 and not hard:
            say(f"  {precision} smoke reported a low init accuracy. Recorded, "
                "not treated as a failure — see the note above SMOKE_PLAN. "
                "Continuing.")
        if rc != 0 and hard and not args.ignore_smoke:
            say()
            say(f"  SMOKE FAILED at {precision} (rc={rc}). Stopping before any "
                "long run. Read the step log above: if the eval-mode accuracy "
                "at init is at chance, the quantizer calibration is broken and "
                "training it would waste the GPU-hours rather than produce a "
                "result. Override with --ignore-smoke only if you have read "
                "the output and disagree.")
            manifest["smoke"] = results
            return False
    manifest["smoke"] = results
    return True


# ─────────────────────────────────────────────────────────────────────────────
# TASK 1 — the ladder
# ─────────────────────────────────────────────────────────────────────────────
def task1(args, manifest):
    banner("TASK 1 — PMABD ladder: M6 -> M7a (W3A2) -> M7b (W2A2)")

    done = {s.get("stage_id") for s in read_stages(LADDER_STAGES)}
    say(f"  already recorded: {sorted(done)}")
    if {"m7a", "m7b"} <= done:
        say("  both m7a and m7b are already in stages.json — nothing to run.")
        manifest["task1"] = {"skipped": True, "wall_hours": 0.0}
        return True
    say("  the runner skips every stage already in stages.json, so m4..m6b are "
        "not touched.")

    cmd = [sys.executable, "run_experiment_imagenet.py",
           "--config", CONFIG, "--start_stage", "m7a",
           "--num_workers", str(args.num_workers)] + data_flag(args, "under")
    if args.teacher_cache_views:
        cmd += ["--teacher_cache_views", str(args.teacher_cache_views)]

    rc, hours = run(cmd, os.path.join(REPO, "logs", "w2a2_suite",
                                      "10_ladder_m7a_m7b.log"), "ladder")
    manifest["task1"] = {"skipped": False, "rc": rc,
                         "wall_hours": round(hours, 3),
                         "teacher_cache_views": args.teacher_cache_views}
    return rc == 0


def stage_ceiling(cfg_key):
    """The `epochs:` ceiling configured for a ladder stage, or None."""
    try:
        import yaml
    except ImportError:
        return None
    try:
        with open(os.path.join(REPO, CONFIG), encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
    except (OSError, ValueError):
        return None
    stage = cfg.get(cfg_key) or {}
    return stage.get("epochs")


def epochs_from_jsonl(stage_name):
    """Distinct epoch indices logged for a stage — the epochs that actually ran.

    Preferred over stages.json's `epochs_run`, which is NOT what its name
    suggests. That field is logger.count_epochs(), a count of ROWS in
    epochs.jsonl whose stage name matches, and epochs.jsonl is opened in append
    mode and survives across processes. So every interrupted attempt leaves its
    rows behind and the count keeps climbing: stage m6a is configured with a
    40-epoch ceiling and records epochs_run = 53.

    Counting distinct epoch indices instead gives the number of epochs the
    final model was actually trained for, whether the stage was resumed
    (indices continue: 1..30, 31..45) or restarted from scratch (indices
    repeat: 1..13, 1..45). Both yield the right answer.
    """
    path = os.path.join(REPO, LADDER_OUT, "logs", "epochs.jsonl")
    if not os.path.isfile(path):
        return None
    seen = set()
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                if str(r.get("stage", "")).startswith(stage_name):
                    ep = r.get("epoch")
                    if isinstance(ep, int):
                        seen.add(ep)
    except OSError:
        return None
    return len(seen) or None


def m7b_epochs():
    """The matched W2A2 budget: epochs PMABD's W2A2 rung actually trained for.

    Read from the ladder's own logs, never from the config's `epochs:` — that
    is a ceiling and the stage early-stops on patience well below it. But it is
    also cross-checked AGAINST that ceiling, because the obvious field to read
    (`epochs_run`) over-counts across interrupted attempts; see
    epochs_from_jsonl. Handing the baselines an inflated number would give them
    strictly more training than PMABD had, which inverts the comparison the
    matched schedule exists to make.
    """
    st = find_stage(LADDER_STAGES, "m7b")
    if st is None:
        return None

    recorded = int(st.get("epochs_run") or 0) or None
    ceiling = stage_ceiling("stage_m7b")
    distinct = epochs_from_jsonl(st.get("name", "M7a->M7b (W2A2)"))

    budget = distinct or recorded
    if budget is None:
        return None

    if distinct and recorded and distinct != recorded:
        say(f"  note: stages.json records epochs_run={recorded}, but only "
            f"{distinct} distinct epochs were logged for m7b — the stage was "
            f"interrupted and epochs_run double-counts the attempts. Using "
            f"{distinct}.")

    if ceiling and budget > ceiling:
        say(f"  note: derived budget {budget} exceeds m7b's configured ceiling "
            f"of {ceiling}, which a single run cannot do. Clamping to "
            f"{ceiling} rather than over-training the baselines.")
        budget = ceiling

    return budget


# ─────────────────────────────────────────────────────────────────────────────
# TASKS 2 and 3 — baselines
# ─────────────────────────────────────────────────────────────────────────────
def run_baseline(method, precision, args, manifest, idx):
    stages = baseline_stages_path(method)
    if find_stage(stages, f"{method}_{precision}") is not None:
        st = find_stage(stages, f"{method}_{precision}")
        say(f"  [skip] {method} {precision} already recorded "
            f"(test top-1 {st.get('test_top1')}, {st.get('hours', 0):.2f} h)")
        return True

    cmd = [sys.executable, "-m", "baselines.run_baseline",
           "--method", method, "--precision", precision,
           "--schedule", "matched", "--config", CONFIG,
           "--num-workers", str(args.num_workers)] + data_flag(args, "dash")

    # Pass the W2A2 budget explicitly instead of letting run_baseline read it
    # back out of stages.json. The driver's m7b_epochs() cross-checks the
    # ladder log against the stage's configured ceiling; the value in
    # stages.json does not, and over-counts across interrupted attempts.
    if precision == "w2a2":
        budget = m7b_epochs()
        if budget:
            cmd += ["--epochs", str(budget)]
            say(f"  matched budget: {budget} epochs (PMABD m7b), passed "
                f"explicitly")
    rc, hours = run(cmd, os.path.join(REPO, "logs", "w2a2_suite",
                                      f"{idx}_{method}_{precision}.log"),
                    f"{method}-{precision}")
    manifest.setdefault("baseline_runs", []).append(
        {"method": method, "precision": precision, "rc": rc,
         "wall_hours": round(hours, 3)})
    if rc != 0:
        say(f"  [FAILED rc={rc}] {method} {precision} — continuing with the "
            "rest of the suite so one bad cell does not cost the others.")
    return rc == 0


def task2(args, manifest):
    banner("TASK 2 — SQAKD and DAQAKD at W2A2 (matched)")
    n = m7b_epochs()
    if n is None:
        say("  m7b is not in the ladder's stages.json, so the matched W2A2 "
            "budget does not exist yet. Skipping Tasks 2 and 3's W2A2 cells.")
        return False
    say(f"  matched budget = {n} epochs — the number m7b actually spent, read "
        f"from {os.path.relpath(LADDER_STAGES, REPO)}.")
    say("  both initialise from M_fp32.pth (fp32_source=kd, the default): the "
        "same model PMABD's own W8A8 rung started from. They are NOT "
        "initialised from their own W3A3 checkpoints — these methods train "
        "each precision independently from FP32.")
    say("  first conv and classifier stay FP32 (the baselines' own "
        "convention). The PMABD ladder holds both at W8A8, which is strictly "
        "harder; --first-last-bits 8,8 equalises it for a sensitivity row.")

    ok = True
    for i, method in enumerate(("sqakd", "daqakd")):
        ok = run_baseline(method, "w2a2", args, manifest, 20 + i) and ok
    return ok


def task3(args, manifest):
    banner("TASK 3 — qat_only (no-distillation control), W8A8 -> W2A2")
    say("  SQAKD's machinery with the KD term switched off: kd_gamma=1.0, "
        "kd_alpha=0.0, cross-entropy on labels only. No teacher is built, so "
        "no teacher forward is paid for.")
    say("  Everything else is held identical to the SQAKD rows — same "
        "quantizer, FP32 init, splits, optimizer, lr, batch size, grad clip "
        "5.0 — so the difference against SQAKD is attributable to the "
        "objective alone.")

    precisions = ["w8a8", "w4a4", "w3a3"]
    if m7b_epochs() is not None:
        precisions.append("w2a2")
    else:
        say("  w2a2 omitted: no matched budget (m7b has not finished).")

    ok = True
    for i, prec in enumerate(precisions):
        ok = run_baseline("qat_only", prec, args, manifest, 30 + i) and ok
    return ok


# ─────────────────────────────────────────────────────────────────────────────
# Final report
# ─────────────────────────────────────────────────────────────────────────────
def report(manifest):
    banner("RESULTS")

    say()
    say("TASK 1 — PMABD ladder, new rungs")
    say(f"  {'stage':<8}{'precision':<12}{'test top-1':>11}{'test top-5':>12}"
        f"{'epochs':>8}{'hours':>8}")
    for sid in ("m7a", "m7b"):
        st = find_stage(LADDER_STAGES, sid)
        if st is None:
            say(f"  {sid:<8}{'—':<12}{'not recorded':>11}")
            continue
        say(f"  {sid:<8}{str(st.get('precision')):<12}"
            f"{st.get('test_top1'):>11}{st.get('test_top5'):>12}"
            f"{st.get('epochs_run'):>8}{st.get('hours', 0):>8.2f}")

    tc = manifest.get("teacher_cache", {})
    if tc.get("used"):
        say(f"  teacher-logit cache: USED (K={tc.get('views')}), build cost "
            f"{tc.get('build_hours', 0):.2f} h"
            + (" (prebuilt, not charged to this run)" if tc.get("prebuilt")
               else ""))
    else:
        say("  teacher-logit cache: NOT used — ladder hours above are the "
            "uncached figures, roughly 3x the cached equivalent.")

    say()
    say("TASKS 2 and 3 — baselines, matched schedule")
    say(f"  {'method':<12}{'precision':<11}{'test top-1':>11}{'test top-5':>12}"
        f"{'epochs':>8}{'hours':>8}")
    rows = [("sqakd", "w2a2"), ("daqakd", "w2a2"),
            ("qat_only", "w8a8"), ("qat_only", "w4a4"),
            ("qat_only", "w3a3"), ("qat_only", "w2a2")]
    for method, prec in rows:
        st = find_stage(baseline_stages_path(method), f"{method}_{prec}")
        if st is None:
            say(f"  {method:<12}{prec:<11}{'not recorded':>11}")
            continue
        say(f"  {method:<12}{prec:<11}{st.get('test_top1'):>11}"
            f"{st.get('test_top5'):>12}{st.get('epochs_run'):>8}"
            f"{st.get('hours', 0):>8.2f}")

    say()
    say("  Reminder for the paper: the baselines keep the first conv and the "
        "classifier at FP32 (their own convention) while the PMABD ladder "
        "holds both at W8A8. That asymmetry favours the baselines and must be "
        "stated; --first-last-bits 8,8 removes it.")
    say("  W2A2 is uncharted — nothing published reports MobileNetV2 W2A2 on "
        "TinyImageNet. A cell at chance (0.50%) is a result, and is reported "
        "as one.")

    path = os.path.join(REPO, "logs", "w2a2_suite", "manifest.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    say()
    say(f"  manifest: {os.path.relpath(path, REPO)}")
    say(f"  full log: {os.path.relpath(LOG.path, REPO)}")


# ─────────────────────────────────────────────────────────────────────────────
def main(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--num-workers", type=int, default=2,
                   help="Default 2. build_loaders sets persistent_workers=True "
                        "so train/val/test hold workers at once; on Windows 8 "
                        "exhausts the pagefile (WinError 1455). Raise on Linux.")
    p.add_argument("--teacher-cache-views", type=int, default=16,
                   help="K for the ladder's teacher-logit cache (Task 1 only). "
                        "Built if missing.")
    p.add_argument("--no-teacher-cache", action="store_true",
                   help="Run the ladder without the cache: no ~1.65 h build, "
                        "but roughly 3x the GPU-hours per epoch in Task 1.")
    p.add_argument("--data-root", default=None,
                   help="Directory containing tiny-imagenet-200/. Omit to "
                        "auto-detect: ./data is preferred, and any other known "
                        "location is used only if ./data is incomplete.")
    p.add_argument("--smoke-steps", type=int, default=30)
    p.add_argument("--smoke-batch", type=int, default=32)
    p.add_argument("--skip-smoke", action="store_true")
    p.add_argument("--ignore-smoke", action="store_true",
                   help="Continue even if a smoke test fails. Only after "
                        "reading the output and disagreeing with it.")
    p.add_argument("--only", default=None,
                   help="Comma list of tasks to run, e.g. '1' or '2,3'.")
    args = p.parse_args(argv)

    if args.no_teacher_cache:
        args.teacher_cache_views = 0

    global LOG
    LOG = Tee(os.path.join(REPO, "logs", "w2a2_suite", "suite.log"))
    t0 = time.time()

    manifest = {
        "started": datetime.now().isoformat(),
        "python": sys.executable,
        "num_workers": args.num_workers,
        "config": CONFIG,
    }
    args.resolved_data_root = None

    banner(f"W2A2 / QAT-ONLY SUITE   {datetime.now():%Y-%m-%d %H:%M:%S}")
    say("Tasks: 1) ladder W3A2+W2A2   2) SQAKD/DAQAKD W2A2   "
        "3) qat_only W8A8..W2A2")

    only = {t.strip() for t in args.only.split(",")} if args.only else {"1", "2", "3"}

    try:
        if not preflight(args):
            manifest["data_root"] = args.resolved_data_root or "./data"
            say()
            say("Preflight failed — nothing was run.")
            return 2

        manifest["data_root"] = args.resolved_data_root or "./data"

        if "1" in only and not ensure_teacher_cache(args, manifest):
            return 3
        if not args.teacher_cache_views:
            manifest.setdefault("teacher_cache",
                                {"used": False, "build_hours": 0.0})

        if not args.skip_smoke and not smoke(args, manifest):
            return 4

        if "1" in only:
            task1(args, manifest)
        if "2" in only:
            task2(args, manifest)
        if "3" in only:
            task3(args, manifest)

        manifest["total_wall_hours"] = round((time.time() - t0) / 3600.0, 3)
        report(manifest)
        return 0
    except KeyboardInterrupt:
        say()
        say("Interrupted. Every finished stage is in its stages.json and an "
            "unfinished one has a per-epoch snapshot; re-run this same command "
            "to continue from where it stopped.")
        return 130
    finally:
        if LOG is not None:
            LOG.close()


if __name__ == "__main__":
    raise SystemExit(main())
