#!/usr/bin/env python
"""
One command for the CIFAR-100 programme (paper Tables 1 and 2).

    python run_cifar_suite.py                  # everything, 25-epoch cap (~10 h)
    python run_cifar_suite.py --max-epochs 30  # ~12 h
    python run_cifar_suite.py --plan           # show what would run, run nothing

Runs, in order, logging everything to logs/cifar_suite/:

  TASK 1  PMABD ResNet-32 ladder extended below W4A4
          M4 (W4A4, 71.22) -> M5a W4A3 -> M5b W3A3 -> M6a W3A2 -> M6b W2A2
  TASK 2  SQAKD / DAQAKD / CMT-KD on ResNet-32 at W2A2, matched to the epochs
          Task 1's W2A2 rung actually ran. W8A8 / W4A4 are already recorded
          (full budget) and are skipped.
  TASK 3  SQAKD / DAQAKD / CMT-KD on ResNet-18 at W8A8, W4A4, W2A2 (Table 2),
          matched to the ResNet-18 ladder's recorded epochs.

Baselines are compared at W8A8 / W4A4 / W2A2 only. PMABD's W4A3, W3A3 and
W3A2 rungs are still trained, because the ladder has to pass through them.

Task 2 needs Task 1's logs for its budgets. Task 3 does not, and still runs if
Task 1 fails.

THE EPOCH CAP (--max-epochs, default 25; ~0.4 h per epoch of cap; 0 = uncapped)
    Same convention as the TinyImageNet W2A2 column: every cell is the best
    val-selected result within the first N epochs, for every method.
      * Baselines: run_matrix --max-epochs N, which RE-FITS the LR decay to N
        (cosine anneals by epoch N, multistep milestones keep their fractions),
        so a capped run is a true N-epoch schedule, not a truncated long one.
      * New PMABD rungs: trained for N epochs, warmup scaled by N/epochs so a
        rung is not all warmup. Patience (>= 40) never fires inside N.
      * PMABD's ResNet-18 rungs were trained at full length; the report reads
        their best-within-N from the ladder's per-epoch log, so the Table 2
        comparison is N epochs per rung for everyone. PMABD still enters each
        rung from a fully trained previous rung -- the same caveat already in
        the TinyImageNet caption, and it must be stated here too.

WHICH LADDER CODE RUNS TASK 1
    The ResNet-18 ladder's runner (W:\\...\\RESNET18\\run_experiment.py),
    through cifar_ladder.py, which restores already-trained rungs QUANTIZED
    (the runner alone strips their quantizers: W4A4 69.15% instead of 71.22%).
    Its pipeline.py is byte-identical to code_v5's, which trained M3/M4a/M4.

RESUMABILITY
    Ladder: a stage whose checkpoint exists is loaded, not retrained; an
    interrupted stage restarts from its first epoch. Baselines: finished runs
    are skipped via stages.json and an interrupted one resumes from its
    per-epoch snapshot. Re-run the same command after any interruption.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime

import yaml

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from baselines.common import setup as bsetup  # noqa: E402
from baselines.common.schedules import _budgets  # noqa: E402

LADDER_CODE = r"W:\01_MS-Thesis\1_WACV_paper_review\RESNET18"
SRC_CKPT = (r"W:\01_MS-Thesis\1_WACV_paper_review\code_v5\outputs"
            r"\resnet32_cifar100_no_saturation")
SEED_CKPTS = ("M1.pth", "M2.pth", "M3.pth", "M4a.pth", "M4.pth")
NEW_STAGES = ("stage_m5a", "stage_m5b", "stage_m6a", "stage_m6b")

LADDER_CFG = os.path.join("configs", "resnet32_cifar100_2bit_ladder.yaml")
LADDER_OUT = os.path.join(REPO, "outputs", "resnet32_cifar100_2bit_ladder")
R32_CFG = os.path.join("configs", "resnet32_cifar100_baselines.yaml")
R18_CFG = os.path.join("configs", "resnet18_cifar100_baselines.yaml")
METHODS = ("sqakd", "daqakd", "cmtkd")
# The precisions baselines are compared at. PMABD's W4A3/W3A3/W3A2 rungs are
# still trained -- the ladder needs them -- but have no baseline column.
BASELINE_PRECISIONS = ("w8a8", "w4a4", "w2a2")
LOG_DIR = os.path.join(REPO, "logs", "cifar_suite")


# ─────────────────────────────────────────────────────────────────────────────
# Logging and subprocesses (same plumbing as run_w2a2_suite.py)
# ─────────────────────────────────────────────────────────────────────────────
class Tee:
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


def run(cmd, step_log, label):
    """Run a subprocess, streaming its output to the console and two files."""
    say()
    say("-" * 78)
    say(f"[{label}] {' '.join(cmd)}")
    say(f"[{label}] step log: {os.path.relpath(step_log, REPO)}")
    say("-" * 78)

    os.makedirs(os.path.dirname(step_log), exist_ok=True)
    t0 = time.time()
    env = dict(os.environ, PYTHONUNBUFFERED="1",
               # The runners print arrows and box-drawing characters; the
               # Windows console codepage would raise on them hours into a run.
               PYTHONIOENCODING="utf-8")
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


def read_yaml(path):
    try:
        with open(path, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except (OSError, ValueError):
        return {}


_EPOCH_RE = re.compile(r"^\[(?P<name>.+?)\] Epoch\s+(?P<ep>\d+) \(cycle ep.*?"
                       r"val=(?P<val>[\d.]+)% test=(?P<test>[\d.]+)%")


def ladder_best_at(log_dir, n):
    """{precision: (val, test, epoch)}: best val within the first n epochs of
    each rung, and the test accuracy at that epoch, from the CIFAR ladder's
    per-epoch console lines. A later log overrides an earlier one for any
    precision it trained. n=0 means no cap."""
    out = {}
    for path in sorted(glob.glob(os.path.join(log_dir, "run_*.log"))):
        rows = {}
        with open(path, encoding="utf-8", errors="replace") as f:
            for line in f:
                m = _EPOCH_RE.match(line)
                pm = m and re.search(r"\(W(\d+)A(\d+)", m.group("name"))
                if not pm:
                    continue
                ep = int(m.group("ep"))
                if n and ep > n:
                    continue
                rows.setdefault(f"w{pm.group(1)}a{pm.group(2)}", []).append(
                    (float(m.group("val")), float(m.group("test")), ep))
        for prec, rs in rows.items():
            # Strict ">" keeps the FIRST epoch reaching the best val, as the
            # runner's own checkpoint selection does.
            best = rs[0]
            for r in rs[1:]:
                if r[0] > best[0]:
                    best = r
            out[prec] = best
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Preflight
# ─────────────────────────────────────────────────────────────────────────────
def preflight(args, only):
    banner("PREFLIGHT")
    ok = True
    say(f"  repo     : {REPO}")
    say(f"  python   : {sys.executable}")
    say(f"  epoch cap: {args.max_epochs or 'none (full budgets)'}")
    try:
        import torch
        cuda = torch.cuda.is_available()
        say(f"  torch    : {torch.__version__}  cuda={cuda}")
        if cuda:
            say(f"  gpu      : {torch.cuda.get_device_name(0)}")
        elif not args.plan:
            say("  gpu      : NONE — refusing to start on CPU. Activate the "
                "torchgpu environment.")
            ok = False
    except ImportError as e:
        say(f"  torch    : NOT IMPORTABLE ({e})")
        ok = False

    cifar = os.path.join(REPO, "data", "cifar-100-python")
    have = all(os.path.isfile(os.path.join(cifar, n))
               for n in ("train", "test", "meta"))
    say(f"  data     : {'ok     ' if have else 'MISSING'} {cifar}")
    ok = ok and have

    if "1" in only:
        for n in ("run_experiment.py", "pipeline.py"):
            p = os.path.join(args.ladder_code, n)
            present = os.path.isfile(p)
            say(f"  ladder   : {'ok     ' if present else 'MISSING'} {p}")
            ok = ok and present
        os.makedirs(LADDER_OUT, exist_ok=True)
        for n in SEED_CKPTS:
            dst, src = os.path.join(LADDER_OUT, n), os.path.join(args.src_ckpt, n)
            if os.path.isfile(dst):
                say(f"  seed ckpt: ok      {n}")
            elif not os.path.isfile(src):
                say(f"  seed ckpt: MISSING {src}")
                ok = False
            elif args.plan:
                say(f"  seed ckpt: would copy {src}")
            else:
                shutil.copy2(src, dst)
                say(f"  seed ckpt: copied  {n}  <- {args.src_ckpt}")

    if "3" in only:
        p = bsetup.fp32_checkpoint_path(bsetup.load_config(R18_CFG))
        present = os.path.isfile(p)
        say(f"  r18 fp32 : {'ok     ' if present else 'MISSING'} {p}")
        ok = ok and present
    return ok


# ─────────────────────────────────────────────────────────────────────────────
# Tasks
# ─────────────────────────────────────────────────────────────────────────────
def smoke(args):
    """ResNet-18 is the one architecture no baseline has run on yet.

    W8A8 is nearly lossless, so eval-mode accuracy at init, after BN
    recalibration and before any step, must land near the FP32 model (78%),
    not at chance. ResNet-32 needs none: six real runs already passed.
    """
    banner("SMOKE TEST — ResNet-18 baselines @ W8A8 (gate)")
    rc, _ = run([sys.executable, "-m", "baselines.smoke_test",
                 "--config", R18_CFG, "--methods", ",".join(METHODS),
                 "--precision", "w8a8", "--steps", "30", "--batch-size", "64"],
                os.path.join(LOG_DIR, "01_smoke_resnet18_w8a8.log"), "smoke-r18")
    return rc == 0


def ladder_config(n):
    """The ladder config to run: as written, or with the new rungs capped.

    Written next to the ladder's outputs so the exact settings that produced
    the checkpoints are kept with them.
    """
    if not n:
        return LADDER_CFG
    cfg = read_yaml(os.path.join(REPO, LADDER_CFG))
    for key in NEW_STAGES:
        st = cfg[key]
        if st["epochs"] > n:
            st["warmup_epochs"] = max(1, round(st["warmup_epochs"] * n
                                               / st["epochs"]))
            st["epochs"] = n
        say(f"  {key}: {st['name']:<28} epochs={st['epochs']:<4} "
            f"warmup={st['warmup_epochs']}")
    path = os.path.join(LADDER_OUT, f"ladder_config_cap{n}.yaml")
    os.makedirs(LADDER_OUT, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# Generated by run_cifar_suite.py --max-epochs {n} from "
                f"{LADDER_CFG}\n")
        yaml.safe_dump(cfg, f, sort_keys=False)
    return os.path.relpath(path, REPO)


def task1(args, manifest):
    banner("TASK 1 — PMABD ResNet-32 ladder: M4 (W4A4) -> W4A3 -> W3A3 -> "
           "W3A2 -> W2A2")
    res = read_yaml(os.path.join(LADDER_OUT, "results.yaml"))
    if "m6b_acc" in res and os.path.isfile(os.path.join(LADDER_OUT, "M6.pth")):
        say("  already finished (results.yaml has m6b) — nothing to run.")
        manifest["task1"] = {"skipped": True}
        return True
    cfg_path = ladder_config(args.max_epochs)
    # Through cifar_ladder.py, not the runner directly: the runner restores a
    # finished rung with its quantizers stripped (M4: 69.15% instead of
    # 71.22%) and would hand that model to the new rungs as a teacher.
    # --unsigned-act: the ladder's activation quantizer is signed, but every
    # quantized conv here is ReLU-fed, so without it A2 rungs get binary
    # activations. On ResNet-18 the fix took W2A2 from 73.01 to 77.70. The
    # restored M3/M4a/M4 teachers keep the signed quantizer they were trained
    # with; only the new rungs use the unsigned range.
    cmd = [sys.executable, "cifar_ladder.py", "--ladder-code", args.ladder_code,
           "--unsigned-act", "--config", cfg_path]
    if args.plan:
        say(f"  would run: {' '.join(cmd)}")
        say("  M3, M4a, M4 already have checkpoints: restored, not retrained.")
        return True
    rc, hours = run(cmd, os.path.join(LOG_DIR, "10_ladder_resnet32.log"),
                    "ladder-r32")
    # The CIFAR runner records no GPU hours of its own. This wall time covers
    # restoring M1..M4 (a minute or two) plus the new rungs, on one GPU with
    # nothing else running, so it is the ladder's GPU time for them.
    manifest["task1"] = {"rc": rc, "wall_hours": round(hours, 3),
                         "config": cfg_path}
    return rc == 0


def matrix(args, cfg_path, label, step):
    cmd = [sys.executable, "-m", "baselines.run_matrix",
           "--config", cfg_path, "--methods", ",".join(METHODS),
           "--schedule", "matched", "--first-last-bits", "8,8",
           "--per-channel-weights", "--num-workers", str(args.num_workers)]
    # Baselines are compared at these precisions only. PMABD still trains its
    # W4A3/W3A3/W3A2 rungs (the ladder needs them), but no baseline runs there.
    cmd += ["--precisions", ",".join(BASELINE_PRECISIONS)]
    if args.max_epochs:
        cmd += ["--max-epochs", str(args.max_epochs)]
    if not args.plan:
        cmd.append("--execute")
    rc, _ = run(cmd, os.path.join(LOG_DIR, f"{step}_{label}.log"), label)
    return rc == 0


def task2(args, manifest):
    banner("TASK 2 — ResNet-32 baselines, W2A2 (matched to Task 1)")
    b = _budgets(bsetup.load_config(R32_CFG))
    missing = [p for p in BASELINE_PRECISIONS if p not in b]
    say(f"  matched budgets: { {p: b[p] for p in BASELINE_PRECISIONS if p in b} }")
    if missing:
        say(f"  no budget yet for {missing}: Task 1 has not logged those rungs, "
            "so they are not planned. W8A8/W4A4 are already done.")
    manifest["task2_budgets"] = b
    return matrix(args, R32_CFG, "baselines_resnet32", 20)


def task3(args, manifest):
    banner("TASK 3 — ResNet-18 baselines, W8A8 + W4A4 + W2A2 (Table 2)")
    b = _budgets(bsetup.load_config(R18_CFG))
    say(f"  matched budgets: {b}  (ResNet-18 ladder, run_20260626_154756.log)"
        + (f", capped at {args.max_epochs}" if args.max_epochs else ""))
    manifest["task3_budgets"] = b
    return matrix(args, R18_CFG, "baselines_resnet18", 30)


# ─────────────────────────────────────────────────────────────────────────────
# Report
# ─────────────────────────────────────────────────────────────────────────────
def report(args, manifest):
    n = args.max_epochs
    banner(f"RESULTS  (test top-1 at best val"
           f"{f', within the first {n} epochs of each rung' if n else ''})")
    for cfg_path, title, ladder_dirs in (
            # ResNet-32 W3A2/W2A2 were retrained at 10x lr into their own
            # folder; read it after the ladder so those rungs override.
            (R32_CFG, "ResNet-32 / CIFAR-100",
             [LADDER_OUT, os.path.join(REPO, "outputs",
                                       "resnet32_cifar100_fixedlr")]),
            # The ResNet-18 W3A2/W2A2 rungs were retrained with the unsigned
            # activation quantizer into their own folder; read it AFTER the
            # original ladder so those rungs override the signed-run numbers.
            (R18_CFG, "ResNet-18 / CIFAR-100",
             [bsetup.load_config(R18_CFG)["experiment"]["output_dir"],
              os.path.join(REPO, "outputs", "resnet18_cifar100_fixedact")])):
        cfg = bsetup.load_config(cfg_path)
        say()
        say(title)
        best = {}
        for d in ladder_dirs:
            best.update(ladder_best_at(d, n))
        pm = "  ".join(f"{p.upper()} {best[p][1]:.2f} (ep {best[p][2]})"
                       for p in ("w8a8", "w4a4", "w3a3", "w2a2") if p in best)
        say(f"  {'PMABD':<8} {pm or '(no per-epoch ladder log here)'}")
        root = bsetup.baseline_output_root(cfg)
        for m in METHODS:
            path = os.path.join(root, f"{m}_matched", "logs", "stages.json")
            try:
                with open(path, encoding="utf-8") as f:
                    stages = json.load(f)
            except (OSError, ValueError):
                stages = []
            cells = "  ".join(
                f"{s['precision']} {s['test_top1']:.2f} ({s['epochs_run']}ep "
                f"{float(s.get('hours') or 0):.2f}h)" for s in stages)
            say(f"  {m:<8} {cells or '(nothing recorded)'}")

    say()
    say("  ResNet-32 W8A8/W4A4 cells were trained at full budget (78/194 "
        "epochs) for every method; the PMABD ResNet-32 W8A8/W4A4 rungs come "
        "from the earlier run and have no per-epoch lines in this ladder's "
        "log: take them from Table 1 (72.14 / 71.22).")
    say("  Caption notes: every baseline starts from the same FP32 model as "
        "PMABD and holds first conv/fc at W8A8 with per-channel weights, as "
        "PMABD does; at W2A2 PMABD also holds the penultimate layer at W4A4. "
        "The ResNet-32 W4A3..W2A2 rungs reuse the ResNet-18 ladder's "
        "hyperparameters.")
    path = os.path.join(LOG_DIR, "manifest.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, default=str)
    say(f"  manifest: {os.path.relpath(path, REPO)}")


# ─────────────────────────────────────────────────────────────────────────────
def main(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--plan", action="store_true",
                   help="Print the plan (and each matrix's plan); train nothing.")
    p.add_argument("--max-epochs", type=int, default=25,
                   help="Cap every new run at N epochs (default 25, ~10 h "
                        "total; each epoch of cap is ~0.4 h, so 30 is ~12 h). "
                        "0 = full budgets.")
    p.add_argument("--only", default=None,
                   help="Comma list of tasks, e.g. '1' or '2,3'. Default all.")
    p.add_argument("--skip-smoke", action="store_true")
    p.add_argument("--num-workers", type=int, default=2,
                   help="Baseline dataloader workers (see run_baseline.py).")
    p.add_argument("--ladder-code", default=LADDER_CODE,
                   help="Directory holding the CIFAR ladder runner to use.")
    p.add_argument("--src-ckpt", default=SRC_CKPT,
                   help="Where M1..M4 of the ResNet-32 run are copied from.")
    args = p.parse_args(argv)
    only = ({t.strip() for t in args.only.split(",")} if args.only
            else {"1", "2", "3"})

    global LOG
    LOG = Tee(os.path.join(LOG_DIR, "suite.log"))
    manifest = {"started": datetime.now().isoformat(),
                "python": sys.executable, "tasks": sorted(only),
                "max_epochs": args.max_epochs, "plan_only": args.plan}
    banner(f"CIFAR-100 SUITE   {datetime.now():%Y-%m-%d %H:%M:%S}"
           f"{'   [PLAN ONLY]' if args.plan else ''}")
    try:
        if not preflight(args, only) and not args.plan:
            say("\nPreflight failed — nothing was run.")
            return 2
        if "3" in only and not (args.plan or args.skip_smoke) and not smoke(args):
            say("\nSMOKE FAILED — stopping before anything long. Read the step "
                "log above; pass --skip-smoke only if you disagree with it.")
            return 4
        if "1" in only:
            task1(args, manifest)
        if "2" in only:
            task2(args, manifest)
        if "3" in only:
            task3(args, manifest)
        report(args, manifest)
        return 0
    except KeyboardInterrupt:
        say("\nInterrupted. Re-run the same command to continue.")
        return 130
    finally:
        LOG.close()


if __name__ == "__main__":
    raise SystemExit(main())
