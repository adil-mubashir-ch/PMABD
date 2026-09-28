#!/usr/bin/env python
"""
Run the full baseline matrix, resumably.

    python -m baselines.run_matrix --schedule matched            # print plan
    python -m baselines.run_matrix --schedule matched --execute  # run it

    # CIFAR-100 ResNet-32, three methods, W8A8..W2A2:
    python -m baselines.run_matrix --config configs/resnet32_cifar100_baselines.yaml \
        --methods sqakd,daqakd,cmtkd --schedule matched --execute

Per-precision methods (sqakd, daqakd, cmtkd) get one run per precision.
Switchable methods (any_precision, instantnet_cdt) get ONE run that serves
W8A8/W4A4/W3A3 together, so they appear once.

A run whose stages.json already contains its stage_id is skipped, so an
interrupted matrix can be restarted with the same command.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from baselines.common import setup as bsetup  # noqa: E402
from baselines.common.schedules import SWITCHABLE  # noqa: E402

PER_PRECISION_METHODS = ["sqakd", "daqakd", "qat_only", "cmtkd"]
TARGET_PRECISIONS = ["w8a8", "w4a4", "w3a3", "w2a2"]

# Shared-weight runs to schedule. Each entry is one training run whose bit_list
# covers several reported cells; "w2a2" additionally trains a 2-bit branch.
# Anything already in stages.json is skipped, so listing both here means the
# finished w3a3 runs are left alone and only the w2a2 ones are launched.
SWITCHABLE_PRECISIONS = ["w3a3", "w2a2"]

# Per-method precision overrides. Empty: every per-precision method runs every
# entry in TARGET_PRECISIONS, and already_done() skips whatever is finished.
#
# CMT-KD now includes W2A2. Its W8A8 cell was produced under an older
# configuration (trained teachers, bs 16) while W4A4/W3A3/W2A2 use frozen
# teachers at bs 32, so that one cell is not comparable with the rest of its
# own row -- it is flagged in its stages.json entry.
METHOD_PRECISIONS = {}

# Order matters for WHEN you can stop, not for total time.
#
# instantnet_cdt runs first because its Cascade Distillation Training is the
# closest prior art to the PMABD ladder ("each bit-width distils from all
# higher bit-widths"), so it is the row the novelty argument turns on. Leaving
# it last meant it would land days after everything else, which is the wrong
# risk to carry. any_precision is the other shared-weight run and fills three
# cells at once, so it comes next. cmtkd is last: three separate runs, and the
# weakest row (no released code, frozen teachers).
#
# qat_only is deliberately absent: that row, and the PMABD ladder itself, are
# being run on the other machine. Adding it here would duplicate the work.
METHOD_ORDER = ["sqakd", "daqakd", "instantnet_cdt", "any_precision", "cmtkd"]

# (method, precision) cells this driver must NOT launch, even though they are
# absent from stages.json and so would otherwise look like work to do.
# TinyImageNet only -- see the dataset check in main().
#
# daqakd/w2a2 is here because it is mid-flight at epoch 54/60 with 9.4 h spent.
# --max-epochs changes the resume signature (engine.py builds it from
# sched.epochs), so launching it under a cap discards that snapshot and
# restarts from epoch 1 -- more wall-clock than letting it finish, for a worse
# number. Finish it uncapped in its own invocation instead:
#
#   python -m baselines.run_baseline --method daqakd --precision w2a2
#       --schedule matched
#       --config configs/mobilenetv2_tinyimagenet_2bit_ladder.yaml
#       --num-workers 2
#
# Once that writes its stages.json entry, already_done() covers it and this
# entry can be dropped.
SKIP_RUNS = {("daqakd", "w2a2")}


def planned_runs(methods=None, precisions=None):
    runs = []
    for m in methods or METHOD_ORDER:
        if m in SWITCHABLE:
            # One run per entry; each bit_list covers several reported rungs.
            for p in SWITCHABLE_PRECISIONS:
                runs.append((m, p))
        else:
            for p in METHOD_PRECISIONS.get(m, TARGET_PRECISIONS):
                runs.append((m, p))
    if precisions:
        runs = [(m, p) for m, p in runs if p in precisions]
    return runs


def already_done(method, precision, schedule, baseline_root) -> bool:
    stages_path = os.path.join(baseline_root, f"{method}_{schedule}",
                               "logs", "stages.json")
    if not os.path.isfile(stages_path):
        return False
    try:
        with open(stages_path, encoding="utf-8") as f:
            stages = json.load(f)
    except (json.JSONDecodeError, OSError):
        return False
    return any(s.get("stage_id") == f"{method}_{precision}" for s in stages)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--schedule", default="matched", choices=("paper", "matched"))
    p.add_argument("--execute", action="store_true",
                   help="Actually run. Without this, only the plan is printed.")
    p.add_argument("--python", default=sys.executable,
                   help="Interpreter to launch each run with (use the GPU env).")
    p.add_argument("--first-last-bits", default=None,
                   help="e.g. '8,8' to match the PMABD ladder's harder "
                        "first/last convention.")
    p.add_argument("--config",
                   default="configs/mobilenetv2_tinyimagenet_2bit_ladder.yaml")
    p.add_argument("--methods", default=None,
                   help="Comma list, e.g. 'sqakd,daqakd,cmtkd'. Default: "
                        "METHOD_ORDER.")
    p.add_argument("--precisions", default=None,
                   help="Comma list, e.g. 'w8a8,w4a4'. Default: all planned.")
    p.add_argument("--num-workers", type=int, default=2,
                   help="Passed to every run. Keep low on Windows — see the "
                        "note in run_baseline.py; 8 exhausts the pagefile.")
    p.add_argument("--baseline-root", default=None,
                   help="Default: baselines/outputs for TinyImageNet, "
                        "baselines/outputs/<dataset> otherwise -- the same "
                        "place run_baseline.py writes to.")
    p.add_argument("--force", action="store_true",
                   help="Re-run even if a result already exists.")
    p.add_argument("--per-channel-weights", action="store_true",
                   help="Passed to every run. See run_baseline.py.")
    p.add_argument("--max-epochs", type=int, default=None,
                   help="Cap EVERY run at N epochs, with the LR decay re-fitted "
                        "to the shorter budget (see schedules._cap_epochs). "
                        "Note this changes the resume signature, so snapshots "
                        "taken under a longer budget restart from epoch 1.")
    p.add_argument("--no-resume", action="store_true",
                   help="Pass --no-resume to each run: ignore per-epoch "
                        "snapshots and restart interrupted stages from epoch 1.")
    args = p.parse_args(argv)

    cfg = bsetup.load_config(args.config)
    baseline_root = args.baseline_root or bsetup.baseline_output_root(cfg)
    skip = (SKIP_RUNS if cfg["experiment"].get("dataset") == "tinyimagenet"
            else set())

    methods = args.methods.split(",") if args.methods else None
    precisions = args.precisions.split(",") if args.precisions else None
    # On CIFAR the matched budgets (declared in the config, or read from the
    # PMABD ladder's logs) define which rungs exist, so default to exactly
    # those rather than planning runs that cannot start.
    from baselines.common.schedules import _budgets, _is_cifar
    if _is_cifar(cfg):
        budgets = _budgets(cfg)
        if precisions is None:
            precisions = [p for p in TARGET_PRECISIONS if p in budgets]
        elif args.schedule == "matched":
            # An explicitly requested rung whose PMABD budget does not exist yet
            # (its ladder stage has not run) cannot start; say so and skip it
            # rather than launching a run that dies on the missing budget.
            unbudgeted = [p for p in precisions if p not in budgets]
            if unbudgeted:
                print(f"\nNo matched budget yet for {unbudgeted} -- the PMABD "
                      f"ladder has not trained those rungs. Not planned.")
            precisions = [p for p in precisions if p in budgets]
    runs = planned_runs(methods, precisions)
    todo = [(m, pr) for m, pr in runs
            if (m, pr) not in skip
            and (args.force or not already_done(m, pr, args.schedule,
                                                baseline_root))]
    n_skip = sum(1 for r in runs if r in skip)

    print(f"\nMatrix: schedule={args.schedule}   config={args.config}\n"
          f"        output={baseline_root}\n"
          f"        {len(todo)} to run, "
          f"{len(runs) - len(todo) - n_skip} already done\n")
    for m, pr in runs:
        if (m, pr) in skip:
            mark = "skip"
        else:
            mark = " " if (m, pr) in todo else "done"
        if m in SWITCHABLE:
            from baselines.common.schedules import _bit_list_for
            served = [f"w{b}a{b}" for b in _bit_list_for(pr) if b != 32]
            note = f"  (serves {'+'.join(served)})"
        else:
            note = ""
        print(f"  [{mark:>4}] {m:<16} {pr}{note}")

    if not args.execute:
        print("\nDry plan only. Re-run with --execute to launch.\n")
        return 0

    failures = []
    for i, (method, precision) in enumerate(todo, 1):
        cmd = [args.python, "-m", "baselines.run_baseline",
               "--method", method, "--precision", precision,
               "--schedule", args.schedule, "--config", args.config,
               "--output-dir", os.path.join(baseline_root,
                                            f"{method}_{args.schedule}")]
        if args.first_last_bits:
            cmd += ["--first-last-bits", args.first_last_bits]
        if args.num_workers is not None:
            cmd += ["--num-workers", str(args.num_workers)]
        if args.per_channel_weights:
            cmd += ["--per-channel-weights"]
        if args.max_epochs is not None:
            cmd += ["--max-epochs", str(args.max_epochs)]
        if args.no_resume:
            cmd += ["--no-resume"]

        print(f"\n{'=' * 78}\n[{i}/{len(todo)}] {' '.join(cmd)}\n{'=' * 78}")
        t0 = time.time()
        rc = subprocess.call(cmd, cwd=_REPO_ROOT)
        dt = (time.time() - t0) / 3600
        if rc != 0:
            failures.append((method, precision, rc))
            print(f"[FAILED rc={rc}] {method} {precision} after {dt:.2f} h — "
                  "continuing with the rest of the matrix.")
        else:
            print(f"[done] {method} {precision} in {dt:.2f} h")

    print(f"\n{'=' * 78}")
    if failures:
        print("Failed runs:")
        for m, pr, rc in failures:
            print(f"  {m} {pr}  (rc={rc})")
    else:
        print("All runs completed.")
    print(f"Now: python -m baselines.aggregate --config {args.config} "
          f"--baseline-root {baseline_root} --csv results/baseline_comparison.csv")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
