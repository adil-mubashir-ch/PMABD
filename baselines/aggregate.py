#!/usr/bin/env python
"""
Collect every baseline run plus the PMABD ladder into one accuracy + GPU-hour
table.

    python -m baselines.aggregate
    python -m baselines.aggregate --csv results/baseline_comparison.csv

Reads stages.json from the ladder's output_dir and from every
baselines/outputs/<method>_<schedule>/logs/, so it reflects whatever has
finished so far. Missing cells print as "-" rather than failing.

TWO COST COLUMNS, AND WHY
    `hours` is what the run cost. `h/prec` divides that by the number of
    precisions the run serves, because Any-Precision and InstantNet train one
    shared-weight model that yields W8A8, W4A4 and W3A3 from a single run,
    while SQAKD, DAQAKD and CMT-KD each train one model per precision.

    For PMABD the honest figure is neither: reaching W3A3 requires descending
    the whole ladder, so the cumulative column is the real cost of that cell.
    All three numbers are printed; use the cumulative one when comparing
    "cost to obtain a W3A3 model".
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

PRECISION_ORDER = ["w8a8", "w8a4", "w4a4", "w4a3", "w3a3", "w2a2"]

# PMABD ladder stage_id -> precision label, and the pre-ladder fixed costs.
LADDER_STAGES = {
    "m4": "w8a8", "m5a": "w8a4", "m5b": "w4a4", "m6a": "w4a3", "m6b": "w3a3",
}
LADDER_SETUP_STAGES = ["pretrain_resnet50", "pretrain_resnet34",
                       "pretrain_resnet18", "pretrain_mobilenet_v2", "m3_kd"]
# teacher_cache.py writes no stages.json entry; measured from file timestamps.
TEACHER_CACHE_HOURS = 1.65


def _read_stages(path):
    if not os.path.isfile(path):
        return []
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return []


def collect_ladder(output_dir):
    stages = _read_stages(os.path.join(output_dir, "logs", "stages.json"))
    by_id = {s.get("stage_id"): s for s in stages}

    setup_hours = sum(float(by_id[s].get("hours") or 0.0)
                      for s in LADDER_SETUP_STAGES if s in by_id)
    setup_hours += TEACHER_CACHE_HOURS

    rows, cumulative = {}, setup_hours
    for stage_id, prec in LADDER_STAGES.items():
        s = by_id.get(stage_id)
        if not s:
            continue
        cumulative += float(s.get("hours") or 0.0)
        rows[prec] = {
            "test_top1": s.get("test_top1"),
            "test_top5": s.get("test_top5"),
            "hours": float(s.get("hours") or 0.0),
            "hours_per_prec": float(s.get("hours") or 0.0),
            "cum_hours": cumulative,
            "epochs": s.get("epochs_run"),
        }
    return rows, setup_hours


def collect_baselines(baseline_root):
    """{(method, schedule): {precision: row}}"""
    out = {}
    if not os.path.isdir(baseline_root):
        return out
    for entry in sorted(os.listdir(baseline_root)):
        stages_path = os.path.join(baseline_root, entry, "logs", "stages.json")
        stages = _read_stages(stages_path)
        if not stages:
            continue
        method, _, schedule = entry.rpartition("_")
        bucket = out.setdefault((method, schedule), {})
        for s in stages:
            hours = float(s.get("hours") or 0.0)
            n_prec = int(s.get("precisions_served") or 1)
            # A switchable run fills several cells from one training run.
            per_prec = s.get("per_precision") or {}
            primary = str(s.get("precision", "")).lower()
            targets = per_prec or {primary: {
                "test_top1": s.get("test_top1"),
                "test_top5": s.get("test_top5")}}
            for label, d in targets.items():
                if label not in PRECISION_ORDER:
                    continue
                bucket[label] = {
                    "test_top1": d.get("test_top1"),
                    "test_top5": d.get("test_top5"),
                    "hours": hours,
                    "hours_per_prec": hours / max(n_prec, 1),
                    "cum_hours": hours,
                    "epochs": s.get("epochs_run"),
                }
    return out


def fmt(v, spec=".2f"):
    if v is None:
        return "-"
    try:
        return format(float(v), spec)
    except (TypeError, ValueError):
        return str(v)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--config",
                   default="configs/mobilenetv2_tinyimagenet_2bit_ladder.yaml")
    p.add_argument("--baseline-root",
                   default=os.path.join(_REPO_ROOT, "baselines", "outputs"))
    p.add_argument("--csv", default=None, help="Also write the table here.")
    args = p.parse_args(argv)

    import yaml
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    ladder_dir = cfg["experiment"]["output_dir"]

    ladder, setup_hours = collect_ladder(ladder_dir)
    baselines = collect_baselines(args.baseline_root)

    header = (f"{'method':<22}{'sched':<9}" +
              "".join(f"{p.upper():>9}" for p in PRECISION_ORDER) +
              f"{'hours':>9}{'h/prec':>9}{'cum h':>9}")
    print("\nMobileNetV2 / TinyImageNet-200 @224 — test top-1 (official 10k val split)")
    print(f"All runs: PMABD splits, same FP32 init, single RTX 3070 Ti.")
    print(f"PMABD fixed setup cost (teacher pretrain + FP32 KD + logit cache): "
          f"{setup_hours:.2f} h\n")
    print(header)
    print("-" * len(header))

    rows_out = []

    def emit(name, schedule, cells):
        line = f"{name:<22}{schedule:<9}"
        for prec in PRECISION_ORDER:
            c = cells.get(prec)
            line += f"{fmt(c['test_top1']) if c else '-':>9}"
        last = cells.get("w3a3") or next(
            (cells[p] for p in reversed(PRECISION_ORDER) if p in cells), None)
        line += (f"{fmt(last['hours']) if last else '-':>9}"
                 f"{fmt(last['hours_per_prec']) if last else '-':>9}"
                 f"{fmt(last['cum_hours']) if last else '-':>9}")
        print(line)
        for prec in PRECISION_ORDER:
            c = cells.get(prec)
            if c:
                rows_out.append({"method": name, "schedule": schedule,
                                 "precision": prec, **c})

    if ladder:
        emit("PMABD (ours)", "-", ladder)
    for (method, schedule), cells in sorted(baselines.items()):
        emit(method, schedule, cells)

    if not baselines:
        print("\n(no baseline runs found yet — run baselines/run_baseline.py)")

    print("\nPublished numbers, for reference only — NOT comparable, since their")
    print("FP32 MobileNetV2 is trained from random init (58.07) while ours is")
    print("fine-tuned from ImageNet weights (74.44):")
    print(f"{'SQAKD (published)':<22}{'-':<9}{'58.13':>9}{'-':>9}{'57.14':>9}{'-':>9}{'52.73':>9}")
    print(f"{'DAQAKD (published)':<22}{'-':<9}{'-':>9}{'-':>9}{'59.48':>9}{'-':>9}{'56.17':>9}")

    if args.csv and rows_out:
        os.makedirs(os.path.dirname(os.path.abspath(args.csv)), exist_ok=True)
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows_out[0]))
            w.writeheader()
            w.writerows(rows_out)
        print(f"\nWrote {args.csv}")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
