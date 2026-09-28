#!/usr/bin/env python
"""
Run one baseline at one precision, on the PMABD splits and FP32 initialisation.

    python -m baselines.run_baseline --method sqakd --precision w3a3 \
        --config configs/mobilenetv2_tinyimagenet_2bit_ladder.yaml \
        --schedule matched

Methods:
    sqakd           SQAKD, AISTATS 2024            (authors' code ported)
    daqakd          SQAKD + strong DA              (no code; partial reimpl.)
    qat_only        PACT/DoReFa-style QAT, no KD   (control, not a paper)
    any_precision   Any-Precision DNNs, AAAI 2021  (authors' code ported)
    instantnet_cdt  InstantNet CDT, DAC 2021       (authors' code ported)
    cmtkd           CMT-KD, WACV 2023              (no code; full reimpl.)

Schedules:
    --schedule paper     each method's own published recipe
    --schedule matched   the epoch budget the PMABD ladder spent at that
                         precision (see baselines/common/schedules.py)

Everything lands in <output_dir>/logs/ in the same epochs.csv / stages.json
format the ladder uses, so GPU hours are directly comparable.
"""

from __future__ import annotations

import argparse
import os
import sys

# Must be set before torch initialises CUDA. Reduces allocator fragmentation
# under memory pressure. NOTE: torch 2.8 reports "expandable_segments not
# supported on this platform" on Windows, so this is currently a no-op here and
# is kept only for Linux runs. The memory problem it was reached for is solved
# instead by gradient checkpointing in methods/cmtkd.py.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from pmabd_logging import RunLogger  # noqa: E402

from baselines.common import setup as bsetup  # noqa: E402
from baselines.common.engine import run_training  # noqa: E402
from baselines.common.schedules import (  # noqa: E402
    PRECISIONS, SWITCHABLE, get_schedule,
)
from baselines.methods.cmtkd import CMTKD  # noqa: E402
from baselines.methods.qat_only import QATOnly  # noqa: E402
from baselines.methods.sqakd import DAQAKD, SQAKD  # noqa: E402
from baselines.methods.switchable import AnyPrecision, InstantNetCDT  # noqa: E402

METHODS = {
    "sqakd": SQAKD,
    "daqakd": DAQAKD,
    "qat_only": QATOnly,
    "any_precision": AnyPrecision,
    "instantnet_cdt": InstantNetCDT,
    "cmtkd": CMTKD,
}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--method", required=True, choices=sorted(METHODS))
    p.add_argument("--precision", required=True, choices=sorted(PRECISIONS),
                   help="Target precision. Switchable methods reject the "
                        "asymmetric rungs (w8a4, w4a3) by design.")
    p.add_argument("--config", default="configs/mobilenetv2_tinyimagenet_2bit_ladder.yaml",
                   help="The ladder config — supplies the splits and output_dir.")
    p.add_argument("--schedule", default="paper", choices=("paper", "matched"))
    p.add_argument("--fp32-source", default="kd", choices=("kd", "pretrained"),
                   help="'kd' = M_fp32.pth, the model our own W8A8 rung started "
                        "from (default, and the fair one). 'pretrained' = the "
                        "plain fine-tune with no PMABD distillation in it.")
    p.add_argument("--first-last-bits", default=None,
                   help="Comma pair, e.g. '8,8', to quantize the stem conv and "
                        "classifier like the PMABD ladder does. Default leaves "
                        "them FP32, which is every baseline's own convention "
                        "and is easier than what our ladder does.")
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--data-root", default=None,
                   help="Override data.data_root from the config. Must be the "
                        "same extraction the ladder used, or the splits stop "
                        "being comparable.")
    p.add_argument("--num-workers", type=int, default=2,
                   help="Default 2, NOT the config's 8. build_loaders sets "
                        "persistent_workers=True, so train/val/test each hold "
                        "their workers alive at once; on Windows every worker "
                        "is a fresh interpreter importing torch, and 3x8 of "
                        "them exhausts the pagefile with WinError 1455 the "
                        "moment the eval loaders spawn. The PMABD ladder runs "
                        "were launched with --num_workers 2 for the same "
                        "reason, so this also keeps dataloading identical "
                        "between our numbers and the baselines'.")
    p.add_argument("--epochs", type=int, default=None,
                   help="Force an exact epoch count (up OR down), overriding "
                        "the schedule. Prefer --max-epochs, which only ever "
                        "shortens and cannot silently lengthen a run.")
    p.add_argument("--max-epochs", type=int, default=None,
                   help="Cap the schedule at N epochs, re-fitting the LR "
                        "decay to the shorter budget so cosine still anneals "
                        "and multistep milestones keep their fractions. A "
                        "schedule already shorter than N is left alone.")
    p.add_argument("--output-dir", default=None,
                   help="Defaults to baselines/outputs/<method>_<schedule>.")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--no-resume", action="store_true",
                   help="Ignore any <stage>_resume.pt snapshot and train the "
                        "stage from epoch 1. By default an interrupted stage "
                        "resumes from its last completed epoch, carrying the "
                        "optimizer, scheduler, best-val selection and elapsed "
                        "GPU hours across the interruption.")
    p.add_argument("--per-channel-weights", action="store_true",
                   help="One learned weight range per output channel instead "
                        "of one per tensor (depthwise convs excluded). This "
                        "matches the PMABD ladder's quantizer "
                        "(pipeline.py:459) and removes the quantizer as a "
                        "confound; without it a W2A2 comparison measures "
                        "per-tensor-vs-per-channel as much as it measures the "
                        "method. It makes the baselines STRONGER than their "
                        "published configuration (SQAKD's own TinyImageNet "
                        "config sets per_channel: False), which is deliberate.")
    p.add_argument("--dry-run", action="store_true",
                   help="Build everything, run no training, print the plan.")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    cfg = bsetup.load_config(args.config)
    if args.data_root:
        cfg["data"]["data_root"] = args.data_root

    w_bits, a_bits = PRECISIONS[args.precision]
    sched = get_schedule(args.method, args.precision, args.schedule,
                         max_epochs=args.max_epochs, cfg=cfg)
    if args.epochs is not None:
        # Exact override: applied after the cap and NOT decay-refitted, because
        # it is the escape hatch for reproducing a specific historical run.
        sched.epochs = args.epochs
    if args.batch_size is not None:
        sched.batch_size = args.batch_size
    if args.per_channel_weights:
        sched.extra = dict(sched.extra, per_channel_w=True)

    first_last = None
    if args.first_last_bits:
        parts = [int(v) for v in args.first_last_bits.split(",")]
        if len(parts) != 2:
            raise SystemExit("--first-last-bits expects 'w,a', e.g. '8,8'")
        first_last = tuple(parts)

    # The PMABD ladder runs with this on (run_experiment_imagenet.py:360) and
    # the baselines were not, which both cost speed and biased the GPU-hour
    # comparison against them. Input size is fixed here, so autotuning pays.
    torch.backends.cudnn.benchmark = True

    # TinyImageNet keeps baselines/outputs/<method>_<schedule>; other datasets
    # get baselines/outputs/<dataset>/..., so the two tables never collide.
    out_dir = args.output_dir or os.path.join(
        bsetup.baseline_output_root(cfg), f"{args.method}_{args.schedule}")
    os.makedirs(out_dir, exist_ok=True)

    run_name = f"{args.method}_{args.precision}_{args.schedule}"
    logger = RunLogger(out_dir, run_name=run_name)
    logger.start_capture()

    try:
        print("=" * 78)
        print(f"BASELINE  {args.method}   precision={args.precision}   "
              f"schedule={args.schedule}")
        print(f"  {sched.describe()}")
        print(f"  fp32 init/teacher : "
              f"{bsetup.fp32_checkpoint_path(cfg, args.fp32_source)} "
              f"({args.fp32_source})")
        print(f"  first/last layers : "
              f"{'FP32 (method convention)' if first_last is None else f'W{first_last[0]}A{first_last[1]} (PMABD convention)'}")
        if args.method in SWITCHABLE:
            covered = sched.extra.get("covered_precisions") or []
            print(f"  shared-weight run — one training serves: "
                  f"{sorted(set(covered + [args.precision]))}")
        print("=" * 78)

        loaders = bsetup.build_loaders(cfg, batch_size=sched.batch_size,
                                       num_workers=args.num_workers)
        train_loader, _val, _test = loaders

        logger.write_meta({
            "config": cfg,
            "baseline_method": args.method,
            "precision": args.precision,
            "schedule_mode": args.schedule,
            "schedule": sched.__dict__,
            "fp32_source": args.fp32_source,
            "first_last_bits": first_last,
            "device": args.device,
        })

        method = METHODS[args.method](
            cfg, sched, args.precision, w_bits, a_bits,
            device=args.device, fp32_source=args.fp32_source,
            first_last_bits=first_last, train_loader=train_loader,
        )

        if args.dry_run:
            n_params = sum(p.numel() for p in method.parameters())
            print(f"[dry-run] method built OK — {n_params / 1e6:.2f}M trainable "
                  f"params, eval targets {method.eval_targets()}")
            print("[dry-run] no training performed.")
            return 0

        result = run_training(
            method, loaders, sched, logger,
            stage_id=f"{args.method}_{args.precision}",
            stage_name=f"{args.method} {args.precision} ({args.schedule})",
            precision=args.precision, device=args.device,
            output_dir=out_dir, resume=not args.no_resume,
        )

        print("\n" + "=" * 78)
        print(f"RESULT  {args.method} {args.precision} ({args.schedule})")
        print(f"  best val top-1 : {result.best_val_top1:.2f} "
              f"(epoch {result.best_epoch})")
        print(f"  test top-1/5   : {result.test_top1:.2f} / {result.test_top5:.2f}")
        for label, d in result.per_precision.items():
            print(f"    {label:<6} test top-1 {d['test_top1']:.2f}")
        print(f"  GPU hours      : {result.hours:.2f}")
        print(f"  checkpoint     : {result.checkpoint}")
        print("=" * 78)
        return 0
    finally:
        logger.stop_capture()
        logger.close()


if __name__ == "__main__":
    raise SystemExit(main())
