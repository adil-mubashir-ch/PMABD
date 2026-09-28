#!/usr/bin/env python
"""
Fast correctness check for every baseline: build it, run a handful of real
training steps, confirm the loss is finite and that gradients actually reach
the parameters the method claims to train.

    python -m baselines.smoke_test                    # all methods, w3a3
    python -m baselines.smoke_test --steps 5 --precision w4a4

This runs on real data (so the loaders and checkpoints are exercised) but only
a few batches, so it finishes in a couple of minutes rather than hours. Run it
before launching anything long.
"""

from __future__ import annotations

import argparse
import itertools
import math
import os
import sys

import torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from baselines.common import setup as bsetup  # noqa: E402
from baselines.common.engine import build_optimizer  # noqa: E402
from baselines.common.schedules import PRECISIONS, get_schedule  # noqa: E402
from baselines.run_baseline import METHODS  # noqa: E402


@torch.no_grad()
def quick_eval(model, loader, device, batches=12, bits=None):
    """Top-1 over a few batches, in EVAL mode (running BN stats, not batch)."""
    if bits is not None:
        from baselines.common.quant_switchable import set_bits
        set_bits(model, bits)
    was_training = model.training
    model.eval()
    correct = seen = 0
    for x, y in itertools.islice(loader, batches):
        x, y = x.to(device), y.to(device)
        correct += (model(x).argmax(1) == y).sum().item()
        seen += y.size(0)
    model.train(was_training)
    return 100.0 * correct / max(seen, 1)


def check(method_name, cfg, precision, steps, device, loaders,
          batch_size, min_acc_factor=4.0, eval_batches=12):
    train_loader, val_loader, _test = loaders
    w_bits, a_bits = PRECISIONS[precision]
    # "matched" needs the ladder to have finished the corresponding rung, which
    # is not true for a precision being smoke-tested BEFORE its ladder run.
    # Falling back to the paper recipe is sound here: over a few dozen steps
    # the epoch count only reaches the loop through the cosine T_max, and this
    # test never steps the scheduler at all. Doing this is what lets every
    # configuration be validated up front rather than after hours of training.
    try:
        sched = get_schedule(method_name, precision, "matched", cfg=cfg)
    except RuntimeError as e:
        print(f"    [{method_name} {precision}] no matched budget yet "
              f"({str(e).split('.')[0]}); using the paper recipe for this "
              f"smoke test only.")
        sched = get_schedule(method_name, precision, "paper", cfg=cfg)
    sched.batch_size = batch_size

    try:
        method = METHODS[method_name](
            cfg, sched, precision, w_bits, a_bits, device=device,
            fp32_source="kd", first_last_bits=None, train_loader=train_loader)
    except ValueError as e:  # switchable methods legitimately reject w8a4/w4a3
        return f"SKIP  ({e.args[0].splitlines()[0][:60]}...)"

    # THE CHECK THAT WAS MISSING and cost four invalid runs.
    #
    # The quantizers calibrate each layer's output_scale so the quantized layer
    # matches the FP32 layer's output magnitude at initialisation. That is what
    # keeps the BatchNorm running statistics inherited from the FP32 checkpoint
    # valid. If it is wrong, training still looks healthy (train mode uses
    # batch statistics) while eval sits at chance for tens of epochs.
    #
    # So: evaluate in EVAL mode before a single gradient step. At W8A8
    # quantization is nearly lossless, so this must land near the FP32 model,
    # not at 1/num_classes.
    eval_bits = w_bits if method_name in ("any_precision", "instantnet_cdt") else None
    bsetup.recalibrate_bn(method.model, train_loader, device, batches=50,
                          bits=eval_bits)
    init_acc = quick_eval(method.model, val_loader, device,
                          batches=eval_batches, bits=eval_bits)
    chance = 100.0 / cfg["experiment"]["num_classes"]
    # W3A3 genuinely is destructive before any training (measured 4.56% for
    # SQAKD), so the floor has to be low enough not to reject a real result.
    # It only has to be far enough above chance to catch a collapse.
    #
    # The default factor of 4 was calibrated on W8A8/W4A4/W3A3. It is NOT a
    # safe floor at W2A2, which is uncharted -- no published result reports
    # MobileNetV2 at W2A2 on TinyImageNet, so there is no basis for expecting
    # any particular pre-training accuracy there, and a factor-4 floor would
    # reject a struggling-but-real quantizer as if it were broken. Callers
    # targeting 2 bits pass a lower factor; what still has to be caught there
    # is a collapse to exactly chance, which means the quantizer is broken
    # rather than merely losing to the bit-width.
    if init_acc < min_acc_factor * chance:
        return (f"FAIL  eval-mode top-1 at init is {init_acc:.2f}% "
                f"(chance {chance:.2f}%, floor {min_acc_factor:g}x) after "
                "BN recalibration - "
                "quantizer calibration is broken")

    optimizer = build_optimizer(method, sched)
    method.model.train()

    losses = []
    for x, y in itertools.islice(train_loader, steps):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        out = method.train_batch(x, y)
        loss = out["loss"]
        if not math.isfinite(loss):
            return f"FAIL  non-finite loss at step {len(losses)}: {loss}"
        losses.append(loss)
        # Must mirror engine.run_training exactly. Omitting this ran the smoke
        # test unclipped while real runs are clipped, which reported divergence
        # that the actual training loop does not have.
        if sched.grad_clip:
            torch.nn.utils.clip_grad_norm_(
                [p for p in method.parameters() if p.grad is not None],
                sched.grad_clip)
        optimizer.step()

    params = [p for p in method.parameters() if p.requires_grad]
    with_grad = sum(1 for p in params if p.grad is not None
                    and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0)
    if with_grad == 0:
        return "FAIL  no parameter received a non-zero finite gradient"

    aux = list(method.aux_parameters())
    aux_grad = sum(1 for p in aux if p.grad is not None and p.grad.abs().sum() > 0)

    # Recalibrate before the post-training eval too, exactly as the engine
    # does. BN statistics go stale within a handful of gradient steps, so
    # skipping this reads 0% and tells you nothing.
    bsetup.recalibrate_bn(method.model, train_loader, device, batches=50,
                          bits=eval_bits)
    post_acc = quick_eval(method.model, val_loader, device,
                          batches=eval_batches, bits=eval_bits)
    trend = "rising" if losses[-1] > losses[0] * 1.2 else "ok"
    return (f"OK    eval {init_acc:5.2f}% -> {post_acc:5.2f}%   "
            f"loss {losses[0]:8.3f} -> {losses[-1]:8.3f} ({trend})   "
            f"grads {with_grad}/{len(params)}")


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--methods", default=",".join(sorted(METHODS)))
    p.add_argument("--precision", default="w3a3", choices=sorted(PRECISIONS))
    p.add_argument("--steps", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--eval-batches", type=int, default=12,
                   help="Batches used for the before/after eval. The default "
                        "12 is fine at w8a8, where the signal is tens of "
                        "points. It is NOT enough near chance: at w2a2 every "
                        "method sits around 0.5%% before training, so 12x32 "
                        "images resolves 0 vs 1 correct prediction and the "
                        "number is noise. Raise it when the expected accuracy "
                        "is low.")
    p.add_argument("--min-acc-factor", type=float, default=4.0,
                   help="Fail if eval-mode top-1 at init is below this "
                        "multiple of chance. Default 4, calibrated on "
                        "w8a8/w4a4/w3a3. Use ~1.5 at w2a2, where the only "
                        "thing that can be asserted is 'above chance'.")
    p.add_argument("--config",
                   default="configs/mobilenetv2_tinyimagenet_2bit_ladder.yaml")
    p.add_argument("--data-root", default=None,
                   help="Override data.data_root from the config.")
    p.add_argument("--device",
                   default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args(argv)

    cfg = bsetup.load_config(args.config)
    if args.data_root:
        cfg["data"]["data_root"] = args.data_root
    loaders = bsetup.build_loaders(cfg, batch_size=args.batch_size,
                                   num_workers=0)

    print(f"\ndevice={args.device}  precision={args.precision}  "
          f"steps={args.steps}  batch={args.batch_size}  "
          f"eval={args.eval_batches} batches  "
          f"init floor={args.min_acc_factor:g}x chance = "
          f"{args.min_acc_factor * 100.0 / cfg['experiment']['num_classes']:.2f}%")
    print()
    failures = 0
    for name in args.methods.split(","):
        name = name.strip()
        try:
            status = check(name, cfg, args.precision, args.steps, args.device,
                           loaders, args.batch_size, args.min_acc_factor,
                           args.eval_batches)
        except Exception as e:  # noqa: BLE001 - smoke test reports, never raises
            import traceback
            traceback.print_exc()
            status = f"ERROR {type(e).__name__}: {e}"
        if status.startswith(("FAIL", "ERROR")):
            failures += 1
        print(f"  {name:<16} {status}")
        if args.device == "cuda":
            torch.cuda.empty_cache()

    print()
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
