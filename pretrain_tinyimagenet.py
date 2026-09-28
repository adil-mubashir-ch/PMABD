"""
pretrain_tinyimagenet.py
========================
Produce the TinyImageNet-200 checkpoints the PMABD ladder needs.

Why this script exists
----------------------
torchvision only publishes 1000-class ImageNet heads. Point the ladder at
TinyImageNet and every backbone gets its classifier replaced by a *random*
200-way Linear. The student would eventually train that head, but the frozen
teachers never would — they would sit in the pool emitting noise at ~0.5%
accuracy, and every distillation number downstream would be meaningless.

So each backbone is fine-tuned here first, from ImageNet-pretrained weights,
on the same train/val/test splits the ladder uses:

  M1.pth                        ResNet-50   teacher
  M2.pth                        ResNet-34   teacher
  M3.pth                        ResNet-18   teacher
  mobilenetv2_tin_pretrained.pth  MobileNetV2 student initialisation

Setup follows SQAKD (Zhao & Zhao, AISTATS 2024) and DAQAKD (Kur & Zhao, 2025),
the works whose TinyImageNet numbers this pipeline is benchmarked against:
64x64 source images upsampled to 224x224, ImageNet-pretrained initialisation,
ImageNet normalisation. Their reference full-precision accuracies on the
official 10k val split are ResNet-18 65.59 / 66.87 and MobileNetV2 58.07 /
58.64, which is the band this script should land in.

Data discipline: optimises on train only; `val` (carved from the official
train split) drives checkpoint selection and early stopping; the official 10k
val split is evaluated as `test` and never influences any decision.

Usage
-----
    python pretrain_tinyimagenet.py --config configs/mobilenetv2_tinyimagenet_2bit_ladder.yaml
    python pretrain_tinyimagenet.py --config <cfg> --models resnet50,resnet34
    python pretrain_tinyimagenet.py --config <cfg> --epochs 40 --batch_size 96
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml

import pipeline as pl
from pipeline_imagenet_patch import (
    assert_teacher_is_useful,
    build_loaders_from_cfg,
    get_classifier,
    load_model_imagenet,
)
from pmabd_logging import RunLogger

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("pretrain_tin")

# arch -> (output checkpoint name, human label)
DEFAULT_TARGETS = [
    ("resnet50",     "M1.pth",                         "teacher M1"),
    ("resnet34",     "M2.pth",                         "teacher M2"),
    ("resnet18",     "M3.pth",                         "teacher M3"),
    ("mobilenet_v2", "mobilenetv2_tin_pretrained.pth", "student init"),
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--models", default=None,
                   help="comma-separated subset, e.g. resnet50,mobilenet_v2")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--lr", type=float, default=None,
                   help="peak LR for the backbone (head gets 10x)")
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--num_workers", type=int, default=None)
    p.add_argument("--image_size", type=int, default=None)
    p.add_argument("--patience", type=int, default=None)
    p.add_argument("--batch_log_every", type=int, default=50,
                   help="print a within-epoch progress line every N batches "
                        "(0 disables)")
    p.add_argument("--force", action="store_true",
                   help="retrain even if the checkpoint already exists")
    p.add_argument("--dry_run", action="store_true")
    return p.parse_args()


def finetune(arch, model, train_loader, val_loader, test_loader, device,
             epochs, lr, weight_decay, label_smoothing, warmup_epochs,
             patience, min_delta, grad_clip, checkpoint_path, logger,
             stage_tag, batch_log_every=50):
    """
    Fine-tune one ImageNet-pretrained backbone on TinyImageNet-200.

    The head is randomly initialised while the backbone is not, so the head
    gets 10x the backbone LR — without that split the backbone is washed out
    by the large early gradients coming through an untrained classifier.
    """
    model = model.to(device)
    head = get_classifier(model, arch)
    head_ids = {id(p) for p in head.parameters()}
    backbone_params = [p for p in model.parameters() if id(p) not in head_ids]

    optimizer = torch.optim.SGD(
        [
            {"params": backbone_params, "lr": lr},
            {"params": list(head.parameters()), "lr": lr * 10.0},
        ],
        momentum=0.9, weight_decay=weight_decay, nesterov=True,
    )
    warmup_sched, cosine_sched, n_warmup = pl.make_scheduler(
        optimizer, epochs, lr, warmup_epochs=warmup_epochs, eta_min_factor=0.001)
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    stopper = pl.EarlyStopper(patience=patience, min_delta=min_delta)

    best_val, best_test, best_test5, best_state, best_epoch = -1.0, 0.0, 0.0, None, 0
    t_stage = time.time()

    print(f"\n{'='*70}")
    print(f"PRETRAIN {arch} on TinyImageNet-200")
    print(f"  epochs={epochs} lr={lr} (head {lr*10:g}) wd={weight_decay} "
          f"warmup={warmup_epochs} patience={patience} ls={label_smoothing}")
    print(f"  checkpoint -> {checkpoint_path}")
    print(f"{'='*70}")

    for epoch in range(epochs):
        t0 = time.time()
        model.train()
        tracker = pl.MetricTracker()
        gsum, nb = 0.0, 0

        n_batches = len(train_loader)
        for bi, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            loss = criterion(logits, targets)
            loss.backward()
            gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()

            bs = inputs.size(0)
            tracker.update("loss", loss.item(), bs)
            tracker.update("correct", logits.argmax(1).eq(targets).sum().item(), 1)
            tracker.update("total", bs, 1)
            gsum += float(gnorm); nb += 1

            # An epoch here is ~900 batches; without a heartbeat there is no way
            # to tell a slow run from a hung one until the epoch ends.
            if batch_log_every and (bi + 1) % batch_log_every == 0:
                done = bi + 1
                rate = done / max(time.time() - t0, 1e-6)
                print(f"    [{stage_tag}] ep {epoch+1}/{epochs} "
                      f"batch {done}/{n_batches} ({100.0*done/n_batches:.0f}%) "
                      f"loss={tracker.avg('loss'):.4f} "
                      f"{rate:.2f} it/s  eta {(n_batches - done)/max(rate,1e-6)/60:.1f} min",
                      flush=True)

        (warmup_sched if epoch < n_warmup else cosine_sched).step()

        train_acc = 100.0 * tracker.avg("correct") / max(tracker.avg("total"), 1)
        val1, val5   = pl.evaluate_topk(model, val_loader,  device)
        test1, test5 = pl.evaluate_topk(model, test_loader, device)
        cur_lr = optimizer.param_groups[0]["lr"]
        secs = time.time() - t0

        print(f"[{stage_tag}] Epoch {epoch+1:3d}/{epochs} | "
              f"loss={tracker.avg('loss'):.4f} | train={train_acc:.2f}% "
              f"val={val1:.2f}%/{val5:.2f}% test={test1:.2f}%/{test5:.2f}% | "
              f"lr={cur_lr:.5f} gnorm={gsum/max(nb,1):.3f} secs={secs:.1f}")

        if logger is not None:
            logger.log_epoch({
                "stage": stage_tag, "cycle": 1, "cycle_epoch": epoch + 1,
                "epoch": epoch + 1, "max_epochs": epochs,
                "bitwidth_w": 32, "bitwidth_a": 32, "lr": cur_lr,
                "loss": tracker.avg("loss"), "loss_ce": tracker.avg("loss"),
                "loss_kd": 0.0, "loss_feat": 0.0, "loss_qat": 0.0,
                "grad_norm": gsum / max(nb, 1), "kd_w": 0.0, "ce_w": 1.0, "T_eff": 0.0,
                "train_top1": train_acc, "val_top1": val1, "val_top5": val5,
                "test_top1": test1, "test_top5": test5,
                "best_val_so_far": max(best_val, val1), "epoch_secs": secs,
            })

        improved = stopper.step(val1)
        if val1 > best_val:
            best_val, best_test, best_test5 = val1, test1, test5
            best_epoch = epoch + 1
            best_state = copy.deepcopy(model.state_dict())
            torch.save(best_state, checkpoint_path)
            print(f"  val={best_val:.2f}% (test={best_test:.2f}%)"
                  f"{' *** New best ***' if improved else ''} -> saved")

        if stopper.should_stop:
            print(f"  [EARLY STOP] no val gain > {min_delta} for {patience} epochs.")
            break

    hours = (time.time() - t_stage) / 3600
    model.load_state_dict(best_state)
    print(f"\n[{stage_tag}] DONE. best val={best_val:.2f}% @ epoch {best_epoch} | "
          f"test top1={best_test:.2f}% top5={best_test5:.2f}% | {hours:.2f} h")
    return {
        "arch": arch, "best_val_top1": best_val, "best_epoch": best_epoch,
        "test_top1": best_test, "test_top5": best_test5,
        "epochs_run": epoch + 1, "hours": hours,
        "checkpoint": str(checkpoint_path),
    }


def main():
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    cfg["experiment"]["dataset"] = "tinyimagenet"
    cfg["experiment"]["num_classes"] = 200
    for key, val in [("batch_size", args.batch_size),
                     ("num_workers", args.num_workers),
                     ("image_size", args.image_size)]:
        if val is not None:
            cfg["data"][key] = val

    num_classes = 200
    output_dir  = cfg["experiment"]["output_dir"]
    seed        = cfg["experiment"].get("seed", 42)
    image_size  = cfg["data"].get("image_size", 224)
    small_input = image_size <= 64
    device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pcfg = cfg.get("pretrain", {})
    epochs          = args.epochs   or pcfg.get("epochs", 40)
    lr              = args.lr       or pcfg.get("lr", 0.01)
    patience        = args.patience or pcfg.get("patience", 10)
    weight_decay    = pcfg.get("weight_decay", 1.0e-4)
    label_smoothing = pcfg.get("label_smoothing", 0.1)
    warmup_epochs   = pcfg.get("warmup_epochs", 2)
    min_delta       = pcfg.get("min_delta", 0.05)
    grad_clip       = pcfg.get("grad_clip", 5.0)

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    random.seed(seed); np.random.seed(seed)

    targets = DEFAULT_TARGETS
    if args.models:
        wanted = {m.strip() for m in args.models.split(",")}
        targets = [t for t in DEFAULT_TARGETS if t[0] in wanted]
        unknown = wanted - {t[0] for t in DEFAULT_TARGETS}
        if unknown:
            sys.exit(f"Unknown model(s): {sorted(unknown)}. "
                     f"Choose from {[t[0] for t in DEFAULT_TARGETS]}")

    logger = RunLogger(output_dir, run_name="pretrain_tin")
    logger.start_capture()
    logger.write_meta({"phase": "pretrain", "config_path": args.config,
                       "config": cfg, "device": str(device),
                       "targets": [t[0] for t in targets],
                       "hyperparams": {
                           "epochs": epochs, "lr": lr, "weight_decay": weight_decay,
                           "label_smoothing": label_smoothing,
                           "warmup_epochs": warmup_epochs, "patience": patience,
                           "image_size": image_size, "small_input_stem": small_input,
                       }})

    print("=" * 70)
    print("PMABD — TinyImageNet backbone pretraining")
    print(f"  Config     : {args.config}")
    print(f"  Device     : {device}")
    print(f"  Output     : {output_dir}")
    print(f"  Image size : {image_size} (small-input stem: {small_input})")
    print(f"  Targets    : {', '.join(a for a, _, _ in targets)}")
    print("=" * 70)

    if args.dry_run:
        print("Dry run: config parsed, imports OK. Nothing trained.")
        logger.close()
        return

    train_loader, val_loader, test_loader = build_loaders_from_cfg(cfg)
    print(f"Batches — train {len(train_loader)} | val {len(val_loader)} | "
          f"test {len(test_loader)}")

    results = []
    for arch, ckpt_name, label in targets:
        ckpt_path = os.path.join(output_dir, ckpt_name)
        if os.path.isfile(ckpt_path) and not args.force:
            print(f"\n[SKIP] {arch} ({label}) — {ckpt_path} already exists. "
                  f"Use --force to retrain.")
            model = load_model_imagenet(arch, num_classes=num_classes,
                                        pretrained=False, checkpoint_path=ckpt_path,
                                        device=device, small_input=small_input)
            t1, t5 = pl.evaluate_topk(model, test_loader, device)
            v1, _  = pl.evaluate_topk(model, val_loader, device)
            print(f"  existing checkpoint: val top1={v1:.2f}%  "
                  f"test top1={t1:.2f}% top5={t5:.2f}%")
            assert_teacher_is_useful(t1, f"{label} ({arch}) @ {ckpt_path}",
                                     num_classes)
            results.append({"arch": arch, "label": label, "skipped": True,
                            "best_val_top1": v1, "test_top1": t1, "test_top5": t5,
                            "checkpoint": ckpt_path})
            del model
            torch.cuda.empty_cache()
            continue

        model = load_model_imagenet(arch, num_classes=num_classes, pretrained=True,
                                    checkpoint_path=None, device=device,
                                    small_input=small_input)
        res = finetune(
            arch, model, train_loader, val_loader, test_loader, device,
            epochs=epochs, lr=lr, weight_decay=weight_decay,
            label_smoothing=label_smoothing, warmup_epochs=warmup_epochs,
            patience=patience, min_delta=min_delta, grad_clip=grad_clip,
            checkpoint_path=ckpt_path, logger=logger,
            stage_tag=f"pretrain_{arch}", batch_log_every=args.batch_log_every,
        )
        res["label"] = label
        results.append(res)
        logger.log_stage(f"pretrain_{arch}", {
            "precision": "FP32", **{k: v for k, v in res.items() if k != "arch"},
            "arch": arch,
        })
        del model
        torch.cuda.empty_cache()

    summary_path = os.path.join(output_dir, "pretrain_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)

    print("\n" + "=" * 78)
    print("TINYIMAGENET FULL-PRECISION BACKBONES")
    print(f"{'model':14s} {'role':14s} {'val top1':>9s} {'test top1':>10s} {'test top5':>10s}")
    print("-" * 78)
    for r in results:
        print(f"{r['arch']:14s} {r.get('label',''):14s} "
              f"{r.get('best_val_top1', float('nan')):9.2f} "
              f"{r.get('test_top1', float('nan')):10.2f} "
              f"{r.get('test_top5', float('nan')):10.2f}")
    print("-" * 78)
    print("Reference points on the official 10k val split:")
    print("  ResNet-18   65.59 (SQAKD, AISTATS 2024) / 66.87 (DAQAKD, 2025)")
    print("  MobileNetV2 58.07 (SQAKD, AISTATS 2024) / 58.64 (DAQAKD, 2025)")
    print(f"Summary -> {summary_path}")
    print("=" * 78)
    logger.close()


if __name__ == "__main__":
    main()
