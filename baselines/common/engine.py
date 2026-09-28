"""
Shared training loop and GPU-hour accounting for every baseline.

All five baselines run through this one loop so that the compute comparison is
not confounded by loop overhead, evaluation frequency, or checkpoint policy.
Logging goes through the ladder's own `pmabd_logging.RunLogger`, so baseline
runs emit the same epochs.csv / stages.json schema as the PMABD run and can be
aggregated by the same scripts.

CHECKPOINT SELECTION
    Best top-1 on the held-out val split (the 10% carved out of official
    train). The reported test number is the official 10k val split, evaluated
    every epoch but never used to choose anything. This is stricter than
    SQAKD's own protocol, which selects on the split it reports; applying the
    stricter rule uniformly keeps the comparison even.

GPU HOURS
    `epoch_secs` is wall-clock per epoch including evaluation, matching how the
    PMABD ladder's hours were measured. A `hours` field lands in stages.json.
    For switchable methods, one run serves several precisions, so stages.json
    also carries `hours_per_precision` = hours / len(precisions_served). Quote
    both: the raw total is what the method costs, the divided figure is what a
    single deployed precision costs.
"""

from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass

import torch
import torch.nn as nn

from .setup import evaluate, recalibrate_bn
from .schedules import Schedule

__all__ = ["TrainResult", "run_training", "build_optimizer", "build_scheduler"]


@dataclass
class TrainResult:
    best_val_top1: float
    best_epoch: int
    test_top1: float
    test_top5: float
    per_precision: dict          # {"w4a4": {"test_top1": .., "test_top5": ..}}
    epochs_run: int
    hours: float
    checkpoint: str


def build_optimizer(method, sched: Schedule):
    """Two param groups: ordinary weights, and the method's auxiliary params.

    "Auxiliary" means learned quantization ranges (SQAKD/EWGS uW/lW/uA/lA) or
    CMT-KD's importance factors pi — both of which their authors train at a
    different learning rate from the weights.
    """
    aux = list(method.aux_parameters())
    aux_ids = {id(p) for p in aux}
    weights = [p for p in method.parameters() if id(p) not in aux_ids]

    groups = [{"params": weights, "lr": sched.lr,
               "weight_decay": sched.weight_decay}]
    if aux:
        groups.append({"params": aux, "lr": sched.aux_lr or sched.lr,
                       "weight_decay": 0.0})

    if sched.optimizer.lower() == "sgd":
        return torch.optim.SGD(groups, lr=sched.lr, momentum=sched.momentum,
                               weight_decay=sched.weight_decay)
    if sched.optimizer.lower() == "adam":
        return torch.optim.Adam(groups, lr=sched.lr,
                                weight_decay=sched.weight_decay)
    raise ValueError(f"Unknown optimizer {sched.optimizer!r}")


def build_scheduler(optimizer, sched: Schedule, steps_per_epoch: int):
    if sched.lr_schedule == "multistep":
        return torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=list(sched.milestones), gamma=sched.gamma)
    if sched.lr_schedule == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(sched.epochs, 1))
    raise ValueError(f"Unknown lr_schedule {sched.lr_schedule!r}")


def _warmup_factor(epoch: int, sched: Schedule) -> float:
    if sched.warmup_epochs <= 0 or epoch >= sched.warmup_epochs:
        return 1.0
    return float(epoch + 1) / float(sched.warmup_epochs)


# ─────────────────────────────────────────────────────────────────────────────
# Mid-stage resume
#
# Power cuts on this machine have repeatedly destroyed 10-40 epochs of work.
# stages.json only records *finished* stages, so an interrupted stage restarted
# from scratch. We now snapshot after every epoch and resume from the next one.
#
# What has to be in the snapshot for a resume to be scientifically identical to
# an uninterrupted run:
#   * every trainable tensor, which for CMT-KD means the teachers and the
#     importance factors too, not just `method.model` (see _method_state)
#   * optimizer state, or SGD momentum restarts from zero
#   * scheduler state, or the LR jumps back to its epoch-0 value
#   * the running best-val selection, or a resumed run could report a worse
#     checkpoint than it had already found
#   * elapsed hours, or the GPU-hour comparison silently undercounts
# ─────────────────────────────────────────────────────────────────────────────
def _method_state(method) -> dict:
    """Everything trainable. Methods with extra trained modules override this."""
    if hasattr(method, "state_dict"):
        return method.state_dict()
    return {"model": method.model.state_dict()}


def _load_method_state(method, state: dict):
    if hasattr(method, "load_state_dict"):
        method.load_state_dict(state)
    else:
        method.model.load_state_dict(state["model"])


def _save_resume(path, payload):
    """Atomic: a crash during the write must not leave a corrupt snapshot."""
    tmp = path + ".tmp"
    torch.save(payload, tmp)
    os.replace(tmp, path)


def run_training(method, loaders, sched: Schedule, logger, *,
                 stage_id: str, stage_name: str, precision: str,
                 device: str = "cuda", output_dir: str = ".",
                 ckpt_name: str | None = None,
                 log_every: int = 50,
                 bn_recal_batches: int = 50,
                 resume: bool = True) -> TrainResult:
    """Train `method` for sched.epochs and return the selected result.

    `method` must provide:
        .model                -> nn.Module to checkpoint and evaluate
        .parameters()         -> all trainable params
        .aux_parameters()     -> params that get sched.aux_lr
        .train_batch(x, y)    -> dict with at least {"loss": float}; the method
                                 does its own forward/backward, the engine owns
                                 the optimizer step
        .eval_targets()       -> [(label, bits_or_None), ...], primary first
    """
    train_loader, val_loader, test_loader = loaders
    optimizer = build_optimizer(method, sched)
    scheduler = build_scheduler(optimizer, sched, len(train_loader))

    ckpt_name = ckpt_name or f"{stage_id}.pth"
    ckpt_path = os.path.join(output_dir, ckpt_name)

    w_bits, a_bits = method.bitwidth
    best_val, best_epoch, best_state = -1.0, -1, None
    best_test = (float("nan"), float("nan"))
    best_per_prec: dict = {}
    start_epoch = 0
    elapsed_prior = 0.0

    # ── resume an interrupted stage ──────────────────────────────────────
    resume_path = os.path.join(output_dir, f"{stage_id}_resume.pt")
    if resume and os.path.isfile(resume_path):
        try:
            ck = torch.load(resume_path, map_location=device, weights_only=False)
        except Exception as e:  # noqa: BLE001 - a corrupt snapshot must not block
            print(f"[resume] ignoring unreadable snapshot {resume_path}: {e}")
            ck = None
        # A snapshot from a different configuration would silently produce a
        # run that is neither the old one nor the new one.
        signature = (stage_id, precision, sched.epochs, sched.describe())
        if ck is not None and tuple(ck.get("signature", ())) != signature:
            print(f"[resume] snapshot does not match this configuration "
                  f"({ck.get('signature')} != {signature}) - starting fresh.")
            ck = None
        if ck is not None:
            _load_method_state(method, ck["method_state"])
            optimizer.load_state_dict(ck["optimizer"])
            scheduler.load_state_dict(ck["scheduler"])
            start_epoch = int(ck["next_epoch"])
            best_val = float(ck["best_val"])
            best_epoch = int(ck["best_epoch"])
            best_state = ck["best_state"]
            best_test = tuple(ck["best_test"])
            best_per_prec = dict(ck["best_per_prec"])
            elapsed_prior = float(ck["elapsed_hours"])
            print(f"[resume] {stage_id}: continuing from epoch "
                  f"{start_epoch + 1}/{sched.epochs} "
                  f"(best val {best_val:.2f} @ ep {best_epoch}, "
                  f"{elapsed_prior:.2f} h already spent)")

    if start_epoch >= sched.epochs:
        print(f"[resume] {stage_id} already completed all {sched.epochs} epochs.")

    stage_t0 = time.time()

    for epoch in range(start_epoch, sched.epochs):
        # Warmup multiplies the scheduler's current lr. Capture the scheduler's
        # value first: reading param_groups back after writing to them would
        # compound the factor every epoch.
        wf = _warmup_factor(epoch, sched)
        sched_lrs = [g["lr"] for g in optimizer.param_groups]
        if wf != 1.0:
            for g, base in zip(optimizer.param_groups, sched_lrs):
                g["lr"] = base * wf

        method.model.train()
        epoch_t0 = time.time()
        agg, n_batches, seen, correct = {}, 0, 0, 0
        grad_norm_acc = 0.0

        for i, (x, y) in enumerate(train_loader):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            out = method.train_batch(x, y)          # forward + backward

            if sched.grad_clip:
                gn = torch.nn.utils.clip_grad_norm_(
                    [p for p in method.parameters() if p.grad is not None],
                    sched.grad_clip)
                grad_norm_acc += float(gn)
            optimizer.step()

            for k, v in out.items():
                if isinstance(v, (int, float)) and not math.isnan(float(v)):
                    agg[k] = agg.get(k, 0.0) + float(v)
            n_batches += 1

            if "logits" in out and out["logits"] is not None:
                with torch.no_grad():
                    correct += (out["logits"].argmax(1) == y).sum().item()
                    seen += y.size(0)

            if log_every and i % log_every == 0:
                loss_v = out.get("loss", float("nan"))
                print(f"  [{stage_id}] ep {epoch + 1}/{sched.epochs} "
                      f"it {i}/{len(train_loader)} loss={loss_v:.4f}")

        # Undo the warmup multiplier before stepping: CosineAnnealingLR's
        # default step() is recursive in group["lr"], so leaving the scaled
        # value in place would fold the warmup into every later epoch.
        if wf != 1.0:
            for g, base in zip(optimizer.param_groups, sched_lrs):
                g["lr"] = base
        scheduler.step()

        # ── evaluation ───────────────────────────────────────────────────
        # BN running statistics must be re-estimated on the quantized network
        # before every eval, once per precision served. Without this, eval sits
        # at chance while training looks healthy — see recalibrate_bn's
        # docstring for the mechanism and the measured numbers.
        targets = method.eval_targets()
        primary_label, primary_bits = targets[0]

        per_prec = {}
        val_top1 = val_top5 = float("nan")
        for label, bits in targets:
            recalibrate_bn(method.model, train_loader, device,
                           batches=bn_recal_batches, bits=bits)
            t1, t5 = evaluate(method.model, test_loader, device, bits=bits)
            per_prec[label] = {"test_top1": round(t1, 2),
                               "test_top5": round(t5, 2)}
            if label == primary_label:
                val_top1, val_top5 = evaluate(method.model, val_loader,
                                              device, bits=bits)
        test_top1 = per_prec[primary_label]["test_top1"]
        test_top5 = per_prec[primary_label]["test_top5"]

        improved = val_top1 > best_val
        if improved:
            best_val, best_epoch = val_top1, epoch + 1
            best_test = (test_top1, test_top5)
            best_state = {k: v.detach().cpu().clone()
                          for k, v in method.model.state_dict().items()}
            best_per_prec = dict(per_prec)

        epoch_secs = time.time() - epoch_t0
        mean = {k: v / max(n_batches, 1) for k, v in agg.items()}
        logger.log_epoch({
            "stage": stage_name,
            "cycle": 1,
            "cycle_epoch": epoch + 1,
            "epoch": epoch + 1,
            "max_epochs": sched.epochs,
            "bitwidth_w": w_bits,
            "bitwidth_a": a_bits,
            "lr": optimizer.param_groups[0]["lr"],
            "loss": mean.get("loss"),
            "loss_ce": mean.get("loss_ce"),
            "loss_kd": mean.get("loss_kd"),
            "loss_feat": mean.get("loss_feat"),
            "loss_qat": mean.get("loss_qat"),
            "grad_norm": grad_norm_acc / max(n_batches, 1),
            "kd_w": method.loss_weights.get("kd"),
            "ce_w": method.loss_weights.get("ce"),
            "T_eff": method.loss_weights.get("T"),
            "train_top1": (100.0 * correct / seen) if seen else None,
            "val_top1": round(val_top1, 2),
            "val_top5": round(val_top5, 2),
            "test_top1": round(test_top1, 2),
            "test_top5": round(test_top5, 2),
            "best_val_so_far": round(best_val, 2),
            "epoch_secs": epoch_secs,
        })

        flag = "  *best" if improved else ""
        print(f"[{stage_id}] epoch {epoch + 1}/{sched.epochs}  "
              f"val {val_top1:.2f}  test {test_top1:.2f}  "
              f"{epoch_secs / 60:.1f} min{flag}")

        # Snapshot AFTER logging, so a crash during the write cannot lose an
        # epoch that epochs.csv already claims happened.
        if resume:
            _save_resume(resume_path, {
                "signature": (stage_id, precision, sched.epochs,
                              sched.describe()),
                "next_epoch": epoch + 1,
                "method_state": _method_state(method),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best_val": best_val,
                "best_epoch": best_epoch,
                "best_state": best_state,
                "best_test": best_test,
                "best_per_prec": best_per_prec,
                "elapsed_hours": elapsed_prior
                                 + (time.time() - stage_t0) / 3600.0,
            })

    # Hours must span every attempt, or an interrupted stage would report only
    # the time since the last resume and undercount the compute comparison.
    hours = elapsed_prior + (time.time() - stage_t0) / 3600.0

    if best_state is not None:
        method.model.load_state_dict(best_state)
        torch.save({"state_dict": best_state,
                    "best_val_top1": best_val,
                    "best_epoch": best_epoch,
                    "precision": precision}, ckpt_path)

    n_prec = len(method.eval_targets())
    logger.log_stage(stage_id, {
        "name": stage_name,
        "precision": precision.upper(),
        "best_val_top1": round(best_val, 2),
        "best_epoch": best_epoch,
        "test_top1": best_test[0],
        "test_top5": best_test[1],
        "per_precision": best_per_prec if best_state is not None else {},
        "epochs_run": sched.epochs,
        "hours": hours,
        "hours_per_precision": hours / max(n_prec, 1),
        "precisions_served": n_prec,
        "schedule": sched.describe(),
        "checkpoint": ckpt_path,
    })

    # The stage is now in stages.json, so run_matrix will skip it. Drop the
    # snapshot: keeping it would waste disk and risk a stale resume if the
    # stage were ever re-run with the same configuration.
    if resume and os.path.isfile(resume_path):
        try:
            os.remove(resume_path)
        except OSError as e:  # noqa: BLE001 - never fail a finished stage on this
            print(f"[resume] could not remove {resume_path}: {e}")

    return TrainResult(
        best_val_top1=best_val, best_epoch=best_epoch,
        test_top1=best_test[0], test_top5=best_test[1],
        per_precision=best_per_prec if best_state is not None else {},
        epochs_run=sched.epochs, hours=hours, checkpoint=ckpt_path,
    )
