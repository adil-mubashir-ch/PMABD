#!/usr/bin/env python
"""
run_experiment.py (v3)
=======================
Run the KD+QAT pipeline on CIFAR-100 (ResNet32) or CIFAR-10 (ResNet20),
with patience-based early stopping and optional multi-cycle saturation
training per precision level.

Data discipline (v3 — see pipeline.py module docstring for full rationale)
----------------------------------------------------------------------------
- val is now carved from the 50k TRAIN set (stratified, default 10%), NOT
  from the test set. This lets the FULL official 10k test set be reported,
  matching standard CIFAR paper methodology (RA's requirement).
- Student stages MUST NOT use pretrained_init (hub weights saw all 50k
  train images, which would leak into the train-derived val split). This
  script raises a configuration error if a student stage sets
  pretrained_init: true. The teacher MAY use it — it is frozen and never
  selected by val accuracy.
- K-Fold CV is available: set data.n_folds > 1 and data.fold to run a
  specific fold. Re-run the same config with different `fold` values (or
  use --fold on the command line to override) to get full K-Fold coverage.

Saturation training (multi-cycle) — "is lowest precision reached?"
----------------------------------------------------------------------------
A stage with `saturating: true` in its config trains the SAME precision
level through repeated cycles (run_distillation_stage_saturating) instead
of a single pass, stopping once val accuracy improvement between cycles
drops below `cycle_min_delta`. This addresses the concern that one training
pass at a given precision may not be the optimal model obtainable at that
precision.

Patience-based stopping — "is training done?"
----------------------------------------------------------------------------
Every stage (saturating or not) uses patience-based early stopping within
each cycle: `patience` epochs without a val improvement greater than
`min_delta` ends that cycle/pass early, rather than always running the
full epoch budget.

Usage:
    python run_experiment.py --config configs/resnet32_cifar100.yaml
    python run_experiment.py --config configs/resnet32_cifar100_smoketest.yaml

    # Specific K-Fold fold (overrides config's data.fold):
    python run_experiment.py --config configs/resnet32_cifar100.yaml --fold 2

    # Skip to a specific stage:
    python run_experiment.py --config configs/resnet32_cifar100.yaml --start-stage stage_m4b
"""

import argparse
import json
import os
import sys

import torch
import yaml

from datetime import datetime

from pipeline import (
    get_cifar100_dataloaders, get_cifar10_dataloaders,
    FeatureExtractorModel,
    extract_fp32_weights, evaluate,
    run_distillation_stage, run_distillation_stage_saturating,
    train_baseline_from_scratch,
    compute_robustness_scores,
)

REPO = "chenyaofo/pytorch-cifar-models"


class Tee:
    """File-like object that mirrors writes to multiple streams."""
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()

    def flush(self):
        for s in self.streams:
            s.flush()


# ============================================================
# MODEL LOADERS
# ============================================================

def load_model(arch, dataset, pretrained=False):
    assert dataset in ("cifar10", "cifar100")
    model_name = f"{dataset}_{arch}"
    return torch.hub.load(REPO, model_name, pretrained=pretrained)


def load_or_cache_teacher(cfg, device):
    """
    Load teacher (M1) from cache or download pretrained weights.

    The teacher MAY use hub-pretrained weights even though they saw all
    50k train images: the teacher is frozen, used only to supply soft
    labels, and is never selected via val accuracy — so this leakage does
    not compromise checkpoint selection for any student.
    """
    ckpt = cfg["teacher"]["checkpoint"]
    arch = cfg["teacher"]["arch"]
    dataset = cfg["experiment"]["dataset"]

    if os.path.exists(ckpt):
        print(f"Loading teacher from cache: {ckpt}")
        model = load_model(arch, dataset, pretrained=False)
        model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    else:
        print(f"Downloading pretrained teacher: {arch}")
        model = load_model(arch, dataset, pretrained=True)
        torch.save(model.state_dict(), ckpt)
        print(f"Saved teacher → {ckpt}")

    model.to(device)
    return model


def load_student_init(stage_cfg, dataset, device, stage_key):
    """
    Load student initial weights for a stage.

    - init_from : load from a checkpoint produced by THIS pipeline (safe —
                  was itself trained only on the train-split's train_eff).
    - pretrained_init : FORBIDDEN for students in v3. Hub weights saw the
                  full 50k train set, which now includes our val split.
                  Using them would silently leak val into the student's
                  initialization, inflating early-epoch val accuracy and
                  corrupting checkpoint selection. Raises here instead of
                  silently proceeding.
    - otherwise : random init.
    """
    if stage_cfg.get("pretrained_init", False):
        raise ValueError(
            f"[{stage_key}] pretrained_init=true is not allowed for student "
            f"stages in v3. Hub-pretrained weights were trained on the full "
            f"50k CIFAR train set, which now includes the validation split "
            f"(val is carved from train, not test, in this version). Using "
            f"pretrained_init here would leak val data into the student's "
            f"starting weights and invalidate checkpoint selection.\n"
            f"Fix: remove pretrained_init from this stage's config and "
            f"either set init_from to a checkpoint produced by an earlier "
            f"pipeline stage, or omit both keys for random initialization."
        )

    arch = stage_cfg["arch"]
    model = load_model(arch, dataset, pretrained=False)

    if "init_from" in stage_cfg:
        src = stage_cfg["init_from"]
        print(f"  Init from: {src}")
        sd = torch.load(src, map_location="cpu")
        model.load_state_dict(extract_fp32_weights(sd), strict=False)
    else:
        print(f"  Random init ({arch})")

    return model


# ============================================================
# ROBUSTNESS SCORE LOADING
# ============================================================

def load_robustness_scores(cfg, teacher_key_order):
    rb_cfg = cfg.get("robustness_weighting")
    if not rb_cfg:
        return {}

    path = rb_cfg.get("scores_path")
    if not path or not os.path.exists(path):
        print(f"  [WARN] robustness_weighting.scores_path not found: {path}")
        print("  Falling back to uniform weighting for any 'robustness' stages.")
        return {}

    with open(path, encoding="utf-8") as f:
        noise_results = json.load(f)

    metric     = rb_cfg.get("metric", "auc_normalized")
    snr_levels = rb_cfg.get("snr_levels", None)

    all_keys = list(dict.fromkeys(teacher_key_order))
    scores_tensor = compute_robustness_scores(
        noise_results, all_keys, snr_levels=snr_levels, metric=metric)
    score_map = {k: scores_tensor[i] for i, k in enumerate(all_keys)}

    print(f"\n  Robustness scores ({metric}, from {os.path.basename(path)}):")
    for k, s in score_map.items():
        print(f"    {k}: {s.item():.4f}")
    return score_map


# ============================================================
# STAGE RUNNER
# ============================================================

def run_stage(stage_key, stage_cfg, teacher_models, teacher_accs, teacher_keys,
              robustness_score_map, train_loader, val_loader, test_loader,
              device, dataset):
    """
    Run a single distillation stage, respecting checkpoint caching.

    Three possible training paths, checked in this order:
      1. stage_cfg["baseline"] = true  → train_baseline_from_scratch.
         No teacher, no KD loss — plain cross-entropy with the hub's
         original SGD+Nesterov+cosine recipe. Used to reproduce the hub's
         own from-scratch training so a healthy FP32 base model can be
         obtained on OUR train/val split (val carved from train, so hub
         weights can't be used without leaking val data).
      2. stage_cfg["saturating"] = true → multi-cycle saturation KD runner.
      3. otherwise → single-pass (but still patience-aware) KD runner.

    Returns (model, val_acc, test_acc).
    """
    ckpt = stage_cfg["checkpoint"]

    if os.path.exists(ckpt):
        print(f"\n[{stage_cfg['name']}] Checkpoint found → loading {ckpt}")
        model = load_model(stage_cfg["arch"], dataset, pretrained=False)
        sd = torch.load(ckpt, map_location="cpu")
        model.load_state_dict(extract_fp32_weights(sd), strict=False)
        model.to(device)
        wrapped = FeatureExtractorModel(model)
        val_acc  = evaluate(wrapped, val_loader, device)
        test_acc = evaluate(wrapped, test_loader, device)
        wrapped.remove_hooks()
        print(f"  Loaded accuracy: val={val_acc:.2f}%  test={test_acc:.2f}%")
        return model, val_acc, test_acc

    student_init = load_student_init(stage_cfg, dataset, device, stage_key)

    # ── Path 1: from-scratch baseline (no teacher, no KD) ──────────────
    if stage_cfg.get("baseline", False):
        model, val_acc, test_acc = train_baseline_from_scratch(
            stage_name=stage_cfg["name"],
            student_model=student_init,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            device=device,
            max_epochs=stage_cfg.get("epochs", 200),
            lr=stage_cfg.get("lr", 0.1),
            momentum=stage_cfg.get("momentum", 0.9),
            weight_decay=stage_cfg.get("weight_decay", 5e-4),
            nesterov=stage_cfg.get("nesterov", True),
            eta_min=stage_cfg.get("eta_min", 0.0),
            label_smoothing=stage_cfg.get("label_smoothing", 0.0),
            patience=stage_cfg.get("patience", 30),
            min_delta=stage_cfg.get("min_delta", 0.05),
            checkpoint_path=ckpt,
        )
        return model, val_acc, test_acc

    feat_channels    = stage_cfg.get("feat_channels", [16, 32, 64])
    feature_strategy = stage_cfg.get("feature_strategy", "none")
    weighting        = stage_cfg.get("weighting_strategy", "uniform")
    patience         = stage_cfg.get("patience", 15)
    min_delta        = stage_cfg.get("min_delta", 0.05)

    rob_scores = None
    if weighting == "robustness" and robustness_score_map:
        rob_scores = torch.stack([
            robustness_score_map.get(k, torch.tensor(0.5)) for k in teacher_keys
        ])
        print(f"  Robustness scores for this stage: "
              f"{[f'{k}={robustness_score_map.get(k, 0):.4f}' for k in teacher_keys]}")

    common_kwargs = dict(
        stage_name=stage_cfg["name"],
        teacher_models=teacher_models,
        student_model=student_init,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
        bitwidth_w=stage_cfg.get("bitwidth_w", 32),
        bitwidth_a=stage_cfg.get("bitwidth_a", 32),
        alpha=stage_cfg.get("alpha", 0.5),
        beta=stage_cfg.get("beta", 0.0),
        temperature=stage_cfg.get("temperature", 4.0),
        weighting_strategy=weighting,
        teacher_accuracies=teacher_accs if weighting == "accuracy" else None,
        teacher_robustness_scores=rob_scores,
        feature_strategy=feature_strategy,
        lambda_qat=stage_cfg.get("lambda_qat", 0.0),
        grad_clip=stage_cfg.get("grad_clip", 0.5),
        eta_min_factor=stage_cfg.get("eta_min_factor", 0.01),
        use_rkd=stage_cfg.get("use_rkd", True),
        warmup_epochs=stage_cfg.get("warmup_epochs", 5),
        label_smoothing=stage_cfg.get("label_smoothing", 0.1),
        quant_log_every=stage_cfg.get("quant_log_every", 10),
        teacher_names=teacher_keys,
        checkpoint_path=ckpt,
        patience=patience,
        min_delta=min_delta,
        student_feat_channels=tuple(feat_channels),
        teacher_feat_channels=tuple(feat_channels),
    )

    # ── Path 2: multi-cycle saturation KD ───────────────────────────────
    if stage_cfg.get("saturating", False):
        model, val_acc, test_acc, n_cycles = run_distillation_stage_saturating(
            epochs_per_cycle=stage_cfg.get(
                "epochs_per_cycle", stage_cfg.get("epochs", 50)),
            lr=stage_cfg["lr"],
            max_cycles=stage_cfg.get("max_cycles", 5),
            cycle_min_delta=stage_cfg.get("cycle_min_delta", 0.10),
            cycle_lr_decay=stage_cfg.get("cycle_lr_decay", 0.6),
            **common_kwargs,
        )
        print(f"  [{stage_key}] Saturated after {n_cycles} cycle(s).")
    # ── Path 3: single-pass KD ───────────────────────────────────────────
    else:
        model, val_acc, test_acc = run_distillation_stage(
            epochs=stage_cfg.get("epochs", stage_cfg.get("epochs_per_cycle", 50)),
            lr=stage_cfg["lr"],
            **common_kwargs,
        )

    return model, val_acc, test_acc


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--binary", action="store_true",
                        help="Enable binary quantizer for W1A1 stages")
    parser.add_argument("--start-stage", default=None,
                        help="Skip to a specific stage key, e.g. stage_m4b")
    parser.add_argument("--fold", type=int, default=None,
                        help="Override data.fold for K-Fold CV (0-indexed)")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = cfg["experiment"]["dataset"]
    out_dir = cfg["experiment"]["output_dir"]

    n_folds = cfg["data"].get("n_folds", 1)
    fold    = args.fold if args.fold is not None else cfg["data"].get("fold", 0)
    if n_folds > 1:
        out_dir = os.path.join(out_dir, f"fold{fold}")

    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(out_dir, f"run_{datetime.now():%Y%m%d_%H%M%S}.log")
    log_file = open(log_path, "w", encoding="utf-8")
    sys.stdout = Tee(sys.stdout, log_file)
    sys.stderr = Tee(sys.stderr, log_file)
    print(f"Logging full console output to {log_path}")

    print(f"Device:  {device}")
    print(f"Dataset: {dataset}")
    print(f"Output:  {out_dir}")
    if n_folds > 1:
        print(f"K-Fold:  fold {fold+1}/{n_folds}")
    if torch.cuda.is_available():
        print(f"GPU:     {torch.cuda.get_device_name(0)}")
        print(f"VRAM:    {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
    if args.binary:
        print("Mode:    BINARY (W1A1 — BinaryQuantizer active)")

    ALL_STAGE_KEYS = [
        "stage_m2", "stage_m3",
        "stage_m4a", "stage_m4b",
        "stage_m4c", "stage_m4d",
        "stage_m4e", "stage_m4f", "stage_m4g",
        "stage_m4h", "stage_m4i",
    ]
    active_stages = [k for k in ALL_STAGE_KEYS if k in cfg]
    if args.start_stage:
        start_key = (args.start_stage if args.start_stage.startswith("stage_")
                    else f"stage_{args.start_stage}")
        if start_key in active_stages:
            idx = active_stages.index(start_key)
            skipped = active_stages[:idx]
            active_stages = active_stages[idx:]
            print(f"--start-stage {args.start_stage}: skipping {skipped}, "
                  f"starting at {start_key}")
        else:
            print(f"[WARN] --start-stage {args.start_stage} not found in "
                  f"config's active stages {active_stages}; ignoring.")

    def prefix(s):
        if os.path.dirname(s) == "":
            return os.path.join(out_dir, s)
        return s

    for key in ["teacher"] + active_stages:
        if key not in cfg:
            continue
        if "checkpoint" in cfg[key]:
            cfg[key]["checkpoint"] = prefix(cfg[key]["checkpoint"])
        if "init_from" in cfg[key]:
            cfg[key]["init_from"] = prefix(cfg[key]["init_from"])

    rb_cfg = cfg.get("robustness_weighting", {})
    if "scores_path" in rb_cfg and os.path.dirname(rb_cfg["scores_path"]) == "":
        rb_cfg["scores_path"] = os.path.join(out_dir, rb_cfg["scores_path"])

    # ── Data ─────────────────────────────────────────────────
    get_loaders = (get_cifar100_dataloaders if dataset == "cifar100"
                   else get_cifar10_dataloaders)
    train_loader, val_loader, test_loader = get_loaders(
        batch_size=cfg["data"]["batch_size"],
        num_workers=cfg["data"]["num_workers"],
        data_root=cfg["data"]["data_root"],
        val_fraction=cfg["data"].get("val_fraction", 0.10),
        val_seed=cfg["data"].get("val_seed", 42),
        n_folds=n_folds,
        fold=fold,
    )

    # ── Load teacher(s) ────────────────────────────────────────
    M1 = load_or_cache_teacher(cfg, device)
    M1_wrapped  = FeatureExtractorModel(M1)
    m1_val_acc  = evaluate(M1_wrapped, val_loader, device)
    m1_test_acc = evaluate(M1_wrapped, test_loader, device)
    M1_wrapped.remove_hooks()
    print(f"M1 ({cfg['teacher']['arch']}) accuracy: "
          f"val={m1_val_acc:.2f}%  test={m1_test_acc:.2f}%  "
          f"[note: val may be leakage-inflated for the frozen teacher — "
          f"only test accuracy is meaningful for M1]")

    extra = cfg.get("existing_checkpoints", {})
    loaded_teachers = {"M1": (M1, m1_val_acc)}

    for name in ["M2", "M3", "M4", "M4d", "M4g"]:
        if name in extra and os.path.exists(extra[name]):
            arch_map = {"M2": "resnet32", "M3": "resnet32",
                        "M4": "resnet32", "M4d": "resnet32", "M4g": "resnet32"}
            m = load_model(arch_map.get(name, "resnet32"), dataset, pretrained=False)
            sd = torch.load(extra[name], map_location="cpu")
            m.load_state_dict(extract_fp32_weights(sd), strict=False)
            m.to(device)
            wrapped    = FeatureExtractorModel(m)
            t_val_acc  = evaluate(wrapped, val_loader, device)
            t_test_acc = evaluate(wrapped, test_loader, device)
            wrapped.remove_hooks()
            print(f"{name} (loaded from {extra[name]}): "
                  f"val={t_val_acc:.2f}%  test={t_test_acc:.2f}%")
            loaded_teachers[name] = (m, t_val_acc)

    all_possible_teacher_keys = ["M1", "M2", "M3"]
    robustness_score_map = load_robustness_scores(cfg, all_possible_teacher_keys)

    results = {"experiment": cfg["experiment"]["name"], "dataset": dataset,
               "fold": fold if n_folds > 1 else None,
               "m1_acc": m1_test_acc, "m1_val_acc": m1_val_acc}

    def get_teachers_for_stage(stage_key):
        """
        Return (teacher_models, teacher_accs, teacher_keys) for a stage.

        NOTE on cross-architecture configs (e.g. MobileNetV2 distilled from
        reused ResNet teachers): `loaded_teachers` is pre-populated from
        `existing_checkpoints` in the config BEFORE any stage runs (see
        above). That means a teacher named "M3" can already be loaded and
        available even while the CURRENT stage_m3 entry in the config is
        busy producing a DIFFERENT model (e.g. M3_mnet) — this happens
        whenever a config reuses an upstream run's checkpoints as its
        teacher pool rather than training its own from M1+M2.
        Each branch below explicitly checks every teacher key that could
        legitimately be available at that point (M1 always; M2 once
        stage_m2 — or existing_checkpoints — has produced/loaded it; M3
        once stage_m3 has produced it OR existing_checkpoints loaded it
        upfront), so a reused M3 is never silently dropped from the
        teacher pool for stage_m3 itself.
        """
        if stage_key in ("stage_m2",):
            models = [M1];        accs = [m1_val_acc]; keys = ["M1"]
        elif stage_key in ("stage_m3",):
            M2m, M2a = loaded_teachers.get("M2", (None, None))
            M3m, M3a = loaded_teachers.get("M3", (None, None))
            models = [M1] + ([M2m] if M2m else []) + ([M3m] if M3m else [])
            accs   = [m1_val_acc] + ([M2a] if M2a else []) + ([M3a] if M3a else [])
            keys   = ["M1"] + (["M2"] if M2m else []) + (["M3"] if M3m else [])
        elif stage_key in ("stage_m4a", "stage_m4b"):
            M2m, M2a = loaded_teachers.get("M2", (None, None))
            M3m, M3a = loaded_teachers.get("M3", (None, None))
            models = [M1] + ([M2m] if M2m else []) + ([M3m] if M3m else [])
            accs   = [m1_val_acc] + ([M2a] if M2a else []) + ([M3a] if M3a else [])
            keys   = ["M1"] + (["M2"] if M2m else []) + (["M3"] if M3m else [])
        else:
            M2m, M2a = loaded_teachers.get("M2", (None, None))
            M3m, M3a = loaded_teachers.get("M3", (None, None))
            models = [M1] + ([M2m] if M2m else []) + ([M3m] if M3m else [])
            accs   = [m1_val_acc] + ([M2a] if M2a else []) + ([M3a] if M3a else [])
            keys   = ["M1"] + (["M2"] if M2m else []) + (["M3"] if M3m else [])
        return (
            [m for m in models if m is not None],
            [a for a in accs   if a is not None],
            [k for k, m in zip(keys, models) if m is not None],
        )

    reused_teacher_names = set(extra.keys()) & {"M2", "M3", "M4", "M4d", "M4g"}
    if reused_teacher_names:
        print(f"\n  Reused teachers from existing_checkpoints: "
              f"{sorted(reused_teacher_names)} — these slots will NOT be "
              f"overwritten by freshly trained stages of the same name "
              f"(important for cross-architecture configs where, e.g., "
              f"stage_m3 trains a different model than the reused M3).")

    for stage_key in active_stages:
        stage_cfg_entry = cfg[stage_key]
        teacher_models, teacher_accs, teacher_keys = get_teachers_for_stage(stage_key)

        model, val_acc, test_acc = run_stage(
            stage_key, stage_cfg_entry,
            teacher_models, teacher_accs, teacher_keys,
            robustness_score_map,
            train_loader, val_loader, test_loader, device, dataset)

        short = stage_key.replace("stage_", "").upper()
        bw_w  = stage_cfg_entry.get("bitwidth_w", 32)
        bw_a  = stage_cfg_entry.get("bitwidth_a", 32)
        label = f"W{bw_w}A{bw_a}" if bw_w < 32 else "FP32"
        print(f"{short} ({label}): val={val_acc:.2f}%  test={test_acc:.2f}%")
        results[f"{short.lower()}_acc"]     = test_acc
        results[f"{short.lower()}_val_acc"] = val_acc

        name_map = {
            "stage_m2": "M2", "stage_m3": "M3",
            "stage_m4b": "M4", "stage_m4d": "M4d", "stage_m4g": "M4g",
        }
        target_name = name_map.get(stage_key)
        if target_name is not None:
            if target_name in reused_teacher_names:
                print(f"  [TEACHER POOL] NOT registering {stage_key}'s output "
                      f"as teacher '{target_name}' — that slot is pinned to "
                      f"the reused checkpoint from existing_checkpoints. "
                      f"{stage_key}'s trained model is still saved to its "
                      f"own checkpoint file and used downstream via "
                      f"init_from, just not as a teacher under '{target_name}'.")
            else:
                loaded_teachers[target_name] = (model, val_acc)

    print("\n" + "="*55)
    print("FINAL RESULTS  (test = full official held-out test set)")
    print("="*55)
    for k, v in results.items():
        if k.endswith("_val_acc"):
            continue
        if k.endswith("_acc"):
            base  = k[:-4]
            label = base.upper()
            val_v = results.get(f"{base}_val_acc")
            if val_v is not None:
                print(f"  {label:8s}: test={v:.2f}%  (val={val_v:.2f}%)")
            else:
                print(f"  {label:8s}: test={v:.2f}%")
    print("="*55)

    results_path = os.path.join(out_dir, "results.yaml")
    with open(results_path, "w", encoding="utf-8") as f:
        yaml.dump(results, f)
    print(f"\nResults saved → {results_path}")


if __name__ == "__main__":
    main()
