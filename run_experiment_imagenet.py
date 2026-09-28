"""
run_experiment_imagenet.py
==========================
Runner for PMABD on ImageNet / TinyImageNet with MobileNetV2.
Calls pipeline.run_distillation_stage / run_distillation_stage_saturating
directly — no intermediate abstraction layer.

Data discipline (three disjoint splits, see pipeline_imagenet_patch):
    train : optimised on.
    val   : carved out of the official train split; drives checkpoint
            selection, early stopping and saturation. Never trained on.
    test  : the official val split (10k for TinyImageNet, 50k for ImageNet);
            evaluated and logged every epoch but never influences a decision.
            This is the number to report.

Usage
-----
    # TinyImageNet — pretrain the backbones first, then run the ladder
    python pretrain_tinyimagenet.py --config configs/mobilenetv2_tinyimagenet_2bit_ladder.yaml
    python run_experiment_imagenet.py --config configs/mobilenetv2_tinyimagenet_2bit_ladder.yaml

    python run_experiment_imagenet.py --config <cfg> --start_stage m5a
    python run_experiment_imagenet.py --config <cfg> --dry_run
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

try:
    import pipeline as pl
except ImportError:
    sys.exit("ERROR: pipeline.py not found. Place this file in the same directory.")

from pipeline_imagenet_patch import (
    MOBILENETV2_TAP_CHANNELS,
    MOBILENETV2_TAP_INDICES,
    assert_teacher_is_useful,
    build_loaders_from_cfg,
    download_and_save_pretrained,
    load_model_imagenet,
    parse_aux_teachers,
)
from pmabd_logging import RunLogger

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("run_imagenet")

# Stage order: (config_key, short_id, label)
STAGE_ORDER = [
    ("stage_m3_kd", "m3_kd", "FP32 KD warm-up"),
    ("stage_m4",    "m4",    "W8A8"),
    ("stage_m5a",   "m5a",   "W8A4"),
    ("stage_m5b",   "m5b",   "W4A4"),
    ("stage_m6a",   "m6a",   "W4A3"),
    ("stage_m6b",   "m6b",   "W3A3"),
    ("stage_m7a",   "m7a",   "W3A2"),
    ("stage_m7b",   "m7b",   "W2A2"),
]

STUDENT_LAYER_NAMES = ()          # MobileNetV2 taps by module, not by name
TEACHER_LAYER_NAMES = ('layer1', 'layer2', 'layer3', 'layer4')  # ResNet-style


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config",      required=True)
    p.add_argument("--dataset",     default=None, help="imagenet | tinyimagenet")
    p.add_argument("--start_stage", default=None, help="e.g. m5a")
    p.add_argument("--only_stage",  default=None,
                   help="run exactly one stage and stop, e.g. m4")
    p.add_argument("--dry_run",     action="store_true")
    p.add_argument("--num_workers", type=int, default=None)
    p.add_argument("--batch_size",  type=int, default=None)
    p.add_argument("--image_size",  type=int, default=None)
    p.add_argument("--data_root",   default=None,
                   help="Override data.data_root. Useful when the in-repo "
                        "./data is incomplete and a full extraction already "
                        "exists elsewhere on the machine — pointing at it "
                        "beats re-downloading or duplicating 110k JPEGs.")
    p.add_argument("--batch_log_every", type=int, default=50,
                   help="print a within-epoch progress line every N batches "
                        "(0 disables)")
    p.add_argument("--epoch_scale", type=float, default=None,
                   help="multiply every stage's epoch budget (e.g. 0.25 for a "
                        "quick end-to-end shakedown)")
    p.add_argument("--teacher_cache_views", type=int, default=None,
                   help="use precomputed base-teacher logits over K fixed "
                        "augmented views (build with teacher_cache.py --views K). "
                        "Caps augmentation diversity at K views per image — "
                        "state it in the paper.")
    return p.parse_args()


def load_cfg(path):
    with open(path) as f:
        return yaml.safe_load(f)


def ckpt_path(output_dir, name):
    return os.path.join(output_dir, name)


def ckpt_exists(output_dir, name):
    return bool(name) and os.path.isfile(ckpt_path(output_dir, name))


def build_student(arch, num_classes, init_from, output_dir, device,
                  small_input=False):
    """
    Load the student for a stage.

    init_from is either "pretrained" (the dataset-specific FP32 initialisation
    produced by pretrain_tinyimagenet.py, or torchvision weights on ImageNet)
    or the filename of a checkpoint this pipeline produced earlier.

    Missing checkpoints are a hard error: silently falling back to ImageNet
    weights would restart the ladder from full precision and quietly invalidate
    every stage after it.
    """
    if init_from == "pretrained":
        if num_classes == 1000:
            path = ckpt_path(output_dir, "mobilenetv2_imagenet_pretrained.pth")
            download_and_save_pretrained(arch, path, num_classes)
        else:
            path = ckpt_path(output_dir, "mobilenetv2_tin_pretrained.pth")
            if not os.path.isfile(path):
                raise FileNotFoundError(
                    f"Student initialisation {path!r} not found. For a "
                    f"{num_classes}-class dataset the FP32 student must be "
                    "fine-tuned first:\n"
                    "    python pretrain_tinyimagenet.py --config <your config>"
                )
        init_from = path
    elif init_from and not os.path.isabs(init_from):
        init_from = ckpt_path(output_dir, init_from)

    if not init_from or not os.path.isfile(init_from):
        raise FileNotFoundError(
            f"Student init checkpoint not found: {init_from!r}. The previous "
            "ladder stage must complete before this one can start.")

    # Stage checkpoints from QAT stages carry LSQ scale params the plain model
    # has no slot for; strip them so the load stays strict about real weights.
    state = torch.load(init_from, map_location="cpu", weights_only=True)
    if isinstance(state, dict):
        state = state.get("state_dict", state.get("model", state))
    clean = pl.extract_fp32_weights(state)

    model = load_model_imagenet(arch, num_classes=num_classes, pretrained=False,
                                checkpoint_path=None, device=torch.device("cpu"),
                                small_input=small_input)
    missing, unexpected = model.load_state_dict(clean, strict=False)
    if missing:
        raise RuntimeError(
            f"Loading {init_from!r} left {len(missing)} parameters uninitialised "
            f"(first: {missing[0]}). The checkpoint does not match arch {arch!r}.")
    if unexpected:
        log.warning("Ignored %d unexpected keys from %s (first: %s).",
                    len(unexpected), init_from, unexpected[0])
    log.info("Student initialised from %s", init_from)
    return model.to(device)


def run_stage(stage_cfg, student_model, teacher_models, teacher_names,
              train_loader, val_loader, test_loader, output_dir, device,
              use_saturation, epoch_callback=None, epoch_scale=None,
              batch_log_every=0):
    """
    Call pipeline.run_distillation_stage or run_distillation_stage_saturating.

    Feature distillation only uses the PRIMARY teacher (teacher_models[0])
    because ResNet50/34/18 have different channel widths at layer1-4 and
    stacking them causes shape mismatches. Logit KD uses all teachers.
    """
    bw_w  = stage_cfg.get("bitwidth_w", 32)
    bw_a  = stage_cfg.get("bitwidth_a", 32)
    beta  = stage_cfg.get("beta", 0.0)
    # feature_strategy other than "projected" means there is no projector to
    # align shapes, so any feature loss would be a shape mismatch waiting to
    # happen — force beta off.
    if stage_cfg.get("feature_strategy", "none") != "projected":
        beta = 0.0
    ckpt  = ckpt_path(output_dir, stage_cfg["checkpoint"])

    student_feat_ch = list(MOBILENETV2_TAP_CHANNELS)
    teacher_feat_ch = [256, 512, 1024, 2048]      # ResNet-50 layer1..layer4

    if beta > 0:
        active_teachers = [teacher_models[0]]
        active_names    = [teacher_names[0]]
    else:
        active_teachers = teacher_models
        active_names    = teacher_names

    # pipeline.FeatureExtractorModel taps ResNets by child name; MobileNetV2's
    # blocks live inside a Sequential and have to be tapped by module object.
    # The taps must come from whatever model instance pipeline hands us — the
    # QAT path deep-copies the student before training, so capturing modules
    # from the pre-quantization model here would hook a network that is never
    # forward-passed and silently yield empty features.
    _orig_fem = pl.FeatureExtractorModel

    class _MBv2FEM(pl.FeatureExtractorModel):
        def __init__(self, model, layer_names=(), hook_modules=None, detach_feats=False):
            # No layer_names and no explicit modules -> this is the student.
            if not layer_names and hook_modules is None:
                hook_modules = [model.features[i] for i in MOBILENETV2_TAP_INDICES]
            super().__init__(model, layer_names=layer_names,
                             hook_modules=hook_modules, detach_feats=detach_feats)

    pl.FeatureExtractorModel = _MBv2FEM

    common = dict(
        stage_name         = stage_cfg.get("name", stage_cfg.get("checkpoint", "stage")),
        teacher_models     = active_teachers,
        student_model      = student_model,
        train_loader       = train_loader,
        val_loader         = val_loader,
        test_loader        = test_loader,
        device             = device,
        lr                 = stage_cfg["lr"],
        bitwidth_w         = bw_w,
        bitwidth_a         = bw_a,
        alpha              = stage_cfg.get("alpha", 0.5),
        beta               = beta,
        temperature        = stage_cfg.get("temperature", 4.0),
        weighting_strategy = stage_cfg.get("weighting_strategy", "entropy"),
        feature_strategy   = stage_cfg.get("feature_strategy", "projected"),
        lambda_qat         = stage_cfg.get("lambda_qat", 0.0),
        grad_clip          = stage_cfg.get("grad_clip", 0.5),
        eta_min_factor     = stage_cfg.get("eta_min_factor", 0.01),
        use_rkd            = stage_cfg.get("use_rkd", True),
        warmup_epochs      = stage_cfg.get("warmup_epochs", 5),
        label_smoothing    = stage_cfg.get("label_smoothing", 0.1),
        quant_log_every    = stage_cfg.get("quant_log_every", 10),
        teacher_names      = active_names,
        checkpoint_path    = ckpt,
        patience           = stage_cfg.get("patience", 30),
        min_delta          = stage_cfg.get("min_delta", 0.05),
        student_feat_channels = student_feat_ch,
        teacher_feat_channels = teacher_feat_ch,
        student_layer_names   = STUDENT_LAYER_NAMES,
        teacher_layer_names   = TEACHER_LAYER_NAMES,
        epoch_callback        = epoch_callback,
        batch_log_every       = batch_log_every,
    )

    def _scaled(n):
        return max(1, int(round(n * epoch_scale))) if epoch_scale else n

    try:
        if use_saturation:
            model_out, best_val, best_test, cycles = pl.run_distillation_stage_saturating(
                epochs_per_cycle = _scaled(stage_cfg.get("sat_epochs_per_cycle",
                                          stage_cfg.get("epochs", 120))),
                max_cycles       = stage_cfg.get("sat_cycles", 2),
                cycle_min_delta  = stage_cfg.get("sat_min_delta", 0.1),
                cycle_lr_decay   = stage_cfg.get("sat_lr_decay", 0.6),
                **common,
            )
        else:
            cycles = 1
            model_out, best_val, best_test = pl.run_distillation_stage(
                epochs = _scaled(stage_cfg.get("epochs", 100)),
                **common,
            )
    finally:
        pl.FeatureExtractorModel = _orig_fem

    return model_out, best_val, best_test, cycles


def _register_as_teacher(stage_cfg, output_dir, num_classes, device,
                         teacher_models, teacher_names, small_input=False,
                         enabled=True, pool_max=0, n_base=3):
    """
    Add a completed stage's student to the teacher pool.

    The checkpoint may be a QAT state dict; its LSQ buffers are stripped so
    what enters the pool is the FP32 latent-weight version of that stage.

    Every teacher costs one extra forward pass per batch, so `pool_max` caps
    the total. The first `n_base` entries (the fine-tuned ResNet teachers) are
    never evicted; the oldest student-teacher is dropped first.
    """
    if not enabled:
        return
    ckpt_name = stage_cfg.get("checkpoint", "")
    if not ckpt_name:
        return
    path = ckpt_path(output_dir, ckpt_name)
    if not os.path.isfile(path):
        return
    key  = Path(ckpt_name).stem
    if key in teacher_names:
        return
    arch = stage_cfg.get("arch", "mobilenet_v2")
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
        if isinstance(state, dict):
            state = state.get("state_dict", state.get("model", state))
        clean = pl.extract_fp32_weights(state)
        model = load_model_imagenet(arch, num_classes=num_classes, pretrained=False,
                                    checkpoint_path=None, device=torch.device("cpu"),
                                    small_input=small_input)
        missing, _ = model.load_state_dict(clean, strict=False)
        if missing:
            raise RuntimeError(f"{len(missing)} params missing (first {missing[0]})")
        model = model.to(device).eval()
        for p in model.parameters():
            p.requires_grad_(False)
        teacher_models.append(model)
        teacher_names.append(key)
        if pool_max and len(teacher_models) > pool_max:
            dropped = teacher_names.pop(n_base)
            teacher_models.pop(n_base)
            log.info("Teacher pool capped at %d — evicted oldest student-teacher %r.",
                     pool_max, dropped)
        log.info("Registered %r as teacher (pool size now %d: %s).",
                 key, len(teacher_models), ", ".join(teacher_names))
    except Exception as e:
        log.warning("Could not register %r as teacher: %s", key, e)


def main():
    args = parse_args()
    cfg  = load_cfg(args.config)

    if args.dataset:
        cfg["experiment"]["dataset"] = args.dataset
    if cfg["experiment"].get("dataset") == "tinyimagenet":
        cfg["experiment"]["num_classes"] = 200
    for key, val in [("num_workers", args.num_workers),
                     ("batch_size", args.batch_size),
                     ("image_size", args.image_size),
                     ("data_root", args.data_root)]:
        if val is not None:
            cfg["data"][key] = val

    dataset     = cfg["experiment"].get("dataset", "imagenet")
    num_classes = cfg["experiment"].get("num_classes", 1000)
    output_dir  = cfg["experiment"]["output_dir"]
    seed        = cfg["experiment"].get("seed", 42)
    image_size  = cfg["data"].get("image_size", 224)
    small_input = (dataset == "tinyimagenet") and image_size <= 64
    device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Input size is constant for the whole ladder, so the autotuner pays for
    # itself after the first few batches. Algorithm selection only — this does
    # not change numerics the way TF32/AMP would.
    torch.backends.cudnn.benchmark = True

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)

    logger = RunLogger(output_dir, run_name=cfg["experiment"].get("name", "ladder"))
    logger.start_capture()
    logger.write_meta({"phase": "ladder", "config_path": args.config, "config": cfg,
                       "device": str(device), "small_input_stem": small_input,
                       "epoch_scale": args.epoch_scale})

    log.info("=" * 70)
    log.info("PMABD ImageNet / TinyImageNet Pipeline")
    log.info("  Config     : %s", args.config)
    log.info("  Dataset    : %s (%d classes)", dataset, num_classes)
    log.info("  Image size : %d (small-input stem: %s)", image_size, small_input)
    log.info("  Device     : %s", device)
    log.info("  Output     : %s", output_dir)
    log.info("  Logs       : %s", logger.log_dir)
    log.info("  Dry run    : %s", args.dry_run)
    log.info("=" * 70)

    if args.dry_run:
        log.info("Dry run: config parsed OK, imports OK.")
        log.info("Run without --dry_run to start training.")
        logger.close()
        return

    # ── Data ──────────────────────────────────────────────────────────────
    log.info("Building data loaders …")
    train_loader, val_loader, test_loader = build_loaders_from_cfg(cfg)

    # Fixed-view teacher cache: replace the train loader with one that renders
    # K deterministic views per image and reports the cache row for each
    # sample, then hang the cache off the loader for train_one_epoch to find.
    # val/test loaders are untouched — no teacher ever runs on them.
    if args.teacher_cache_views:
        from torch.utils.data import DataLoader as _DL
        import teacher_cache as tc

        k = args.teacher_cache_views
        base = train_loader.dataset
        expect = tc.expected_meta(cfg)
        expect["n_train"] = len(base)
        expect["num_views"] = k
        cache = tc.TeacherLogitCache.load(
            output_dir, k, tc.teacher_ckpt_paths(cfg), expect)

        nw = cfg["data"].get("num_workers", 0)
        if args.num_workers is not None:
            nw = args.num_workers
        view_ds = tc.FixedViewDataset(base, k, seed)
        sampler = tc.FixedViewSampler(len(base), k, seed)
        train_loader = _DL(view_ds, batch_size=cfg["data"]["batch_size"],
                           sampler=sampler, num_workers=nw, pin_memory=True,
                           persistent_workers=nw > 0, drop_last=True)
        train_loader.teacher_cache = cache
        log.info("Teacher cache ON — %d base teachers read from disk, "
                 "%d fixed views per image. Augmentation diversity is capped; "
                 "report this in the paper.", cache.n_base, k)

    log.info("Batches — train %d | val %d | test %d",
             len(train_loader), len(val_loader), len(test_loader))
    if val_loader is test_loader:
        log.warning("val and test loaders are the SAME object — checkpoint "
                    "selection is leaking into the reported number. Set "
                    "data.val_fraction > 0 before producing paper results.")

    # ── Teachers ───────────────────────────────────────────────────────────
    teacher_cfg = cfg["teacher"]
    t_ckpt      = ckpt_path(output_dir, teacher_cfg["checkpoint"])
    download_and_save_pretrained(teacher_cfg["arch"], t_ckpt, num_classes,
                                 small_input=small_input)
    primary_teacher = load_model_imagenet(
        teacher_cfg["arch"], num_classes=num_classes, pretrained=True,
        checkpoint_path=t_ckpt, device=device, small_input=small_input)
    primary_teacher.eval()
    for p in primary_teacher.parameters():
        p.requires_grad_(False)
    t1, t5 = pl.evaluate_topk(primary_teacher, test_loader, device)
    log.info("Primary teacher (%s) loaded — test top1=%.2f%% top5=%.2f%%",
             teacher_cfg["arch"], t1, t5)
    # A floor of None uses the automatic 10x-chance rule; lower it only for
    # smoke tests on synthetic data.
    min_teacher_top1 = cfg["experiment"].get("min_teacher_top1")
    assert_teacher_is_useful(t1, f"primary teacher M1 ({teacher_cfg['arch']})",
                             num_classes, min_teacher_top1)

    aux_list = parse_aux_teachers(cfg, output_dir, num_classes, device,
                                  small_input=small_input)

    teacher_models = [primary_teacher] + [m for _, m in aux_list]
    teacher_names  = ["M1"]            + [n for n, _ in aux_list]
    n_base_teachers = len(teacher_models)

    reg_kwargs = dict(
        small_input = small_input,
        enabled     = cfg["experiment"].get("register_stage_teachers", True),
        pool_max    = cfg["experiment"].get("teacher_pool_max", 0),
        n_base      = n_base_teachers,
    )

    teacher_report = [{"name": "M1", "arch": teacher_cfg["arch"],
                       "test_top1": t1, "test_top5": t5}]
    for (name, model), entry in zip(aux_list, cfg.get("aux_teachers", [])):
        a1, a5 = pl.evaluate_topk(model, test_loader, device)
        log.info("Aux teacher %s (%s) — test top1=%.2f%% top5=%.2f%%",
                 name, entry["arch"], a1, a5)
        assert_teacher_is_useful(a1, f"aux teacher {name} ({entry['arch']})",
                                 num_classes, min_teacher_top1)
        teacher_report.append({"name": name, "arch": entry["arch"],
                               "test_top1": a1, "test_top5": a5})
    logger.log_stage("teachers", {"precision": "FP32", "teachers": teacher_report,
                                  "best_val_top1": float("nan"),
                                  "test_top1": t1, "test_top5": t5,
                                  "epochs_run": 0, "hours": 0.0})

    # ── Stage loop ─────────────────────────────────────────────────────────
    start_key = args.start_stage.lower() if args.start_stage else None
    only_key  = args.only_stage.lower() if args.only_stage else None
    skip      = start_key is not None
    total_start = time.time()

    for cfg_key, stage_id, label in STAGE_ORDER:
        stage_cfg = cfg.get(cfg_key)
        if stage_cfg is None:
            log.info("Stage %s not in config — skipping.", cfg_key)
            continue

        if only_key and stage_id != only_key:
            if ckpt_exists(output_dir, stage_cfg.get("checkpoint", "")):
                _register_as_teacher(stage_cfg, output_dir, num_classes, device,
                                     teacher_models, teacher_names, **reg_kwargs)
            continue

        if skip:
            if stage_id == start_key:
                skip = False
            else:
                if ckpt_exists(output_dir, stage_cfg.get("checkpoint", "")):
                    log.info("Skipping %s — checkpoint exists.", cfg_key)
                    _register_as_teacher(stage_cfg, output_dir, num_classes, device,
                                         teacher_models, teacher_names, **reg_kwargs)
                    continue
                log.warning("start_stage=%s but %s checkpoint missing — "
                            "running from here.", start_key, cfg_key)
                skip = False

        # Idempotent resume
        if ckpt_exists(output_dir, stage_cfg.get("checkpoint", "")):
            log.info("Stage %s already complete — skipping.", cfg_key)
            _register_as_teacher(stage_cfg, output_dir, num_classes, device,
                                 teacher_models, teacher_names, **reg_kwargs)
            continue

        log.info("")
        log.info("--- Stage: %s : %s", cfg_key, label)
        stage_start = time.time()

        student = build_student(
            arch        = stage_cfg.get("arch", "mobilenet_v2"),
            num_classes = num_classes,
            init_from   = stage_cfg.get("init_from"),
            output_dir  = output_dir,
            device      = device,
            small_input = small_input,
        )

        use_sat = stage_cfg.get("saturation", False)

        student_out, best_val, best_test, cycles = run_stage(
            stage_cfg     = stage_cfg,
            student_model = student,
            teacher_models= teacher_models,
            teacher_names = teacher_names,
            train_loader  = train_loader,
            val_loader    = val_loader,
            test_loader   = test_loader,
            output_dir    = output_dir,
            device        = device,
            use_saturation= use_sat,
            epoch_callback= logger.log_epoch,
            epoch_scale   = args.epoch_scale,
            batch_log_every = args.batch_log_every,
        )

        # Final held-out numbers for the selected checkpoint.
        test1, test5 = pl.evaluate_topk(student_out, test_loader, device)
        elapsed = (time.time() - stage_start) / 3600
        log.info("--- Stage %s done. val %.2f%% | test top1 %.2f%% top5 %.2f%% | %.2f h",
                 cfg_key, best_val, test1, test5, elapsed)

        logger.log_stage(stage_id, {
            "config_key":    cfg_key,
            "name":          stage_cfg.get("name", ""),
            "precision":     f"W{stage_cfg.get('bitwidth_w',32)}A{stage_cfg.get('bitwidth_a',32)}",
            "best_val_top1": best_val,
            "test_top1":     test1,
            "test_top5":     test5,
            "cycles":        cycles,
            "epochs_run":    logger.count_epochs(
                                 stage_cfg.get("name", stage_cfg["checkpoint"])),
            "hours":         elapsed,
            "checkpoint":    ckpt_path(output_dir, stage_cfg["checkpoint"]),
            "teacher_pool":  list(teacher_names),
        })

        del student
        torch.cuda.empty_cache()

        _register_as_teacher(stage_cfg, output_dir, num_classes, device,
                             teacher_models, teacher_names, **reg_kwargs)

        if only_key:
            break

    total_elapsed = (time.time() - total_start) / 3600
    log.info("Pipeline complete. Total wall time: %.2f h", total_elapsed)
    log.info("Checkpoints in: %s", output_dir)
    print("\n" + "=" * 70)
    print(f"RESULTS — {cfg['experiment'].get('name','')} ({dataset})")
    print("=" * 70)
    print(logger.stage_table())
    print("=" * 70)
    logger.close()


if __name__ == "__main__":
    main()
