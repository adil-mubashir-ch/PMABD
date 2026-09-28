"""
Data splits and FP32 initialisation shared by every baseline.

This module exists so that no baseline can accidentally differ from the PMABD
ladder on the two things that make the comparison meaningful: the data split
and the starting weights. Both come from the same code paths the ladder uses.

THE POINT OF THE WHOLE EXERCISE
    SQAKD trains its TinyImageNet MobileNetV2 teacher *from random init*
    (their run_tinyimagenet_mobilenetV2.sh `mobilenet_v2_fp` branch passes no
    --pretrained flag) for 100 epochs and reaches 58.07 top-1. The PMABD
    backbone is fine-tuned from ImageNet-pretrained weights and reaches 74.44.
    That single difference — initialisation, not protocol — accounts for
    essentially the whole 16-point gap between our table and theirs, and it is
    why their published quantized numbers cannot be quoted against ours.
    Everything here exists to remove that confound.
"""

from __future__ import annotations

import copy
import os
import sys

import torch
import torch.nn as nn
import yaml

# The ladder's own modules are the source of truth for splits and models.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from pipeline_imagenet_patch import (  # noqa: E402
    build_loaders_from_cfg, load_model_imagenet,
)

__all__ = ["load_config", "build_loaders", "build_fp32_mobilenetv2",
           "load_ladder_teacher", "CKPT", "is_cifar_hub", "dataset_norm",
           "baseline_output_root", "fp32_checkpoint_path"]

# CIFAR ResNets come from the same hub the PMABD CIFAR runner uses
# (run_experiment.py), so the baselines get the identical architecture.
CIFAR_HUB_REPO = "chenyaofo/pytorch-cifar-models"
CIFAR_HUB_ARCHS = ("resnet20", "resnet32", "resnet44", "resnet56")

_NORM = {
    "cifar100": ((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    "cifar10": ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
}
_IMAGENET_NORM = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))


def dataset_norm(cfg: dict):
    """(mean, std) the loaders normalise with — pipeline.py for CIFAR."""
    return _NORM.get(cfg["experiment"].get("dataset"), _IMAGENET_NORM)


def is_cifar_hub(cfg: dict) -> bool:
    return (cfg["experiment"].get("dataset") in _NORM
            and cfg.get("model", {}).get("arch") in CIFAR_HUB_ARCHS)


def baseline_output_root(cfg: dict) -> str:
    """Where baseline runs land.

    `baselines.output_root` in the config wins (relative to the repo root).
    Otherwise TinyImageNet keeps the historical baselines/outputs/ so its
    recorded stages.json stay put, and every other setup gets
    baselines/outputs/<dataset>_<arch>. The arch must be in the path: two
    architectures on one dataset share stage ids (e.g. sqakd_w4a4), so a
    shared folder would mark one's runs as the other's and skip them.
    """
    explicit = (cfg.get("baselines") or {}).get("output_root")
    if explicit:
        return os.path.join(_REPO_ROOT, explicit)
    root = os.path.join(_REPO_ROOT, "baselines", "outputs")
    ds = cfg["experiment"].get("dataset", "tinyimagenet")
    if ds == "tinyimagenet":
        return root
    arch = cfg.get("model", {}).get("arch", "mobilenet_v2")
    return os.path.join(root, f"{ds}_{arch}")


def fp32_checkpoint_path(cfg: dict, source: str = "kd") -> str:
    """`baselines.fp32_checkpoint` in the config wins; else output_dir/CKPT."""
    explicit = (cfg.get("baselines") or {}).get("fp32_checkpoint")
    if explicit:
        return explicit
    return os.path.join(_output_dir(cfg), CKPT[source])


def _load_cifar_hub(cfg: dict, arch: str, ckpt_path: str, device) -> nn.Module:
    """A chenyaofo CIFAR ResNet loaded with a PMABD checkpoint.

    Loaded strictly: run_experiment.py loads with strict=False, which would
    silently leave a random layer if a key were renamed -- not acceptable for
    the one checkpoint every baseline starts from.
    """
    ds = cfg["experiment"]["dataset"]
    local = os.path.join(torch.hub.get_dir(),
                         CIFAR_HUB_REPO.replace("/", "_") + "_master")
    if os.path.isdir(local):  # offline-safe: the ladder already cached it
        model = torch.hub.load(local, f"{ds}_{arch}", source="local",
                               pretrained=False)
    else:
        model = torch.hub.load(CIFAR_HUB_REPO, f"{ds}_{arch}",
                               pretrained=False, trust_repo=True)
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    from pipeline import extract_fp32_weights
    sd = {k[len("model."):] if k.startswith("model.") else k: v
          for k, v in extract_fp32_weights(sd).items()}
    model.load_state_dict(sd, strict=True)
    return model.to(device)


# Checkpoints produced by the PMABD run, relative to the config's output_dir.
CKPT = {
    "pretrained": "mobilenetv2_tin_pretrained.pth",  # plain FP32 fine-tune, 74.44
    "kd": "M_fp32.pth",                              # after FP32 multi-teacher KD, 75.57
    "resnet50": "M1.pth",
    "resnet34": "M2.pth",
    "resnet18": "M3.pth",
}


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_loaders(cfg: dict, batch_size: int | None = None,
                  num_workers: int | None = None):
    """Identical train/val/test splits to the ladder.

    train = 90% of official train, val = the held-out 10%, test = the official
    10k val split. The carve is class-stratified and seeded by data.val_seed,
    so it reproduces exactly across baselines and across reruns.

    Note this is stricter than SQAKD's own protocol, which selects its best
    checkpoint on the same 10k split it reports. Ours never lets the reported
    split influence a decision. Say so in the paper — under our harness the
    baselines get the stricter treatment too, so the comparison stays even.
    """
    cfg = copy.deepcopy(cfg)
    if batch_size is not None:
        cfg["data"]["batch_size"] = batch_size
    if num_workers is not None:
        cfg["data"]["num_workers"] = num_workers
    return build_loaders_from_cfg(cfg)


def _output_dir(cfg: dict) -> str:
    return cfg["experiment"]["output_dir"]


def build_fp32_mobilenetv2(cfg: dict, source: str = "kd",
                           device: str = "cuda") -> nn.Module:
    """The FP32 MobileNetV2 every baseline starts from (and distils from).

    source="kd"          -> M_fp32.pth, the model PMABD's own W8A8 rung was
                            initialised from. This is the default because it
                            is the fair one: giving the baselines a weaker
                            starting point than our own ladder had would hand
                            us a free head start.
    source="pretrained"  -> mobilenetv2_tin_pretrained.pth, the plain
                            ImageNet->TinyImageNet fine-tune with no PMABD
                            distillation in it at all. Use this for the
                            stricter reading in which M_fp32 counts as part of
                            our method rather than as shared setup.

    Report which one was used. The two differ by ~1.1 points top-1, which is
    smaller than the effects under study but not negligible.
    """
    if source not in ("kd", "pretrained"):
        raise ValueError(f"source must be 'kd' or 'pretrained', got {source!r}")

    num_classes = cfg["experiment"]["num_classes"]
    # Architecture comes from the config so the same baseline code serves
    # MobileNetV2 on TinyImageNet and a ResNet on CIFAR-100. Defaults to
    # mobilenet_v2, so existing configs are unaffected.
    arch = cfg.get("model", {}).get("arch", "mobilenet_v2")
    ckpt_path = fp32_checkpoint_path(cfg, source)
    if is_cifar_hub(cfg):
        # CIFAR: the PMABD ladder's FP32 student (M2.pth) is the model its W8A8
        # rung was initialised from, so it is the fair start for every
        # baseline. There is no separate "pretrained" model, so --fp32-source
        # is ignored here.
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(
                f"CIFAR FP32 checkpoint not found: {ckpt_path}\n"
                "Set baselines.fp32_checkpoint in the config to the PMABD "
                "run's M2.pth.")
        return _load_cifar_hub(cfg, arch, ckpt_path, device)
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(
            f"FP32 MobileNetV2 checkpoint not found: {ckpt_path}\n"
            "Run the PMABD pretraining first:\n"
            "  python pretrain_tinyimagenet.py --config "
            "configs/mobilenetv2_tinyimagenet_2bit_ladder.yaml"
        )

    # A 32x32 input through an ImageNet stem (32x downsample) would leave a
    # 1x1 final feature map, so CIFAR-sized inputs get the stem rewritten. This
    # must happen before the checkpoint load so the shapes line up.
    small = int(cfg["data"].get("image_size", 224)) <= 64
    model = load_model_imagenet(
        arch=arch, pretrained=False, num_classes=num_classes,
        checkpoint_path=ckpt_path, device=device, small_input=small,
    )
    return model


def load_ladder_teacher(cfg: dict, arch: str, device: str = "cuda") -> nn.Module:
    """One of the PMABD ResNet teachers (M1/M2/M3), frozen and in eval mode.

    Only CMT-KD-style experiments that want a cross-architecture teacher need
    this. SQAKD, DAQAKD, Any-Precision and InstantNet all distil from a model
    of the student's own architecture, so they use build_fp32_mobilenetv2.
    """
    if arch not in ("resnet50", "resnet34", "resnet18"):
        raise ValueError(f"Not a ladder teacher arch: {arch!r}")
    ckpt_path = os.path.join(_output_dir(cfg), CKPT[arch])
    model = load_model_imagenet(
        arch=arch, pretrained=False,
        num_classes=cfg["experiment"]["num_classes"],
        checkpoint_path=ckpt_path, device=device,
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def freeze(model: nn.Module) -> nn.Module:
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


@torch.no_grad()
def recalibrate_bn(model: nn.Module, loader, device: str = "cuda",
                   batches: int = 50, bits=None) -> nn.Module:
    """Re-estimate BatchNorm running statistics on the *quantized* network.

    WHY THIS IS MANDATORY, NOT AN OPTIMISATION
        Every baseline here starts from an FP32 checkpoint and inherits its BN
        running statistics. Both quantizer families then change the activation
        distribution in a way those statistics do not describe:

          * The EWGS/SQAKD quantizer maps activations as (x - lA)/(uA - lA).
            When lA != 0 that is an *affine* map, and the -lA/span constant
            injects a bias no single output_scale can undo. The authors' models
            feed every quantized conv from a ReLU, so lA = 0 and the map is
            purely multiplicative. MobileNetV2's inverted residuals feed each
            expand conv from a *linear* bottleneck, which is signed — measured
            lA of -60.08 at features.2.conv.0.0 and -13.30 at features.18.0.
          * The switchable DoReFa quantizer has the same unsigned assumption.

        BatchNorm absorbs the shift in train mode because it subtracts the
        batch mean, so training looks perfectly healthy. Evaluation reads the
        stale running statistics and collapses to chance. Measured at init:
        W8A8 0.00% -> 69.53% after recalibration, W4A4 0.00% -> 51.17%.

        Both source repos do this too — InstantNet ships calibrate_bn.py and
        Any-Precision has an update_bn path — so it is standard practice in
        this literature rather than a fix invented here.

    momentum=None makes BatchNorm accumulate a cumulative average rather than
    an exponential one, which is the correct estimator for a fresh pass.
    """
    bn_types = (nn.BatchNorm1d, nn.BatchNorm2d)

    if bits is not None:
        from .quant_switchable import set_bits
        set_bits(model, bits)

    # A switchable model keeps one BN per bit-width. Resetting every BN in the
    # tree would wipe the branches belonging to the other precisions, which are
    # separately calibrated and still needed — so touch only the active branch.
    from .quant_switchable import SwitchBatchNorm2d
    switch_mods = [m for m in model.modules()
                   if isinstance(m, SwitchBatchNorm2d)]
    branch_ids = {id(bn) for m in switch_mods for bn in m.bn_dict.values()}
    bns = [m.bn_dict[str(m.abit)] for m in switch_mods]
    bns += [m for m in model.modules()
            if isinstance(m, bn_types) and id(m) not in branch_ids]
    if not bns:
        return model

    saved_momentum = [m.momentum for m in bns]
    for m in bns:
        m.reset_running_stats()
        m.momentum = None

    was_training = model.training
    model.train()
    for i, (x, _y) in enumerate(loader):
        model(x.to(device, non_blocking=True))
        if i + 1 >= batches:
            break
    model.train(was_training)

    for m, mom in zip(bns, saved_momentum):
        m.momentum = mom
    return model


@torch.no_grad()
def evaluate(model: nn.Module, loader, device: str = "cuda",
             bits: int | None = None) -> tuple[float, float]:
    """top-1 / top-5 percentages.

    `bits` switches a shared-weight switchable model to that precision first;
    it is ignored by per-precision models.
    """
    if bits is not None:
        from .quant_switchable import set_bits
        set_bits(model, bits)

    was_training = model.training
    model.eval()
    top1 = top5 = n = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        _, pred = logits.topk(5, dim=1)
        correct = pred.eq(y.view(-1, 1))
        top1 += correct[:, :1].sum().item()
        top5 += correct.sum().item()
        n += y.size(0)
    model.train(was_training)
    return 100.0 * top1 / max(n, 1), 100.0 * top5 / max(n, 1)
