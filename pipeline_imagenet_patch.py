"""
pipeline_imagenet_patch.py
==========================
Drop this file alongside pipeline.py and run_experiment.py.
Import it at the TOP of run_experiment.py with:

    from pipeline_imagenet_patch import (
        load_model_imagenet,
        build_imagenet_loaders,
        build_tinyimagenet_loaders,
        download_and_save_pretrained,
        get_feat_channels_for_arch,
        parse_aux_teachers,
        saturation_aware_run_stage,
    )

Then replace the relevant sections in run_experiment.py as described
in the integration comments throughout this file.

Changes required in pipeline.py itself are minimal and listed at the
bottom of this file under PIPELINE_PY_CHANGES.
"""

import os
import math
import copy
import logging
from pathlib import Path

import torch
import torch.nn as nn
import torchvision
import torchvision.models as tv_models
import torchvision.transforms as T
from torch.utils.data import DataLoader, random_split, Subset
from torchvision.datasets import ImageFolder

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# 1.  MODEL LOADING
# ─────────────────────────────────────────────────────────────────────────────

_TV_ARCH_MAP = {
    # torchvision arch string → (constructor, default weights enum)
    "resnet18":    (tv_models.resnet18,    "ResNet18_Weights.IMAGENET1K_V1"),
    "resnet34":    (tv_models.resnet34,    "ResNet34_Weights.IMAGENET1K_V1"),
    "resnet50":    (tv_models.resnet50,    "ResNet50_Weights.IMAGENET1K_V2"),
    "resnet101":   (tv_models.resnet101,   "ResNet101_Weights.IMAGENET1K_V2"),
    "mobilenet_v2":(tv_models.mobilenet_v2,"MobileNet_V2_Weights.IMAGENET1K_V2"),
}

# Feature-tap channel widths for each arch
_FEAT_CHANNELS = {
    "resnet18":    [64,  128, 256, 512],
    "resnet34":    [64,  128, 256, 512],
    "resnet50":    [256, 512, 1024, 2048],
    "resnet101":   [256, 512, 1024, 2048],
    "mobilenet_v2":[32,  64,  96,  320],
}


def get_feat_channels_for_arch(arch: str) -> list[int]:
    if arch not in _FEAT_CHANNELS:
        raise ValueError(f"Unknown arch for feat_channels lookup: {arch!r}. "
                         f"Add it to _FEAT_CHANNELS in pipeline_imagenet_patch.py.")
    return _FEAT_CHANNELS[arch]


def load_model_imagenet(
    arch: str,
    num_classes: int = 1000,
    pretrained: bool = True,
    checkpoint_path: str | None = None,
    device: torch.device | None = None,
    small_input: bool = False,
    strict: bool = True,
) -> nn.Module:
    """
    Load a model for ImageNet / TinyImageNet experiments.

    Priority:
      1. checkpoint_path exists on disk  -> load that state dict
      2. pretrained=True                 -> torchvision ImageNet weights
      3. otherwise                       -> random init

    For TinyImageNet (num_classes=200) the final Linear is replaced. Note that
    a replaced head is RANDOM until something trains it — see
    `pretrain_tinyimagenet.py`, and `assert_teacher_is_useful` below.

    small_input=True rewrites the stem for 64px inputs (see
    adapt_model_for_small_input). It is applied BEFORE the checkpoint load so
    the checkpoint's stem shapes line up.

    strict=False is needed when loading a QAT state dict into a plain model;
    prefer pipeline.extract_fp32_weights and keep strict=True.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if arch not in _TV_ARCH_MAP:
        raise ValueError(f"Arch {arch!r} not supported for ImageNet. "
                         f"Supported: {list(_TV_ARCH_MAP)}")

    constructor, weights_str = _TV_ARCH_MAP[arch]

    def _fresh_pretrained():
        try:
            weights_cls_str, attr = weights_str.rsplit(".", 1)
            weights_cls = getattr(tv_models, weights_cls_str)
            return constructor(weights=getattr(weights_cls, attr))
        except AttributeError:
            log.warning("torchvision weights= API not available; using pretrained=True")
            return constructor(pretrained=True)  # noqa: deprecated fallback

    if checkpoint_path and os.path.isfile(checkpoint_path):
        log.info("Loading %s from checkpoint: %s", arch, checkpoint_path)
        model = constructor(weights=None)
        _replace_classifier_if_needed(model, arch, num_classes)
        if small_input:
            adapt_model_for_small_input(model, arch)
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if isinstance(state, dict):
            state = state.get("state_dict", state.get("model", state))
        missing, unexpected = model.load_state_dict(state, strict=strict)
        if missing or unexpected:
            log.warning("Checkpoint %s loaded non-strictly: %d missing, %d unexpected "
                        "keys (first missing: %s)", checkpoint_path, len(missing),
                        len(unexpected), missing[0] if missing else "-")
    elif pretrained:
        log.info("Loading %s with pretrained ImageNet weights.", arch)
        model = _fresh_pretrained()
        _replace_classifier_if_needed(model, arch, num_classes)
        if small_input:
            adapt_model_for_small_input(model, arch)
    else:
        log.info("Random-init %s (num_classes=%d).", arch, num_classes)
        model = constructor(weights=None)
        _replace_classifier_if_needed(model, arch, num_classes)
        if small_input:
            adapt_model_for_small_input(model, arch)

    return model.to(device)


def _replace_classifier_if_needed(model: nn.Module, arch: str, num_classes: int):
    """Replace the final head if num_classes != 1000."""
    if num_classes == 1000:
        return
    if arch.startswith("resnet"):
        in_f = model.fc.in_features
        model.fc = nn.Linear(in_f, num_classes)
    elif arch == "mobilenet_v2":
        in_f = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_f, num_classes)


def get_classifier(model: nn.Module, arch: str) -> nn.Module:
    """Return the final Linear layer for a supported arch."""
    if arch.startswith("resnet"):
        return model.fc
    if arch == "mobilenet_v2":
        return model.classifier[1]
    raise ValueError(f"get_classifier: unsupported arch {arch!r}")


def download_and_save_pretrained(arch: str, save_path: str, num_classes: int = 1000,
                                 allow_random_head: bool = False,
                                 small_input: bool = False):
    """
    Materialise a starting checkpoint at save_path if one is not there already.

    For num_classes == 1000 this is just the torchvision ImageNet checkpoint
    and is exactly what a teacher should be.

    For any other num_classes the torchvision head does not fit and gets
    replaced by a RANDOM Linear. Saving that as a "teacher" would silently
    give you a teacher that predicts noise — which is the single most
    expensive mistake available here — so it refuses unless the caller opts in
    with allow_random_head=True (only `pretrain_tinyimagenet.py` should).
    """
    if os.path.isfile(save_path):
        log.info("Checkpoint already exists, skipping download: %s", save_path)
        return
    if num_classes != 1000 and not allow_random_head:
        raise FileNotFoundError(
            f"No checkpoint at {save_path!r} and num_classes={num_classes} != 1000.\n"
            "torchvision only ships 1000-class heads, so auto-downloading here would "
            "hand you a teacher with a RANDOM classifier (~0.5% accuracy) and every "
            "distillation number downstream would be meaningless.\n"
            "Fine-tune the backbones on this dataset first:\n"
            "    python pretrain_tinyimagenet.py --config <your config>\n"
            "That writes M1.pth / M2.pth / M3.pth / mobilenetv2_tin_pretrained.pth "
            "into the output directory."
        )
    log.info("Downloading pretrained %s -> %s", arch, save_path)
    model = load_model_imagenet(arch, num_classes=num_classes, pretrained=True,
                                small_input=small_input, device=torch.device("cpu"))
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_path)
    log.info("Saved.")


def assert_teacher_is_useful(top1: float, name: str, num_classes: int,
                             min_top1: float | None = None):
    """
    Refuse a teacher that is no better than chance.

    torchvision only ships 1000-class heads, so on TinyImageNet a backbone
    loaded without a fine-tuned checkpoint carries a RANDOM 200-way classifier.
    Frozen in the teacher pool it emits pure noise at ~0.5% accuracy, and every
    distillation number downstream is quietly worthless. Measured accuracy is
    the only unambiguous way to catch that, so this is checked against real
    data rather than inferred from the weights.

    The floor is deliberately far below any usable teacher (10x chance, or 5%,
    whichever is larger) — it separates "random" from "trained", not "weak"
    from "strong".
    """
    chance = 100.0 / max(num_classes, 1)
    floor = min_top1 if min_top1 is not None else max(5.0, 10.0 * chance)
    if top1 < floor:
        raise RuntimeError(
            f"{name}: top-1 accuracy is {top1:.2f}%, below the {floor:.2f}% floor "
            f"(chance is {chance:.2f}%). This model was never trained on this "
            "dataset — most likely its classifier head is still the random "
            "replacement torchvision's 1000-class head was swapped for. "
            "Distilling from it would produce noise. "
            "Fix: python pretrain_tinyimagenet.py --config <your config>"
        )
    log.info("%s: top-1 %.2f%% (floor %.2f%%) — teacher is trained.",
             name, top1, floor)


# ─────────────────────────────────────────────────────────────────────────────
# 2.  DATA LOADERS
# ─────────────────────────────────────────────────────────────────────────────

# Normalisation: every backbone is initialised from torchvision ImageNet
# weights, so ImageNet statistics MUST be used here — dataset-specific stats
# would shift the input distribution away from what those pretrained features
# expect and cost several points of accuracy.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def _imagenet_transforms(split: str, image_size: int = 224):
    if split == "train":
        return T.Compose([
            T.RandomResizedCrop(image_size),
            T.RandomHorizontalFlip(),
            T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
    else:  # val / test
        return T.Compose([
            T.Resize(int(image_size * 256 / 224)),
            T.CenterCrop(image_size),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])


def _tinyimagenet_transforms(split: str, image_size: int = 224):
    """
    TinyImageNet source images are 64x64. Following SQAKD (Zhao & Zhao,
    AISTATS 2024) and DAQAKD (Kur & Zhao, 2025) — the two works whose
    TinyImageNet numbers we benchmark against — images are upsampled to
    `image_size` (224 by default) and the ImageNet-pretrained backbones are
    used unmodified. Set image_size=64 for the cheap native-resolution
    ablation, which also needs `adapt_model_for_small_input`.

    Augmentation is deliberately light (random crop, horizontal flip,
    normalise) to match those reference setups.
    """
    if split == "train":
        if image_size <= 64:
            return T.Compose([
                T.RandomCrop(image_size, padding=image_size // 8),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ])
        return T.Compose([
            T.Resize(image_size),
            T.RandomCrop(image_size, padding=image_size // 16),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
    # val / test — deterministic, no augmentation
    return T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def _stratified_split(targets, num_classes, val_fraction, seed):
    """
    Class-balanced deterministic split of a labelled set into
    (train_indices, val_indices). Same algorithm as
    pipeline._make_stratified_kfold_split with n_folds=1, kept here so this
    module stays importable without pipeline.py.
    """
    g = torch.Generator().manual_seed(seed)
    by_class: dict[int, list[int]] = {c: [] for c in range(num_classes)}
    for idx, t in enumerate(targets):
        by_class[int(t)].append(idx)

    train_idx, val_idx = [], []
    for c in range(num_classes):
        idxs = torch.tensor(by_class[c], dtype=torch.long)
        if len(idxs) == 0:
            continue
        perm = idxs[torch.randperm(len(idxs), generator=g)]
        n_val = max(1, int(round(val_fraction * len(idxs))))
        val_idx.extend(perm[:n_val].tolist())
        train_idx.extend(perm[n_val:].tolist())
    return sorted(train_idx), sorted(val_idx)


def _make_loaders(train_subset, val_subset, test_set, batch_size,
                  num_workers, pin_memory, seed=42):
    gen = torch.Generator().manual_seed(seed)
    persist = num_workers > 0
    train_loader = DataLoader(
        train_subset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory,
        persistent_workers=persist, drop_last=True, generator=gen,
    )
    val_loader = DataLoader(
        val_subset, batch_size=batch_size * 2, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
        persistent_workers=persist,
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size * 2, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
        persistent_workers=persist,
    )
    return train_loader, val_loader, test_loader


def build_imagenet_loaders(
    data_root: str,
    batch_size: int = 128,
    num_workers: int = 8,
    image_size: int = 224,
    pin_memory: bool = True,
    val_fraction: float = 0.02,
    val_seed: int = 42,
):
    """
    Build ImageNet train/val/test loaders with no overlap between the three.

      train : (1 - val_fraction) of the official train split, augmented.
      val   : val_fraction of the official train split, clean transform.
              Drives checkpoint selection / early stopping / saturation.
      test  : the official 50k val split, clean transform. Reported only.

    val_fraction=0.0 restores the legacy behaviour where val and test are the
    same official split. That leaks the reported number into checkpoint
    selection, so it warns loudly.

    Returns (train_loader, val_loader, test_loader).
    """
    train_dir = os.path.join(data_root, "train")
    val_dir   = os.path.join(data_root, "val")

    if not os.path.isdir(train_dir) or not os.path.isdir(val_dir):
        raise FileNotFoundError(
            f"ImageNet not found at {data_root!r}. "
            "Expected subdirectories: train/ and val/ in standard ImageFolder layout."
        )

    train_aug   = ImageFolder(train_dir, transform=_imagenet_transforms("train", image_size))
    train_clean = ImageFolder(train_dir, transform=_imagenet_transforms("val",   image_size))
    test_set    = ImageFolder(val_dir,   transform=_imagenet_transforms("val",   image_size))

    if val_fraction <= 0.0:
        log.warning("val_fraction=0 — val and test are BOTH the official val split. "
                    "The reported test number is no longer held out; use this only "
                    "for quick debugging, never for a paper table.")
        train_subset, val_subset = train_aug, test_set
    else:
        tr_idx, va_idx = _stratified_split(
            train_aug.targets, len(train_aug.classes), val_fraction, val_seed)
        train_subset = Subset(train_aug, tr_idx)
        val_subset   = Subset(train_clean, va_idx)

    log.info("ImageNet splits: train=%d  val=%d  test=%d (official val, untouched)",
             len(train_subset), len(val_subset), len(test_set))
    return _make_loaders(train_subset, val_subset, test_set,
                         batch_size, num_workers, pin_memory, val_seed)


def build_tinyimagenet_loaders(
    data_root: str,
    batch_size: int = 128,
    num_workers: int = 0,
    image_size: int = 224,
    pin_memory: bool = True,
    val_fraction: float = 0.10,
    val_seed: int = 42,
):
    """
    Build TinyImageNet train/val/test loaders as three DISJOINT splits.

    TinyImageNet ships only train (100k labelled) and val (10k labelled); the
    official test split has no public labels, so every published paper reports
    on the 10k val split. To report on that split honestly we must not also use
    it for checkpoint selection, so the val that drives training decisions is
    carved out of the official train split instead:

      train : 90% of the official train split (450 img/class), augmented.
      val   : 10% of the official train split ( 50 img/class), clean transform.
              Drives checkpoint selection, early stopping, saturation.
      test  : the official 10k val split (50 img/class), clean transform.
              Evaluated and logged only — never influences any decision.
              This is the number directly comparable to published work.

    The train/val carve is class-stratified and seeded, so it is identical
    across every stage of the ladder and across reruns.

    Expects data_root/tiny-imagenet-200/{train,val} (standard extraction);
    auto-downloads if absent.

    Returns (train_loader, val_loader, test_loader).
    """
    tiny_root = os.path.join(data_root, "tiny-imagenet-200")

    if not os.path.isdir(tiny_root):
        log.info("tiny-imagenet-200 not found at %s — attempting download.", tiny_root)
        _download_tinyimagenet(data_root)
    else:
        log.info("Found existing TinyImageNet at %s", tiny_root)

    train_dir = os.path.join(tiny_root, "train")
    val_images_dir = os.path.join(tiny_root, "val", "images")
    val_organised_dir = os.path.join(tiny_root, "val_organised")

    if not os.path.isdir(train_dir):
        raise FileNotFoundError(
            "TinyImageNet train/ not found at " + str(train_dir) +
            ". Expected: <data_root>/tiny-imagenet-200/train/<class_id>/*.JPEG"
        )

    # Reorganise flat val/images/ into val_organised/<class_id>/ for ImageFolder
    if os.path.isdir(val_images_dir):
        _reorganise_tinyimagenet_val(tiny_root, val_images_dir)

    if os.path.isdir(val_organised_dir):
        test_dir = val_organised_dir
    elif os.path.isdir(os.path.join(tiny_root, "val", "n01443537")):
        # Some extractions already have class subfolders directly under val/
        test_dir = os.path.join(tiny_root, "val")
    else:
        raise FileNotFoundError(
            "Cannot find usable TinyImageNet val split. "
            "Make sure val/images/ and val/val_annotations.txt exist."
        )

    tf_train = _tinyimagenet_transforms("train", image_size)
    tf_clean = _tinyimagenet_transforms("val",   image_size)

    train_aug   = ImageFolder(train_dir, transform=tf_train)
    train_clean = ImageFolder(train_dir, transform=tf_clean)
    test_set    = ImageFolder(test_dir,  transform=tf_clean)

    # A silent class-order mismatch between the two ImageFolders would relabel
    # the whole test set, so check it rather than trusting it.
    if train_aug.classes != test_set.classes:
        raise RuntimeError(
            "TinyImageNet train and val class lists differ — the label spaces do "
            "not line up and every test number would be meaningless.\n"
            f"  train[:3]={train_aug.classes[:3]} ({len(train_aug.classes)} classes)\n"
            f"  val[:3]  ={test_set.classes[:3]} ({len(test_set.classes)} classes)\n"
            "This usually means train/ and val/ came from different sources "
            "(e.g. the HuggingFace export mixed with the Stanford zip). "
            f"Delete {tiny_root!r} and re-extract from a single source."
        )

    if val_fraction <= 0.0:
        log.warning("val_fraction=0 — val and test are BOTH the official val split. "
                    "The reported test number is no longer held out; use this only "
                    "for quick debugging, never for a paper table.")
        train_subset, val_subset = train_aug, test_set
    else:
        tr_idx, va_idx = _stratified_split(
            train_aug.targets, len(train_aug.classes), val_fraction, val_seed)
        train_subset = Subset(train_aug, tr_idx)
        val_subset   = Subset(train_clean, va_idx)

    log.info("TinyImageNet splits [stratified, seed=%d, val_fraction=%.2f, "
             "image_size=%d]: train=%d  val=%d  test=%d (official val, untouched)",
             val_seed, val_fraction, image_size,
             len(train_subset), len(val_subset), len(test_set))
    return _make_loaders(train_subset, val_subset, test_set,
                         batch_size, num_workers, pin_memory, val_seed)


def _download_tinyimagenet(data_root: str):
    """
    Download TinyImageNet.

    The canonical Stanford zip is tried first because it is the only source
    that gives wnid class folders for BOTH train and val, guaranteeing a
    consistent label space. The HuggingFace export is the fallback; it is
    written out with synthetic `cls_%03d` folder names applied identically to
    both splits so the two still agree.
    """
    import urllib.request, zipfile

    tiny_root = os.path.join(data_root, "tiny-imagenet-200")
    os.makedirs(data_root, exist_ok=True)

    urls = [
        "http://cs231n.stanford.edu/tiny-imagenet-200.zip",
        "https://image-net.org/data/tiny-imagenet-200.zip",
    ]
    zip_path = os.path.join(data_root, "tiny-imagenet-200.zip")
    for url in urls:
        try:
            log.info("Downloading TinyImageNet from %s …", url)
            urllib.request.urlretrieve(url, zip_path)
            with zipfile.ZipFile(zip_path) as z:
                log.info("Extracting …")
                z.extractall(data_root)
            os.remove(zip_path)
            log.info("TinyImageNet extracted to %s", tiny_root)
            return
        except Exception as e:
            log.warning("URL %s failed: %s", url, e)
            if os.path.isfile(zip_path):
                os.remove(zip_path)

    try:
        from datasets import load_dataset
        log.info("Falling back to HuggingFace datasets …")
        ds = load_dataset("zh-plus/tiny-imagenet")
        os.makedirs(tiny_root, exist_ok=True)

        for split_name, hf_key in [("train", "train"), ("val_organised", "valid")]:
            hf_split = ds[hf_key]
            split_dir = os.path.join(tiny_root, split_name)
            log.info("Writing %s split (%d images) …", split_name, len(hf_split))
            for i, sample in enumerate(hf_split):
                # Zero-padded synthetic wnid keeps train and val folder names
                # identical AND lexicographically ordered by label index, so
                # ImageFolder assigns the same integer to the same class in both.
                cls_dir = os.path.join(split_dir, f"cls_{int(sample['label']):03d}")
                os.makedirs(cls_dir, exist_ok=True)
                sample["image"].convert("RGB").save(
                    os.path.join(cls_dir, f"{split_name}_{i:06d}.jpg"))
                if i % 10000 == 0 and i > 0:
                    log.info("  … %d/%d", i, len(hf_split))
        with open(os.path.join(tiny_root, "val_organised", ".complete"), "w") as f:
            f.write("written from huggingface\n")
        log.info("TinyImageNet written from HuggingFace to %s", tiny_root)
        return
    except ImportError:
        log.warning("HuggingFace 'datasets' library not installed.")
    except Exception as e:
        log.warning("HuggingFace download failed: %s", e)

    raise RuntimeError(
        "Could not download TinyImageNet automatically.\n"
        "Download http://cs231n.stanford.edu/tiny-imagenet-200.zip manually and\n"
        "extract it so that this path exists: " + os.path.join(tiny_root, "train")
    )


def _reorganise_tinyimagenet_val(tiny_root: str, val_images_dir: str):
    """Convert flat TinyImageNet val into ImageFolder-compatible class subdirs."""
    out_dir = os.path.join(tiny_root, "val_organised")
    done_marker = os.path.join(out_dir, ".complete")
    if os.path.isfile(done_marker):
        return
    import shutil
    annotations = os.path.join(tiny_root, "val", "val_annotations.txt")
    if not os.path.isfile(annotations):
        raise FileNotFoundError(
            f"{annotations} missing — cannot label the flat val/images folder.")
    with open(annotations) as f:
        rows = [line.strip().split("\t") for line in f if line.strip()]
    n = 0
    for fname, cls, *_ in rows:
        cls_dir = os.path.join(out_dir, cls)
        os.makedirs(cls_dir, exist_ok=True)
        src = os.path.join(val_images_dir, fname)
        dst = os.path.join(cls_dir, fname)
        if os.path.isfile(src) and not os.path.isfile(dst):
            shutil.copy2(src, dst)
            n += 1
    with open(done_marker, "w") as f:
        f.write(f"{len(rows)} images organised\n")
    log.info("TinyImageNet val reorganised -> %s (%d images copied)", out_dir, n)


# ─────────────────────────────────────────────────────────────────────────────
# 2.5  SMALL-INPUT STEM ADAPTATION (only for the native-64px ablation)
# ─────────────────────────────────────────────────────────────────────────────

def adapt_model_for_small_input(model: nn.Module, arch: str) -> nn.Module:
    """
    Rework the aggressive ImageNet stem so a 64x64 input does not collapse to a
    2x2 final feature map.

      resnet*      : conv1 7x7/s2 -> 3x3/s1 (initialised from the centre 3x3 of
                     the pretrained kernel, rescaled), maxpool -> Identity.
                     A 64x64 input then yields an 8x8 layer4 output — the same
                     ratio a 224px input gives through the original stem.
      mobilenet_v2 : first conv stride 2 -> 1 (weights reused unchanged),
                     giving a 4x4 final feature map.

    ONLY use this when image_size <= 64. At the default 224 the pretrained
    stems are correct as-is and adapting them discards pretrained weights for
    no benefit.
    """
    if arch.startswith("resnet"):
        old = model.conv1
        new = nn.Conv2d(old.in_channels, old.out_channels, kernel_size=3,
                        stride=1, padding=1, bias=False)
        with torch.no_grad():
            # Centre 3x3 slice of the 7x7 kernel, rescaled so the total
            # response magnitude of the original kernel is preserved.
            centre = old.weight[:, :, 2:5, 2:5]
            scale = old.weight.abs().sum() / centre.abs().sum().clamp(min=1e-8)
            new.weight.copy_(centre * scale)
        new = new.to(old.weight.device)
        model.conv1 = new
        model.maxpool = nn.Identity()
    elif arch == "mobilenet_v2":
        model.features[0][0].stride = (1, 1)
    else:
        raise ValueError(f"adapt_model_for_small_input: unsupported arch {arch!r}")
    log.info("Adapted %s stem for small (<=64px) inputs.", arch)
    return model


# ─────────────────────────────────────────────────────────────────────────────
# 3.5  MOBILENETV2 FEATURE HOOKS  (self-contained — no pipeline.py edit needed)
# ─────────────────────────────────────────────────────────────────────────────

# Empirically verified tap indices for torchvision MobileNetV2 (224×224 input):
#   features[0]  -> 32ch  (Conv2dNormActivation, 112×112)
#   features[7]  -> 64ch  (InvertedResidual, 14×14)
#   features[11] -> 96ch  (InvertedResidual, 14×14)
#   features[17] -> 320ch (InvertedResidual, 7×7)
MOBILENETV2_TAP_INDICES = [0, 7, 11, 17]
MOBILENETV2_TAP_CHANNELS = [32, 64, 96, 320]

# ResNet tap layers (used by pipeline.py's existing hook logic):
#   resnet18/34:  layer1→64ch, layer2→128ch, layer3→256ch, layer4→512ch
#   resnet50/101: layer1→256ch, layer2→512ch, layer3→1024ch, layer4→2048ch
RESNET_TAP_ATTRS = ["layer1", "layer2", "layer3", "layer4"]


def register_mobilenetv2_feature_hooks(
    model: nn.Module,
) -> tuple[list, dict]:
    """
    Register forward hooks on the correct MobileNetV2 tap layers.

    Returns
    -------
    handles : list of hook handles (call h.remove() to clean up)
    feat_store : dict that will be populated with {tap_name: tensor} on each
                 forward pass. Keys are "feat_0" … "feat_3".

    Usage in a training loop
    ------------------------
        handles, feat_store = register_mobilenetv2_feature_hooks(student)
        out = student(x)
        # feat_store is now populated: feat_store["feat_0"] has shape [B, 32, H, W]
        # ... compute feature distillation loss ...
        # Clean up hooks at end of training:
        for h in handles:
            h.remove()
    """
    feat_store: dict[str, torch.Tensor] = {}
    handles = []

    for i, layer_idx in enumerate(MOBILENETV2_TAP_INDICES):
        key = f"feat_{i}"

        def make_hook(k):
            def hook(module, inp, output):
                feat_store[k] = output
            return hook

        layer = model.features[layer_idx]
        handles.append(layer.register_forward_hook(make_hook(key)))

    return handles, feat_store


def register_resnet_feature_hooks(
    model: nn.Module,
) -> tuple[list, dict]:
    """
    Register forward hooks on ResNet layer1–layer4.
    Works for resnet18, resnet34, resnet50, resnet101.
    """
    feat_store: dict[str, torch.Tensor] = {}
    handles = []

    for i, attr in enumerate(RESNET_TAP_ATTRS):
        key = f"feat_{i}"
        layer = getattr(model, attr)

        def make_hook(k):
            def hook(module, inp, output):
                feat_store[k] = output
            return hook

        handles.append(layer.register_forward_hook(make_hook(key)))

    return handles, feat_store


def register_feature_hooks_for_arch(
    model: nn.Module, arch: str
) -> tuple[list, dict]:
    """
    Dispatch to the correct hook registration for a given arch string.
    Call this instead of build_feature_hooks() in pipeline.py when working
    with ImageNet architectures — no pipeline.py edit required.
    """
    if arch == "mobilenet_v2":
        return register_mobilenetv2_feature_hooks(model)
    elif arch in ("resnet18", "resnet34", "resnet50", "resnet101"):
        return register_resnet_feature_hooks(model)
    else:
        raise ValueError(
            f"register_feature_hooks_for_arch: unsupported arch {arch!r}. "
            "Add it to pipeline_imagenet_patch.py."
        )




def parse_aux_teachers(cfg: dict, output_dir: str, num_classes: int, device,
                       small_input: bool = False) -> list[tuple[str, nn.Module]]:
    """
    Parse the aux_teachers list from config and return a list of
    (name, frozen_model) tuples ready to register in the teacher pool.

    Each teacher's checkpoint must already exist when num_classes != 1000 —
    see download_and_save_pretrained for why. The caller is responsible for
    running assert_teacher_is_useful on each returned model once it has an
    evaluation loader.

    cfg["aux_teachers"] example:
      - arch: resnet34
        pretrained: true
        source: torchvision
        checkpoint: M2.pth
        register_as: M2
    """
    out = []
    for entry in cfg.get("aux_teachers", []):
        arch       = entry["arch"]
        name       = entry.get("register_as", arch)
        ckpt       = entry.get("checkpoint", f"{name}.pth")
        ckpt_path  = os.path.join(output_dir, ckpt)

        download_and_save_pretrained(arch, ckpt_path, num_classes,
                                     small_input=small_input)
        model = load_model_imagenet(arch, num_classes=num_classes,
                                    pretrained=True, checkpoint_path=ckpt_path,
                                    device=device, small_input=small_input)
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        out.append((name, model))
        log.info("Registered aux teacher %r (%s) from %s", name, arch, ckpt_path)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 4.  SATURATION-AWARE STAGE RUNNER
# ─────────────────────────────────────────────────────────────────────────────

def saturation_aware_run_stage(run_stage_fn, stage_cfg: dict, *args, **kwargs):
    """
    Wraps run_stage with multi-cycle saturation if stage_cfg["saturation"] is True.

    run_stage_fn signature (same as existing pipeline.run_stage):
        run_stage(stage_cfg, train_loader, val_loader, teacher_pool,
                  device, output_dir, **overrides) -> best_acc

    Saturation params (all in stage_cfg):
        saturation            : bool  (default False)
        sat_cycles            : int   (default 2)
        sat_epochs_per_cycle  : int   (default 120)
        sat_min_delta         : float (default 0.1)

    Behaviour:
        - Cycle 1 always runs.
        - Cycle N+1 runs only if (best_acc_cycle_N - best_acc_cycle_N-1) >= sat_min_delta.
        - Each cycle reloads the checkpoint produced by the previous cycle
          (so we always restart from the best weights, not the end-of-cycle weights).
        - After all cycles, the best checkpoint across all cycles is written to
          stage_cfg["checkpoint"].
    """
    if not stage_cfg.get("saturation", False):
        return run_stage_fn(stage_cfg, *args, **kwargs)

    max_cycles    = stage_cfg.get("sat_cycles", 2)
    epochs_cycle  = stage_cfg.get("sat_epochs_per_cycle", 120)
    min_delta     = stage_cfg.get("sat_min_delta", 0.1)
    output_dir    = kwargs.get("output_dir", args[3] if len(args) > 3 else ".")

    # Stash original checkpoint name; use per-cycle temp names
    final_ckpt = stage_cfg["checkpoint"]
    best_acc   = 0.0
    prev_acc   = 0.0
    best_cycle = 0

    for cycle in range(1, max_cycles + 1):
        cycle_ckpt = final_ckpt.replace(".pth", f"_sat{cycle}.pth")
        log.info("Saturation cycle %d/%d — checkpoint: %s", cycle, max_cycles, cycle_ckpt)

        cycle_cfg = {**stage_cfg,
                     "epochs":     epochs_cycle,
                     "checkpoint": cycle_ckpt}

        # After cycle 1, init from the best-so-far checkpoint
        if cycle > 1:
            cycle_cfg["init_from"] = best_ckpt_path

        cycle_acc = run_stage_fn(cycle_cfg, *args, **kwargs)
        cycle_ckpt_path = os.path.join(output_dir, cycle_ckpt)

        if cycle_acc > best_acc:
            best_acc = cycle_acc
            best_cycle = cycle
            best_ckpt_path = cycle_ckpt_path

        improvement = cycle_acc - prev_acc
        log.info("Cycle %d accuracy: %.2f%% (improvement: %.2f%%)",
                 cycle, cycle_acc, improvement)

        if cycle > 1 and improvement < min_delta:
            log.info("Improvement %.2f%% < threshold %.2f%% — stopping saturation early.",
                     improvement, min_delta)
            break
        prev_acc = cycle_acc

    # Promote best cycle checkpoint to final name
    import shutil
    best_src = os.path.join(output_dir, best_ckpt_path)
    best_dst = os.path.join(output_dir, final_ckpt)
    if os.path.isfile(best_src) and best_src != best_dst:
        shutil.copy2(best_src, best_dst)
    log.info("Saturation complete. Best cycle: %d, best accuracy: %.2f%%",
             best_cycle, best_acc)
    return best_acc


# ─────────────────────────────────────────────────────────────────────────────
# 5.  CONVENIENCE: AUTO-BUILD LOADER FROM CONFIG
# ─────────────────────────────────────────────────────────────────────────────

def resolve_tinyimagenet_root(data_root: str) -> str:
    """
    Accept a data_root pointing at either ./data or ./data/imagenet and return
    the directory that does (or will) contain tiny-imagenet-200/.
    """
    if os.path.isdir(os.path.join(data_root, "tiny-imagenet-200")):
        return data_root
    parent = os.path.dirname(os.path.abspath(data_root))
    if os.path.isdir(os.path.join(parent, "tiny-imagenet-200")):
        return parent
    return data_root  # will trigger the download path


def build_loaders_from_cfg(cfg: dict):
    """
    Single entry point — reads experiment.dataset and returns
    (train_loader, val_loader, test_loader) as three disjoint splits.

    val  drives checkpoint selection / early stopping / saturation.
    test is evaluated and logged but never influences any decision.
    """
    dataset     = cfg["experiment"].get("dataset", "cifar100")
    data_cfg    = cfg["data"]
    data_root   = data_cfg["data_root"]
    batch_size  = data_cfg["batch_size"]
    num_workers = data_cfg.get("num_workers", 4)
    image_size  = data_cfg.get("image_size", 224)
    val_seed    = data_cfg.get("val_seed", 42)

    # Windows spawns dataloader workers via a fresh interpreter; that needs the
    # caller to be under a __main__ guard. run_experiment_imagenet.py is, so
    # workers are allowed — but default to 0 if the config did not say.
    import platform
    if platform.system() == "Windows" and num_workers is None:
        num_workers = 0

    if dataset == "tinyimagenet":
        return build_tinyimagenet_loaders(
            resolve_tinyimagenet_root(data_root),
            batch_size=batch_size,
            num_workers=num_workers,
            image_size=image_size,
            val_fraction=data_cfg.get("val_fraction", 0.10),
            val_seed=val_seed,
        )
    elif dataset == "imagenet":
        return build_imagenet_loaders(
            data_root,
            batch_size=batch_size,
            num_workers=num_workers,
            image_size=image_size,
            val_fraction=data_cfg.get("val_fraction", 0.02),
            val_seed=val_seed,
        )
    elif dataset == "cifar100":
        # Delegated to pipeline.py, which already implements the exact split
        # and augmentation the PMABD CIFAR-100 runs used: 32x32 native,
        # RandomCrop(32, pad=4) + HFlip + AutoAugment(CIFAR10), val carved
        # from train by the same stratified seed, full 10k test untouched.
        # Routing through here rather than reimplementing keeps the baselines
        # bit-identical to the ladder on data, which is the whole point of
        # this module.
        from pipeline import get_cifar100_dataloaders
        return get_cifar100_dataloaders(
            batch_size=batch_size, num_workers=num_workers,
            data_root=data_root,
            val_fraction=data_cfg.get("val_fraction", 0.10),
            val_seed=val_seed,
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset!r}")


# ─────────────────────────────────────────────────────────────────────────────
# 6.  PIPELINE.PY CHANGES SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
#
# The following changes are needed in pipeline.py itself. They are all
# additive / guard-clause additions — no existing logic is removed.
#
# A) LSQQuantizer.forward() — tighten percentile at <=2-bit:
#    (Already implemented in the companion patch from our CIFAR 2-bit work.)
#
# B) apply_quantization() — depthwise conv handling for MobileNetV2:
#    When arch == "mobilenet_v2", depthwise convolutions (groups == in_channels)
#    should use per-tensor rather than per-channel quantization to avoid
#    per-group scale explosion. Add a check:
#
#       if isinstance(m, nn.Conv2d) and m.groups == m.in_channels:
#           quantizer = LSQQuantizer(..., per_channel=False)
#
# C) build_feature_hooks() — MobileNetV2 tap points:
#    Empirically verified channel map (torchvision MobileNetV2, 224x224 input):
#       features[ 0]  Conv2dNormActivation  ->  32ch  (112x112)
#       features[ 1]  InvertedResidual      ->  16ch
#       features[ 2]  InvertedResidual      ->  24ch
#       features[ 3]  InvertedResidual      ->  24ch
#       features[ 4]  InvertedResidual      ->  32ch
#       features[ 5]  InvertedResidual      ->  32ch
#       features[ 6]  InvertedResidual      ->  32ch
#       features[ 7]  InvertedResidual      ->  64ch  ← tap 2
#       features[ 8]  InvertedResidual      ->  64ch
#       features[ 9]  InvertedResidual      ->  64ch
#       features[10]  InvertedResidual      ->  64ch
#       features[11]  InvertedResidual      ->  96ch  ← tap 3
#       features[12]  InvertedResidual      ->  96ch
#       features[13]  InvertedResidual      ->  96ch
#       features[14]  InvertedResidual      -> 160ch
#       features[15]  InvertedResidual      -> 160ch
#       features[16]  InvertedResidual      -> 160ch
#       features[17]  InvertedResidual      -> 320ch  ← tap 4
#       features[18]  Conv2dNormActivation  -> 1280ch
#
#    Correct tap indices for feat_channels=[32, 64, 96, 320]:
#       [features[0], features[7], features[11], features[17]]
#
#    NOTE: features[3]=24ch and features[6]=32ch were WRONG in the original
#    documentation. The smoke test caught this. Always use [0, 7, 11, 17].
#
# D) ProjectionHead alignment:
#    MobileNetV2 spatial feature maps at features[0] are 112×112 (ImageNet).
#    Teacher ResNet50 at layer1 output is also 56×56 (or 28×28 after pool).
#    The existing AdaptiveAvgPool2d→Linear projection handles arbitrary
#    spatial sizes, so no change is needed here.
#
# E) run_stage() — read saturation flag from stage_cfg:
#    saturation_aware_run_stage() in this patch handles the outer loop.
#    run_stage() itself does not need to know about saturation — the wrapper
#    calls it with a per-cycle cfg that already has epochs set correctly.
#
# F) num_classes propagation:
#    All model constructors in pipeline.py should accept num_classes from cfg.
#    Add: num_classes = cfg.get("experiment", {}).get("num_classes", 100)
#    and pass it to every load_model() call.
