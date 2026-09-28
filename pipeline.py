#!/usr/bin/env python
"""
pipeline.py — Core KD + QAT pipeline library (v3).

All shared components: quantizers, losses, trainer, evaluator, pipeline runner.
Import this in experiment scripts; do not run directly.

NOTE ON DATA SPLITS (v3 — supersedes v2's test-derived val split)
-------------------------------------------------------------------
Per RA recommendation, the full official 10k test set must be reported as
final test accuracy to be comparable with published CIFAR results. That
means validation can no longer come from the test set.

Validation now comes from the 50k TRAIN set instead:
  - train_eff : 45k images (or (1 - 1/k) fraction under K-Fold), augmented —
                the only data any student model is ever optimised on.
  - val       : 5k images (or 1/k fraction under K-Fold), clean transform,
                carved from train — drives checkpoint selection, patience,
                and saturation decisions. NEVER used for gradient updates.
  - test      : the full, untouched, official 10k test set, clean transform —
                used ONLY for final reporting. Never influences any decision.

CRITICAL CAVEAT — hub-pretrained weights:
  chenyaofo/pytorch-cifar-models hub checkpoints were trained on the FULL
  50k train set. That means a val split carved from train is NOT "unseen"
  for any model initialised via `pretrained_init=True`. To keep val honest:
    - The teacher (M1) MAY use pretrained_init — it is a fixed, frozen soft
      label source, never selected by val accuracy, so leakage into its own
      val number is harmless (we only ever report its test accuracy).
    - Student stages MUST NOT use pretrained_init. They start from either
      `init_from` (a checkpoint produced by THIS pipeline) or random init.
      This keeps every student's val accuracy an honest, unseen signal.
  `load_student_init` in run_experiment.py enforces this — pretrained_init
  for a student stage now raises a configuration error rather than silently
  leaking.

K-Fold option:
  get_cifar100_dataloaders / get_cifar10_dataloaders accept `fold` and
  `n_folds`. With n_folds=1 (default) behaviour is the classic single
  stratified 45k/5k split. With n_folds>1, the requested fold's 1/k slice
  of train becomes val and the rest becomes train_eff, enabling K-Fold CV
  by re-invoking the loader with different `fold` values across runs.
"""

import copy
import math
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset


# ============================================================
# DATA
# ============================================================

def _make_stratified_kfold_split(targets, num_classes, n_folds=1, fold=0,
                                  val_fraction=0.10, seed=42):
    """
    Build a class-balanced train/val split from a labelled set.

    n_folds == 1 (default):
        Single deterministic stratified split. val gets
        round(val_fraction * len(targets) / num_classes) examples per class.

    n_folds > 1:
        True K-Fold: each class's examples are partitioned into n_folds
        equal-ish contiguous chunks (after a single deterministic shuffle).
        `fold` (0-indexed) selects which chunk is held out as val; the
        remaining (n_folds - 1) chunks form train_eff. Iterating fold over
        0..n_folds-1 across separate runs gives full K-Fold CV coverage.

    Returns (train_indices, val_indices), both sorted, disjoint, covering
    the whole input set.
    """
    g = torch.Generator().manual_seed(seed)
    by_class = {c: [] for c in range(num_classes)}
    for idx, t in enumerate(targets):
        by_class[int(t)].append(idx)

    train_idx, val_idx = [], []

    if n_folds <= 1:
        per_class_val = max(1, round(val_fraction * (len(targets) / num_classes)))
        for c in range(num_classes):
            idxs = torch.tensor(by_class[c], dtype=torch.long)
            perm = idxs[torch.randperm(len(idxs), generator=g)]
            val_idx.extend(perm[:per_class_val].tolist())
            train_idx.extend(perm[per_class_val:].tolist())
    else:
        if not (0 <= fold < n_folds):
            raise ValueError(f"fold={fold} out of range for n_folds={n_folds}")
        for c in range(num_classes):
            idxs = torch.tensor(by_class[c], dtype=torch.long)
            perm = idxs[torch.randperm(len(idxs), generator=g)]
            # Split this class's indices into n_folds nearly-equal chunks
            chunks = torch.chunk(perm, n_folds)
            for i, chunk in enumerate(chunks):
                if i == fold:
                    val_idx.extend(chunk.tolist())
                else:
                    train_idx.extend(chunk.tolist())

    return sorted(train_idx), sorted(val_idx)


def get_cifar100_dataloaders(batch_size=256, num_workers=4, data_root="./data",
                             val_fraction=0.10, val_seed=42,
                             n_folds=1, fold=0):
    """
    CIFAR-100 loaders with val carved from TRAIN (not test).

    - train_eff : (1 - val_fraction) of the 50k train set (or (k-1)/k under
                  K-Fold), augmented. The only data any student trains on.
    - val       : val_fraction of train (or 1/k under K-Fold), clean
                  transform, carved from train. Drives checkpoint selection.
    - test      : full, untouched 10k official test set, clean transform.
                  Reported only — never used for any decision.

    See module docstring for the pretrained-init leakage caveat.
    """
    NUM_CLASSES = 100
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.CIFAR10),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5071, 0.4867, 0.4408),
                             std=(0.2675, 0.2565, 0.2761)),
    ])
    transform_clean = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5071, 0.4867, 0.4408),
                             std=(0.2675, 0.2565, 0.2761)),
    ])

    train_aug   = torchvision.datasets.CIFAR100(
        root=data_root, train=True, download=True, transform=transform_train)
    train_clean = torchvision.datasets.CIFAR100(
        root=data_root, train=True, download=True, transform=transform_clean)
    test_set    = torchvision.datasets.CIFAR100(
        root=data_root, train=False, download=True, transform=transform_clean)

    train_idx, val_idx = _make_stratified_kfold_split(
        train_aug.targets, NUM_CLASSES, n_folds=n_folds, fold=fold,
        val_fraction=val_fraction, seed=val_seed)

    fold_desc = (f"fold {fold+1}/{n_folds}" if n_folds > 1
                else f"{val_fraction:.0%} holdout")
    print(f"CIFAR-100 split [{fold_desc}, seed={val_seed}]: "
          f"train={len(train_idx)}  val={len(val_idx)}  "
          f"test={len(test_set)} (full, untouched)")

    train_loader = DataLoader(
        Subset(train_aug, train_idx), batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(
        Subset(train_clean, val_idx), batch_size=batch_size * 2, shuffle=False,
        num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(
        test_set, batch_size=batch_size * 2, shuffle=False,
        num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader, test_loader


def get_cifar10_dataloaders(batch_size=256, num_workers=4, data_root="./data",
                            val_fraction=0.10, val_seed=42,
                            n_folds=1, fold=0):
    """CIFAR-10 counterpart of get_cifar100_dataloaders. See its docstring."""
    NUM_CLASSES = 10
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.CIFAR10),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.4914, 0.4822, 0.4465),
                             std=(0.2023, 0.1994, 0.2010)),
    ])
    transform_clean = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.4914, 0.4822, 0.4465),
                             std=(0.2023, 0.1994, 0.2010)),
    ])

    train_aug   = torchvision.datasets.CIFAR10(
        root=data_root, train=True, download=True, transform=transform_train)
    train_clean = torchvision.datasets.CIFAR10(
        root=data_root, train=True, download=True, transform=transform_clean)
    test_set    = torchvision.datasets.CIFAR10(
        root=data_root, train=False, download=True, transform=transform_clean)

    train_idx, val_idx = _make_stratified_kfold_split(
        train_aug.targets, NUM_CLASSES, n_folds=n_folds, fold=fold,
        val_fraction=val_fraction, seed=val_seed)

    fold_desc = (f"fold {fold+1}/{n_folds}" if n_folds > 1
                else f"{val_fraction:.0%} holdout")
    print(f"CIFAR-10 split [{fold_desc}, seed={val_seed}]: "
          f"train={len(train_idx)}  val={len(val_idx)}  "
          f"test={len(test_set)} (full, untouched)")

    train_loader = DataLoader(
        Subset(train_aug, train_idx), batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(
        Subset(train_clean, val_idx), batch_size=batch_size * 2, shuffle=False,
        num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(
        test_set, batch_size=batch_size * 2, shuffle=False,
        num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader, test_loader


# ============================================================
# QUANTIZATION LAYERS (LSQ)
# ============================================================

class LSQQuantizer(nn.Module):
    """
    Learned Step Size Quantization (Esser et al. 2020, ICLR).
    Scale is an nn.Parameter — gets real gradients every step,
    eliminating EMA lag that causes progressive noise accumulation.
    """
    def __init__(self, bitwidth=4, n_features=1):
        super().__init__()
        self.bitwidth = bitwidth
        self.qmin = -(2 ** (bitwidth - 1))
        self.qmax = (2 ** (bitwidth - 1)) - 1
        self.grad_scale_factor = 1.0 / ((n_features * self.qmax) ** 0.5)
        self.scale = nn.Parameter(torch.ones(1))
        self.register_buffer("initialized", torch.tensor(False))
        self.register_buffer("running_max", torch.tensor(1.0))
        self.register_buffer("num_updates", torch.tensor(0, dtype=torch.long))
        self.percentile = (99.5 if bitwidth <= 2 else
                           99.9 if bitwidth <= 4 else 99.99)

    def forward(self, x):
        if self.bitwidth >= 32:
            return x
        if not self.initialized:
            return x
        s = self.scale.abs().clamp(min=1e-8)
        s_scaled = s * self.grad_scale_factor
        x_scaled = x / s_scaled
        x_clipped = x_scaled.clamp(self.qmin, self.qmax)
        x_quant = x_clipped + (x_clipped.round() - x_clipped).detach()
        return x_quant * s_scaled

    def extra_repr(self):
        return f"bitwidth={self.bitwidth}, qmin={self.qmin}, qmax={self.qmax}"


class LSQQuantizerPerChannel(nn.Module):
    """
    Per-channel LSQ quantizer for Conv2d weights.

    Why per-channel for weights:
    - Each output filter has a different weight magnitude distribution
    - A single per-tensor scale wastes levels on channels that need less range
    - Per-channel gives each filter its own optimal scale
    - Activations stay per-tensor (per-channel activations break residual adds)

    Shape: weight [C_out, C_in, kH, kW] → one scale per C_out
    """
    def __init__(self, bitwidth=4, n_channels=1, n_features_per_channel=1):
        super().__init__()
        self.bitwidth = bitwidth
        self.qmin = -(2 ** (bitwidth - 1))
        self.qmax = (2 ** (bitwidth - 1)) - 1
        self.grad_scale_factor = 1.0 / ((n_features_per_channel * self.qmax) ** 0.5)
        self.scale = nn.Parameter(torch.ones(n_channels))
        self.register_buffer("initialized", torch.tensor(False))
        self.register_buffer("running_max", torch.zeros(n_channels))
        self.register_buffer("num_updates", torch.tensor(0, dtype=torch.long))
        self.percentile = (99.5 if bitwidth <= 2 else
                           99.9 if bitwidth <= 4 else 99.99)
        self.n_channels = n_channels

    def forward(self, x):
        if self.bitwidth >= 32 or not self.initialized:
            return x
        s = self.scale.abs().clamp(min=1e-8) * self.grad_scale_factor
        s = s.view(-1, *([1] * (x.dim() - 1)))
        x_scaled  = x / s
        x_clipped = x_scaled.clamp(self.qmin, self.qmax)
        x_quant   = x_clipped + (x_clipped.round() - x_clipped).detach()
        return x_quant * s

    def extra_repr(self):
        return (f"bitwidth={self.bitwidth}, n_channels={self.n_channels}, "
                f"qmin={self.qmin}, qmax={self.qmax}")


class BinaryQuantizer(nn.Module):
    """
    Binary (1-bit) quantizer using sign function + STE gradient.
    Replaces LSQQuantizer when bitwidth=1.
    """
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(1))
        self.register_buffer("initialized", torch.tensor(True))
        self.bitwidth = 1
        self.percentile = 99.9
        self.register_buffer("running_max", torch.tensor(1.0))
        self.register_buffer("num_updates", torch.tensor(0, dtype=torch.long))
        self.grad_scale_factor = 1.0

    def forward(self, x):
        s = self.scale.abs().clamp(min=1e-8)
        x_clipped = x.clamp(-1.0, 1.0)
        x_sign = x_clipped.sign()
        x_binary = x_sign + (x_clipped - x_clipped).detach()
        return x_binary * s

    def extra_repr(self):
        return "bitwidth=1 (binary, sign-STE)"


class QuantizedConv2d(nn.Conv2d):
    """
    Conv2d with per-channel LSQ for weights, per-tensor LSQ for activations,
    BinaryQuantizer when bitwidth=1.
    """
    def __init__(self, *args, weight_bitwidth=8, act_bitwidth=4,
                 per_channel_weights=True, **kwargs):
        super().__init__(*args, **kwargs)
        w_numel = self.weight.numel()
        n_out   = self.out_channels
        feats_per_ch = w_numel // n_out

        self.act_q = (BinaryQuantizer() if act_bitwidth == 1
                      else LSQQuantizer(act_bitwidth, n_features=w_numel))

        if weight_bitwidth == 1:
            self.weight_q = BinaryQuantizer()
        elif per_channel_weights:
            self.weight_q = LSQQuantizerPerChannel(
                weight_bitwidth, n_channels=n_out,
                n_features_per_channel=feats_per_ch)
        else:
            self.weight_q = LSQQuantizer(weight_bitwidth, n_features=w_numel)

        self.register_buffer("act_noise", torch.tensor(0.0))
        self.register_buffer("weight_noise", torch.tensor(0.0))

    def forward(self, x):
        x_q = self.act_q(x)
        w_q = self.weight_q(self.weight)
        if self.training:
            with torch.no_grad():
                self.act_noise.copy_(F.mse_loss(x_q.detach(), x.detach()))
                self.weight_noise.copy_(F.mse_loss(w_q.detach(), self.weight.detach()))
        return F.conv2d(x_q, w_q, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)


class QuantizedLinear(nn.Linear):
    """
    Linear with per-channel LSQ for weights, per-tensor LSQ for activations,
    BinaryQuantizer when bitwidth=1.
    """
    def __init__(self, *args, weight_bitwidth=8, act_bitwidth=8,
                 per_channel_weights=True, **kwargs):
        super().__init__(*args, **kwargs)
        w_numel = self.out_features * self.in_features
        n_out   = self.out_features

        self.act_q = (BinaryQuantizer() if act_bitwidth == 1
                      else LSQQuantizer(act_bitwidth, n_features=w_numel))

        if weight_bitwidth == 1:
            self.weight_q = BinaryQuantizer()
        elif per_channel_weights:
            self.weight_q = LSQQuantizerPerChannel(
                weight_bitwidth, n_channels=n_out,
                n_features_per_channel=self.in_features)
        else:
            self.weight_q = LSQQuantizer(weight_bitwidth, n_features=w_numel)

        self.register_buffer("act_noise", torch.tensor(0.0))
        self.register_buffer("weight_noise", torch.tensor(0.0))

    def forward(self, x):
        x_q = self.act_q(x)
        w_q = self.weight_q(self.weight)
        if self.training:
            with torch.no_grad():
                self.act_noise.copy_(F.mse_loss(x_q.detach(), x.detach()))
                self.weight_noise.copy_(F.mse_loss(w_q.detach(), self.weight.detach()))
        return F.linear(x_q, w_q, self.bias)


# ============================================================
# QUANTIZATION UTILITIES
# ============================================================

def replace_with_fake_quantization(model, weight_bitwidth=8, act_bitwidth=4):
    """
    Replace Conv2d/Linear with quantized versions.
    Boundary layers (first conv, last linear) always use W8A8.
    """
    if weight_bitwidth >= 32 and act_bitwidth >= 32:
        return model

    print(f"Applying QAT: W{weight_bitwidth}A{act_bitwidth} (I/O at W8A8)")
    model = copy.deepcopy(model)

    conv_linear_modules = [
        name for name, m in model.named_modules()
        if isinstance(m, (nn.Conv2d, nn.Linear))
    ]
    first_name = conv_linear_modules[0] if conv_linear_modules else None
    last_name  = conv_linear_modules[-1] if conv_linear_modules else None
    # penultimate layer: held at W4A4 minimum when interior bitwidth < 3-bit
    penultimate_name = (conv_linear_modules[-2]
                        if len(conv_linear_modules) >= 2 else None)

    def _get_parent(model, name):
        parts = name.split('.')
        m = model
        for p in parts[:-1]:
            m = getattr(m, p)
        return m, parts[-1]

    for full_name in conv_linear_modules:
        parent, attr = _get_parent(model, full_name)
        child = getattr(parent, attr)
        is_boundary = (full_name == first_name or full_name == last_name)
        # Penultimate layer held at W4A4 min when interior precision < 3-bit
        # (prevents catastrophic accuracy collapse at the network's final feature gate)
        is_penultimate = (full_name == penultimate_name
                          and (weight_bitwidth < 3 or act_bitwidth < 3))
        # Depthwise conv: groups == in_channels → use per-tensor weight quantization
        # Per-channel on a depthwise with 1 weight per group causes scale instability
        is_depthwise = (isinstance(child, nn.Conv2d)
                        and child.groups == child.in_channels
                        and child.in_channels > 1)

        if is_boundary:
            w_bw, a_bw = 8, 8
        elif is_penultimate:
            w_bw = max(weight_bitwidth, 4)
            a_bw = max(act_bitwidth, 4)
        else:
            w_bw, a_bw = weight_bitwidth, act_bitwidth

        if isinstance(child, nn.Conv2d):
            new_layer = QuantizedConv2d(
                child.in_channels, child.out_channels, child.kernel_size,
                child.stride, child.padding, child.dilation, child.groups,
                bias=(child.bias is not None),
                weight_bitwidth=w_bw, act_bitwidth=a_bw,
                per_channel_weights=(not is_depthwise),  # per-tensor for depthwise
            )
            new_layer.weight.data.copy_(child.weight.data)
            if child.bias is not None:
                new_layer.bias.data.copy_(child.bias.data)
        else:
            new_layer = QuantizedLinear(
                child.in_features, child.out_features,
                bias=(child.bias is not None),
                weight_bitwidth=w_bw, act_bitwidth=a_bw,
            )
            new_layer.weight.data.copy_(child.weight.data)
            if child.bias is not None:
                new_layer.bias.data.copy_(child.bias.data)

        setattr(parent, attr, new_layer)
        if is_boundary:
            tag = " [BOUNDARY W8A8]"
        elif is_penultimate:
            tag = f" [PENULTIMATE W{w_bw}A{a_bw} floor]"
        elif is_depthwise:
            tag = f" [DEPTHWISE W{w_bw}A{a_bw} per-tensor]"
        else:
            tag = f" [W{w_bw}A{a_bw}]"
        print(f"  Replaced {full_name}{tag}")

    return model


def split_batch(batch):
    """Loaders normally yield (inputs, targets). The fixed-view train loader
    used by the teacher cache (teacher_cache.py) yields a third element, the
    cache row for each sample. Every consumer of a loader goes through here so
    that adding the cache did not have to touch each unpack site by hand."""
    if len(batch) == 3:
        return batch[0], batch[1], batch[2]
    return batch[0], batch[1], None


def calibrate_quantizers(model, train_loader, device, num_batches=50):
    """
    Two-phase LSQ calibration:
    1. Weight scales from weight tensor percentiles (no forward pass needed).
    2. Activation scales from observed forward pass ranges via pre-hooks.

    Skips quantizers that are already initialized (e.g. carried over from a
    previous training cycle at the same precision — see run_distillation_stage
    multi-cycle support), so calibration only touches genuinely fresh scales.
    """
    print("Initializing LSQ weight scales from weight distributions...")
    n_w = 0
    n_w_skipped = 0
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, (QuantizedConv2d, QuantizedLinear)):
                wq = module.weight_q
                if isinstance(wq, BinaryQuantizer):
                    wq.initialized.fill_(True)
                    n_w += 1
                    continue
                if wq.initialized:
                    n_w_skipped += 1
                    continue

                # Always work on CPU for quantile — avoids device mismatch
                # when model is on CUDA (q_pct tensor must match weight device)
                w = module.weight.detach().float().cpu()
                q_pct_val = wq.percentile / 100.0  # plain Python float, no device issue

                if isinstance(wq, LSQQuantizerPerChannel):
                    w_per_ch = w.view(w.size(0), -1).abs()
                    w_maxes = torch.quantile(
                        w_per_ch, q_pct_val, dim=1
                    ).clamp(min=1e-5)
                    s_init = (2.0 * w_maxes / (wq.qmax - wq.qmin)) / wq.grad_scale_factor
                    wq.scale.data.copy_(s_init.to(wq.scale.device))
                    wq.running_max.copy_(w_maxes.to(wq.running_max.device))
                else:
                    w_max = torch.quantile(w.abs().flatten(), q_pct_val).clamp(min=1e-5)
                    s_init = (2.0 * w_max / (wq.qmax - wq.qmin)) / wq.grad_scale_factor
                    wq.scale.data.fill_(s_init.item())
                    wq.running_max.fill_(w_max.item())

                wq.initialized.fill_(True)
                wq.num_updates.fill_(1)
                n_w += 1
    if n_w_skipped:
        print(f"  Skipped {n_w_skipped} already-initialized weight scales "
              f"(carried over from previous cycle).")
    print(f"  Initialized {n_w} weight LSQ scales (per-channel where applicable).")

    act_maxes = {}
    hooks = []

    def make_hook(name):
        def hook(module, args):
            x = args[0].detach().abs().float().flatten()
            q = torch.tensor(module.percentile / 100.0,
                             device=x.device, dtype=torch.float32)
            MAX_ELEMS = 1_000_000
            if x.numel() > MAX_ELEMS:
                idx = torch.randperm(x.numel(), device=x.device)[:MAX_ELEMS]
                x = x[idx]
            val = torch.quantile(x, q).clamp(min=1e-5)
            act_maxes[name] = torch.maximum(act_maxes[name], val) \
                if name in act_maxes else val
        return hook

    n_a_to_calibrate = 0
    for name, module in model.named_modules():
        if isinstance(module, LSQQuantizer) and not module.initialized:
            hooks.append(module.register_forward_pre_hook(make_hook(name)))
            n_a_to_calibrate += 1

    if n_a_to_calibrate == 0:
        print("  All activation LSQ scales already initialized — skipping "
              "calibration forward passes.")
    else:
        model.train()
        model = model.to(device)  # ensure model is on correct device before forward passes
        print(f"Calibrating activation LSQ scales ({num_batches} batches)...")
        with torch.no_grad():
            for i, batch in enumerate(train_loader):
                if i >= num_batches:
                    break
                inputs, _, _ = split_batch(batch)
                model(inputs.to(device))

    for h in hooks:
        h.remove()

    n_a = 0
    for name, module in model.named_modules():
        if isinstance(module, LSQQuantizer) and not module.initialized:
            if name in act_maxes:
                a_max = act_maxes[name].clamp(min=1e-5)
                s_init = (2.0 * a_max / (module.qmax - module.qmin))
                s_init = s_init / module.grad_scale_factor
                module.scale.data.fill_(s_init.item())
                module.running_max.fill_(a_max.item())
            module.initialized.fill_(True)
            module.num_updates.fill_(1)
            n_a += 1
    if n_a:
        print(f"Calibration done. Initialized {n_a} activation + {n_w} weight LSQ scales.")

    act_scales, w_scales = [], []
    for module in model.modules():
        if isinstance(module, (QuantizedConv2d, QuantizedLinear)):
            if module.act_q.bitwidth <= 4 and module.act_q.initialized:
                act_scales.append(
                    (module.act_q.scale.abs() * module.act_q.grad_scale_factor).mean().item())
            if module.weight_q.bitwidth <= 4 and module.weight_q.initialized:
                w_scales.append(
                    (module.weight_q.scale.abs() * module.weight_q.grad_scale_factor).mean().item())
    if act_scales:
        print(f"  Effective act scales: "
              f"min={min(act_scales):.4f} mean={sum(act_scales)/len(act_scales):.4f} "
              f"max={max(act_scales):.4f}")
    if w_scales:
        print(f"  Effective wgt scales: "
              f"min={min(w_scales):.4f} mean={sum(w_scales)/len(w_scales):.4f} "
              f"max={max(w_scales):.4f}")
    if act_scales and max(act_scales) > 1.0:
        print("  [WARN] Some act scales still high — outlier layers present.")
    return model


def extract_fp32_weights(quantized_state_dict):
    """
    Strip LSQ/QAT-specific buffers so a QAT state dict loads into a plain model.
    Keeps: weight, bias, BN running stats.
    Drops: act_q.*, weight_q.*, act_noise, weight_noise.
    """
    DROP_SUFFIXES = (
        ".act_noise", ".weight_noise",
        ".act_q.running_max", ".act_q.num_updates", ".act_q.initialized",
        ".act_q.scale",
        ".weight_q.running_max", ".weight_q.num_updates", ".weight_q.initialized",
        ".weight_q.scale",
    )
    return {k: v for k, v in quantized_state_dict.items()
            if not any(k.endswith(s) for s in DROP_SUFFIXES)}


def extract_full_qat_weights_keep_scales(quantized_state_dict):
    """
    Strip only the noise-diagnostic buffers, KEEPING LSQ scale parameters
    and 'initialized' flags. Used for multi-cycle warm restart at the same
    precision: cycle k+1 should inherit cycle k's learned quantizer scales
    rather than recalibrating from scratch, which is what prevents the sharp
    gradient spikes a full reset would otherwise cause.
    """
    DROP_SUFFIXES = (".act_noise", ".weight_noise")
    return {k: v for k, v in quantized_state_dict.items()
            if not any(k.endswith(s) for s in DROP_SUFFIXES)}


def log_quantization_health(model, epoch, stage_name):
    """Print per-layer quantization noise summary."""
    q_layers = []
    for name, m in model.named_modules():
        if isinstance(m, (QuantizedConv2d, QuantizedLinear)):
            eff_act_scale = ((m.act_q.scale.abs() * m.act_q.grad_scale_factor).item()
                             if m.act_q.initialized else 0.0)
            if isinstance(m.weight_q, LSQQuantizerPerChannel) and m.weight_q.initialized:
                eff_w_scale = (m.weight_q.scale.abs() * m.weight_q.grad_scale_factor).mean().item()
            elif hasattr(m.weight_q, "scale") and m.weight_q.initialized:
                eff_w_scale = (m.weight_q.scale.abs() * m.weight_q.grad_scale_factor).item()
            else:
                eff_w_scale = 0.0
            q_layers.append({
                "name": name,
                "act_noise": m.act_noise.item(),
                "w_noise": m.weight_noise.item(),
                "act_scale": eff_act_scale,
                "w_scale_mean": eff_w_scale,
                "act_bw": m.act_q.bitwidth,
            })
    if not q_layers:
        return
    total_act = sum(l["act_noise"] for l in q_layers)
    total_w   = sum(l["w_noise"]   for l in q_layers)
    worst     = max(q_layers, key=lambda l: l["act_noise"])
    print(f"\n  [QUANT HEALTH | {stage_name} | Epoch {epoch}]")
    print(f"    Total act noise:    {total_act:.6f}")
    print(f"    Total weight noise: {total_w:.6f}")
    print(f"    Worst act layer:    {worst['name']} "
          f"(noise={worst['act_noise']:.6f}, act_scale={worst['act_scale']:.6f}, "
          f"w_scale_mean={worst.get('w_scale_mean', 0.0):.6f}, bw={worst['act_bw']})")


# ============================================================
# MODEL WRAPPERS
# ============================================================

class FeatureExtractorModel(nn.Module):
    """
    Wraps any model and captures intermediate feature maps via hooks.
    Returns {"logits": Tensor, "feats": List[Tensor]} from forward().

    Supports two tap modes (can be mixed):
      layer_names : list of named child attribute strings — works for ResNet
                    ('layer1', 'layer2', 'layer3', 'layer4').
      hook_modules: list of nn.Module objects to tap directly — works for
                    MobileNetV2 (model.features[0], model.features[7], ...)
                    and any architecture where layers are not direct named
                    children (e.g. Sequential blocks, features[i]).

    Usage for MobileNetV2:
        taps = [model.features[0], model.features[7],
                model.features[11], model.features[17]]
        extractor = FeatureExtractorModel(model, layer_names=(), hook_modules=taps)

    Usage for ResNet (backward-compatible, default):
        extractor = FeatureExtractorModel(model,
                        layer_names=('layer1','layer2','layer3','layer4'))
    """
    def __init__(self, model, layer_names=('layer1', 'layer2', 'layer3'),
                 hook_modules=None, detach_feats=False):
        super().__init__()
        self.model = model
        self.layer_names = list(layer_names) if layer_names else []
        self.hook_modules = list(hook_modules) if hook_modules else []
        self.detach_feats = detach_feats
        self._feats = {}
        self.hooks = []
        self._n_module = len(self.hook_modules)
        self._register_hooks()

    def _register_hooks(self):
        def make_hook(key):
            def hook(module, inp, out):
                self._feats[key] = out.detach() if self.detach_feats else out
            return hook
        # Named-child taps (ResNet-style)
        for name, module in self.model.named_children():
            if name in self.layer_names:
                self.hooks.append(
                    module.register_forward_hook(make_hook(name)))
        # Direct-module taps (MobileNetV2-style: pass module objects directly)
        for i, module in enumerate(self.hook_modules):
            key = f"__hook_module_{i}__"
            self.hooks.append(module.register_forward_hook(make_hook(key)))

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()

    def forward(self, x):
        self._feats = {}
        logits = self.model(x)
        # Named taps first, then indexed taps — preserves insertion order
        feats = [self._feats[n]
                 for n in self.layer_names if n in self._feats]
        feats += [self._feats[f"__hook_module_{i}__"]
                  for i in range(self._n_module)
                  if f"__hook_module_{i}__" in self._feats]
        return {"logits": logits, "feats": feats}


class FeatureProjector(nn.Module):
    """1x1 conv + BN projector for student→teacher feature alignment.
    No ReLU — preserves full feature distribution for RKD/AT losses."""
    def __init__(self, student_channels, teacher_channels):
        super().__init__()
        self.projectors = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(sc, tc, kernel_size=1, bias=False),
                nn.BatchNorm2d(tc),
            )
            for sc, tc in zip(student_channels, teacher_channels)
        ])

    def forward(self, student_feats):
        return [proj(sf) for proj, sf in zip(self.projectors, student_feats)]


# ============================================================
# LOSSES
# ============================================================

def kd_loss_kl(s_logits, t_logits, temperature):
    """KL-divergence KD loss (Hinton 2015)."""
    s_log_prob = F.log_softmax(s_logits / temperature, dim=1)
    t_prob     = F.softmax(t_logits  / temperature, dim=1)
    return F.kl_div(s_log_prob, t_prob, reduction='batchmean') * (temperature ** 2)


def rkd_loss(s_feats, t_feats):
    """
    Relational KD (Park et al. 2019).
    Transfers pairwise distance + angle structure — robust to quantization noise
    since it doesn't require matching absolute feature magnitudes.
    """
    def _flat(f):
        return f.view(f.size(0), -1)

    def _pdist(e, eps=1e-12):
        e_sq = (e * e).sum(dim=1, keepdim=True)
        prod = torch.mm(e, e.t())
        return (e_sq + e_sq.t() - 2 * prod).clamp(min=eps).sqrt()

    total = 0.0
    for sf, tf in zip(s_feats, t_feats):
        sf = F.normalize(_flat(sf), dim=1)
        tf = F.normalize(_flat(tf), dim=1)
        s_d = _pdist(sf); s_d = s_d / (s_d.mean() + 1e-8)
        t_d = _pdist(tf); t_d = t_d / (t_d.mean() + 1e-8)
        loss_d = F.smooth_l1_loss(s_d, t_d)
        s_cos = torch.mm(sf, sf.t()).clamp(-1 + 1e-7, 1 - 1e-7)
        t_cos = torch.mm(tf, tf.t()).clamp(-1 + 1e-7, 1 - 1e-7)
        loss_a = F.smooth_l1_loss(s_cos, t_cos)
        total += loss_d + 2.0 * loss_a
    return total / max(len(s_feats), 1)


def att_loss(s_feats, t_feats):
    """Attention Transfer loss (Zagoruyko & Komodakis 2017)."""
    def attn(f):
        return F.normalize(f.pow(2).mean(1).view(f.size(0), -1), dim=1)
    total = sum(F.mse_loss(attn(sf), attn(tf)) for sf, tf in zip(s_feats, t_feats))
    return total / max(len(s_feats), 1)


# ============================================================
# TEACHER WEIGHTING
# ============================================================

def compute_entropy_weights(t_logits_list, temperature=1.0):
    entropies = []
    for logits in t_logits_list:
        probs = F.softmax(logits / temperature, dim=1)
        entropies.append(-(probs * (probs + 1e-8).log()).sum(dim=1))
    ent_stack = torch.stack(entropies)
    return F.softmax(-ent_stack, dim=0).unsqueeze(-1)


def compute_weighted_teacher_logits(t_logits_list, weights, temperature=None):
    if isinstance(weights, str) and weights == "entropy":
        dynamic_w = compute_entropy_weights(t_logits_list, temperature or 1.0)
        probs = torch.stack([F.softmax(l / (temperature or 1.0), dim=1)
                             for l in t_logits_list])
        return (dynamic_w * probs).sum(dim=0), "probs"
    else:
        w = weights.view(-1, 1, 1).to(t_logits_list[0].device)
        return (w * torch.stack(t_logits_list)).sum(dim=0), "logits"


def compute_robustness_scores(noise_results, teacher_keys, snr_levels=None,
                              metric="auc_normalized"):
    """
    Compute a scalar robustness score for each teacher from noise sweep results.
    See run_experiment.py for how this is wired into weighting_strategy="robustness".
    """
    scores = []
    for key in teacher_keys:
        r = noise_results.get(key, {})
        clean = r.get("clean", 1.0)
        if clean <= 0:
            scores.append(0.0)
            continue

        if snr_levels is not None:
            pairs = [(s, r.get(str(float(s)), r.get(str(s), clean)))
                     for s in sorted(snr_levels, reverse=True)]
        else:
            pairs = sorted(
                [(float(k), v) for k, v in r.items() if k != "clean"],
                reverse=True)

        if not pairs:
            scores.append(1.0)
            continue

        retentions = [acc / clean for _, acc in pairs]

        if metric == "auc_normalized":
            if len(retentions) == 1:
                score = retentions[0]
            else:
                auc = sum(0.5 * (retentions[i] + retentions[i + 1])
                          for i in range(len(retentions) - 1))
                score = auc / (len(retentions) - 1)
        elif metric == "mean_retention":
            score = sum(retentions) / len(retentions)
        elif metric == "worst_retention":
            score = min(retentions)
        else:
            raise ValueError(f"Unknown robustness metric: {metric!r}.")
        scores.append(score)

    return torch.tensor(scores, dtype=torch.float32)


def get_teacher_weights(strategy, num_teachers, accuracies=None,
                        temperature=5.0, robustness_scores=None):
    """Return teacher weights for a given strategy."""
    if strategy == "entropy":
        return "entropy"
    if strategy == "accuracy" and accuracies:
        acc = torch.tensor(accuracies, dtype=torch.float32) / 100.0
        return F.softmax(acc / temperature, dim=0)
    if strategy == "robustness" and robustness_scores is not None:
        return F.softmax(robustness_scores / temperature, dim=0)
    return torch.ones(num_teachers) / num_teachers


# ============================================================
# EVALUATOR
# ============================================================

@torch.no_grad()
def evaluate_topk(model, dataloader, device, verbose=False):
    """Return (top1, top5) accuracy percentages."""
    model.eval()
    correct = total = top5_correct = 0
    for batch in dataloader:
        inputs, targets, _ = split_batch(batch)
        inputs, targets = inputs.to(device), targets.to(device)
        out = model(inputs)
        logits = out["logits"] if isinstance(out, dict) else out
        _, pred = logits.max(1)
        correct += pred.eq(targets).sum().item()
        total   += targets.size(0)
        _, top5  = logits.topk(min(5, logits.size(1)), dim=1)
        top5_correct += top5.eq(targets.unsqueeze(1)).any(dim=1).sum().item()
    top1 = 100.0 * correct / total
    top5 = 100.0 * top5_correct / total
    if verbose:
        print(f"  Top-1: {top1:.2f}%  Top-5: {top5:.2f}%")
    return top1, top5


def evaluate(model, dataloader, device, verbose=False):
    """Top-1 accuracy only (back-compatible wrapper around evaluate_topk)."""
    return evaluate_topk(model, dataloader, device, verbose)[0]


# ============================================================
# TRAINER
# ============================================================

class MetricTracker:
    def __init__(self): self.reset()
    def reset(self): self._s = {}; self._c = {}
    def update(self, k, v, n=1):
        self._s[k] = self._s.get(k, 0.0) + v * n
        self._c[k] = self._c.get(k, 0) + n
    def avg(self, k): return self._s.get(k, 0.0) / max(self._c.get(k, 1), 1)


def train_one_epoch(student, teachers, dataloader, optimizer, device,
                    teacher_weights, temperature=4.0, alpha=0.5, beta=0.0,
                    projector=None, lambda_qat=0.0, epoch=0, total_epochs=1,
                    grad_clip=0.5, label_smoothing=0.1, use_rkd=True,
                    batch_log_every=0, stage_tag=""):
    student.train()
    criterion_ce = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    tracker = MetricTracker()
    total_grad_norm = 0.0
    num_batches = 0

    progress = epoch / max(total_epochs - 1, 1)
    T_eff  = temperature
    KD_FLOOR = 0.15
    kd_w  = alpha * (KD_FLOOR + (1.0 - KD_FLOOR) *
                     (0.5 + 0.5 * math.cos(math.pi * progress)))
    ce_w  = 1.0 - kd_w

    # Fixed-view teacher cache (see teacher_cache.py). The runner attaches it
    # to the loader object rather than threading it through four call
    # signatures. When present, the first n_base teachers are read from disk
    # instead of being run; teachers pooled from finished ladder stages are
    # still computed live, so the pool order must keep the base teachers first.
    cache = getattr(dataloader, "teacher_cache", None)
    n_cached = cache.n_base if cache is not None else 0
    if cache is not None:
        sampler = getattr(dataloader, "sampler", None)
        if hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)      # selects view e % K for this epoch
        if len(teachers) < n_cached:
            raise RuntimeError(
                f"Teacher cache holds {n_cached} teachers but the pool has "
                f"{len(teachers)}; the cache does not line up with the pool.")

    # Teacher feature maps are only read by the feature loss below. When that
    # loss is off (beta == 0, as in every ladder stage) holding them costs
    # ~90 MB per teacher per batch at bs=32/224 for nothing.
    need_feats = beta > 0 and projector is not None
    if need_feats and cache is not None:
        raise RuntimeError(
            "The teacher cache stores logits only, so the feature loss "
            "(beta > 0) cannot be used with it. Rerun without --teacher_cache_views.")

    n_batches = len(dataloader)
    t_epoch = time.time()
    for batch_i, batch in enumerate(dataloader):
        inputs, targets, rows = split_batch(batch)
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()

        s_out = student(inputs)
        s_logits, s_feats = s_out["logits"], s_out["feats"]

        t_logits_list, t_feats_list = [], []
        if cache is not None:
            t_logits_list.extend(cache.lookup(rows, device))
        with torch.no_grad():
            for t in teachers[n_cached:]:
                t_out = t(inputs)
                t_logits_list.append(t_out["logits"])
                if need_feats:
                    t_feats_list.append(t_out["feats"])

        loss_ce = criterion_ce(s_logits, targets)

        avg_teacher, mode = compute_weighted_teacher_logits(
            t_logits_list, teacher_weights, T_eff)
        if mode == "logits":
            loss_kd = kd_loss_kl(s_logits, avg_teacher, T_eff)
        else:
            s_log_prob = F.log_softmax(s_logits / T_eff, dim=1)
            loss_kd = F.kl_div(s_log_prob, avg_teacher,
                               reduction='batchmean') * (T_eff ** 2)

        loss_feat = torch.tensor(0.0, device=device)
        if beta > 0 and projector is not None and len(s_feats) > 0:
            s_proj = projector(s_feats)
            t_avg_feats = [
                torch.stack([tf[i] for tf in t_feats_list]).mean(0)
                for i in range(len(s_feats))
            ]
            if use_rkd:
                loss_feat = loss_feat + rkd_loss(s_proj, t_avg_feats)
            loss_feat = loss_feat + att_loss(s_proj, t_avg_feats)

        loss_qat = torch.tensor(0.0, device=device)
        if lambda_qat > 0:
            for module in student.modules():
                if isinstance(module, (QuantizedConv2d, QuantizedLinear)):
                    loss_qat = loss_qat + module.act_noise + module.weight_noise

        loss = ce_w * loss_ce + kd_w * loss_kd + beta * loss_feat + lambda_qat * loss_qat
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=grad_clip)
        optimizer.step()

        bs = inputs.size(0)
        tracker.update("loss", loss.item(), bs)
        tracker.update("loss_ce", loss_ce.item(), bs)
        tracker.update("loss_kd", loss_kd.item(), bs)
        tracker.update("loss_feat", loss_feat.item(), bs)
        tracker.update("loss_qat", loss_qat.item(), bs)
        total_grad_norm += grad_norm.item()
        num_batches += 1
        _, pred = s_logits.max(1)
        tracker.update("correct", pred.eq(targets).sum().item(), 1)
        tracker.update("total", bs, 1)

        # An epoch is hundreds of batches with several teacher forwards each;
        # without a heartbeat a slow run is indistinguishable from a hung one.
        if batch_log_every and (batch_i + 1) % batch_log_every == 0:
            done = batch_i + 1
            rate = done / max(time.time() - t_epoch, 1e-6)
            print(f"    [{stage_tag}] ep {epoch+1}/{total_epochs} "
                  f"batch {done}/{n_batches} ({100.0*done/n_batches:.0f}%) "
                  f"loss={tracker.avg('loss'):.4f} "
                  f"{rate:.2f} it/s  eta {(n_batches - done)/max(rate,1e-6)/60:.1f} min",
                  flush=True)

    train_acc = 100.0 * tracker.avg("correct") / max(tracker.avg("total"), 1)
    return tracker.avg("loss"), train_acc, {
        "loss_ce":   tracker.avg("loss_ce"),
        "loss_kd":   tracker.avg("loss_kd"),
        "loss_feat": tracker.avg("loss_feat"),
        "loss_qat":  tracker.avg("loss_qat"),
        "grad_norm": total_grad_norm / max(num_batches, 1),
        "T_eff": T_eff, "kd_w": kd_w, "ce_w": ce_w,
    }


# ============================================================
# SCHEDULER
# ============================================================

def make_scheduler(optimizer, epochs, lr, warmup_epochs=5, eta_min_factor=0.01):
    """Linear warmup → cosine decay. No restarts (restarts cause val drops in QAT)."""
    warmup_epochs = max(1, min(warmup_epochs, epochs))
    def lr_lambda(epoch):
        return (epoch + 1) / warmup_epochs if epoch < warmup_epochs else 1.0
    warmup_sched  = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    cosine_sched  = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(epochs - warmup_epochs, 1), eta_min=lr * eta_min_factor)
    return warmup_sched, cosine_sched, warmup_epochs


# ============================================================
# EARLY STOPPING
# ============================================================

class EarlyStopper:
    """
    Patience-based early stopping on validation accuracy.

    improved(val_acc) returns True if val_acc is a new best (beyond
    min_delta), in which case the internal patience counter resets.
    Otherwise the counter increments. should_stop becomes True once the
    counter reaches `patience`.
    """
    def __init__(self, patience=15, min_delta=0.05):
        self.patience   = patience
        self.min_delta  = min_delta
        self.best        = -float("inf")
        self.num_bad     = 0
        self.should_stop = False

    def step(self, val_acc):
        if val_acc > self.best + self.min_delta:
            self.best    = val_acc
            self.num_bad = 0
            return True   # improved
        self.num_bad += 1
        if self.num_bad >= self.patience:
            self.should_stop = True
        return False      # did not improve


# ============================================================
# FROM-SCRATCH BASELINE TRAINER (no KD, no teacher)
# ============================================================
#
# Why this exists: the chenyaofo/pytorch-cifar-models hub checkpoints were
# themselves trained from scratch (pretrained=False in their own configs)
# using a specific recipe: SGD + Nesterov momentum, lr=0.1, weight_decay
# =5e-4, CosineAnnealingLR over 200 epochs, no warmup, plain cross-entropy
# (no distillation). Our KD pipeline trains students with AdamW + KD loss,
# which is a reasonable recipe for distillation stages but is NOT what gets
# a from-scratch ResNet32 to the hub's reported ~69-70% baseline — it needs
# the hub's own optimizer recipe to land near that number.
#
# train_baseline_from_scratch reproduces that recipe as closely as possible
# so a clean FP32 baseline can be trained on OUR train/val split (45k/5k,
# carved from the 50k train set) without touching the test set or using
# hub-pretrained weights (which would leak val data into the model). Once
# this baseline is healthy, it becomes the init_from source for the rest of
# the KD/QAT chain (M3, M4a, M4b), instead of those stages starting from a
# weak from-scratch-via-KD M2.

def train_baseline_from_scratch(
    stage_name, student_model, train_loader, val_loader, test_loader, device,
    max_epochs=200, lr=0.1, momentum=0.9, weight_decay=5e-4, nesterov=True,
    eta_min=0.0, label_smoothing=0.0,
    patience=30, min_delta=0.05,
    checkpoint_path=None,
):
    """
    Train a plain classifier from scratch with the hub's original recipe:
    SGD + Nesterov momentum, CosineAnnealingLR (no warmup), pure
    cross-entropy loss (no KD, no teacher). Patience-based early stopping
    still applies on top, since the hub's fixed 200-epoch schedule is a
    ceiling, not a requirement — we stop once val saturates.

    Data discipline: identical to the rest of the pipeline. Optimizes only
    on train_loader; val_loader drives checkpoint selection and early
    stopping; test_loader is evaluated and logged every epoch but never
    influences any decision.

    Returns (student_model, best_val_acc, test_acc_at_best_val).
    """
    print(f"\n{'='*70}")
    print(f"BASELINE (FROM SCRATCH, NO KD): {stage_name}")
    print(f"  Optimizer: SGD lr={lr} momentum={momentum} nesterov={nesterov} "
          f"weight_decay={weight_decay}")
    print(f"  Scheduler: CosineAnnealingLR T_max={max_epochs} eta_min={eta_min}")
    print(f"  max_epochs={max_epochs} patience={patience} min_delta={min_delta} "
          f"label_smoothing={label_smoothing}")
    print(f"{'='*70}")

    student_model = student_model.to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    optimizer = torch.optim.SGD(
        student_model.parameters(), lr=lr, momentum=momentum,
        weight_decay=weight_decay, nesterov=nesterov,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max_epochs, eta_min=eta_min)

    stopper = EarlyStopper(patience=patience, min_delta=min_delta)
    best_acc, best_test_acc, best_state = -1.0, 0.0, None

    for epoch in range(max_epochs):
        student_model.train()
        tracker = MetricTracker()
        total_grad_norm = 0.0
        num_batches = 0

        for batch in train_loader:
            inputs, targets, _ = split_batch(batch)
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            logits = student_model(inputs)
            loss = criterion(logits, targets)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                student_model.parameters(), max_norm=5.0)
            optimizer.step()

            bs = inputs.size(0)
            tracker.update("loss", loss.item(), bs)
            total_grad_norm += grad_norm.item()
            num_batches += 1
            _, pred = logits.max(1)
            tracker.update("correct", pred.eq(targets).sum().item(), 1)
            tracker.update("total", bs, 1)

        scheduler.step()

        train_acc = 100.0 * tracker.avg("correct") / max(tracker.avg("total"), 1)
        val_acc  = evaluate(student_model, val_loader, device)
        test_acc = evaluate(student_model, test_loader, device)
        current_lr = optimizer.param_groups[0]['lr']
        avg_gnorm = total_grad_norm / max(num_batches, 1)

        print(f"[{stage_name}] Epoch {epoch+1:4d}/{max_epochs} | "
              f"loss={tracker.avg('loss'):.4f} | "
              f"train={train_acc:.2f}% val={val_acc:.2f}% test={test_acc:.2f}% | "
              f"lr={current_lr:.5f} gnorm={avg_gnorm:.3f}")

        improved = stopper.step(val_acc)
        if val_acc > best_acc:
            best_acc      = val_acc
            best_test_acc = test_acc
            best_state    = copy.deepcopy(student_model.state_dict())
            tag = " *** New best (val) ***" if improved else ""
            print(f"  val={best_acc:.2f}% (test={best_test_acc:.2f}%){tag}")

        if stopper.should_stop:
            print(f"  [EARLY STOP] No val improvement > {min_delta} for "
                  f"{patience} epochs. Stopping.")
            break

    print(f"\n[{stage_name}] Best Val Acc: {best_acc:.2f}% | "
          f"Test Acc @ best val: {best_test_acc:.2f}%")
    student_model.load_state_dict(best_state)
    if checkpoint_path is not None:
        torch.save(best_state, checkpoint_path)
        print(f"  [CKPT] Saved best model → {checkpoint_path}")
    return student_model, best_acc, best_test_acc


# ============================================================
# PIPELINE RUNNER — single training run (one cycle)
# ============================================================

def _run_single_cycle(
    stage_name, teacher_models, student_model, train_loader, val_loader,
    test_loader, device,
    max_epochs, lr, alpha=0.5, beta=0.0,
    temperature=4.0, weighting_strategy="uniform", teacher_accuracies=None,
    teacher_robustness_scores=None,
    feature_strategy="none", lambda_qat=0.0, grad_clip=0.5, eta_min_factor=0.01,
    use_rkd=True, warmup_epochs=5, label_smoothing=0.1, quant_log_every=10,
    teacher_names=None,
    patience=15, min_delta=0.05,
    apply_qat=False, bitwidth_w=32, bitwidth_a=32,
    student_feat_channels=(16, 32, 64),
    teacher_feat_channels=(16, 32, 64),
    student_layer_names=('layer1', 'layer2', 'layer3'),
    teacher_layer_names=('layer1', 'layer2', 'layer3'),
    epoch_offset=0,
    epoch_callback=None,
    cycle_index=1,
    batch_log_every=0,
):
    """
    Run ONE training cycle to patience-based convergence (or max_epochs,
    whichever comes first). This is the inner loop used by both the
    single-pass `run_distillation_stage` and the multi-cycle saturation loop.

    student_model is assumed to ALREADY have quantizers applied and
    calibrated if apply_qat=True — this function does not touch
    quantization setup, only training.

    Returns (best_state_dict, best_val_acc, test_acc_at_best_val,
             epochs_run, stopped_early: bool).
    """
    if teacher_names is None:
        teacher_names = [f"T{i}" for i in range(len(teacher_models))]

    weights = get_teacher_weights(
        weighting_strategy, len(teacher_models),
        accuracies=teacher_accuracies, temperature=5.0,
        robustness_scores=teacher_robustness_scores,
    )
    if not isinstance(weights, str):
        weights = weights.to(device)
        print(f"  Teacher weights: {weights.cpu().tolist()}")

    student  = FeatureExtractorModel(student_model, layer_names=student_layer_names).to(device)
    teachers = []
    for t in teacher_models:
        t.to(device).eval()
        teachers.append(
            FeatureExtractorModel(t, layer_names=teacher_layer_names,
                                  detach_feats=True).to(device))

    projector = None
    if feature_strategy == "projected" and beta > 0:
        projector = FeatureProjector(student_feat_channels, teacher_feat_channels).to(device)
        print("  FeatureProjector enabled.")

    scale_params, other_params = [], []
    for name, p in student_model.named_parameters():
        if 'act_q.scale' in name or 'weight_q.scale' in name:
            scale_params.append(p)
        else:
            other_params.append(p)
    if projector is not None:
        other_params += list(projector.parameters())

    if scale_params:
        optimizer = torch.optim.AdamW([
            {'params': other_params, 'lr': lr,       'weight_decay': 5e-4},
            {'params': scale_params, 'lr': lr * 2.0, 'weight_decay': 0.0},
        ])
        print(f"  Optimizer: {len(other_params)} weight params (wd=5e-4) + "
              f"{len(scale_params)} scale params (wd=0, lr={lr*2:.2e})")
    else:
        optimizer = torch.optim.AdamW(other_params, lr=lr, weight_decay=5e-4)

    warmup_sched, cosine_sched, n_warmup = make_scheduler(
        optimizer, max_epochs, lr, warmup_epochs=warmup_epochs,
        eta_min_factor=eta_min_factor)

    stopper = EarlyStopper(patience=patience, min_delta=min_delta)
    best_acc, best_test_acc, best_state = -1.0, 0.0, None

    print(f"  Cycle training loop: up to {max_epochs} epochs, "
          f"patience={patience}, min_delta={min_delta}")

    epochs_run = 0
    for epoch in range(max_epochs):
        epochs_run = epoch + 1
        global_epoch = epoch_offset + epoch + 1

        epoch_t0 = time.time()
        loss, train_acc, info = train_one_epoch(
            student, teachers, train_loader, optimizer, device,
            teacher_weights=weights, temperature=temperature,
            alpha=alpha, beta=beta, projector=projector,
            lambda_qat=lambda_qat, epoch=epoch, total_epochs=max_epochs,
            grad_clip=grad_clip, label_smoothing=label_smoothing, use_rkd=use_rkd,
            batch_log_every=batch_log_every, stage_tag=stage_name,
        )
        (warmup_sched if epoch < n_warmup else cosine_sched).step()

        val_acc,  val_top5  = evaluate_topk(student, val_loader,  device)
        # The test split is reported, never decided on. The only test number
        # ever quoted is the one belonging to the best-val checkpoint, so it is
        # evaluated on new-best epochs only; other epochs log NaN. Halves the
        # per-epoch eval cost (val and test are both 10k images).
        is_best = val_acc > best_acc
        if is_best:
            test_acc, test_top5 = evaluate_topk(student, test_loader, device)
        else:
            test_acc, test_top5 = float("nan"), float("nan")
        current_lr = optimizer.param_groups[0]['lr']
        epoch_secs = time.time() - epoch_t0

        print(f"[{stage_name}] Epoch {global_epoch:4d} (cycle ep {epoch+1:3d}/{max_epochs}) | "
              f"loss={loss:.4f} (ce={info['loss_ce']:.3f} kd={info['loss_kd']:.3f} "
              f"feat={info['loss_feat']:.3f} qat={info['loss_qat']:.4f}) | "
              f"train={train_acc:.2f}% val={val_acc:.2f}% test={test_acc:.2f}% | "
              f"lr={current_lr:.6f} gnorm={info['grad_norm']:.3f} "
              f"T_eff={info['T_eff']:.2f} kd_w={info['kd_w']:.3f} "
              f"secs={epoch_secs:.1f}")

        if epoch_callback is not None:
            epoch_callback({
                "stage":       stage_name,
                "cycle":       cycle_index,
                "cycle_epoch": epoch + 1,
                "epoch":       global_epoch,
                "max_epochs":  max_epochs,
                "bitwidth_w":  bitwidth_w,
                "bitwidth_a":  bitwidth_a,
                "lr":          current_lr,
                "loss":        loss,
                "loss_ce":     info["loss_ce"],
                "loss_kd":     info["loss_kd"],
                "loss_feat":   info["loss_feat"],
                "loss_qat":    info["loss_qat"],
                "grad_norm":   info["grad_norm"],
                "kd_w":        info["kd_w"],
                "ce_w":        info["ce_w"],
                "T_eff":       info["T_eff"],
                "train_top1":  train_acc,
                "val_top1":    val_acc,
                "val_top5":    val_top5,
                "test_top1":   test_acc,
                "test_top5":   test_top5,
                "best_val_so_far": max(best_acc, val_acc),
                "epoch_secs":  epoch_secs,
            })

        if apply_qat and (epoch + 1) % quant_log_every == 0:
            log_quantization_health(student_model, global_epoch, stage_name)

        improved = stopper.step(val_acc)
        if is_best:
            best_acc      = val_acc
            best_test_acc = test_acc
            best_state    = copy.deepcopy(student_model.state_dict())
            tag = " *** New best (val) ***" if improved else ""
            print(f"  val={best_acc:.2f}% (test={best_test_acc:.2f}%){tag}")

        if stopper.should_stop:
            print(f"  [EARLY STOP] No val improvement > {min_delta} for "
                  f"{patience} epochs (cycle ep {epoch+1}). Stopping cycle.")
            break

    stopped_early = stopper.should_stop
    student.remove_hooks()
    for t in teachers:
        t.remove_hooks()
    return best_state, best_acc, best_test_acc, epochs_run, stopped_early


# ============================================================
# PIPELINE RUNNER — single-pass wrapper (backward compatible)
# ============================================================

def run_distillation_stage(
    stage_name, teacher_models, student_model, train_loader, val_loader,
    test_loader, device,
    epochs, lr, bitwidth_w=32, bitwidth_a=32, alpha=0.5, beta=0.0,
    temperature=4.0, weighting_strategy="uniform", teacher_accuracies=None,
    teacher_robustness_scores=None,
    feature_strategy="none", lambda_qat=0.0, grad_clip=0.5, eta_min_factor=0.01,
    use_rkd=True, warmup_epochs=5, label_smoothing=0.1, quant_log_every=10,
    teacher_names=None, checkpoint_path=None,
    patience=None, min_delta=0.05,
    student_feat_channels=(16, 32, 64),
    teacher_feat_channels=(16, 32, 64),
    student_layer_names=('layer1', 'layer2', 'layer3'),
    teacher_layer_names=('layer1', 'layer2', 'layer3'),
    epoch_callback=None,
    batch_log_every=0,
):
    """
    Single-pass training entry point (back-compatible with v2 callers).

    `epochs` is now treated as max_epochs for the patience-based stopper.
    If patience is None, defaults to epochs (i.e. no early stop — trains
    the full budget, matching old behaviour).

    Data discipline:
      - The model is optimized ONLY on train_loader.
      - Checkpoints are selected on val_loader.
      - test_loader is reported every epoch but NEVER drives any decision.

    Returns (student_model, best_val_acc, test_acc_at_best_val).
    """
    print(f"\n{'='*70}")
    print(f"STAGE: {stage_name}")
    print(f"  Bitwidth: W{bitwidth_w}A{bitwidth_a} | Weighting: {weighting_strategy}")
    print(f"  alpha={alpha} beta={beta} T={temperature} lr={lr} max_epochs={epochs}")
    print(f"  lambda_qat={lambda_qat} grad_clip={grad_clip} use_rkd={use_rkd}")
    print(f"{'='*70}")

    if patience is None:
        patience = epochs  # no early stop — exhaust full budget like v2

    apply_qat = (bitwidth_w < 32) or (bitwidth_a < 32)
    if apply_qat:
        student_model = replace_with_fake_quantization(student_model, bitwidth_w, bitwidth_a)
        student_model = student_model.to(device)
        # More calibration batches at ultra-low precision: 4-level (2-bit)
        # quantization grid is highly sensitive to outliers in the scale estimate
        calib_batches = 80 if (bitwidth_w <= 2 or bitwidth_a <= 2) else 30
        student_model = calibrate_quantizers(student_model, train_loader, device,
                                             num_batches=calib_batches)
    else:
        student_model = student_model.to(device)

    best_state, best_acc, best_test_acc, epochs_run, stopped_early = _run_single_cycle(
        stage_name=stage_name, teacher_models=teacher_models,
        student_model=student_model, train_loader=train_loader,
        val_loader=val_loader, test_loader=test_loader, device=device,
        max_epochs=epochs, lr=lr, alpha=alpha, beta=beta,
        temperature=temperature, weighting_strategy=weighting_strategy,
        teacher_accuracies=teacher_accuracies,
        teacher_robustness_scores=teacher_robustness_scores,
        feature_strategy=feature_strategy, lambda_qat=lambda_qat,
        grad_clip=grad_clip, eta_min_factor=eta_min_factor, use_rkd=use_rkd,
        warmup_epochs=warmup_epochs, label_smoothing=label_smoothing,
        quant_log_every=quant_log_every, teacher_names=teacher_names,
        patience=patience, min_delta=min_delta,
        apply_qat=apply_qat, bitwidth_w=bitwidth_w, bitwidth_a=bitwidth_a,
        student_feat_channels=student_feat_channels,
        teacher_feat_channels=teacher_feat_channels,
        student_layer_names=student_layer_names,
        teacher_layer_names=teacher_layer_names,
        epoch_callback=epoch_callback,
        batch_log_every=batch_log_every,
    )

    print(f"\n[{stage_name}] Best Val Acc: {best_acc:.2f}% | "
          f"Test Acc @ best val: {best_test_acc:.2f}% | "
          f"Epochs run: {epochs_run}{' (early stopped)' if stopped_early else ''}")
    student_model.load_state_dict(best_state)
    if checkpoint_path is not None:
        torch.save(best_state, checkpoint_path)
        print(f"  [CKPT] Saved best model → {checkpoint_path}")
    return student_model, best_acc, best_test_acc


# ============================================================
# PIPELINE RUNNER — multi-cycle saturation loop
# ============================================================

def run_distillation_stage_saturating(
    stage_name, teacher_models, student_model, train_loader, val_loader,
    test_loader, device,
    epochs_per_cycle, lr, bitwidth_w=32, bitwidth_a=32, alpha=0.5, beta=0.0,
    temperature=4.0, weighting_strategy="uniform", teacher_accuracies=None,
    teacher_robustness_scores=None,
    feature_strategy="none", lambda_qat=0.0, grad_clip=0.5, eta_min_factor=0.01,
    use_rkd=True, warmup_epochs=5, label_smoothing=0.1, quant_log_every=10,
    teacher_names=None, checkpoint_path=None,
    patience=15, min_delta=0.05,
    max_cycles=5, cycle_min_delta=0.10, cycle_lr_decay=0.6,
    student_feat_channels=(16, 32, 64),
    teacher_feat_channels=(16, 32, 64),
    student_layer_names=('layer1', 'layer2', 'layer3'),
    teacher_layer_names=('layer1', 'layer2', 'layer3'),
    epoch_callback=None,
    batch_log_every=0,
):
    """
    Train a student at a FIXED precision through repeated cycles until the
    best validation accuracy saturates across cycles, rather than assuming
    a single training pass reaches the precision's ceiling.

    Mechanics, addressing the "is lowest precision reached?" problem:
      - Cycle 1 trains student_model (freshly quantized + calibrated if
        QAT) from scratch via _run_single_cycle with full patience-based
        early stopping.
      - At the end of each cycle, if the cycle's best val acc improved over
        the running best by more than `cycle_min_delta`, we continue: cycle
        k+1 WARM-STARTS from cycle k's best checkpoint.
      - If a cycle fails to improve the running best by cycle_min_delta, OR
        max_cycles is reached, the loop stops and the running best is
        returned as this precision's final model.

    Avoiding sharp restarts between cycles (your concern):
      - The optimizer (including momentum/Adam state) is rebuilt fresh each
        cycle — this is unavoidable since a new cycle is a new
        train_one_epoch loop — BUT:
          * LSQ quantizer scale parameters are NOT reinitialized between
            cycles. extract_full_qat_weights_keep_scales + strict=False
            loading carries the learned scales forward, so the quantization
            grid the student "lives in" doesn't reset.
          * Each new cycle's peak LR is `lr * (cycle_lr_decay ** (cycle-1))`
            — strictly decaying, so later cycles make smaller, gentler
            updates rather than repeating cycle 1's aggressive exploration.
          * Each cycle still gets its own short warmup (same warmup_epochs),
            so even the decayed peak LR is approached gradually rather than
            applied as a step change on cycle epoch 1.

    Returns (student_model, best_val_acc, test_acc_at_best_val, n_cycles_run).
    """
    print(f"\n{'#'*70}")
    print(f"SATURATING STAGE: {stage_name}")
    print(f"  Bitwidth: W{bitwidth_w}A{bitwidth_a} | max_cycles={max_cycles} "
          f"| epochs_per_cycle={epochs_per_cycle} | patience={patience}")
    print(f"  cycle_min_delta={cycle_min_delta} | cycle_lr_decay={cycle_lr_decay}")
    print(f"{'#'*70}")

    apply_qat = (bitwidth_w < 32) or (bitwidth_a < 32)
    if apply_qat:
        student_model = replace_with_fake_quantization(student_model, bitwidth_w, bitwidth_a)
        student_model = student_model.to(device)
        # More calibration batches at ultra-low precision: 4-level (2-bit)
        # quantization grid is highly sensitive to outliers in the scale estimate
        calib_batches = 80 if (bitwidth_w <= 2 or bitwidth_a <= 2) else 30
        student_model = calibrate_quantizers(student_model, train_loader, device,
                                             num_batches=calib_batches)
    else:
        student_model = student_model.to(device)

    running_best_acc, running_best_test, running_best_state = -1.0, 0.0, None
    total_epochs_run = 0
    cycles_run = 0

    for cycle in range(1, max_cycles + 1):
        cycles_run = cycle
        cycle_lr = lr * (cycle_lr_decay ** (cycle - 1))
        print(f"\n  >>> CYCLE {cycle}/{max_cycles} | lr={cycle_lr:.3e} "
              f"(decay factor {cycle_lr_decay ** (cycle - 1):.3f}) <<<")

        if cycle > 1:
            # Warm-start from the running best, keeping LSQ scales intact.
            carried = extract_full_qat_weights_keep_scales(running_best_state)
            student_model.load_state_dict(carried, strict=False)
            print(f"  Warm-started cycle {cycle} from previous cycle's best "
                  f"checkpoint (LSQ scales carried over, optimizer reset).")

        cycle_state, cycle_val, cycle_test, cycle_epochs, cycle_stopped = _run_single_cycle(
            stage_name=f"{stage_name} [cycle {cycle}]",
            teacher_models=teacher_models, student_model=student_model,
            train_loader=train_loader, val_loader=val_loader,
            test_loader=test_loader, device=device,
            max_epochs=epochs_per_cycle, lr=cycle_lr, alpha=alpha, beta=beta,
            temperature=temperature, weighting_strategy=weighting_strategy,
            teacher_accuracies=teacher_accuracies,
            teacher_robustness_scores=teacher_robustness_scores,
            feature_strategy=feature_strategy, lambda_qat=lambda_qat,
            grad_clip=grad_clip, eta_min_factor=eta_min_factor, use_rkd=use_rkd,
            warmup_epochs=warmup_epochs, label_smoothing=label_smoothing,
            quant_log_every=quant_log_every, teacher_names=teacher_names,
            patience=patience, min_delta=min_delta,
            apply_qat=apply_qat, bitwidth_w=bitwidth_w, bitwidth_a=bitwidth_a,
            student_feat_channels=student_feat_channels,
            teacher_feat_channels=teacher_feat_channels,
            student_layer_names=student_layer_names,
            teacher_layer_names=teacher_layer_names,
            epoch_offset=total_epochs_run,
            epoch_callback=epoch_callback,
            cycle_index=cycle,
            batch_log_every=batch_log_every,
        )
        total_epochs_run += cycle_epochs

        improvement = cycle_val - running_best_acc
        print(f"  Cycle {cycle} result: best_val={cycle_val:.2f}% "
              f"(test={cycle_test:.2f}%) over {cycle_epochs} epochs"
              f"{' [early stopped]' if cycle_stopped else ''} | "
              f"vs running best {running_best_acc:.2f}% "
              f"(Δ={improvement:+.2f}%)")

        if cycle_val > running_best_acc:
            running_best_acc, running_best_test = cycle_val, cycle_test
            running_best_state = cycle_state
            student_model.load_state_dict(running_best_state)
            if checkpoint_path is not None:
                torch.save(running_best_state, checkpoint_path)
                print(f"  [CKPT] Saved new running-best model → {checkpoint_path}")

        if improvement < cycle_min_delta:
            print(f"  [SATURATED] Cycle {cycle} improvement ({improvement:+.2f}%) "
                  f"< cycle_min_delta ({cycle_min_delta}%). "
                  f"Precision saturated after {cycle} cycle(s).")
            break
    else:
        print(f"  [MAX CYCLES] Reached max_cycles={max_cycles} without "
              f"saturating below cycle_min_delta — using best result found.")

    print(f"\n[{stage_name}] SATURATED after {cycles_run} cycle(s), "
          f"{total_epochs_run} total epochs.")
    print(f"  Final Best Val Acc: {running_best_acc:.2f}% | "
          f"Test Acc @ best val: {running_best_test:.2f}%")

    student_model.load_state_dict(running_best_state)
    return student_model, running_best_acc, running_best_test, cycles_run
