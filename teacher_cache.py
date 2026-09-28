"""
teacher_cache.py
================
Precomputed teacher logits over a fixed bank of augmented views.

WHY
---
The three base teachers (M1/M2/M3) are frozen, but their *input* is not: the
train transform re-randomises the crop and flip every epoch, so there is no
stable key to cache on. The three ResNets are ~9.5 GFLOPs/image of forward
against MobileNetV2's ~0.3, so they dominate every training step and are paid
for again on all 310 remaining ladder epochs.

This module fixes the augmentation to K deterministic views per image. View k
of image i is a pure function of (seed, i, k), so the teachers can be run over
all N*K views once, up front, and every later epoch reads logits instead of
recomputing them. Epoch e uses view e % K.

WHAT IT COSTS
-------------
Augmentation diversity is capped at K views per image instead of unbounded.
At K=16 with the 224px TinyImageNet transform (Resize -> RandomCrop pad=14 ->
HFlip) the underlying view space is ~1700 distinct views, so K=16 samples it
coarsely. STATE THIS IN THE PAPER. It is a real change to the training setup,
not an implementation detail.

Everything else is untouched: cached logits match a live forward to float16
rounding (see --verify), teachers pooled from finished ladder stages are still
computed live each step, and the entropy weighting sees the same numbers it
would have seen.

USAGE
-----
    python teacher_cache.py --config <cfg> --views 16
    python teacher_cache.py --config <cfg> --views 16 --verify
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from typing import List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler


# ============================================================
# DETERMINISTIC FIXED-VIEW DATASET
# ============================================================

def _view_seed(base_seed: int, index: int, view: int) -> int:
    """Stable integer hash -> the RNG seed that renders view `view` of sample
    `index`. Explicit arithmetic rather than hash(), which is not stable."""
    h = (base_seed * 1000003) ^ (index * 1009 + 0x9E3779B9) ^ (view * 2654435761)
    return h % (2 ** 31 - 1)


def _logits(out):
    """Teachers are raw modules here (bare tensor), but pipeline wraps them
    in FeatureExtractorModel during a stage (dict). The wrapper passes logits
    through unchanged, so either shape carries the same numbers. Same idiom
    as pipeline.evaluate_topk."""
    return out["logits"] if isinstance(out, dict) else out


class FixedViewDataset(Dataset):
    """
    Wraps the train Subset so flat index j == i * K + k yields view k of
    sample i, rendered deterministically.

    Returns (image, target, j). The trailing j is the cache row, which the
    training loop uses to look the teacher logits up.
    """

    def __init__(self, base, num_views: int, base_seed: int = 42):
        self.base = base
        self.k = int(num_views)
        self.base_seed = int(base_seed)

    def __len__(self):
        return len(self.base) * self.k

    def __getitem__(self, j: int):
        i, k = divmod(int(j), self.k)
        # torchvision's RandomCrop / RandomHorizontalFlip draw from the global
        # torch RNG, so seeding it here fully determines the view. State is
        # saved and restored so nothing else in the worker is perturbed.
        state = torch.get_rng_state()
        try:
            torch.manual_seed(_view_seed(self.base_seed, i, k))
            img, target = self.base[i]
        finally:
            torch.set_rng_state(state)
        return img, target, j


class FixedViewSampler(Sampler):
    """
    Yields one shuffled pass over all N samples, all rendered at view e % K.

    The view travels in the emitted indices rather than in dataset state:
    with persistent_workers=True the workers hold their own copies of the
    dataset and would never observe a set_epoch() on the main-process object.
    The sampler is iterated in the main process, so this is safe.
    """

    def __init__(self, n: int, num_views: int, seed: int = 42):
        self.n = int(n)
        self.k = int(num_views)
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def __len__(self):
        return self.n

    def __iter__(self):
        view = self.epoch % self.k
        g = torch.Generator().manual_seed(self.seed + 7919 * self.epoch)
        for i in torch.randperm(self.n, generator=g).tolist():
            yield i * self.k + view


# ============================================================
# CACHE FILE
# ============================================================

def _sha256(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def cache_paths(out_dir: str, num_views: int):
    stem = os.path.join(out_dir, "teacher_logits_K%d" % num_views)
    return stem + ".npy", stem + ".json"


class TeacherLogitCache:
    """
    Memory-mapped reader. Lives in the MAIN process only — worker processes
    never touch it, which keeps it clear of the Windows shared-memory commit
    limit the dataloader workers already push against.
    """

    def __init__(self, npy_path: str, meta: dict):
        self.meta = meta
        self.n_base = len(meta["teachers"])
        self.num_views = meta["num_views"]
        self.array = np.load(npy_path, mmap_mode="r")   # (n_base, N*K, C) fp16

    @classmethod
    def load(cls, out_dir: str, num_views: int, teacher_ckpts: List[str],
             expect: dict) -> "TeacherLogitCache":
        npy, js = cache_paths(out_dir, num_views)
        if not (os.path.isfile(npy) and os.path.isfile(js)):
            raise FileNotFoundError(
                "No teacher cache at %s. Build it first:\n"
                "  python teacher_cache.py --config <cfg> --views %d"
                % (npy, num_views))
        meta = json.load(open(js, encoding="utf-8"))

        # Any of these changing silently invalidates every cached logit.
        for key, want in expect.items():
            if meta.get(key) != want:
                raise RuntimeError(
                    "Teacher cache is stale: %s was %r when the cache was built, "
                    "is %r now. Rebuild it." % (key, meta.get(key), want))
        for path in teacher_ckpts:
            name = os.path.basename(path)
            rec = meta["teachers"].get(name)
            if rec is None:
                raise RuntimeError("Cache holds no logits for teacher %r." % name)
            if rec["sha256"] != _sha256(path):
                raise RuntimeError(
                    "Teacher checkpoint %r changed since the cache was built. "
                    "Rebuild it, or the KD targets belong to a different model "
                    "than the one in the pool." % name)
        return cls(npy, meta)

    def lookup(self, rows: torch.Tensor, device) -> List[torch.Tensor]:
        """rows: LongTensor [B] of flat view indices -> list of n_base [B, C]."""
        idx = rows.cpu().numpy()
        order = np.argsort(idx)             # memmap gather prefers sorted reads
        inv = np.argsort(order)
        gathered = self.array[:, idx[order], :][:, inv, :]
        t = torch.from_numpy(np.ascontiguousarray(gathered)).to(
            device, dtype=torch.float32)
        return [t[i] for i in range(t.shape[0])]


def expected_meta(cfg: dict) -> dict:
    """The identity of a cache: anything here changing means rebuild."""
    exp, data = cfg["experiment"], cfg["data"]
    return {
        "n_train": None,        # filled by the builder / checked by the runner
        "num_classes": exp.get("num_classes", 200),
        "image_size": data.get("image_size", 224),
        "base_seed": exp.get("seed", 42),
        "val_seed": data.get("val_seed", 42),
        "val_fraction": data.get("val_fraction", 0.10),
        "dataset": exp.get("dataset"),
    }


def teacher_ckpt_paths(cfg: dict) -> List[str]:
    out_dir = cfg["experiment"]["output_dir"]
    specs = [cfg["teacher"]["checkpoint"]]
    specs += [a["checkpoint"] for a in cfg.get("aux_teachers", [])]
    return [os.path.join(out_dir, c) for c in specs]


# ============================================================
# BUILD
# ============================================================

def build(cfg_path: str, num_views: int, batch_size: Optional[int],
          num_workers: int, verify_only: bool, data_root: Optional[str] = None):
    import yaml
    from pipeline_imagenet_patch import build_loaders_from_cfg, load_model_imagenet

    cfg = yaml.safe_load(open(cfg_path, encoding="utf-8"))
    exp, data = cfg["experiment"], cfg["data"]
    out_dir = exp["output_dir"]
    num_classes = exp.get("num_classes", 200)
    image_size = data.get("image_size", 224)
    small_input = exp.get("dataset") == "tinyimagenet" and image_size <= 64
    base_seed = exp.get("seed", 42)

    if batch_size:
        data["batch_size"] = batch_size
    if data_root:
        data["data_root"] = data_root
    data["num_workers"] = num_workers

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise SystemExit("No CUDA device — refusing to build the cache on CPU.")
    print("device: %s (%s)" % (device, torch.cuda.get_device_name(0)))
    torch.backends.cudnn.benchmark = True

    # Teacher pool, in the exact order the ladder registers them.
    specs = [(cfg["teacher"]["checkpoint"], cfg["teacher"]["arch"])]
    specs += [(a["checkpoint"], a["arch"]) for a in cfg.get("aux_teachers", [])]

    teachers, ckpt_paths = [], []
    for ckpt, arch in specs:
        path = os.path.join(out_dir, ckpt)
        if not os.path.isfile(path):
            raise SystemExit("Missing teacher checkpoint: %s" % path)
        m = load_model_imagenet(arch, num_classes=num_classes, pretrained=False,
                                checkpoint_path=path, device=device,
                                small_input=small_input)
        m.eval()
        for p in m.parameters():
            p.requires_grad_(False)
        teachers.append(m)
        ckpt_paths.append(path)
        print("  teacher %-10s %s" % (ckpt, arch))

    train_loader, _, _ = build_loaders_from_cfg(cfg)
    train_subset = train_loader.dataset
    n = len(train_subset)
    bs = data["batch_size"]
    print("train samples: %d  views: %d  rows: %d" % (n, num_views, n * num_views))

    view_ds = FixedViewDataset(train_subset, num_views, base_seed)
    npy_path, json_path = cache_paths(out_dir, num_views)

    expect = expected_meta(cfg)
    expect["n_train"] = n
    expect["num_views"] = num_views

    if verify_only:
        cache = TeacherLogitCache.load(out_dir, num_views, ckpt_paths, expect)
        _verify(cache, view_ds, teachers, device, bs, n, num_views)
        return

    os.makedirs(out_dir, exist_ok=True)
    arr = np.lib.format.open_memmap(
        npy_path, mode="w+", dtype=np.float16,
        shape=(len(teachers), n * num_views, num_classes))
    print("writing %s (%.2f GB, float16)" % (npy_path, arr.nbytes / 1e9))

    t0 = time.time()
    for k in range(num_views):
        idx = [i * num_views + k for i in range(n)]
        loader = DataLoader(view_ds, batch_size=bs, sampler=idx,
                            num_workers=num_workers, pin_memory=True,
                            persistent_workers=False)
        done = 0
        for images, _, rows in loader:
            images = images.to(device, non_blocking=True)
            with torch.no_grad():
                for ti, t in enumerate(teachers):
                    arr[ti, rows.numpy(), :] = \
                        _logits(t(images)).half().cpu().numpy()
            done += images.shape[0]
            if done % (bs * 200) < bs:
                el = time.time() - t0
                frac = (k * n + done) / float(num_views * n)
                print("  view %d/%d  %d/%d  (%.1f%% overall, eta %.0f min)"
                      % (k + 1, num_views, done, n, 100 * frac,
                         el / max(frac, 1e-9) * (1 - frac) / 60), flush=True)
        arr.flush()
        print("  view %d/%d done (%.1f min elapsed)"
              % (k + 1, num_views, (time.time() - t0) / 60), flush=True)

    meta = dict(expect)
    meta["dtype"] = "float16"
    meta["teachers"] = {}
    for i, (path, (_, arch)) in enumerate(zip(ckpt_paths, specs)):
        meta["teachers"][os.path.basename(path)] = {
            "arch": arch, "sha256": _sha256(path), "slot": i}
    json.dump(meta, open(json_path, "w", encoding="utf-8"), indent=2)
    del arr
    print("\ncache written: %s\nmeta: %s" % (npy_path, json_path))

    cache = TeacherLogitCache.load(out_dir, num_views, ckpt_paths, expect)
    _verify(cache, view_ds, teachers, device, bs, n, num_views)


def _verify(cache, view_ds, teachers, device, bs, n, num_views, n_batches=3):
    """Recompute a few batches live and compare against the cache.

    This is the guard that makes the scheme safe: if deterministic view
    rendering ever drifts from what was cached, KD would silently train
    against logits belonging to a different crop. Here that fails loudly.
    """
    print("\nverifying cached logits against live forwards...")
    g = torch.Generator().manual_seed(0)
    worst = 0.0
    for b in range(n_batches):
        base = torch.randperm(n, generator=g)[:bs]
        view = torch.randint(num_views, (bs,), generator=g)
        picks = base * num_views + view
        items = [view_ds[int(j)] for j in picks]
        images = torch.stack([it[0] for it in items]).to(device)
        rows = torch.tensor([it[2] for it in items])
        cached = cache.lookup(rows, device)
        with torch.no_grad():
            for ti, t in enumerate(teachers):
                live = _logits(t(images))
                d = (live - cached[ti]).abs().max().item()
                worst = max(worst, d)
                print("  batch %d teacher %d: max|live-cached| = %.5f" % (b, ti, d))
    tol = 0.05   # float16 storage of logits in the +-20 range
    if worst > tol:
        raise SystemExit(
            "VERIFY FAILED: max deviation %.5f > %.2f. The cached logits do not "
            "match a live forward — do not train on this cache." % (worst, tol))
    print("verify OK (worst %.5f <= %.2f, float16 rounding only)" % (worst, tol))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--views", type=int, default=16)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--data_root", default=None,
                   help="Override data.data_root. Must match what the ladder "
                        "runs with: the cache is indexed by training-set "
                        "position, so a different extraction would silently "
                        "pair each image with another image's teacher logits.")
    p.add_argument("--verify", action="store_true",
                   help="check an existing cache instead of building one")
    a = p.parse_args()
    build(a.config, a.views, a.batch_size, a.num_workers, a.verify,
          a.data_root)


if __name__ == "__main__":
    main()
