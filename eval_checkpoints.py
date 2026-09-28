"""
eval_checkpoints.py
===================
Evaluate saved checkpoints on the val and test splits without training.

Use it to check what a pretraining run actually produced — especially after an
interrupted run, where the checkpoint on disk holds the best-val state at the
moment of interruption, which is not necessarily the last epoch you saw.

    python eval_checkpoints.py --config configs/mobilenetv2_tinyimagenet_2bit_ladder.yaml
    python eval_checkpoints.py --config <cfg> --only M3.pth
"""

from __future__ import annotations

import argparse
import logging
import os

import torch
import yaml

import pipeline as pl
from pipeline_imagenet_patch import build_loaders_from_cfg, load_model_imagenet

logging.basicConfig(level=logging.WARNING)

# checkpoint name -> (arch, role)
KNOWN = [
    ("M1.pth",                         "resnet50",     "teacher M1"),
    ("M2.pth",                         "resnet34",     "teacher M2"),
    ("M3.pth",                         "resnet18",     "teacher M3"),
    ("mobilenetv2_tin_pretrained.pth", "mobilenet_v2", "student init"),
    ("M_fp32.pth",                     "mobilenet_v2", "student FP32 (KD)"),
]

REFERENCES = {
    "resnet18":     "65.59 (SQAKD) / 66.87 (DAQAKD)",
    "mobilenet_v2": "58.07 (SQAKD) / 58.64 (DAQAKD)",
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--only", default=None, help="evaluate just this checkpoint name")
    p.add_argument("--num_workers", type=int, default=0)
    args = p.parse_args()

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    cfg["data"]["num_workers"] = args.num_workers
    out_dir = cfg["experiment"]["output_dir"]
    num_classes = cfg["experiment"].get("num_classes", 200)
    image_size = cfg["data"].get("image_size", 224)
    small_input = cfg["experiment"].get("dataset") == "tinyimagenet" and image_size <= 64

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}"
          f"{' (' + torch.cuda.get_device_name(0) + ')' if device.type == 'cuda' else ''}")

    _, val_loader, test_loader = build_loaders_from_cfg(cfg)

    targets = [t for t in KNOWN if args.only is None or t[0] == args.only]
    print(f"\n{'checkpoint':32s} {'arch':13s} {'role':18s} "
          f"{'val@1':>7s} {'TEST@1':>7s} {'TEST@5':>7s}")
    print("-" * 96)

    for ckpt, arch, role in targets:
        path = os.path.join(out_dir, ckpt)
        if not os.path.isfile(path):
            print(f"{ckpt:32s} {arch:13s} {role:18s} {'—  missing':>23s}")
            continue
        model = load_model_imagenet(arch, num_classes=num_classes, pretrained=False,
                                    checkpoint_path=path, device=device,
                                    small_input=small_input)
        model.eval()
        v1, _ = pl.evaluate_topk(model, val_loader, device)
        t1, t5 = pl.evaluate_topk(model, test_loader, device)
        print(f"{ckpt:32s} {arch:13s} {role:18s} {v1:7.2f} {t1:7.2f} {t5:7.2f}")
        del model
        torch.cuda.empty_cache()

    print("-" * 96)
    print("Published full-precision references on the official 10k val split:")
    for arch, ref in REFERENCES.items():
        print(f"  {arch:13s} {ref}")


if __name__ == "__main__":
    main()
