#!/usr/bin/env python
"""
Teacher-pool / warm-start ablation for ResNet-18 W2A2 on CIFAR-100.

    python run_teacher_pool_ablation.py
    python run_teacher_pool_ablation.py --plan

Trains the same W2A2 student three times under the fixedact regime
(50 epochs, unsigned activations, identical optim/KD hyperparams), varying
only which already-trained teachers are in the pool and which checkpoint
warm-starts the student:

  A  FP32 only          teachers: M1            warm-start: M2 (same-arch FP32)
  B  FP32 + W8A8        teachers: M1, M3        warm-start: M3
  C  FP32 + W8A8 + W4A4 teachers: M1, M3, M4    warm-start: M4

The published PMABD cell (M1+M3+M4+M5, warm-start M6a, test 77.70%) is NOT
retrained; it is read from outputs/resnet18_cifar100_fixedact/ and printed
as the full-pool reference.

WHY THESE TEACHERS
    The CIFAR runner's W2A2 stage always requests M1+M3+M4+M5 and drops any
    name that was never loaded. Omitting stage_m3 / m4b / m5b from the config
    is therefore enough to shrink the pool - no runner fork required.
    Warm-start uses the lowest-precision teacher in the pool; when that is
    M1 alone (cross-arch ResNet-56), we fall back to M2 (ResNet-18 FP32).

RESUMABILITY
    Each cell writes to its own output_dir. A finished cell (M6.pth +
    results.yaml m6b_acc) is skipped. Re-run the same command after a crash.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime

import yaml

REPO = os.path.dirname(os.path.abspath(__file__))
LADDER_CODE = r"W:\01_MS-Thesis\1_WACV_paper_review\RESNET18"
LADDER_CKPT = (r"W:\01_MS-Thesis\1_WACV_paper_review\RESNET18\outputs"
               r"\resnet18_cifar100_2bit_ladder")
FIXEDACT_OUT = os.path.join(REPO, "outputs", "resnet18_cifar100_fixedact")
LOG_DIR = os.path.join(REPO, "logs", "teacher_pool_ablation")

# (id, config relative to REPO, human label, expected teachers, warm-start file)
CELLS = (
    ("fp32",
     os.path.join("configs", "resnet18_cifar100_ablation_fp32.yaml"),
     "A  FP32 only -> W2A2",
     ("M1",),
     "M2.pth"),
    ("fp32_w8",
     os.path.join("configs", "resnet18_cifar100_ablation_fp32_w8.yaml"),
     "B  FP32 + W8A8 -> W2A2",
     ("M1", "M3"),
     "M3.pth"),
    ("fp32_w8_w4",
     os.path.join("configs", "resnet18_cifar100_ablation_fp32_w8_w4.yaml"),
     "C  FP32 + W8A8 + W4A4 -> W2A2",
     ("M1", "M3", "M4"),
     "M4.pth"),
)

SEED_CKPTS = ("M1.pth", "M2.pth", "M3.pth", "M4.pth", "M4a.pth")


# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────
class Tee:
    def __init__(self, path):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.fh = open(path, "a", encoding="utf-8", buffering=1)

    def write(self, text):
        try:
            sys.__stdout__.write(text)
        except UnicodeEncodeError:
            sys.__stdout__.write(text.encode("ascii", "replace").decode("ascii"))
        sys.__stdout__.flush()
        self.fh.write(text)

    def close(self):
        self.fh.close()


LOG = None


def say(msg=""):
    line = str(msg) + "\n"
    if LOG is not None:
        LOG.write(line)
    else:
        sys.__stdout__.write(line)


def banner(title):
    say()
    say("=" * 78)
    say(title)
    say("=" * 78)


def run(cmd, step_log, label):
    say()
    say("-" * 78)
    say(f"[{label}] {' '.join(cmd)}")
    say(f"[{label}] step log: {os.path.relpath(step_log, REPO)}")
    say("-" * 78)

    os.makedirs(os.path.dirname(step_log), exist_ok=True)
    t0 = time.time()
    env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
    with open(step_log, "a", encoding="utf-8", buffering=1) as fh:
        fh.write(f"\n### {datetime.now().isoformat()}  {' '.join(cmd)}\n")
        proc = subprocess.Popen(
            cmd, cwd=REPO, env=env, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, encoding="utf-8",
            errors="replace", bufsize=1)
        for line in proc.stdout:
            sys.__stdout__.write(line)
            sys.__stdout__.flush()
            fh.write(line)
            if LOG is not None:
                LOG.fh.write(line)
        rc = proc.wait()
    dt = (time.time() - t0) / 3600.0
    say(f"[{label}] exit={rc}  wall={dt:.2f} h")
    return rc, dt


def read_yaml(path):
    try:
        with open(path, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except (OSError, ValueError):
        return {}


# ─────────────────────────────────────────────────────────────────────────────
# Preflight / cells
# ─────────────────────────────────────────────────────────────────────────────
def cell_out_dir(cfg_path):
    cfg = read_yaml(os.path.join(REPO, cfg_path))
    out = cfg.get("experiment", {}).get("output_dir", "")
    if not out:
        return None
    return out if os.path.isabs(out) else os.path.join(REPO, out)


def cell_done(cfg_path):
    out = cell_out_dir(cfg_path)
    if not out:
        return False
    res = read_yaml(os.path.join(out, "results.yaml"))
    return ("m6b_acc" in res
            and os.path.isfile(os.path.join(out, "M6.pth")))


def preflight(args):
    banner("PREFLIGHT")
    ok = True
    say(f"  repo      : {REPO}")
    say(f"  python    : {sys.executable}")
    say(f"  regime    : 50 epochs, unsigned-act, fixedact W2A2 hyperparams")
    try:
        import torch
        cuda = torch.cuda.is_available()
        say(f"  torch     : {torch.__version__}  cuda={cuda}")
        if cuda:
            say(f"  gpu       : {torch.cuda.get_device_name(0)}")
        elif not args.plan:
            say("  gpu       : NONE - refusing to start on CPU.")
            ok = False
    except ImportError as e:
        say(f"  torch     : NOT IMPORTABLE ({e})")
        ok = False

    cifar = os.path.join(REPO, "data", "cifar-100-python")
    have = all(os.path.isfile(os.path.join(cifar, n))
               for n in ("train", "test", "meta"))
    say(f"  data      : {'ok     ' if have else 'MISSING'} {cifar}")
    ok = ok and have

    for n in ("run_experiment.py", "pipeline.py"):
        p = os.path.join(args.ladder_code, n)
        present = os.path.isfile(p)
        say(f"  ladder    : {'ok     ' if present else 'MISSING'} {p}")
        ok = ok and present

    for n in SEED_CKPTS:
        p = os.path.join(LADDER_CKPT, n)
        present = os.path.isfile(p)
        say(f"  teacher   : {'ok     ' if present else 'MISSING'} {n}")
        ok = ok and present

    ref = os.path.join(FIXEDACT_OUT, "results.yaml")
    say(f"  reference : {'ok     ' if os.path.isfile(ref) else 'MISSING'} "
        f"fixedact results (full pool, not retrained)")

    for cid, cfg, label, teachers, warm in CELLS:
        path = os.path.join(REPO, cfg)
        present = os.path.isfile(path)
        status = "DONE   " if present and cell_done(cfg) else (
            "ok     " if present else "MISSING")
        say(f"  cell {cid:<10}: {status}  teachers={'+'.join(teachers)}  "
            f"warm={warm}  - {label}")
        ok = ok and present
    return ok


def run_cell(args, cid, cfg, label, teachers, warm, manifest):
    banner(f"CELL {cid} - {label}")
    say(f"  teachers   : {' + '.join(teachers)}")
    say(f"  warm-start : {warm}")
    say(f"  config     : {cfg}")

    if cell_done(cfg):
        out = cell_out_dir(cfg)
        res = read_yaml(os.path.join(out, "results.yaml"))
        say(f"  already finished - test={res.get('m6b_acc')}%  "
            f"val={res.get('m6b_val_acc')}%  (skipping)")
        manifest["cells"][cid] = {
            "skipped": True,
            "test": res.get("m6b_acc"),
            "val": res.get("m6b_val_acc"),
            "teachers": list(teachers),
            "warm_start": warm,
            "output_dir": out,
        }
        return True

    cmd = [sys.executable, "cifar_ladder.py",
           "--ladder-code", args.ladder_code,
           "--unsigned-act",
           "--config", cfg]
    if args.plan:
        say(f"  would run: {' '.join(cmd)}")
        manifest["cells"][cid] = {
            "planned": True,
            "teachers": list(teachers),
            "warm_start": warm,
            "cmd": cmd,
        }
        return True

    rc, hours = run(cmd, os.path.join(LOG_DIR, f"{cid}.log"), f"abl-{cid}")
    out = cell_out_dir(cfg)
    res = read_yaml(os.path.join(out, "results.yaml")) if out else {}
    manifest["cells"][cid] = {
        "rc": rc,
        "wall_hours": round(hours, 3),
        "test": res.get("m6b_acc"),
        "val": res.get("m6b_val_acc"),
        "teachers": list(teachers),
        "warm_start": warm,
        "output_dir": out,
        "config": cfg,
    }
    return rc == 0


def report(manifest):
    banner("ABLATION SUMMARY - ResNet-18 CIFAR-100 W2A2")
    say(f"  {'cell':<12} {'teachers':<18} {'warm':<8} {'val':>7} {'test':>7}")
    say(f"  {'-'*12} {'-'*18} {'-'*8} {'-'*7} {'-'*7}")

    for cid, _, label, teachers, warm in CELLS:
        info = manifest.get("cells", {}).get(cid, {})
        val = info.get("val")
        test = info.get("test")
        val_s = f"{val:6.2f}%" if isinstance(val, (int, float)) else "   -  "
        test_s = f"{test:6.2f}%" if isinstance(test, (int, float)) else "   -  "
        say(f"  {cid:<12} {'+'.join(teachers):<18} {warm:<8} {val_s} {test_s}")

    ref = read_yaml(os.path.join(FIXEDACT_OUT, "results.yaml"))
    if "m6b_acc" in ref:
        say(f"  {'full_pmabd':<12} {'M1+M3+M4+M5':<18} {'M6a.pth':<8} "
            f"{ref.get('m6b_val_acc', 0):6.2f}% {ref['m6b_acc']:6.2f}%")
        say("  (full_pmabd = existing unsigned-act fixedact run, 50 epochs; "
            "not retrained here)")
    else:
        say("  full_pmabd  (missing fixedact results.yaml)")

    path = os.path.join(LOG_DIR, "manifest.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    say(f"\n  manifest -> {path}")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--plan", action="store_true",
                   help="Print what would run; train nothing.")
    p.add_argument("--ladder-code", default=LADDER_CODE,
                   help="Directory with the CIFAR run_experiment.py + pipeline.py")
    p.add_argument("--only", default="fp32,fp32_w8,fp32_w8_w4",
                   help="Comma-separated cell ids to run "
                        "(default: all three).")
    return p.parse_args()


def main():
    global LOG
    args = parse_args()
    os.makedirs(LOG_DIR, exist_ok=True)
    LOG = Tee(os.path.join(LOG_DIR, "suite.log"))

    banner(f"TEACHER-POOL ABLATION   {datetime.now():%Y-%m-%d %H:%M:%S}")
    say("ResNet-18 / CIFAR-100 / W2A2 - varying intermediate teachers + "
        "warm-start")
    say("Reference regime: fixedact 50-epoch unsigned-act W2A2 (test 77.70%)")

    only = {c.strip() for c in args.only.split(",") if c.strip()}
    unknown = only - {c[0] for c in CELLS}
    if unknown:
        say(f"Unknown --only cells: {sorted(unknown)}")
        return 2

    if not preflight(args):
        say("\nPreflight failed.")
        return 1

    manifest = {
        "started": datetime.now().isoformat(),
        "plan": bool(args.plan),
        "regime": {
            "epochs": 50,
            "unsigned_act": True,
            "matched_to": "outputs/resnet18_cifar100_fixedact (W2A2)",
        },
        "cells": {},
    }

    ok = True
    for cid, cfg, label, teachers, warm in CELLS:
        if cid not in only:
            continue
        ok = run_cell(args, cid, cfg, label, teachers, warm, manifest) and ok

    ref = read_yaml(os.path.join(FIXEDACT_OUT, "results.yaml"))
    if "m6b_acc" in ref:
        manifest["full_pmabd_reference"] = {
            "teachers": ["M1", "M3", "M4", "M5"],
            "warm_start": "M6a.pth",
            "test": ref["m6b_acc"],
            "val": ref.get("m6b_val_acc"),
            "output_dir": FIXEDACT_OUT,
            "note": "Existing run; not retrained by this suite.",
        }

    report(manifest)
    LOG.close()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
