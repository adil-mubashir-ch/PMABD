#!/usr/bin/env python
"""
Run the CIFAR ladder runner with already-trained rungs restored QUANTIZED.

    python cifar_ladder.py --config configs/resnet32_cifar100_2bit_ladder.yaml

This is the ResNet-18 ladder's own runner (--ladder-code, default
W:\\...\\RESNET18\\run_experiment.py, with its own pipeline.py) with exactly
one behaviour changed: what happens when a stage's checkpoint already exists.

WHY
    The runner loads a finished stage via extract_fp32_weights() into a plain
    model, i.e. it strips the quantizers and evaluates the weights as FP32.
    Measured on ResNet-32's W4A4 rung: 69.15% that way against its real 71.22%.
    Two things go wrong from that one line:
      * the "Loaded accuracy" printed and written to results.yaml is not the
        rung's accuracy at its precision; and
      * that de-quantized model is what joins the teacher pool for the next
        rungs, whereas a ladder trained in one go (ResNet-18) hands on the
        quantized model run_distillation_stage returns.
    Extending the ResNet-32 ladder from its saved W4A4 rung is exactly the
    loaded-from-checkpoint case, so both would bias the new W4A3..W2A2 rungs.

WHAT THIS DOES INSTEAD
    For a finished stage below 32 bits it rebuilds the quantized model the same
    way run_distillation_stage built it (replace_with_fake_quantization at the
    stage's bit-widths), loads the checkpoint INCLUDING the learned LSQ scales
    and their `initialized` flags, and evaluates and returns that. Stages that
    still need training are untouched and go through the runner unchanged.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys

LADDER_CODE = r"W:\01_MS-Thesis\1_WACV_paper_review\RESNET18"
_NOISE = (".act_noise", ".weight_noise")


def _set_unsigned(lp, model, loader, device, batches: int = 5,
                  fresh_only: bool = False):
    """Switch activation quantizers whose input is never negative to the
    unsigned range [0, 2^b-1]. Returns (switched, candidates).

    qmin/qmax are plain attributes, NOT in the state dict, so a checkpoint
    trained with --unsigned-act restores as signed unless this is re-applied:
    its learned scales would then be read against the wrong range. Detection
    runs in eval mode so a restored teacher's BatchNorm statistics are left
    exactly as trained.
    """
    import torch

    qlayers = (lp.QuantizedConv2d, lp.QuantizedLinear)
    cand = [(n, m.act_q) for n, m in model.named_modules()
            if isinstance(m, qlayers) and isinstance(m.act_q, lp.LSQQuantizer)
            and m.act_q.bitwidth < 32
            and not (fresh_only and m.act_q.initialized)]
    mins, hooks = {}, []

    def make_hook(name):
        def hook(module, args):
            v = float(args[0].detach().min())
            mins[name] = min(mins.get(name, v), v)
        return hook

    for name, aq in cand:
        hooks.append(aq.register_forward_pre_hook(make_hook(name)))
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            for i, (x, _) in enumerate(loader):
                model(x.to(device))
                if i + 1 >= batches:
                    break
    finally:
        model.train(was_training)
        for h in hooks:
            h.remove()

    switched = []
    for name, aq in cand:
        if mins.get(name, -1.0) >= 0.0:
            aq.qmin = 0
            aq.qmax = 2 ** aq.bitwidth - 1
            switched.append(aq)
    return switched, len(cand)


def _load_runner(ladder_code: str):
    # The ladder's `from pipeline import ...` must resolve to ITS pipeline.py,
    # not this repo's (a different, later version), so its directory goes
    # first on sys.path, and the runner is loaded under a distinct module name
    # so it cannot collide with this repo's own run_experiment.py.
    sys.path.insert(0, os.path.abspath(ladder_code))
    import pipeline as lp  # noqa: E402  (the ladder's pipeline)
    spec = importlib.util.spec_from_file_location(
        "ladder_run_experiment", os.path.join(ladder_code, "run_experiment.py"))
    rx = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rx)
    return rx, lp


def _patch_cached_stages(rx, lp):
    import torch

    original = rx.run_stage

    def run_stage(stage_key, stage_cfg, teacher_models, teacher_accs,
                  teacher_keys, robustness_score_map, train_loader, val_loader,
                  test_loader, device, dataset):
        ckpt = stage_cfg["checkpoint"]
        w = stage_cfg.get("bitwidth_w", 32)
        a = stage_cfg.get("bitwidth_a", 32)
        if not os.path.exists(ckpt) or (w >= 32 and a >= 32):
            return original(stage_key, stage_cfg, teacher_models, teacher_accs,
                            teacher_keys, robustness_score_map, train_loader,
                            val_loader, test_loader, device, dataset)

        print(f"\n[{stage_cfg['name']}] Checkpoint found → restoring QUANTIZED "
              f"W{w}A{a} model from {ckpt}")
        model = lp.replace_with_fake_quantization(
            rx.load_model(stage_cfg["arch"], dataset, pretrained=False), w, a)
        sd = torch.load(ckpt, map_location="cpu")
        missing, unexpected = model.load_state_dict(sd, strict=False)
        # Only the noise diagnostics may differ; anything else means the
        # checkpoint does not describe this quantized architecture.
        bad = [k for k in list(missing) + list(unexpected)
               if not k.endswith(_NOISE)]
        if bad:
            raise RuntimeError(
                f"[{stage_key}] {ckpt} does not match a W{w}A{a} "
                f"{stage_cfg['arch']}: {bad[:5]}")
        if not all(bool(b) for n, b in model.named_buffers()
                   if n.endswith(".initialized")):
            raise RuntimeError(f"[{stage_key}] {ckpt} has uncalibrated "
                               "quantizers; it cannot be restored as trained.")
        model.to(device)
        if stage_cfg.get("unsigned_act"):
            switched, n = _set_unsigned(lp, model, train_loader, device)
            print(f"  [unsigned-act] restored {len(switched)}/{n} activation "
                  f"quantizers with the unsigned range they were trained with")
        wrapped = rx.FeatureExtractorModel(model)
        val_acc = rx.evaluate(wrapped, val_loader, device)
        test_acc = rx.evaluate(wrapped, test_loader, device)
        wrapped.remove_hooks()
        print(f"  Loaded quantized accuracy: val={val_acc:.2f}%  "
              f"test={test_acc:.2f}%")
        return model, val_acc, test_acc

    rx.run_stage = run_stage



def _patch_unsigned_act(lp, batches: int = 5):
    """Quantize non-negative activations over [0, max] instead of [-max, max].

    THE BUG THIS FIXES
        pipeline.LSQQuantizer is symmetric signed: qmin = -2^(b-1),
        qmax = 2^(b-1)-1. Every quantized conv in a ResNet is fed by a ReLU,
        so its input is >= 0 and the whole negative half of the range is
        unusable. Measured on the ResNet-18 ladder's own checkpoints: the
        W3A3 rung used 4 of its 8 activation levels, and the W3A2/W2A2 rungs
        used 2 of 4 -- i.e. binary activations. Each rung was effectively one
        activation bit below its label, which is where the ladder collapses
        (W3A3 78.62 -> W3A2 74.37).

    WHAT IT DOES
        A short calibration pass records each activation quantizer's input
        minimum. Quantizers that never see a negative value switch to the
        unsigned range (qmin=0, qmax=2^b-1) -- the standard LSQ treatment of
        ReLU outputs. The first conv, fed the normalised image, keeps the
        signed range. Weight quantizers are untouched: weights are signed.

    The scale initialiser in calibrate_quantizers assumes the symmetric range
    (s = 2*a_max/(qmax-qmin)), so an unsigned quantizer's scale is halved
    afterwards to give s = a_max/qmax.
    """
    import torch

    original = lp.calibrate_quantizers

    def calibrate_quantizers(model, train_loader, device, num_batches=50):
        unsigned, n = _set_unsigned(lp, model, train_loader, device, batches,
                                    fresh_only=True)
        print(f"  [unsigned-act] {len(unsigned)}/{n} activation "
              f"quantizers switched to [0, 2^b-1]; the rest see negative "
              f"inputs and stay signed.")

        model = original(model, train_loader, device, num_batches=num_batches)
        with torch.no_grad():
            for aq in unsigned:      # undo the symmetric-range factor of 2
                aq.scale.data.mul_(0.5)
        return model

    lp.calibrate_quantizers = calibrate_quantizers
    # run_distillation_stage imported the name directly at module import time
    rx_mod = sys.modules.get("ladder_run_experiment")
    for mod in (lp, rx_mod):
        if mod is not None and hasattr(mod, "calibrate_quantizers"):
            mod.calibrate_quantizers = calibrate_quantizers


def main():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--ladder-code", default=LADDER_CODE)
    p.add_argument("--unsigned-act", action="store_true",
                   help="Quantize non-negative activations over [0, max]; "
                        "see _patch_unsigned_act.")
    ours, rest = p.parse_known_args()
    rx, lp = _load_runner(ours.ladder_code)
    _patch_cached_stages(rx, lp)
    if ours.unsigned_act:
        _patch_unsigned_act(lp)
    sys.argv = [os.path.join(ours.ladder_code, "run_experiment.py")] + rest
    rx.main()


if __name__ == "__main__":
    main()
