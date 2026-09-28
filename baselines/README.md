# Baseline reruns — same FP32, same splits, same GPU

Five competing methods retrained from **our** FP32 MobileNetV2 on **our**
TinyImageNet splits, so their numbers can be put in the same table as PMABD's.

---

## Why this exists

Our published-number comparison was invalid, and not in a small way.

SQAKD trains its TinyImageNet MobileNetV2 teacher **from random initialisation**
— their `run_tinyimagenet_mobilenetV2.sh`, branch `mobilenet_v2_fp`, passes no
`--pretrained` flag — for 100 epochs, reaching 58.07 top-1. Our backbone is
fine-tuned from ImageNet-pretrained weights and reaches 74.44. That single
difference, initialisation rather than protocol, accounts for essentially the
entire 16-point gap between our table and theirs.

The consequence is uncomfortable and needs stating plainly:

| method | FP32 | W4A4 | W3A3 | drop at W3A3 |
|---|---|---|---|---|
| **PMABD** | 75.57 | 68.78 | 62.65 | **−12.92** |
| DAQAKD (published) | 58.64 | 59.48 | 56.17 | −2.47 |
| SQAKD (published) | 58.07 | 57.14 | 52.73 | −5.34 |
| PACT/DoReFa (published) | 58.07 | 50.33 | 47.77 | −10.30 |

We win every absolute cell and lose the relative column to everything,
including plain PACT. A reviewer reads that as "strong pretraining, weak
quantization". These reruns are the only way to find out which reading is
right — and they may not settle it in our favour.

---

## What is being run, and how faithful each row is

| method | paper | code | fidelity |
|---|---|---|---|
| `sqakd` | AISTATS 2024 | [kaiqi123/SQAKD](https://github.com/kaiqi123/SQAKD) | **Ported.** Quantizer transcribed from their `CIFAR/models/custom_modules.py`; loss weights read off their TinyImageNet run names (`gamma0_alpha100`). |
| `daqakd` | arXiv 2509.03850 | none | **Partial reimplementation.** Objective is SQAKD's; the DA *selection metric* is not reproduced. See below. |
| `any_precision` | AAAI 2021 | [SHI-Labs/Any-Precision-DNNs](https://github.com/SHI-Labs/Any-Precision-DNNs) | **Ported.** Recursive supervision and SwitchBatchNorm from their `train.py` / `models/quan_ops.py`. |
| `instantnet_cdt` | DAC 2021 | [GATECH-EIC/InstantNet](https://github.com/GATECH-EIC/InstantNet) | **Ported.** Cascade Distillation Training from their `train.py` (`bit_schedule='avg_loss'`, `cascad=True`). NAS half not used, matching their own CDT ablation. |
| `cmtkd` | WACV 2023 | none | **Full reimplementation** from the paper. |

Clones live in `third_party/` (gitignored) as provenance. They are **not**
imported wholesale: SQAKD's TinyImageNet path needs NVIDIA DALI (Linux-only)
and a torch-1.10-era MQBench, and we run Windows + torch 2.8. The *methods* are
ported; only the plumbing is ours.

### InstantNet is the one that matters for novelty

CDT makes every bit-width distil from **all higher** bit-widths — structurally
the same idea as our ladder, where each finished rung joins the teacher pool.

Any-Precision does *not* cover the same ground, despite looking similar. Read
the two training loops side by side:

- **Any-Precision** — each precision is taught by the *one* precision above it.
  `target_soft` is reassigned as the loop descends, so it is a **chain**.
- **InstantNet CDT** — each precision is taught by *all* higher precisions
  simultaneously, so it is a **dense pool**.

PMABD is the dense pool. CDT is therefore the only baseline in this set that
shares our actual mechanism; Any-Precision is the sparse-chain cousin. Dropping
CDT would leave the nearest-neighbour claim untested.

**On excluding the NAS.** InstantNet is CDT + SP-NAS; we run CDT only. That is
not a compromise, it is required: every row in this table is a fixed
MobileNetV2, and running the architecture search would compare a searched
architecture against fixed ones. The paper's own ablation evaluates CDT
standalone. Sentence for the paper: *"we evaluate the Cascade Distillation
Training component on a fixed MobileNetV2; the architecture-search component is
excluded because all methods here are compared at fixed architecture, following
the paper's own CDT ablation."*

What actually differs between CDT and PMABD:

- CDT shares one weight tensor across precisions; PMABD trains a separate model
  per rung, so a rung can specialise rather than compromise.
- CDT distils with MSE on raw logits; PMABD uses entropy-weighted KL over a pool
  that also contains other architectures (the ResNet teachers).
- CDT's precisions are symmetric; the ladder passes through W8A4 and W4A3,
  which this family cannot express at all.

None of those is self-evidently decisive. That is what the experiment is for.

### Honesty notes to carry into the paper

- **DAQAKD** is a lower bound, not the method. Its contribution is a policy
  *search* driven by Contextual Mutual Information; what runs here is SQAKD
  under a fixed strong policy (RandAugment n=2, m=9) from the family that
  search draws on. Label the row "SQAKD + strong DA". If it already beats
  PMABD, the lower bound suffices; if it does not, the gap to the real method
  is unknown and must not be claimed as a win.
- **CMT-KD** deviates twice: it uses our EWGS quantizer rather than the paper's
  HWGQ (so the table varies the *distillation strategy*, not the quantizer),
  and its fusion points had to be chosen for MobileNetV2, which the paper never
  evaluates.
- **BatchNorm recalibration (applies to every baseline).** All five start from
  an FP32 checkpoint and inherit its BN running statistics, which the quantized
  network invalidates. The EWGS/SQAKD quantizer maps activations as
  `(x - lA)/(uA - lA)`; when `lA != 0` that is an *affine* map whose constant
  term no single `output_scale` can undo. The authors' ResNets/VGGs feed every
  quantized conv from a ReLU, so `lA = 0` and the map is purely multiplicative.
  MobileNetV2's inverted residuals feed each expand conv from a **linear**
  bottleneck, which is signed — measured `lA = -60.08` at
  `features.2.conv.0.0`, `-13.30` at `features.18.0`. BatchNorm absorbs the
  shift in train mode, so training looks healthy while evaluation collapses to
  chance. Measured at initialisation, before/after recalibration: W8A8
  0.00% → 69.53%, W4A4 0.00% → 51.17%, W3A3 0.00% → 4.56%. We therefore
  re-estimate BN statistics on the quantized network before every evaluation
  (50 batches, cumulative average). Both source repos do the same —
  InstantNet ships `calibrate_bn.py`, Any-Precision has an `update_bn` path —
  so this is standard practice here, not an invention.
- **Gradient clipping — a measured, necessary deviation.** Of the five, only
  InstantNet clips in its own code (`config_train.py: C.grad_clip = 5`);
  `grep -rn clip_grad` finds nothing in the others. But running SQAKD
  faithfully unclipped diverges here: `alpha=100` against gradients of norm
  ~560 gives SGD steps of norm ~0.28 at lr 5e-4, and over 120 iterations at
  W8A8 the loss went **114.98 → 124.13 → 122.77** — up to the degenerate
  plateau where the student emits a constant. The same run clipped at 5.0 went
  **47.14 → 38.77 → 29.45**. We therefore clip every method at 5.0, matching
  InstantNet's value. This is a deviation from four of the five sources and
  must be stated; the alternative (re-tuning `alpha` or the learning rate)
  would be a larger departure from their published recipe.
- **Switchable activation ranges.** Both switchable repos quantize activations
  as `clamp(x, 0, 1)`, valid for their ResNets whose nonlinearity is built to
  land in [0,1]. On stock MobileNetV2 that is wrong twice: ReLU6 spans [0,6],
  and each block's expand conv is fed by a signed linear projection. Run
  verbatim, **every parameter in `features.0`–`features.5` got exactly zero
  gradient in the 3/4/8-bit branches** while the 32-bit branch trained fine.
  We learn the range per layer per bit-width instead — the same treatment the
  SQAKD rows already get, so no method is handicapped by a mismatched range.
- **First/last layers.** Every baseline keeps them FP32 (their own convention).
  The PMABD ladder holds them at W8A8, which is strictly harder. Pass
  `--first-last-bits 8,8` for the apples-to-apples version; run both if the
  budget allows, because this is the one asymmetry that currently favours them.

---

## Running

Prerequisite: the ladder's own checkpoints must exist (`pretrain_tinyimagenet.py`
then the ladder). Use the GPU env — the PATH python is CPU-only torch:

```bash
~/anaconda3/envs/torchgpu/python.exe -m baselines.smoke_test
```

That builds all five, runs three real steps each, and checks gradients actually
reach the parameters. Run it before anything long.

One method, one precision:

```bash
~/anaconda3/envs/torchgpu/python.exe -m baselines.run_baseline --method sqakd --precision w3a3 --schedule matched
```

The whole matrix (prints the plan; `--execute` to run it):

```bash
~/anaconda3/envs/torchgpu/python.exe -m baselines.run_matrix --schedule matched
```

Then the comparison table:

```bash
~/anaconda3/envs/torchgpu/python.exe -m baselines.aggregate --csv results/baseline_comparison.csv
```

### Interruptions

Two levels of resume, both automatic:

- **Between stages** — `run_matrix` skips anything already in `stages.json`.
- **Within a stage** — the engine writes `<stage_id>_resume.pt` into the output
  directory after *every* epoch, and a restarted stage continues from the next
  one. Nothing extra to pass; just re-run the same command.

The snapshot carries everything needed for the resumed run to be equivalent to
an uninterrupted one: all trainable tensors (for CMT-KD that means the three
teachers and the importance factors, not just the student), optimizer state
(else SGD momentum restarts at zero), scheduler state (else the LR jumps back
to its epoch-0 value), the running best-val selection, and **elapsed GPU hours**
— without that last one an interrupted stage would report only the time since
the last resume and silently undercount the compute comparison.

It is written atomically (`.tmp` then `os.replace`) so a power cut during the
write cannot leave a corrupt file, and it is validated against
`(stage_id, precision, epochs, schedule)` before use — a snapshot from a
different configuration is rejected rather than producing a run that is neither
the old one nor the new one. Completing a stage deletes it.

Pass `--no-resume` to ignore snapshots and restart stages from epoch 1.

### Schedules

`--schedule paper` uses each method's own published recipe. `--schedule matched`
uses the epoch budget our ladder actually spent at that precision (W8A8 13,
W4A4 17, W3A3 45). Neither is "the" fair one — run both, and let the logged GPU
hours carry the compute argument.

For switchable methods one run serves W8A8/W4A4/W3A3 together, so `matched`
gives them the max over those rather than the sum; otherwise they would get
three times the gradient steps at each precision.

### Cost

| schedule | runs | rough total on one RTX 3070 Ti |
|---|---|---|
| `matched` | 11 | ~40–55 h |
| `paper` | 11 | ~110–140 h |

`matched` is the one to run first. CMT-KD is the expensive row in either mode:
it trains three teachers alongside the student, so every step is four forward
and backward passes.

---

## Layout

```
baselines/
├── common/
│   ├── setup.py            splits + FP32 init (shared by all, so none can drift)
│   ├── quant_sqakd.py      EWGS learned-range quantizer  [SQAKD, DAQAKD, CMT-KD]
│   ├── quant_switchable.py shared-weight switchable ops   [Any-Precision, CDT]
│   ├── schedules.py        the paper/matched flag + measured ladder budgets
│   └── engine.py           training loop, val-based selection, GPU-hour logging
├── methods/                one file per method family
├── run_baseline.py         single run
├── run_matrix.py           the full matrix
├── aggregate.py            accuracy + GPU-hour comparison table
└── smoke_test.py           fast correctness check
```

Logging goes through the ladder's own `pmabd_logging.RunLogger`, so baseline
runs emit the same `epochs.csv` / `stages.json` schema and the GPU-hour
comparison is measured the same way on both sides.

Checkpoint selection is best top-1 on the held-out val split; the reported
number is the official 10k val split and never influences a decision. This is
stricter than SQAKD's own protocol, which selects on the split it reports —
applying the stricter rule uniformly keeps the comparison even.
