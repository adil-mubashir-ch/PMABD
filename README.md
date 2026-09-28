# PMABD — Progressive Multi-teacher Adaptive Bit Distillation

PMABD trains a low-bit network by walking it **down a precision ladder**
(FP32 → W8A8 → W8A4 → W4A4 → … → W2A2). At every rung the student is
quantization-aware trained (LSQ fake quantization) while being distilled from a
**pool of teachers**. That pool is a frozen FP32 teacher plus every
higher-precision rung the ladder has already finished. Each finished rung then
warm-starts the next one.

This repository holds the training code, the experiment configs and the code
used to re-run competing methods under the same conditions. It contains no
datasets, checkpoints or logs. Those are produced locally.

---

## How it works

```
             ┌──────────── teacher pool (frozen) ─────────────┐
             │  M1  FP32 teacher (larger arch, e.g. ResNet56) │
             │  M2  FP32 student-arch baseline                │
             │  M3  W8A8 rung, M4 W4A4 rung, …  (added as     │
             │      each rung finishes)                       │
             └───────────────────────┬────────────────────────┘
                                     │ weighted KD + feature KD + RKD
                                     ▼
   M2 (FP32) ──init──▶ W8A8 ──init──▶ W8A4 ──init──▶ W4A4 ──init──▶ … ──▶ W2A2
```

Each rung minimises

```
loss = (1-α)·CE(student, labels)
     +   α  ·Σ_t w_t · KL(student ‖ teacher_t, temperature T)
     +   β  ·feature distillation (projected intermediate features)
     + RKD relational term (use_rkd) + λ_qat · quantization regulariser
```

- **Teacher weighting** (`weighting_strategy`): `accuracy` (weights teachers
  by their val accuracy), `entropy` (per-sample, confident teachers weigh
  more) or `robustness` (weights read from a precomputed score file).
- **Feature distillation** (`feature_strategy: projected`): a learned 1×1
  projector aligns student and teacher feature maps at each stage
  (`feat_channels`).
- **Quantization**: LSQ learned-step-size quantizers on weights and
  activations (`pipeline.py`: `LSQQuantizer`, `replace_with_fake_quantization`).
  The first conv and last linear layer are held at **W8A8**.
- **Saturation training** (`saturating: true`): a rung is trained in repeated
  cycles at the same precision. Each cycle warm-starts from the previous best,
  keeps the learned quantizer scales and decays the LR by `cycle_lr_decay`.
  Cycles stop once val accuracy improves by less than `cycle_min_delta`, or
  after `max_cycles`.
- **Early stopping**: every stage/cycle stops after `patience` epochs without a
  val improvement greater than `min_delta`.

### Data discipline

Three disjoint splits. The reported number never drives a training decision:

| split | CIFAR-10 / 100 | TinyImageNet | ImageNet | role |
|---|---|---|---|---|
| `train` | 90% of official train | 90% of official train | 98% of official train | optimisation |
| `val` | 10% of official train (stratified) | 10% of official train | 2% of official train | checkpoint selection, early stopping, saturation |
| `test` | official 10k test | official 10k val | official 50k val | reported only |

The val carve is class-stratified and seeded (`data.val_seed`), so it is
identical across rungs and reruns. Student stages may **not** start from
hub-pretrained CIFAR weights (`pretrained_init`). Those weights saw the whole
training set, including our val split, and the runner raises an error if a
config tries. The frozen teacher may use them, because it is never selected on
val.

---

## Repository layout

```
.
├── pipeline.py                  core engine: quantizers, KD losses, teacher weighting,
│                                training loops (single-pass and saturating), CIFAR loaders
├── run_experiment.py            CIFAR-10 / CIFAR-100 runner (ResNet20/32/56 via torch.hub)
├── pipeline_imagenet_patch.py   ImageNet / TinyImageNet loaders, torchvision models, stem adaptation
├── run_experiment_imagenet.py   ImageNet / TinyImageNet ladder runner (MobileNetV2 student)
├── pretrain_tinyimagenet.py     fine-tunes teachers + student init on TinyImageNet (required first)
├── teacher_cache.py             optional precomputed teacher-logit cache (~3× faster ladder)
├── pmabd_logging.py             run log, epochs.csv/jsonl, stages.json, run metadata
├── eval_checkpoints.py          evaluate saved checkpoints on val/test without training
├── build_benchmark_sheet.py     CSV/XLSX table: published numbers vs. this run's stages.json
├── cifar_ladder.py              CIFAR ladder runner that restores finished rungs *quantized*
├── run_cifar_suite.py           one-command CIFAR-100 programme (ladder + baselines)
├── run_w2a2_suite.py            one-command TinyImageNet W2A2 + QAT-only programme
├── run_teacher_pool_ablation.py teacher-pool / warm-start ablation (ResNet-18, CIFAR-100)
├── configs/                     one YAML per experiment (see below)
├── baselines/                   re-implementations of competing methods on our splits
│   ├── run_baseline.py          one method × one precision
│   ├── run_matrix.py            the full method × precision matrix, resumable
│   ├── aggregate.py             accuracy + GPU-hour table across all runs
│   ├── smoke_test.py            few-step correctness check for every method
│   ├── common/  methods/        quantizers, schedules, engine; sqakd, cmtkd, switchable, qat_only
│   └── README.md                per-method provenance and declared deviations
└── notebooks/                   Colab driver for the CIFAR-100 baselines
```

### Configs

| config | runner | what it trains |
|---|---|---|
| `resnet20_cifar10.yaml` | `run_experiment.py` | ResNet20 on CIFAR-10, FP32 → W4A4, saturating 4-bit rungs, ResNet56 teacher |
| `resnet32_cifar100_2bit_ladder.yaml`, `resnet32_cifar100_2bit_fixedlr.yaml` | `cifar_ladder.py` | ResNet32 CIFAR-100 ladder extended below W4A4 to W2A2 |
| `resnet18_cifar100_2bit_fixedact.yaml`, `resnet18_cifar100_ablation_*.yaml` | `cifar_ladder.py` | ResNet18 CIFAR-100 W2A2 and teacher-pool ablations |
| `mobilenetv2_tinyimagenet_2bit_ladder{,_sat}.yaml` | `run_experiment_imagenet.py` | MobileNetV2 on TinyImageNet @224, FP32 → W2A2, ResNet50/34/18 teachers, without / with saturation |
| `mobilenetv2_imagenet_2bit_ladder{,_sat}.yaml` | `run_experiment_imagenet.py` | Same ladder on ImageNet-1k |
| `*_baselines.yaml` | `baselines/run_matrix.py` | Competing methods on the matching dataset/arch |

> Some CIFAR configs and `cifar_ladder.py` / `run_cifar_suite.py` /
> `run_teacher_pool_ablation.py` contain absolute paths (`W:/…`) to
> checkpoints and to an external ResNet-18 ladder runner (`--ladder-code`).
> Point those at your own `outputs/` and runner before using them.

---

## 1. Environment

Python 3.10 or 3.11 and a CUDA GPU.

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

```bash
pip install pyyaml numpy tqdm pandas openpyxl datasets
```

`datasets` is only needed for the TinyImageNet HuggingFace fallback download,
and `openpyxl` only for `build_benchmark_sheet.py`.

## 2. Data

Everything lives under `./data` (`data.data_root` in each config):

```
data/
├── cifar-10-batches-py/        torchvision CIFAR-10 (downloaded automatically)
├── cifar-100-python/           torchvision CIFAR-100 (downloaded automatically)
├── tiny-imagenet-200/          http://cs231n.stanford.edu/tiny-imagenet-200.zip
│   ├── train/<wnid>/images/*.JPEG
│   ├── val/images/*.JPEG  val/val_annotations.txt  wnids.txt
└── imagenet/train/<wnid>/*.JPEG  imagenet/val/<wnid>/*.JPEG   (registration required)
```

TinyImageNet is downloaded automatically if missing. `val_organised/` is built
on first use.

## 3. Run: CIFAR (ResNet)

```bash
python run_experiment.py --config configs/resnet20_cifar10.yaml
```

Stages run in order: `stage_m2` (FP32 from scratch, SGD + cosine) →
`stage_m3` (W8A8) → `stage_m4a` (W8A4) → `stage_m4b` (W4A4). The teacher (M1)
is downloaded from `chenyaofo/pytorch-cifar-models` on first run and cached as
`M1.pth`.

| flag | effect |
|---|---|
| `--start-stage stage_m4a` | skip earlier stages (their checkpoints must exist) |
| `--fold k` | K-fold run when `data.n_folds > 1` |
| `--binary` | W1A1 with the binary quantizer |

**Resuming.** A stage whose `checkpoint` file already exists is loaded and
skipped. A saturating stage writes its checkpoint after **every cycle**. If a
saturating run is interrupted mid-stage, move that stage's `.pth` aside before
re-running, or the stage is treated as finished after the last completed cycle.

## 4. Run: TinyImageNet / ImageNet (MobileNetV2)

```bash
python pretrain_tinyimagenet.py --config configs/mobilenetv2_tinyimagenet_2bit_ladder.yaml
```

```bash
python run_experiment_imagenet.py --config configs/mobilenetv2_tinyimagenet_2bit_ladder.yaml
```

The pretrain step is **required for TinyImageNet**. torchvision ships
1000-class heads only, so it fine-tunes the ResNet50/34/18 teachers and the
MobileNetV2 student init on the same splits. The ladder refuses to start if a
teacher scores below 5% top-1. ImageNet needs no pretrain step.

Ladder stages: `m3_kd` (FP32 KD warm-up) → `m4` W8A8 → `m5a` W8A4 → `m5b` W4A4
→ `m6a` W4A3 → `m6b` W3A3 → `m7a` W3A2 → `m7b` W2A2. Use the `_sat` config for
saturation training.

| flag | effect |
|---|---|
| `--start_stage m5b` / `--only_stage m4` | resume from / run exactly one stage |
| `--epoch_scale 0.25` | scale every epoch budget (quick end-to-end shakedown) |
| `--batch_size 64` / `--num_workers 8` | resources |
| `--image_size 128` | ~3× cheaper, but no longer comparable to published @224 numbers |
| `--teacher_cache_views K` | read base-teacher logits from the cache below |
| `--dry_run` | parse config and exit |

Re-running the same command resumes. Any stage with a checkpoint on disk is
skipped and joins the teacher pool.

**Optional teacher-logit cache.** The three frozen ResNet teachers dominate
step time. `teacher_cache.py` fixes augmentation to K deterministic views per
image and precomputes their logits once:

```bash
python teacher_cache.py --config configs/mobilenetv2_tinyimagenet_2bit_ladder.yaml --views 16
```

```bash
python run_experiment_imagenet.py --config configs/mobilenetv2_tinyimagenet_2bit_ladder.yaml --teacher_cache_views 16
```

This caps augmentation diversity at K views per image, which changes the
training setup, so report it if you use it. The cache stores checkpoint hashes
and refuses to load if a teacher changed (`--verify` re-checks it). It stores
logits only, so it cannot be combined with feature distillation (`beta > 0`).

## 5. Run: baselines

Competing methods retrained from **our** FP32 model on **our** splits, so they
share a table with PMABD. Provenance, fidelity and deviations for each method
are in [`baselines/README.md`](baselines/README.md).

```bash
python -m baselines.smoke_test
```

```bash
python -m baselines.run_matrix --schedule matched --execute
```

```bash
python -m baselines.aggregate --csv results/baseline_comparison.csv
```

`--schedule matched` gives each method the epoch budget PMABD spent at that
precision, and `--schedule paper` uses each method's published recipe. The
upstream repositories the ports were taken from are cloned into
`baselines/third_party/`, which is not tracked in git.

## 6. Outputs

Everything goes to `experiment.output_dir` (default `./outputs/<name>/`):

- `M*.pth`: one checkpoint per rung (quantized rungs include LSQ scales)
- `run_<timestamp>.log`: full console output
- `logs/epochs.csv`, `logs/epochs.jsonl`: one row per epoch, every stage
  (loss terms, LR, grad norm, KD weight, top-1/top-5 on train/val/test, epoch time)
- `logs/stages.json`: final per-stage results, used by `aggregate.py` and
  `build_benchmark_sheet.py`
- `logs/run_meta_<timestamp>.json`: config snapshot, git commit, GPU, argv

Evaluate checkpoints without training:

```bash
python eval_checkpoints.py --config configs/mobilenetv2_tinyimagenet_2bit_ladder.yaml
```

---

## Benchmark notes (TinyImageNet)

The TinyImageNet setup follows SQAKD (Zhao & Zhao, AISTATS 2024) and DAQAKD
(Kur & Zhao, arXiv:2509.03850). Both upsample images to 224×224 and report on
the official 10k val split. Two differences matter when comparing numbers:

- Their FP32 MobileNetV2 is trained from random init. Ours is fine-tuned from
  ImageNet weights. Published quantized numbers are therefore not directly
  comparable to ours, and `baselines/` re-runs their methods from our FP32
  model.
- They keep the first conv and last FC at full precision. PMABD holds them at
  W8A8.

## Troubleshooting

| symptom | fix |
|---|---|
| `top-1 accuracy is X%, below the 5.00% floor` | a teacher head was never trained; run `pretrain_tinyimagenet.py` |
| `num_classes=200 != 1000` | same as above |
| `train and val class lists differ` | re-extract `tiny-imagenet-200/` from a single source |
| `CUDA out of memory` | lower `data.batch_size` (64 or 48), or `--image_size 128` |
| stage crashed mid-run | re-run the same command; finished stages are skipped (see the saturation caveat in §3) |
