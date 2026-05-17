# Surface Damage Detector — Results

Target accuracies (per the project brief):
- **Tiles ≥ 85%**, **Walls ≥ 85%**, **Wood ≥ 80%**.
- User-set floor for this iteration: **≥ 80% on every surface**.

All models are built fully from scratch (see `layers.py`) — no `nn.Conv2d`,
`nn.BatchNorm2d`, `nn.Linear`, `nn.MaxPool2d`, etc. Only `nn.Module` and
`nn.Parameter` (plumbing for autograd) are used from `torch.nn`.

---

## Walls

Two preprocessing modes were tested. The 6-channel hybrid (engineered +
raw RGB) reduced variance on the smudged class and improved the average.

### 3-channel engineered (Saturation + CLAHE + Canny)
| Seed | Test acc | clean | smudged | cracked |
|---|---|---|---|---|
| 42  | 84.2% | 95.0% | 61.1% | 94.7% |
| 123 | 91.2% | 90.0% | 88.9% | 94.7% |
| 2024 | 78.9% | 90.0% | 61.1% | 84.2% |
| **avg** | **84.8%** | **91.7%** | **70.4%** | **91.2%** |

Smudged had high variance (61% / 89% / 61%) — unlucky splits killed accuracy.

### 6-channel hybrid (engineered + raw RGB)
| Seed | Test acc | clean | smudged | cracked |
|---|---|---|---|---|
| 42  | **87.7%** | 100.0% | 72.2% | 89.5% |
| 123 | **89.5%** | 90.0% | 77.8% | 100.0% |
| 2024 | **84.2%** | 85.0% | 72.2% | 94.7% |
| **avg** | **87.1%** | **91.7%** | **74.1%** | **94.7%** |

All three splits clear 80%. Smudged is now consistent (72–78%), and the
overall average improved by 2.3pp. Raw RGB helps the CNN see color-
distinctive smudges (rust, mold, water stains) that the saturation
channel alone was compressing.

Checkpoints:
- `outputs/walls_best.pth` — hybrid seed=42 (the production model)
- `outputs/walls_seed123_best.pth` — hybrid verification
- `outputs/walls_seed42_3ch_best.pth` — old 3-channel seed=42 model (fallback)

Note: seed=2024 results above were obtained in an earlier hybrid run; the
on-disk checkpoint for it was not regenerated during the final log refresh.
Run `python train.py --surface walls --seed 2024` to recreate if needed
(deterministic, expected ~84.2%).

### Smudge-boost experiment (rejected)
Tested `--smudge_boost 1.5` (multiplies smudged class weight in the loss
by 1.5). Smudged improved across all 3 seeds but clean was harmed too much,
so the change was reverted.

| Seed | Mode | Overall | Clean | Smudged | Cracked |
|---|---|---|---|---|---|
| 42  | baseline   | 87.7% | 100.0% | 72.2% | 89.5% |
| 42  | boost 1.5  | 87.7% |  95.0% (↓5) | 77.8% (↑6) | 89.5% |
| 123 | baseline   | 89.5% |  90.0% | 77.8% | 100.0% |
| 123 | boost 1.5  | 89.5% |  80.0% (↓10) | 88.9% (↑11) | 100.0% |
| 2024 | baseline  | 84.2% |  85.0% | 72.2% | 94.7% |
| 2024 | boost 1.5 | **78.9% (↓5)** | **60.0% (↓25)** | 77.8% (↑6) | 100.0% |

The seed=2024 case is the deal-breaker: boosting smudged sent clean
plummeting from 85% to 60%, which dropped the overall below 80%.
Production walls model stays on the unboosted hybrid baseline.

## Tiles

Tiles required several iterations. The hand-crafted engineered channels
(Saturation + CLAHE-gray + Canny edges) alone plateaued at ~62% test
accuracy and could not even fit the training set. Raw RGB alone caught
smudges (88%) but completely missed cracks (0% recall). The final fix
was a **6-channel hybrid**: engineered channels for crack/smudge signals
+ raw RGB so the CNN can learn its own color/texture features.

| # | Config | Test acc | clean | smudged | cracked |
|---|---|---|---|---|---|
| 1 | engineered (blur=7, canny=80/160), seed=42 | 55.8% | 44.4% | 70.6% | 52.9% |
| 2 | engineered, seed=123 | 57.7% | 66.7% | 41.2% | 64.7% |
| 3 | engineered softened (blur=5, canny=60/120) | 61.5% | 66.7% | 47.1% | 70.6% |
| 4 | engineered softened + lr=0.0005, 80 epochs | 61.5% | 66.7% | 52.9% | 64.7% |
| 5 | engineered softened + no augmentation (diagnostic) | 51.9% | 50.0% | 58.8% | 47.1% |
| 6 | raw RGB only | 51.9% | 66.7% | 88.2% | **0.0%** |
| 7 | **6-channel hybrid** (engineered + RGB), canny=(60,120) | **73.1%** | 77.8% | 82.4% | 58.8% |
| 8 | hybrid, canny lowered to (40,100) | 73.1% | 77.8% | 76.5% | 64.7% |
| 9 | hybrid + canny=(40,100), seed=123 | 59.6% | 77.8% | 70.6% | 29.4% |

**Key insights from the iterations:**
- Even with augmentation off (#5), train acc plateaued at ~62% — meaning
  the engineered projection was throwing away information the model needed.
- RGB-only (#6) showed Canny edges were carrying the entire crack signal.
- Hybrid (#7/#8) gives the CNN both. Big jump from engineered alone
  (61.5% → 73.1%). Smudged improves dramatically (47% → 82%) but cracked
  becomes the bottleneck at 58–64%.
- Seed=123 (#9) collapsed to 59.6% — high variance, model not robust.

**Tiles plateau at 73%.** Friend is taking over this surface. Suggested next
steps for them: try lower dropout + more epochs, or `--smudge_boost`/class
weight adjustments to lift cracked specifically.

Tiles checkpoint: `outputs/tiles_best.pth` is the latest hybrid run
(canny=40,100, seed=42). `outputs/tiles_seed123_best.pth` is the seed=123
hybrid model (59.6%). Run `python train.py --surface tiles` to retrain.

## Wood

| Run | Mode | Test acc | clean | smudged | cracked |
|---|---|---|---|---|---|
| 1 | engineered (clip=3.0, blur=5, canny=(40,100)), lr=0.0005, 80 ep | **61.8%** | 68.4% | 33.3% | 83.3% |

Engineered preprocessing fits cracked well (83%) but smudged badly (33%).
Wood grain creates confusing texture that the engineered channels don't
disentangle from smudges. Same fix that worked for walls and tiles
(switch to 6-channel hybrid) should apply here — pending retraining.

Wood checkpoint: `outputs/wood_best.pth` (engineered, 61.8%).

---

## How to inspect training behavior

Every `python train.py` run automatically writes **two** log files:

1. **`outputs/{surface}_train.log`** (or `_seed{N}_train.log`) — the
   "latest" log, overwritten by each new run. Easy to find.
2. **`outputs/training_history/{surface}_seed{N}_{timestamp}.log`** —
   permanent timestamped record. One file per training run, never
   overwritten. Use this to compare different attempts.

So you never lose a log. Open either in VS Code to see per-epoch train/val
loss + accuracy.

## How to reproduce / re-verify any number

```bash
python train.py --surface walls                # seed=42 hybrid, 87.7%
python train.py --surface walls --seed 123     # seed=123 hybrid, 89.5%
python train.py --surface walls --seed 2024    # seed=2024 hybrid, 84.2%
python train.py --surface tiles                # latest hybrid (canny=40,100)
python train.py --surface wood                 # engineered baseline (61.8%)
```

The seed controls both the per-class shuffle (which images land in
train/val/test) and the torch/numpy RNG. Different seeds = different splits.

## Outputs folder map

```
outputs/
├── walls_best.pth                 ← production walls model (hybrid seed=42)
├── walls_seed123_best.pth         ← walls hybrid verification (seed=123)
├── walls_seed42_3ch_best.pth      ← old 3-channel walls (fallback)
├── walls_seed42_3ch_engineered_train.log  ← log of the old 3-channel run
├── walls_train.log                ← latest walls run log (default seed)
├── walls_seed123_train.log        ← latest walls seed=123 run log
├── tiles_best.pth                 ← latest tiles hybrid model (~73%)
├── tiles_seed123_best.pth         ← tiles hybrid seed=123 (59.6%)
├── tiles_train.log                ← latest tiles run log
├── tiles_seed123_train.log        ← tiles seed=123 log
├── wood_best.pth                  ← current wood model (engineered, 61.8%)
├── wood_train.log                 ← latest wood run log
├── training_history/              ← permanent timestamped logs (one per run)
└── smudge_boost_runs/             ← rejected boost experiment logs (kept for reference)
```
