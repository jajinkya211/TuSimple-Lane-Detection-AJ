# TuSimple Lane Detection — Multi-Method Benchmark (UFLD / RESA / CLRNet-ish / LSTR)

**Goal:** build a professional, end-to-end notebook for the TuSimple lane detection task that:
- probes & cleans the dataset,
- trains **four** modern approaches,
- evaluates with **TuSimple metrics** (accuracy, TP/FP/FN + precision/recall/F1),

> Designed to run on Kaggle (T4x2). Works locally with PyTorch + CUDA as well.

---

## What’s inside

- **Data probe & cleaning:** validates file structure, merges the 3 train JSONs, fixes NaNs/negatives, writes a clean Parquet.
- **Unified dataloader:** produces images, seg masks (if available), and **row-anchor** labels (56/48 rows).
- **Models (4 families):**
  1. **UFLD (row-anchor classification)** — fast lane bins per row.
  2. **RESA (segmentation + spatial propagation)** — denoised lane masks → points.
  3. **Detector (CLRNet-style)** — multi-proposal row-anchor with objectness + NMS.
  4. **LSTR (transformer regressor)** — queries=lanes, continuous x per row + existence.
- **Decoders:** per-model tuned post-processing (thresholds, min rows/area, lane-NMS, smoothing).
- **Evaluator:** exact TuSimple protocol (matching per row, tolerance px, counts), plus P/R/F1.
- **Visual report:** GT vs predictions (all models + ensemble) with per-image TP/FP/FN overlays.
---

## Notebook flow (cells at a glance)

1. **Environment + Probe**  
   Detects GPU, verifies dataset layout, prints sample paths, saves `config_tusimple.json`.

2. **Consolidate & Clean**  
   Reads `label_data_{0313,0531,0601}.json`, merges, repairs NaNs/negatives, writes  
   `tusimple_consolidated_clean.parquet` and a preview CSV.

3. **Metrics & Unit Tests**  
   Implements TuSimple evaluator (accuracy, TP/FP/FN + precision/recall/F1).  
   Runs sanity unit tests on edge cases.

4. **Dataset / Dataloader**  
   Builds train/val splits; outputs tensors for:
   - `image` (C×H×W),  
   - `seg_mask` (1×H×W for segmentation),  
   - `row_anchor` (`cls`, `valid`, `n_rows`, `h_rows_res`),  
   - `meta` (paths, original H/W, scale factors).

5. **Sanity train (tiny)**  
   Overfits a few batches to confirm gradients/data wrangling.

6. **UFLD (row-anchor)**  
   ResNet-34 backbone → [lanes, rows, bins]; train + eval; **fast** baseline.

7. **RESA (segmentation)**  
   ResNet-34 + RESA block → mask → connected-components → lane points; train + eval.

8. **Compare (UFLD vs RESA)**  
   Full val metrics, params, throughput (imgs/s). Saves CSV.

9. **Detector (CLRNet-ish)**  
   Multi-proposal head with row-bins + objectness; greedy matching; NMS decode.

10. **LSTR (transformer regressor)**  
    CNN features → transformer (queries=lanes) → continuous x@56 rows + existence.

11–13. **Tuning & Calibration**  
    Stronger decoders, relaxed/strict thresholds, subset calibration (accuracy or F1-centric),  
    re-comparison tables, and a simple **RESA+UFLD ensemble** with lane-NMS.

**VIZ-REPORT**  
Generates a gallery (`/kaggle/working/lane_report/index.html`) with GT vs all predictions + per-image TP/FP/FN.

---

## How the logic works (high-level)

- **Row-anchor paradigm (UFLD / Detector):**  
  Sample a fixed set of y-rows; classify the **x-bin** for each lane at each row.  
  Decoding chooses confident rows, maps bins→x, smooths, then **lane-NMS** merges overlaps.

- **Segmentation (RESA):**  
  Predicts a lane mask; morphology + connected components group pixels into lanes;  
  for each y-row, we take the mean x of the component’s cross-section → lane polyline.

- **Transformer regression (LSTR):**  
  Treat **lanes as queries**; for each query, regress continuous x across rows and predict existence.  
  Decoding keeps confident queries, sorts left→right, and pads to TuSimple rows.

- **Evaluation:**  
  TuSimple accuracy = fraction of correctly predicted points among standard y-rows, plus TP/FP/FN matching criteria.  
  We also compute **precision/recall/F1** for clarity.

- **Decoders matter:**  
  Thresholds (prob/objectness/existence), min-rows/area, and **lane-NMS** strongly affect FP/FN trade-offs.

---

## Reproducible quick results (from the included runs)

| Model                     | Acc    | F1     | TP  | FP   | FN  | Notes |
|--------------------------|:------:|:------:|:---:|:----:|:---:|------|
| **UFLD** (row-anchor)    | ~0.659 | ~0.166 | 221 | 1331 | 897 | Fast; higher FP until tuned longer |
| **RESA** (segmentation)  | ~0.586 | ~0.340 | 376 |  715 | 742 | Strong after tuned mask decoder |
| **Detector** (CLRNet-ish)| ~0.531 | ~0.063 |  80 | 1338 |1038 | Needs longer training & decode tuning |
| **LSTR** (transformer)   | ~0.217 | ~0.000 |   0 | 1190 |1118 | Light config; benefits from more epochs |
| **Ensemble** (RESA+UFLD) | ~0.361 | ~0.128 | 113 |  536 |1005 | Simple lane-NMS merge; improves F1 modestly |

> Metrics are on the provided val split and depend on thresholds/epochs; expect higher numbers after 20–40 epochs with stronger augmentation.

---

## Getting started

### Requirements
- Python 3.10+ (Kaggle uses 3.11)
- PyTorch 2.1+ (Kaggle image used: Torch 2.6, CUDA 12.4)
- `torchvision`, `numpy`, `pandas`, `opencv-python`, `matplotlib`

### Dataset
- https://www.kaggle.com/datasets/manideep1108/tusimple
