# Semantic Segmentation (geoskill-semantic-segmentation)

> Per-pixel sklearn Random Forest classification + sliding-window stitching + post-processing, outputting a classification raster (offline numpy equivalent implementation).

---

## 1. Overview

Performs per-pixel semantic segmentation on multispectral remote sensing imagery, outputting a class raster and per-class area statistics. Supports both unsupervised (KMeans) and supervised (RandomForest, requires a `--labels` label raster) modes. This skill is an **offline numpy equivalent implementation** of FCN/U-Net semantic segmentation networks: it reproduces the semantic segmentation pipeline without relying on deep learning frameworks, using "feature construction -> per-pixel sklearn classifier -> sliding-window tiled prediction and stitching -> majority-filter post-processing"; consistency between tiled and full-image predictions, post-processing denoising effects, and unsupervised class-permutation matching accuracy are all covered by unit tests.

## 3. Quick Start

```bash
pip install -r requirements.txt
python geoskill-semantic-segmentation.py --bbox 116 39 117 40 --synthetic --output-dir ./out
```

## 4. CLI Parameters

Run `python <skill>.py --help` for the full list. Common parameters:

| Parameter | Type | Description |
|---|---|---|
| `--bbox` | `float[4]` | WGS84 bounding box `min_lon min_lat max_lon max_lat` |
| `--input` | `path` | Local input file (GeoJSON/GeoTIFF/etc.) |
| `--output-dir` | `path` | Output directory (default `./output`) |
| `--synthetic` | `flag` | Use synthetic data instead of real input |
| `--quiet` | `flag` | Suppress non-essential stdout |

## 5. Input / Output

| File | Format | Description |
|---|---|---|
| `segmentation.tif` | GeoTIFF | Class raster (one integer class per pixel) |
| `class_stats.json` | JSON | Per-class pixel counts and shares |
| `output-manifest.json` | JSON | Run manifest (inputs/outputs/QA/exit code) |

## 6. Technical Principle

(see SKILL.md for details)

## 8. License

MIT License. See [`LICENSE`](./LICENSE) for full text.

---

> 中文完整版：[references/SKILL.zh-CN.md](references/SKILL.zh-CN.md)
