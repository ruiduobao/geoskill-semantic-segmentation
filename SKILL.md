---
name: geoskill-semantic-segmentation
description: '逐像元sklearn随机森林分类+滑窗拼接+后处理，输出分类栅格（离线numpy等价实现）'
---

# 语义分割 | Semantic Segmentation

Performs per-pixel semantic segmentation of multispectral remote sensing imagery and outputs a class raster with per-class area statistics. Two modes are supported: unsupervised (KMeans) and supervised (RandomForest, requiring a label raster via `--labels`).

This skill is an **offline numpy equivalent implementation** of FCN/U-Net semantic segmentation networks: without depending on any deep learning framework, it reproduces the semantic segmentation pipeline through "feature construction -> per-pixel sklearn classifier -> sliding-window tiled prediction and stitching -> majority-filter post-processing". Unit tests cover the consistency between tiled and whole-scene prediction, the denoising effect of post-processing, and the unsupervised class-remapping matching accuracy.

## Dependencies / 依赖

```bash
pip install numpy rasterio scipy scikit-learn geopandas shapely
```

## Usage / 使用方法

### Example 1 (Synthetic Data, Offline)

```bash
python geoskill-semantic-segmentation.py --bbox 116.0 39.0 117.0 40.0 --synthetic --output-dir ./out
```

### Example 2: Real-Image Tiled Segmentation

```bash
python geoskill-semantic-segmentation.py --input scene.tif --n-classes 5 --tile 64 --smooth 5 --output-dir ./out
```

### Example 3: Supervised Random Forest (Synthetic Ground Truth)

```bash
python geoskill-semantic-segmentation.py --bbox 116.0 39.0 117.0 40.0 --synthetic --method rf --output-dir ./out
```

### Example 4: Supervised Random Forest (Real Labels)

```bash
python geoskill-semantic-segmentation.py --input scene.tif --labels labels.tif --method rf --output-dir ./out
```

## Output / 输出

| File | Format | Description |
|---|---|---|
| `segmentation.tif` | GeoTIFF | Class raster (one integer class per pixel) |
| `class_stats.json` | JSON | Per-class pixel counts and fractions |
| `output-manifest.json` | JSON | Run manifest (inputs/outputs/QA/exit code) |

## Data Source / 数据源 / Source

A local multi-band GeoTIFF, or a `--synthetic` cube of three land-cover types (vegetation/soil/water) with built-in ground truth for accuracy QA.

## Privacy / 隐私声明 / Privacy

- Runs offline by default; `--synthetic` mode requires no network at all.
- All processing is done locally; no user data is uploaded.

## License / License

MIT

---

> 中文完整版：需要时再读 [references/SKILL.zh-CN.md](references/SKILL.zh-CN.md)，不要默认全量加载。
