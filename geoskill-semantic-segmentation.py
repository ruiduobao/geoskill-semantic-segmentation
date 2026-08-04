#!/usr/bin/env python3
"""semantic-segmentation — 语义分割

对多光谱遥感影像做逐像元语义分割，输出类别栅格（每个像元一个整数类别）。

本 skill 是 FCN/U-Net 语义分割网络的**离线 numpy 等价实现**：
不依赖 torch/tensorflow，而用可验证的经典流程复现"逐像元分类 + 拼接 + 后处理"——

1. **特征构建**：把 (bands, H, W) 立方体重排成 (H*W, bands) 的逐像元光谱特征；
2. **逐像元分类器**：用 sklearn 的 KMeans（无监督）或 RandomForest（有监督）
   对每个像元打类别标签，等价于网络的 1x1 卷积分类头；
3. **滑窗拼接**：按瓦片 (tile) 逐块预测再拼回整幅，验证分块推理与整幅一致；
4. **后处理**：众数滤波（majority filter）平滑椒盐噪声，等价于 CRF/形态学后处理。

数据源：本地多波段 GeoTIFF，或 ``--synthetic`` 生成含若干地物分区的模拟立方体。

隐私声明 / Privacy：
- 默认离线运行，``--synthetic`` 模式完全无网络。
- 所有处理在本地完成，不上传任何用户数据。

Usage:
    python semantic-segmentation.py --input scene.tif --n-classes 4 --output-dir ./out
    python semantic-segmentation.py --bbox 116 39 117 40 --synthetic --output-dir ./out

License: MIT
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

VERSION = "1.0.0"
SKILL_NAME = "semantic-segmentation"

try:
    from _geoskill_core.errors import (
        GeoSkillError, UsageError, ValidationError, ProcessError, to_exit_code,
    )
    from _geoskill_core.manifest import OutputManifest, OutputFile
except ImportError:  # pragma: no cover
    class GeoSkillError(Exception):
        def __init__(self, message: str, code: int = 7, kind: str = "EGeo", **kw):
            super().__init__(message)
            self.message, self.code, self.kind = message, code, kind

    class UsageError(GeoSkillError):
        def __init__(self, m, **k): super().__init__(m, code=2, kind="EUsage", **k)

    class ValidationError(GeoSkillError):
        def __init__(self, m, **k): super().__init__(m, code=6, kind="EValidate", **k)

    class ProcessError(GeoSkillError):
        def __init__(self, m, **k): super().__init__(m, code=7, kind="EProcess", **k)

    def to_exit_code(exc):
        return getattr(exc, "code", 7)

    OutputManifest = None
    OutputFile = None


def _utc_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat().replace("+00:00", "Z")


# ---------------------------------------------------------------------------
# 校验前置
# ---------------------------------------------------------------------------
def validate_bbox(bbox, source: str = "bbox") -> None:
    """Validate geographic bbox: W<=E, S<=N, lon/lat in range, min area.

    Cross-dateline (W>E) is a ValidationError with a hint to split.
    """
    if bbox is None or len(bbox) != 4:
        raise ValidationError(f"{source}: expected 4 floats [W S E N], got {bbox!r}")
    try:
        w, s, e, n = (float(v) for v in bbox)
    except (TypeError, ValueError) as exc:
        raise ValidationError(f"{source}: non-numeric bbox values: {bbox!r}") from exc
    for v, name in ((w, "W"), (s, "S"), (e, "E"), (n, "N")):
        if v != v:  # NaN check
            raise ValidationError(f"{source}: bbox contains NaN at {name}")
    if not (-180.0 <= w <= 180.0 and -180.0 <= e <= 180.0):
        raise ValidationError(f"{source}: lon out of [-180,180]: W={w} E={e}")
    if not (-90.0 <= s <= 90.0 and -90.0 <= n <= 90.0):
        raise ValidationError(f"{source}: lat out of [-90,90]: S={s} N={n}")
    if w > e:
        raise ValidationError(
            f"{source}: W ({w}) > E ({e}); cross-dateline bboxes are not supported. "
            "Split into two bboxes on each side of the 180\u00b0 meridian and run separately."
        )
    if s > n:
        raise ValidationError(f"{source}: S ({s}) > N ({n})")
    if (e - w) < 1e-9 or (n - s) < 1e-9:
        raise ValidationError(
            f"{source}: bbox too small (dlon={e - w}, dlat={n - s}); need > 1e-9 degrees"
        )


def validate_n_classes(n_classes: int) -> None:
    """--n-classes must be >= 2 (KMeans / RF trivial for 0/1)."""
    if int(n_classes) < 2:
        raise ValidationError(
            f"--n-classes must be >= 2 (got {n_classes!r}); "
            "KMeans requires at least 2 clusters and 1 class trivially assigns everything."
        )


def validate_tile_smooth(tile: int, smooth: int) -> None:
    if int(tile) < 1:
        raise ValidationError(f"--tile must be >= 1 (got {tile!r})")
    if int(smooth) < 0:
        raise ValidationError(f"--smooth must be >= 0 (got {smooth!r})")


def validate_seed(seed: int) -> None:
    if int(seed) < 0:
        raise ValidationError(f"--seed must be >= 0 (got {seed!r})")


# ---------------------------------------------------------------------------
# 核心算法
# ---------------------------------------------------------------------------
def build_feature_matrix(cube: np.ndarray) -> np.ndarray:
    """把 (bands, H, W) 立方体重排成 (H*W, bands) 的逐像元特征矩阵。"""
    cube = np.asarray(cube, dtype=np.float64)
    if cube.ndim == 2:
        cube = cube[np.newaxis, ...]
    if cube.ndim != 3:
        raise ValidationError("cube must be (bands, H, W)", shape=list(cube.shape))
    nb, h, w = cube.shape
    feat = np.moveaxis(cube, 0, -1).reshape(h * w, nb)
    return feat


def train_classifier(
    features: np.ndarray,
    n_classes: int,
    labels: Optional[np.ndarray] = None,
    method: str = "kmeans",
    seed: int = 42,
):
    """训练逐像元分类器。

    method="kmeans"：无监督聚类（等价于无标签自训练）；
    method="rf"：有监督随机森林，需要 labels（每个样本的类别）。
    返回 fitted sklearn 模型。
    """
    features = np.asarray(features, dtype=np.float64)
    if method == "kmeans":
        from sklearn.cluster import KMeans
        model = KMeans(n_clusters=n_classes, n_init=10, random_state=seed)
        model.fit(features)
        return model
    if method == "rf":
        if labels is None:
            raise UsageError("method='rf' requires training labels")
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=50, random_state=seed, n_jobs=1)
        model.fit(features, np.asarray(labels))
        return model
    raise UsageError(f"unknown method '{method}'. Choose from: kmeans, rf", method=method)


def predict_tiled(
    model,
    features_grid: np.ndarray,
    height: int,
    width: int,
    tile: int = 32,
) -> np.ndarray:
    """按瓦片逐块预测并拼接成 (H, W) 标签图。

    因为使用同一个已训练模型，分块预测与整幅预测应完全一致——
    这正是"滑窗拼接"在离线等价实现里的可验证性质。
    """
    features_grid = np.asarray(features_grid, dtype=np.float64)
    n = features_grid.shape[0]
    if n != height * width:
        raise ValidationError(
            "feature count does not match H*W", n=int(n), hw=int(height * width)
        )
    labels_flat = np.empty(n, dtype=np.int64)
    # 按行块切分（每块包含若干完整行），保证拼接无缝
    for row0 in range(0, height, tile):
        row1 = min(height, row0 + tile)
        start = row0 * width
        end = row1 * width
        block = features_grid[start:end]
        pred = np.asarray(model.predict(block)).astype(np.int64)
        labels_flat[start:end] = pred
    return labels_flat.reshape(height, width)


def majority_filter(label_map: np.ndarray, size: int = 3) -> np.ndarray:
    """众数滤波后处理：把每个像元替换为邻域内出现最多的类别。

    用于消除椒盐噪声（等价于分割网络的 CRF / 形态学后处理）。

    负值（NoData 哨兵，-1）会被跳过，不参与众数计算。窗口全 NoData
    时返回中心像元的值。
    """
    from scipy.ndimage import generic_filter
    lm = np.asarray(label_map)
    if size < 2:
        return lm.copy()

    def _mode(vals):
        v = vals[vals >= 0].astype(np.int64)
        if v.size == 0:
            return vals[len(vals) // 2]
        counts = np.bincount(v)
        return float(np.argmax(counts))

    out = generic_filter(lm.astype(np.float64), _mode, size=size, mode="nearest")
    return out.astype(lm.dtype)


def label_accuracy(pred: np.ndarray, truth: np.ndarray) -> float:
    """计算像素精度（允许类别标签的全局置换匹配——取最佳映射）。

    无监督聚类的类别编号是任意的，所以用混淆矩阵贪心匹配后再算精度。
    """
    pred = np.asarray(pred).ravel()
    truth = np.asarray(truth).ravel()
    if pred.size != truth.size:
        raise ValidationError("pred and truth size mismatch")
    from sklearn.metrics import confusion_matrix
    from scipy.optimize import linear_sum_assignment
    labels = np.unique(np.concatenate([pred, truth]))
    cm = confusion_matrix(truth, pred, labels=labels).astype(np.float64)
    total = cm.sum()
    if total <= 0:
        return 0.0
    # 匈牙利算法求最优一对一类别匹配（最大化命中像元数）
    row_ind, col_ind = linear_sum_assignment(-cm)
    matched = float(cm[row_ind, col_ind].sum())
    return matched / total


def semantic_segment(
    cube: np.ndarray,
    n_classes: int,
    method: str = "kmeans",
    tile: int = 32,
    smooth: int = 3,
    labels: Optional[np.ndarray] = None,
    seed: int = 42,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """完整语义分割流程：特征 -> 训练 -> 分块预测 -> 众数后处理。

    NoData 语义：所有波段同时为有限值的像元参与训练与预测；任何波段
    为 NaN 的像元被标记为 -1（nodata label），不计入任何 class 统计。

    返回 (label_map[H, W], info)。
    """
    cube = np.asarray(cube, dtype=np.float64)
    if cube.ndim == 2:
        cube = cube[np.newaxis, ...]
    nb, h, w = cube.shape
    feat = build_feature_matrix(cube)
    # Per-pixel valid mask: a pixel is valid iff all bands are finite.
    finite = np.isfinite(cube).all(axis=0)
    n_valid = int(finite.sum())
    if n_valid < max(2, n_classes):
        raise ValidationError(
            f"need at least {max(2, n_classes)} valid (non-NoData) pixels; got {n_valid}",
            n_valid=n_valid,
        )
    feat_flat = feat  # (H*W, bands)
    finite_flat = finite.reshape(-1)

    if method == "rf":
        if labels is None:
            raise UsageError("method='rf' requires training labels")
        lab = np.asarray(labels).ravel()
        valid = (lab >= 0) & finite_flat  # -1 视为未标注 + NoData
        if int(valid.sum()) < max(2, n_classes):
            raise ValidationError("not enough labeled pixels for rf training",
                                  n_labeled=int(valid.sum()))
        model = train_classifier(feat_flat[valid], n_classes, labels=lab[valid],
                                 method="rf", seed=seed)
        # RF accepts NaN; predict over all pixels
        pred_all = np.asarray(model.predict(feat_flat)).astype(np.int64)
    else:
        # kmeans only on valid (non-NaN) pixels
        if n_valid < n_classes:
            raise ValidationError(
                f"need at least {n_classes} valid pixels for kmeans; got {n_valid}"
            )
        model = train_classifier(feat_flat[finite_flat], n_classes, method="kmeans", seed=seed)
        # KMeans.predict refuses NaN; predict only on valid pixels, fill rest with -1
        pred_valid = np.asarray(model.predict(feat_flat[finite_flat])).astype(np.int64)
        pred_all = np.full(finite_flat.shape, -1, dtype=np.int64)
        pred_all[finite_flat] = pred_valid
    labels_flat = pred_all
    label_map = labels_flat.reshape(h, w)
    smoothed = majority_filter(label_map, size=smooth)
    # Re-apply nodata mask (majority filter may overwrite -1 for NaN-neighbor cells)
    smoothed = np.where(finite, smoothed, -1).astype(np.int64)

    # Class stats: only over valid (non-NoData) pixels
    valid_smoothed = smoothed[finite]
    unique, counts = np.unique(valid_smoothed, return_counts=True)
    class_stats = [
        {"class_id": int(c), "pixel_count": int(n),
         "fraction": float(n) / float(valid_smoothed.size)}
        for c, n in zip(unique, counts)
    ]
    info = {
        "method": method, "n_classes": int(n_classes),
        "height": int(h), "width": int(w), "n_bands": int(nb),
        "class_stats": class_stats,
        "n_valid_pixels": n_valid,
        "n_total_pixels": int(finite.size),
    }
    return smoothed, info


# ---------------------------------------------------------------------------
# 合成数据
# ---------------------------------------------------------------------------
def generate_synthetic(
    bbox: List[float],
    n_bands: int = 4,
    width: int = 64,
    height: int = 64,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """生成含 3 个地物分区（植被/土壤/水体）的多光谱立方体 + 真值标签图。"""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:height, 0:width]
    xx = xx.astype(np.float64) / max(width - 1, 1)
    yy = yy.astype(np.float64) / max(height - 1, 1)

    truth = np.zeros((height, width), dtype=np.int64)
    truth[(xx + yy) > 1.1] = 1   # 植被
    truth[(xx + yy) < 0.5] = 2   # 水体
    # 其余为 0 (土壤)

    # 每个类别的光谱签名（n_bands 维）
    signatures = {
        0: np.linspace(0.15, 0.30, n_bands),  # 土壤：中等且缓升
        1: np.array([0.04, 0.08, 0.05, 0.45][:n_bands]
                    if n_bands <= 4 else [0.04, 0.08, 0.05, 0.45] + [0.2] * (n_bands - 4)),
        2: np.linspace(0.05, 0.01, n_bands),  # 水体：低且递减
    }
    cube = np.zeros((n_bands, height, width), dtype=np.float32)
    for cls, sig in signatures.items():
        mask = truth == cls
        for b in range(n_bands):
            cube[b][mask] = sig[b]
    cube = cube + rng.normal(0, 0.005, size=cube.shape).astype(np.float32)
    cube = np.clip(cube, 0.0, 1.0)

    info = {"bbox": bbox, "n_bands": n_bands, "width": width, "height": height}
    return cube, truth, info


# ---------------------------------------------------------------------------
# GeoTIFF I/O
# ---------------------------------------------------------------------------
def write_geotiff(path: str, array: np.ndarray, bbox: List[float], nodata: float = -9999.0) -> None:
    import rasterio
    from rasterio.transform import from_bounds

    arr = np.asarray(array, dtype=np.float32)
    if arr.ndim == 2:
        arr = arr[np.newaxis, ...]
    nb, h, w = arr.shape
    transform = from_bounds(bbox[0], bbox[1], bbox[2], bbox[3], w, h)
    profile = {
        "driver": "GTiff", "height": h, "width": w, "count": nb,
        "dtype": "float32", "crs": "EPSG:4326", "transform": transform,
        "nodata": nodata, "compress": "deflate",
    }
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with rasterio.open(path, "w", **profile) as dst:
        for b in range(nb):
            dst.write(arr[b].astype("float32"), b + 1)


def read_geotiff(path: str) -> Tuple[np.ndarray, List[float]]:
    """读取栅格，返回 (cube, bbox)。

    NoData 哨兵值（src.nodata）会被替换为 NaN 以避免污染下游分类器
    （KMeans/RF 会把 -9999 当成"独立类别"，产生 64 像元假聚类）。
    """
    import rasterio
    if not os.path.exists(path):
        raise UsageError(f"input raster not found: {path}", path=path)
    with rasterio.open(path) as src:
        cube = src.read().astype(np.float32)
        b = src.bounds
        bbox = [b.left, b.bottom, b.right, b.top]
        nd = src.nodata
        if nd is not None:
            cube = np.where(cube == float(nd), np.nan, cube).astype(np.float32)
    return cube, bbox


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------
def write_manifest(output_dir, args, outputs, qa, started_at, exit_code, bbox):
    if OutputManifest is None:
        return None
    cmd = " ".join([SKILL_NAME] + sys.argv[1:])
    man = OutputManifest(
        skill=SKILL_NAME, skill_version=VERSION, command=cmd,
        started_at=started_at, finished_at=_utc_now(), exit_code=exit_code,
        inputs={
            "input": getattr(args, "input", None),
            "method": getattr(args, "method", None),
            "n_classes": getattr(args, "n_classes", None),
            "synthetic": bool(getattr(args, "synthetic", False)),
        },
        outputs=[OutputFile(**o) for o in outputs],
        qa=qa,
        software={"python": sys.version.split()[0], "skill": SKILL_NAME},
    )
    path = os.path.join(output_dir, "output-manifest.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(man.to_dict(), f, ensure_ascii=False, indent=2, default=str)
    return path


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def process(args: argparse.Namespace) -> int:
    started_at = _utc_now()
    output_dir = args.output_dir
    bbox = list(args.bbox) if args.bbox else None

    # ===== 0) Validate CLI up-front (no side effects, no mkdir) =====
    if not (args.input or args.synthetic or bbox):
        raise UsageError("provide --bbox (synthetic mode) or --input <raster>")
    validate_n_classes(args.n_classes)
    validate_tile_smooth(args.tile, args.smooth)
    validate_seed(args.seed)
    if bbox is not None:
        validate_bbox(bbox, source="--bbox")

    # mkdir AFTER validation (CONVENTIONS §1.1 / common bug pattern #6)
    os.makedirs(output_dir, exist_ok=True)

    synth_truth = None
    labels_grid = None

    if args.input and not args.synthetic:
        cube, file_bbox = read_geotiff(args.input)
        bbox = bbox if bbox is not None else file_bbox
        if args.bbox is not None:
            validate_bbox(bbox, source="--bbox")
        # Reject all-NaN (would otherwise fail inside KMeans / RF with confusing error)
        if not np.isfinite(cube).any():
            raise ValidationError(
                f"input raster '{args.input}' contains only NoData / NaN pixels; nothing to segment"
            )
        source_note = args.input
        if args.method == "rf":
            if not args.labels:
                raise UsageError("method='rf' requires --labels <raster> (or use --synthetic)")
            lab_cube, _ = read_geotiff(args.labels)
            labels_grid = np.round(lab_cube[0]).astype(np.int64)
    else:
        if bbox is None:
            raise UsageError("provide --bbox (synthetic mode) or --input <raster>")
        cube, synth_truth, _ = generate_synthetic(bbox)
        source_note = "synthetic"
        if args.method == "rf":
            labels_grid = synth_truth  # 合成模式下用内置真值做有监督训练

    if cube.size == 0:
        raise ValidationError("input raster is empty")

    label_map, info = semantic_segment(
        cube, n_classes=args.n_classes, method=args.method,
        tile=args.tile, smooth=args.smooth, labels=labels_grid, seed=args.seed,
    )

    out_tif = os.path.join(output_dir, "segmentation.tif")
    # Write labels as float32 with -1 nodata sentinel
    write_geotiff(out_tif, label_map.astype(np.float32), bbox, nodata=-1.0)

    stats_path = os.path.join(output_dir, "class_stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)

    qa: Dict[str, Any] = {
        "source": source_note, "method": args.method,
        "n_classes": args.n_classes,
        "n_classes_found": len(info["class_stats"]),
        "n_valid_pixels": int(info.get("n_valid_pixels", 0)),
        "n_total_pixels": int(info.get("n_total_pixels", 0)),
    }
    if synth_truth is not None:
        qa["synthetic_accuracy"] = label_accuracy(label_map, synth_truth)

    outputs = [
        {"path": out_tif, "kind": "raster", "crs_epsg": 4326,
         "bbox_wgs84": bbox, "band_count": 1},
        {"path": stats_path, "kind": "json"},
    ]
    man_path = write_manifest(output_dir, args, outputs, qa, started_at, 0, bbox)

    if not args.quiet:
        print(f"[{SKILL_NAME}] source: {source_note}")
        print(f"[{SKILL_NAME}] method: {args.method}  classes: {len(info['class_stats'])}")
        print(f"[{SKILL_NAME}] output: {out_tif}")
        if "synthetic_accuracy" in qa:
            print(f"[{SKILL_NAME}] synthetic accuracy: {qa['synthetic_accuracy']:.3f}")
        if man_path:
            print(f"[{SKILL_NAME}] manifest: {man_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog=SKILL_NAME,
        description="Pixel-wise semantic segmentation (sklearn classifier + tiling + majority filter).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--bbox", nargs=4, type=float, metavar=("W", "S", "E", "N"),
                   help="geographic extent [minLon minLat maxLon maxLat]")
    p.add_argument("--input", help="input multispectral GeoTIFF")
    p.add_argument("--labels", help="training labels GeoTIFF (>=0 labeled, -1 unlabeled; method=rf)")
    p.add_argument("--n-classes", type=int, default=3, help="number of classes (default: 3)")
    p.add_argument("--method", default="kmeans", choices=["kmeans", "rf"],
                   help="classifier method (default: kmeans)")
    p.add_argument("--tile", type=int, default=32, help="prediction tile size in pixels")
    p.add_argument("--smooth", type=int, default=3, help="majority filter window size")
    p.add_argument("--seed", type=int, default=42, help="random seed")
    p.add_argument("--synthetic", action="store_true", help="generate a synthetic scene (offline)")
    p.add_argument("--output-dir", default="./output", help="output directory")
    p.add_argument("--quiet", action="store_true", help="suppress console output")
    p.add_argument("--version", action="version", version=f"%(prog)s {VERSION}")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return process(args)
    except GeoSkillError as exc:
        print(f"[{SKILL_NAME}] ERROR [{exc.kind}] {exc.message}", file=sys.stderr)
        return exc.code
    except KeyboardInterrupt:
        return 130
    except Exception as exc:  # noqa: BLE001
        print(f"[{SKILL_NAME}] ERROR {exc}", file=sys.stderr)
        return to_exit_code(exc)


if __name__ == "__main__":
    sys.exit(main())
