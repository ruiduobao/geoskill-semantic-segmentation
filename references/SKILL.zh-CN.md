# 语义分割 | Semantic Segmentation

对多光谱遥感影像做逐像元语义分割，输出类别栅格与各类面积统计。支持无监督（KMeans）与有监督（RandomForest，需 --labels 标注栅格）两种模式。

本 skill 是 FCN/U-Net 语义分割网络的**离线 numpy 等价实现**：不依赖深度学习框架，用"特征构建 -> 逐像元 sklearn 分类器 -> 滑窗分块预测拼接 -> 众数滤波后处理"复现语义分割流程；分块预测与整幅预测的一致性、后处理去噪效果、无监督类别置换匹配精度均有单元测试覆盖。

## 依赖

```bash
pip install numpy rasterio scipy scikit-learn geopandas shapely
```

## 使用方法

### 示例 1（合成数据，离线）

```bash
python geoskill-semantic-segmentation.py --bbox 116.0 39.0 117.0 40.0 --synthetic --output-dir ./out
```

### 示例 2：真实影像分块分割

```bash
python geoskill-semantic-segmentation.py --input scene.tif --n-classes 5 --tile 64 --smooth 5 --output-dir ./out
```

### 示例 3：有监督随机森林（合成真值）

```bash
python geoskill-semantic-segmentation.py --bbox 116.0 39.0 117.0 40.0 --synthetic --method rf --output-dir ./out
```

### 示例 4：有监督随机森林（真实标注）

```bash
python geoskill-semantic-segmentation.py --input scene.tif --labels labels.tif --method rf --output-dir ./out
```

## 输出

| 文件 | 格式 | 说明 |
|---|---|---|
| `segmentation.tif` | GeoTIFF | 类别栅格（每像元一个整数类别） |
| `class_stats.json` | JSON | 各类像元数与占比 |
| `output-manifest.json` | JSON | 运行清单（输入/输出/QA/退出码） |

## 数据源 / Source

本地多波段 GeoTIFF，或 --synthetic 合成三体物（植被/土壤/水体）立方体（自带真值用于精度 QA）。

## 隐私声明 / Privacy

- 默认离线运行，`--synthetic` 模式完全无网络。
- 所有处理在本地完成，不上传任何用户数据。

## License

MIT
