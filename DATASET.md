# UltraG-Ray 数据集要求

## 目录结构

```text
datasets/ultragray/
  pig_shoulder_v2/
    conf.json
    images_train.npy
    images_val.npy
    poses_train.npy
    poses_val.npy
  spine_phantom/
    conf.json
    images_train.npy
    images_val_v2.npy
    poses_train.npy
    poses_val_v2.npy
```

## 已验证张量

| 数据集 | 图像 | 位姿 |
|---|---|---|
| Pig Shoulder v2 train | `(219,732,746,3)` uint8 | `(219,4,4)` float64 |
| Pig Shoulder v2 val | `(104,732,746,3)` uint8 | `(104,4,4)` float64 |
| Spine Phantom train | `(1845,658,372,3)` uint8 | `(1845,4,4)` float64 |
| Spine Phantom val | `(93,658,372,3)` uint8 | `(93,4,4)` float64 |

图像和位姿的首维必须相等；每个位姿必须为有限的齐次 `4x4` 矩阵。RGB 图像接入灰度 pipeline 前应使用固定转换规则，例如

\[
I=0.299R+0.587G+0.114B.
\]

原始 NPY 数据不能直接传给当前 NPZ pipeline：需要根据 `conf.json` 明确像素间距、坐标系与 pose 方向，再转换为 [DATA_FORMAT.md](DATA_FORMAT.md) 定义的字段，避免得到尺度正确但空间方向错误的重建。

数据下载受原发布方许可约束。本仓库仅提供下载脚本，不重新分发原始数据。
