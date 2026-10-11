# Pipeline NPZ 数据格式

`pipeline_god_file.py` 接收单个 `.npz` 文件。

## 必需字段

| 字段 | shape | 类型/含义 |
|---|---|---|
| `images` | `(N,H,W)` 或 `(N,1,H,W)` | 灰度超声帧；float `[0,1]` 或 uint8 `[0,255]` |

## 推荐字段

| 字段 | shape | 含义 |
|---|---|---|
| `pixel_positions_world` | `(N,H*W,3)` 或 `(N,H,W,3)` | 像素中心世界坐标，单位 mm |
| `probe_origin_world` | `(N,3)` 或 `(3,)` | 探头原点，单位 mm |
| `axis_lateral_world` | `(N,3)` 或 `(3,)` | lateral 单位轴 |
| `axis_axial_world` | `(N,3)` 或 `(3,)` | axial/depth 单位轴 |
| `axis_elevational_world` | `(N,3)` 或 `(3,)` | out-of-plane 单位轴 |

若未给出逐像素坐标，配置必须提供 `delta_lateral_mm`、`delta_axial_mm`、`delta_elevational_mm`、`near_mm`、`far_mm` 和 `width_mm`，pipeline 将据此生成坐标。

UltraGray 可额外提供 `ray_origins (N,W,3)` 与 `ray_dirs (N,W,3)`；UltraGS 可额外提供 `viewmatrix`、`projmatrix`、`camera_center`、`tanfovx` 和 `tanfovy`。

所有空间长度统一使用毫米。位姿矩阵必须明确其方向，并在转换前检查旋转正交性及相邻帧平移尺度。
