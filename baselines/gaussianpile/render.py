"""GaussianPile 简化渲染接口。

这个文件刻意不依赖原项目里的 DatasetMeta、SliceMeta、SceneFactory 或 Hydra。
外部合作者只要按下面 dataclass 准备张量，就可以调用 GaussianPile renderer。
所有空间长度单位默认是 mm，图像强度默认在 [0, 1]。
"""

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Dict, Optional

import torch

from gaussianplie_renderer import GaussianPlieRenderConfig, GaussianPlieRenderer
from gaussianplie_renderer.grid_geometry import build_slice_grid_pack


@dataclass
class SliceGeometryInput:
    # 每个像素中心在 world 坐标系下的 3D 位置，shape=(H*W, 3)，单位 mm。
    # 来源：由 probe pose、像素间距和 lateral/axial 方向计算得到。
    pixel_positions_world: torch.Tensor

    # 当前帧探头/切片坐标系原点在 world 坐标中的位置，shape=(3,)，单位 mm。
    # 必须和 pixel_positions_world 使用同一个 world 坐标系。
    probe_origin_world: torch.Tensor

    # 图像横向/lateral 方向的 world-space 单位向量，shape=(3,)，无单位。
    axis_lateral_world: torch.Tensor

    # 图像深度/axial 方向的 world-space 单位向量，shape=(3,)，无单位。
    axis_axial_world: torch.Tensor

    # 切片法向/elevational/out-of-plane 方向单位向量，shape=(3,)，无单位。
    axis_elevational_world: torch.Tensor

    # world 到 probe/slice 局部坐标的刚体变换，通常 shape=(4,4)。
    # 旋转无单位，平移单位 mm；由 tracked probe pose 或仿真 pose 的逆变换得到。
    pose_world_to_probe: torch.Tensor


@dataclass
class GaussianPileInput:
    # Gaussian 中心点，shape=(G,3)，world 坐标，单位 mm。
    means: torch.Tensor

    # Gaussian 三个主轴方向的标准差/尺度，shape=(G,3)，单位 mm。
    scales: torch.Tensor

    # Gaussian 旋转四元数，shape=(G,4)，通常为 (w,x,y,z)，无单位。
    quats: torch.Tensor

    # 每个 Gaussian 的贡献强度/透明度，shape=(G,) 或 (G,1)，数值通常在 [0,1]。
    # GaussianPile 不单独学习 intensity channel，主要由 alpha 控制加性贡献。
    alphas: torch.Tensor


@dataclass
class GaussianPileRenderSettings:
    image_height: int  # H，图像高度/axial 像素数，单位 pixel。
    image_width: int  # W，图像宽度/lateral 像素数，单位 pixel。
    lateral_spacing_mm_per_pix: float  # 相邻列像素的 lateral 物理间距，单位 mm/pixel。
    axial_spacing_mm_per_pix: float  # 相邻行像素的 axial/depth 物理间距，单位 mm/pixel。
    axial_near_mm: float  # 第 0 行像素对应的起始深度 near，单位 mm。
    lateral_col_offset: float = 0.0  # lateral=0 对应的列偏移；居中图像常用 (W-1)/2。
    backend: str = "ours_focus"  # renderer 后端名，通常保持默认。
    tile_size_px: int = 16  # CUDA tile 边长，单位 pixel。
    radius_sigma: float = 3.0  # 2D footprint 截断半径，单位为 Gaussian 标准差倍数。
    sigma_z_mm: float = 1.0  # elevational PSF/切片厚度近似，单位 mm；常取 delta_elevational_mm。
    prefilter_elevational_radius_mm: float = 0.0  # 可选粗筛半径，单位 mm；0 表示关闭。
    det_eps: float = 1.0e-6  # 2D covariance determinant 的数值稳定项，无物理单位。
    cov_eps: float = 1.0e-6  # covariance 数值稳定项，单位约为 mm^2。
    opacity_cull_threshold: float = 1.0e-6  # alpha 小于该值的 Gaussian 可被忽略，无单位。
    render_chunk_size: int = 250_000  # 一次送入 renderer 的 Gaussian 数量上限。
    enable_input_checks: bool = False  # 是否启用 renderer 输入合法性检查。


@dataclass
class RenderOutput:
    image: torch.Tensor  # 渲染图像，通常 shape=(1,H,W)，强度范围建议 [0,1]。
    depth: Optional[torch.Tensor] = None  # GaussianPile 无真实 depth，这里返回 zero tensor 保持接口统一。
    extras: Dict[str, torch.Tensor] = field(default_factory=dict)  # 额外诊断信息。


def render_gaussianpile(
    gaussians: GaussianPileInput,
    geometry: SliceGeometryInput,
    settings: GaussianPileRenderSettings,
) -> RenderOutput:
    """渲染一帧 ultrasound slice。

    输入 Gaussian 和当前帧几何信息，输出与输入图像网格对齐的单通道预测图。
    """

    cfg = GaussianPlieRenderConfig(
        H=int(settings.image_height),
        W=int(settings.image_width),
        lateral_spacing_mm_per_pix=float(settings.lateral_spacing_mm_per_pix),
        axial_spacing_mm_per_pix=float(settings.axial_spacing_mm_per_pix),
        axial_near_mm=float(settings.axial_near_mm),
        lateral_col_offset=float(settings.lateral_col_offset),
        backend=str(settings.backend),
        tile_size_px=int(settings.tile_size_px),
        radius_sigma=float(settings.radius_sigma),
        sigma_z_mm=float(settings.sigma_z_mm),
        prefilter_elevational_radius_mm=float(settings.prefilter_elevational_radius_mm),
        det_eps=float(settings.det_eps),
        cov_eps=float(settings.cov_eps),
        opacity_cull_threshold=float(settings.opacity_cull_threshold),
        render_chunk_size=int(settings.render_chunk_size),
        enable_input_checks=bool(settings.enable_input_checks),
    )
    renderer = GaussianPlieRenderer(cfg)
    grid = build_slice_grid_pack_from_settings(settings, gaussians.means.device)
    renderer.set_slice_grid_geometry(grid)

    kwargs = {
        "pixel_positions_world": geometry.pixel_positions_world,
        "origin_world": geometry.probe_origin_world,
        "axis_lateral_world": geometry.axis_lateral_world,
        "axis_axial_world": geometry.axis_axial_world,
        "axis_elevational_world": geometry.axis_elevational_world,
    }
    image = renderer(
        gaussians.means,
        gaussians.scales,
        gaussians.quats,
        gaussians.alphas,
        geometry.pose_world_to_probe,
        **kwargs,
    )
    depth = torch.zeros_like(image)
    return RenderOutput(image=image, depth=depth, extras={})


def build_slice_grid_pack_from_settings(settings: GaussianPileRenderSettings, device: torch.device) -> torch.Tensor:
    """构造 gaussianplie_renderer 需要的最小 slice grid。

    原项目中这个 helper 通常接收 DatasetMeta；这里用 SimpleNamespace 模拟所需字段。
    """

    meta = SimpleNamespace(
        H=settings.image_height,
        W=settings.image_width,
        delta_lateral_mm=settings.lateral_spacing_mm_per_pix,
        delta_axial_mm=settings.axial_spacing_mm_per_pix,
        near_mm=settings.axial_near_mm,
        center_lateral=settings.lateral_col_offset != 0.0,
    )
    return build_slice_grid_pack(meta, device=device)
