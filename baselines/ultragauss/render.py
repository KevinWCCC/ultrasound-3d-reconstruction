"""UltraGauss 简化渲染接口。

UltraGauss 是 probe-plane intersection 模型：每个像素在当前 probe plane
上取一个 3D world 坐标点，然后对候选 Gaussian 计算 3D Mahalanobis 响应。
所有空间长度单位默认是 mm。
"""

from dataclasses import dataclass, field
from typing import Dict, Optional

import torch

from ultragauss_renderer import SliceGeometry, UltraGaussRenderConfig, UltraGaussRenderer


@dataclass
class ProbeSliceInput:
    # 每个像素中心的 world-space 3D 坐标，shape=(H*W,3)，单位 mm。
    pixel_positions_world: torch.Tensor

    # 当前 probe/slice 原点，shape=(3,)，world 坐标，单位 mm。
    probe_origin_world: torch.Tensor

    # lateral 方向单位向量，shape=(3,)，无单位。
    axis_lateral_world: torch.Tensor

    # axial/depth 方向单位向量，shape=(3,)，无单位。
    axis_axial_world: torch.Tensor

    # elevational/out-of-plane 方向单位向量，shape=(3,)，无单位。
    axis_elevational_world: torch.Tensor


@dataclass
class UltraGaussInput:
    # Gaussian 中心，shape=(G,3)，world 坐标，单位 mm。
    means: torch.Tensor

    # 下三角 precision factor 的 6 个自由参数，shape=(G,6)。
    # 它定义 precision matrix Lambda=L L^T；precision 单位约为 1/mm^2。
    # 与 scale+quat 不同，UltraGauss 直接学习 precision 参数化。
    precision_tril: torch.Tensor

    # Gaussian 最大响应/贡献权重 alpha，shape=(G,)，数值通常在 [0,1]。
    opacities: torch.Tensor

    # Gaussian 灰度强度 c，shape=(G,)，数值通常在 [0,1]。
    intensities: torch.Tensor


@dataclass
class UltraGaussRenderSettings:
    image_height: int  # H，图像高度/axial 像素数，单位 pixel。
    image_width: int  # W，图像宽度/lateral 像素数，单位 pixel。
    lateral_spacing_mm_per_pix: float  # lateral 像素间距，单位 mm/pixel。
    axial_spacing_mm_per_pix: float  # axial/depth 像素间距，单位 mm/pixel。
    axial_near_mm: float  # 第 0 行像素的起始深度，单位 mm。
    lateral_col_offset: float = 0.0  # lateral=0 的列偏移，居中图像常用 (W-1)/2。
    chi2_threshold: float = 7.815  # Gaussian 有效范围的卡方阈值；7.815 约等于 3D 95% 椭球。
    mapper_sort_by_depth: bool = True  # 候选 Gaussian 是否按深度排序，影响可选 shadow 近似。
    fixed_bg_alpha: float = 1.0e-6  # 背景 alpha，避免归一化除零，无单位。
    fixed_bg_intensity: float = 0.0  # 背景灰度强度，通常在 [0,1]。
    diag_beta: float = 1.0e-4  # precision_tril 对角正值偏置，保证 precision 正定。
    enable_shadow: bool = False  # 是否启用简化 shadow/累计衰减近似。
    shadow_delta: float = 0.0  # shadow 强度参数，无单位。


@dataclass
class RenderOutput:
    image: torch.Tensor  # 渲染图像，通常 shape=(1,H,W) 或 (H,W)。
    depth: Optional[torch.Tensor] = None  # renderer 返回的 depth-like 辅助图。
    extras: Dict[str, torch.Tensor] = field(default_factory=dict)  # 包含 alpha_image 和中间诊断信息。


def render_ultragauss(
    gaussians: UltraGaussInput,
    geometry: ProbeSliceInput,
    settings: UltraGaussRenderSettings,
) -> RenderOutput:
    """使用 UltraGauss 渲染一帧 ultrasound slice。"""

    renderer = UltraGaussRenderer(
        UltraGaussRenderConfig(
            H=int(settings.image_height),
            W=int(settings.image_width),
            lateral_spacing_mm_per_pix=float(settings.lateral_spacing_mm_per_pix),
            axial_spacing_mm_per_pix=float(settings.axial_spacing_mm_per_pix),
            axial_near_mm=float(settings.axial_near_mm),
            lateral_col_offset=float(settings.lateral_col_offset),
            chi2_threshold=float(settings.chi2_threshold),
            mapper_sort_by_depth=bool(settings.mapper_sort_by_depth),
            fixed_bg_alpha=float(settings.fixed_bg_alpha),
            fixed_bg_intensity=float(settings.fixed_bg_intensity),
            diag_beta=float(settings.diag_beta),
            enable_shadow=bool(settings.enable_shadow),
            shadow_delta=float(settings.shadow_delta),
        )
    )
    image, alpha, depth, intermediates = renderer(
        gauss_means_world=gaussians.means,
        gauss_precision_tril=gaussians.precision_tril,
        gauss_opacities=gaussians.opacities,
        gauss_intensities=gaussians.intensities,
        pixel_positions_world=geometry.pixel_positions_world,
        geometry=SliceGeometry(
            origin_world=geometry.probe_origin_world,
            axis_lateral_world=geometry.axis_lateral_world,
            axis_axial_world=geometry.axis_axial_world,
            axis_elevational_world=geometry.axis_elevational_world,
        ),
        return_intermediates=True,
    )
    intermediates["alpha_image"] = alpha
    return RenderOutput(image=image, depth=depth, extras=intermediates)
