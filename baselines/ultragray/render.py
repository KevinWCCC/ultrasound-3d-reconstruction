"""UltraGray / UltraG-Ray 简化渲染接口。

UltraGray 是 ray casting 模型：每个像素沿 acoustic ray 计算 transmission，
再乘以该 3D 位置处的 echo intensity。所有空间长度单位默认是 mm。
"""

from dataclasses import dataclass, field
from typing import Dict, Optional

import torch

from ultragray_renderer import ProbeGeometry, UltragrayRenderConfig, UltragrayRenderer


@dataclass
class RaySliceInput:
    # 每个像素中心的 3D 位置，renderer 原生布局通常为 (W,H,3)，单位 mm。
    # 需要和 ray_origins/ray_dirs 满足 pixel ~= origin + depth * dir。
    pixel_positions: torch.Tensor

    # 每条 lateral scan line 的 ray 起点，shape=(W,3)，world 坐标，单位 mm。
    ray_origins: torch.Tensor

    # 每条 scan line 的 ray 方向，shape=(W,3)，单位向量，无单位。
    ray_dirs: torch.Tensor

    # probe/transducer 原点，shape=(3,)，world 坐标，单位 mm。
    probe_origin_world: torch.Tensor

    # lateral 方向单位向量，shape=(3,)，无单位。
    axis_lateral_world: torch.Tensor

    # axial/depth 方向单位向量，shape=(3,)，无单位。
    axis_axial_world: torch.Tensor

    # elevational/out-of-plane 方向单位向量，shape=(3,)，无单位。
    axis_elevational_world: torch.Tensor


@dataclass
class UltraGrayInput:
    # Gaussian 中心点，shape=(G,3)，world 坐标，单位 mm。
    means: torch.Tensor

    # Gaussian 三个主轴尺度/标准差，shape=(G,3)，单位 mm。
    scales: torch.Tensor

    # Gaussian 旋转四元数，shape=(G,4)，无单位。
    quats: torch.Tensor

    # transmission/attenuation 的 raw logits，shape=(G,)。
    # renderer 内部通常通过 sigmoid 映射成 tau in [0,1]。
    attenuations: torch.Tensor

    # degree-1 灰度 SH 系数，shape=(G,4)，对应 c0+c1*dx+c2*dy+c3*dz。
    sh_coeffs: torch.Tensor


@dataclass
class UltraGrayRenderSettings:
    image_width: int  # W，图像 lateral 像素数，单位 pixel。
    image_height: int  # H，图像 axial/depth 像素数，单位 pixel。
    width_mm: float  # 图像横向/lateral 视野宽度，单位 mm。
    near_mm: float  # 第 0 行像素的起始深度，单位 mm。
    lateral_col_offset: float = 0.0  # lateral=0 的列偏移，居中图像常用 (W-1)/2。
    elev_cutoff_margin_mm: float = 0.0  # elevational 剔除时额外保留 margin，单位 mm。
    k_sigma: float = 3.0  # mapper 中 Gaussian 安全半径倍数，单位为标准差倍数。
    ops_max_offset_mm: float = 0.0  # OPS ray origin elevational 随机扰动最大幅度，单位 mm；0 表示关闭。
    quadrature_window_radius_mm: float = 2.0  # ray-Gaussian 积分窗口半径，单位 mm。
    local_transmission_window_only: bool = False  # 是否只在局部窗口内累计 transmission。
    local_transmission_max_window_mm: float = 17.0  # local transmission 最大窗口长度，单位 mm。


@dataclass
class RenderOutput:
    image: torch.Tensor  # 渲染 B-mode 图像，通常 shape=(H,W)。
    depth: Optional[torch.Tensor] = None  # UltraGray 当前不输出 depth。
    extras: Dict[str, torch.Tensor] = field(default_factory=dict)  # 预留诊断信息。


def render_ultragray(
    gaussians: UltraGrayInput,
    geometry: RaySliceInput,
    settings: UltraGrayRenderSettings,
) -> RenderOutput:
    """使用 ray attenuation 和 echo intensity 渲染一帧超声图像。"""

    renderer = UltragrayRenderer(
        UltragrayRenderConfig(
            image_width=int(settings.image_width),
            image_height=int(settings.image_height),
            width_mm=float(settings.width_mm),
            near_mm=float(settings.near_mm),
            lateral_col_offset=float(settings.lateral_col_offset),
            elev_cutoff_margin_mm=float(settings.elev_cutoff_margin_mm),
            k_sigma=float(settings.k_sigma),
            ops_max_offset_mm=float(settings.ops_max_offset_mm),
            quadrature_window_radius_mm=float(settings.quadrature_window_radius_mm),
            local_transmission_window_only=bool(settings.local_transmission_window_only),
            local_transmission_max_window_mm=float(settings.local_transmission_max_window_mm),
        )
    )
    image = renderer(
        means=gaussians.means,
        scales=gaussians.scales,
        quats=gaussians.quats,
        attenuations=gaussians.attenuations,
        sh_coeffs=gaussians.sh_coeffs,
        pixel_positions=geometry.pixel_positions,
        ray_origins=geometry.ray_origins,
        ray_dirs=geometry.ray_dirs,
        geometry=ProbeGeometry(
            origin_world=geometry.probe_origin_world,
            axis_lateral_world=geometry.axis_lateral_world,
            axis_elevational_world=geometry.axis_elevational_world,
            axis_axial_world=geometry.axis_axial_world,
        ),
    )
    return RenderOutput(image=image, depth=None, extras={})
