"""UltraGS baseline 简化渲染接口。

这个 baseline 使用 UltraGS/3DGS 风格 rasterizer。它需要外部代码提供
已经和超声物理切片对齐的 virtual camera 参数。所有空间平移单位默认是 mm。
"""

from dataclasses import dataclass, field
from typing import Dict, Optional

import torch


@dataclass
class UltraGSCameraInput:
    image_height: int  # H，图像高度，单位 pixel。
    image_width: int  # W，图像宽度，单位 pixel。
    tanfovx: float  # tan(FoVx/2)，水平视场半角正切值，无单位。
    tanfovy: float  # tan(FoVy/2)，垂直视场半角正切值，无单位。
    viewmatrix: torch.Tensor  # world->camera 矩阵，shape=(4,4)，平移单位 mm。
    projmatrix: torch.Tensor  # camera->clip/screen 投影矩阵，shape=(4,4)，无单位。
    camera_center: torch.Tensor  # camera center 的 world 坐标，shape=(3,)，单位 mm。


@dataclass
class UltraGSBaselineInput:
    # Gaussian 中心点，shape=(G,3)，world 坐标，单位 mm。
    means: torch.Tensor

    # UltraGS/3DGS rasterizer 使用的 2D disk scale，shape=(G,2)，单位随 renderer 相机尺度定义。
    # 在本项目超声适配里它表示投影平面上的横向/纵向 splat 尺度。
    scales_2d: torch.Tensor

    # disk/Gaussian 旋转四元数，shape=(G,4)，无单位。
    quats: torch.Tensor

    # opacity/alpha，shape=(G,) 或 (G,1)，通常在 [0,1]。
    opacities: torch.Tensor

    # spherical harmonics 系数，布局需符合 ultrags_baseline_renderer 的 shs 输入要求。
    sh_coeffs: torch.Tensor

    # 当前启用的 SH degree；灰度或 DC-only 时通常为 0。
    active_sh_degree: int = 0


@dataclass
class UltraGSRenderSettings:
    bg_color: float = 0.0  # 背景强度，通常在 [0,1]。
    scale_modifier: float = 1.0  # 全局 scale 缩放，无单位；通常保持 1。
    output_grayscale: bool = True  # True 时将 RGB rasterizer 输出转成单通道灰度。
    debug: bool = False  # 是否启用 renderer debug 分支。


@dataclass
class RenderOutput:
    image: torch.Tensor  # 渲染图像，shape=(1,H,W) 或 (3,H,W)。
    depth: Optional[torch.Tensor] = None  # allmap 中的 expected depth，若 renderer 提供。
    extras: Dict[str, torch.Tensor] = field(default_factory=dict)  # radii、alpha_image、allmap 等诊断信息。


def render_ultrags_baseline(
    gaussians: UltraGSBaselineInput,
    camera: UltraGSCameraInput,
    settings: UltraGSRenderSettings,
) -> RenderOutput:
    """使用 UltraGS/3DGS 风格 rasterizer 渲染一帧。"""

    from ultrags_baseline_renderer import GaussianRasterizationSettings, GaussianRasterizer

    device = gaussians.means.device
    screenspace_points = torch.zeros_like(gaussians.means, requires_grad=True, device=device)
    try:
        screenspace_points.retain_grad()
    except RuntimeError:
        pass

    background = torch.full((3,), float(settings.bg_color), dtype=torch.float32, device=device)
    raster_settings = GaussianRasterizationSettings(
        image_height=int(camera.image_height),
        image_width=int(camera.image_width),
        tanfovx=float(camera.tanfovx),
        tanfovy=float(camera.tanfovy),
        bg=background,
        scale_modifier=float(settings.scale_modifier),
        viewmatrix=camera.viewmatrix,
        projmatrix=camera.projmatrix,
        sh_degree=int(gaussians.active_sh_degree),
        campos=camera.camera_center,
        prefiltered=False,
        debug=bool(settings.debug),
    )
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    rendered_rgb, radii, allmap = rasterizer(
        means3D=gaussians.means,
        means2D=screenspace_points,
        shs=gaussians.sh_coeffs,
        colors_precomp=None,
        opacities=gaussians.opacities,
        scales=gaussians.scales_2d,
        rotations=gaussians.quats,
        cov3D_precomp=None,
    )
    image = rendered_rgb
    if settings.output_grayscale:
        image = (0.299 * rendered_rgb[0] + 0.587 * rendered_rgb[1] + 0.114 * rendered_rgb[2]).unsqueeze(0)

    alpha = allmap[1:2] if allmap.shape[0] > 1 else None
    depth = allmap[0:1] if allmap.shape[0] > 0 else None
    if depth is not None and alpha is not None:
        depth = torch.nan_to_num(depth / torch.clamp(alpha, min=1.0e-8), 0.0, 0.0)

    return RenderOutput(
        image=image,
        depth=depth,
        extras={
            "viewspace_points": screenspace_points,
            "radii": radii,
            "alpha_image": alpha,
            "allmap": allmap,
        },
    )
