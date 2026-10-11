"""slice-adapted 3DGS 简化渲染接口。

这个 baseline 使用 3D Gaussian 几何，但渲染时按超声物理切片厚度筛选，
再把 Gaussian 投影成 2D ellipse 做 additive splat。所有空间单位默认是 mm。
"""

from dataclasses import dataclass, field
from typing import Dict, Optional

import torch


@dataclass
class SliceCameraInput:
    image_height: int  # H，图像高度/axial 像素数，单位 pixel。
    image_width: int  # W，图像宽度/lateral 像素数，单位 pixel。
    half_width_mm: float  # 切片平面横向半宽，单位 mm；通常约为 width_mm/2。
    half_height_mm: float  # 切片平面轴向半高，单位 mm；通常约为成像深度范围/2。
    focal_depth_mm: float  # camera space 中切片平面的 z 位置，单位 mm。
    world_view_transform: torch.Tensor  # world->camera 矩阵，shape=(4,4)，平移单位 mm。


@dataclass
class SliceAdapted3DGSInput:
    # Gaussian 中心点，shape=(G,3)，world 坐标，单位 mm。
    means: torch.Tensor

    # 3D Gaussian 三个主轴标准差/尺度，shape=(G,3)，单位 mm。
    scales_3d: torch.Tensor

    # Gaussian 旋转四元数，shape=(G,4)，无单位。
    quats: torch.Tensor

    # Gaussian opacity/alpha，shape=(G,) 或 (G,1)，通常在 [0,1]。
    opacities: torch.Tensor

    # Gaussian 颜色或灰度扩展成 RGB，shape=(G,3)，强度通常在 [0,1]。
    colors: torch.Tensor


@dataclass
class SliceAdaptedRenderSettings:
    slice_thickness_mm: float = 0.3  # 物理切片厚度/elevational 厚度近似，单位 mm；常取 delta_elevational_mm。
    cull_std: float = 3.0  # 保留距离切片平面 cull_std 个 sigma 内的 Gaussian。
    thickness_scale: float = 1.0  # 对切片厚度筛选半径的额外缩放，无单位。
    min_covariance: float = 1.0e-4  # covariance 下限，单位约 mm^2，用于数值稳定。
    normalize_by_footprint: bool = True  # 是否按 2D footprint 面积做归一化。
    clamp_output: bool = True  # 是否把输出图像 clamp 到 [0,1]。
    bg_color: float = 0.0  # 背景强度，通常在 [0,1]。
    output_grayscale: bool = True  # True 时把 RGB 输出转成单通道灰度。


@dataclass
class RenderOutput:
    image: torch.Tensor  # 渲染图像，shape=(1,H,W) 或 (3,H,W)。
    depth: Optional[torch.Tensor] = None  # 当前简化接口不输出真实 depth。
    extras: Dict[str, torch.Tensor] = field(default_factory=dict)  # 包含 visibility_filter 和 alpha_image。


def render_slice_adapted_3dgs(
    gaussians: SliceAdapted3DGSInput,
    camera: SliceCameraInput,
    settings: SliceAdaptedRenderSettings,
) -> RenderOutput:
    """渲染一张有限厚度物理切片。"""

    device = gaussians.means.device
    dtype = gaussians.means.dtype
    view = camera.world_view_transform.to(device=device, dtype=dtype)
    means_cam = _view_points(gaussians.means, view)
    rel_z = means_cam[:, 2] - float(camera.focal_depth_mm)

    cov_cam = _camera_covariances(gaussians.scales_3d, gaussians.quats, view)
    min_cov = float(settings.min_covariance)
    cov2d = cov_cam[:, :2, :2] + torch.eye(2, dtype=dtype, device=device).unsqueeze(0) * min_cov
    var_z = torch.clamp(cov_cam[:, 2, 2], min=min_cov)
    sigma_z = torch.sqrt(var_z + float(settings.slice_thickness_mm) ** 2)

    keep = torch.abs(rel_z) <= float(settings.cull_std) * sigma_z * float(settings.thickness_scale)
    if not bool(keep.any()):
        image = torch.full((3, camera.image_height, camera.image_width), float(settings.bg_color), device=device, dtype=dtype)
        return _finalize(image, None, keep, settings)

    means2d = means_cam[keep, :2]
    cov2d = cov2d[keep]
    rel_z = rel_z[keep]
    sigma_z = sigma_z[keep]
    colors = gaussians.colors[keep].to(dtype=dtype)
    opacities = gaussians.opacities.reshape(-1, 1)[keep].to(dtype=dtype)

    det = torch.clamp(cov2d[:, 0, 0] * cov2d[:, 1, 1] - cov2d[:, 0, 1] * cov2d[:, 1, 0], min=min_cov)
    inv = torch.empty_like(cov2d)
    inv[:, 0, 0] = cov2d[:, 1, 1] / det
    inv[:, 1, 1] = cov2d[:, 0, 0] / det
    inv[:, 0, 1] = -cov2d[:, 0, 1] / det
    inv[:, 1, 0] = -cov2d[:, 1, 0] / det

    weights = opacities * torch.exp(-0.5 * (rel_z / torch.clamp(sigma_z, min=1.0e-6)).square()).unsqueeze(-1)
    if settings.normalize_by_footprint:
        weights = weights / torch.sqrt(det).unsqueeze(-1)

    if device.type != "cuda" or dtype != torch.float32:
        raise RuntimeError("This simplified delivery renderer expects CUDA float32 tensors for additive_splat_bounded.")

    from slice_adapted_3dgs_renderer import additive_splat_bounded

    image_flat, alpha_flat = additive_splat_bounded(
        int(camera.image_height),
        int(camera.image_width),
        float(camera.half_width_mm),
        float(camera.half_height_mm),
        means2d,
        inv,
        weights,
        colors,
        float(settings.cull_std),
    )
    image_flat = image_flat + float(settings.bg_color)
    if settings.clamp_output:
        image_flat = image_flat.clamp(0.0, 1.0)
    image = image_flat.transpose(0, 1).reshape(3, int(camera.image_height), int(camera.image_width))
    alpha = alpha_flat.transpose(0, 1).reshape(1, int(camera.image_height), int(camera.image_width))
    return _finalize(image, alpha, keep, settings)


def _finalize(image: torch.Tensor, alpha: Optional[torch.Tensor], keep: torch.Tensor, settings: SliceAdaptedRenderSettings) -> RenderOutput:
    if settings.output_grayscale:
        image = (0.299 * image[0] + 0.587 * image[1] + 0.114 * image[2]).unsqueeze(0)
    return RenderOutput(image=image, depth=None, extras={"visibility_filter": keep, "alpha_image": alpha})


def _view_points(points: torch.Tensor, view: torch.Tensor) -> torch.Tensor:
    """把 world 坐标点变换到 camera 坐标。"""
    ones = torch.ones((points.shape[0], 1), dtype=points.dtype, device=points.device)
    return torch.cat([points, ones], dim=-1).matmul(view)[..., :3]


def _quat_to_matrix(q: torch.Tensor) -> torch.Tensor:
    """四元数转旋转矩阵。"""
    q = torch.nn.functional.normalize(q, dim=-1, eps=1.0e-8)
    w, x, y, z = q.unbind(-1)
    return torch.stack(
        [
            1 - 2 * (y * y + z * z),
            2 * (x * y - z * w),
            2 * (x * z + y * w),
            2 * (x * y + z * w),
            1 - 2 * (x * x + z * z),
            2 * (y * z - x * w),
            2 * (x * z - y * w),
            2 * (y * z + x * w),
            1 - 2 * (x * x + y * y),
        ],
        dim=-1,
    ).reshape(q.shape[:-1] + (3, 3))


def _camera_covariances(scales: torch.Tensor, quats: torch.Tensor, view: torch.Tensor) -> torch.Tensor:
    """根据 scale+quat 构造 world covariance，并变换到 camera 坐标系。"""
    rot = _quat_to_matrix(quats)
    cov_world = rot.matmul(torch.diag_embed(scales * scales)).matmul(rot.transpose(-1, -2))
    r_wc = view[:3, :3]
    return r_wc.transpose(0, 1).unsqueeze(0).matmul(cov_world).matmul(r_wc.unsqueeze(0))
