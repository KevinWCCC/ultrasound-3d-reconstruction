"""一个故意塞在单文件里的 baseline finetune pipeline。

这个文件的目标是让外部合作者不依赖原项目的 engine/scene/Hydra/callback
框架，也能跑通一个最小训练闭环。代码组织故意扁平：配置、数据读取、
Gaussian 参数、渲染分发、loss、callbacks、checkpoint、CLI 全都在这里。

数据入口是一个 .npz 文件，格式见 DATA_FORMAT.md。
所有空间单位默认是 mm，图像强度默认是 [0, 1]。
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parent


def _prepend_local_paths() -> None:
    """把本交付目录里的算子包加入 PYTHONPATH。"""

    candidates = [
        ROOT,
        ROOT / "baselines" / "gaussianpile",
        ROOT / "baselines" / "slice_adapted_3dgs",
        ROOT / "baselines" / "ultrags_baseline",
        ROOT / "baselines" / "ultragauss",
        ROOT / "baselines" / "ultragray",
        ROOT / "ops" / "gaussianplie_renderer",
        ROOT / "ops" / "slice_adapted_3dgs_renderer",
        ROOT / "ops" / "ultrags_baseline_renderer",
        ROOT / "ops" / "ultragauss_renderer",
        ROOT / "ops" / "ultragray_renderer",
    ]
    for path in reversed(candidates):
        s = str(path)
        if s not in sys.path:
            sys.path.insert(0, s)


_prepend_local_paths()


@dataclass
class GodConfig:
    # 选择 baseline: gaussianpile, slice_adapted_3dgs, ultrags_baseline, ultragauss, ultragray
    method: str = "ultragauss"

    # 输入 .npz 路径；格式见 DATA_FORMAT.md。
    data_npz: str = "dataset.npz"

    # 输出目录，保存 config、日志、checkpoint。
    output_dir: str = "runs/debug"

    # cuda 或 cpu。注意多数 renderer 需要 CUDA 扩展，正式训练建议 cuda。
    device: str = "cuda"

    # 随机种子。
    seed: int = 42

    # 训练迭代次数。每次迭代随机取一帧。
    iterations: int = 1000

    # 日志、验证、保存间隔。
    log_every: int = 10
    eval_every: int = 100
    save_every: int = 500

    # 初始 Gaussian 数量；如果 npz 有 init_means，会优先使用 init_means。
    num_init: int = 100000

    # 全部参数共用一个 Adam learning rate；这是最小 pipeline，不做复杂 param group。
    learning_rate: float = 1.0e-3

    # loss = loss_l1_weight * L1 + loss_ssim_weight * (1 - SSIM)
    loss_l1_weight: float = 1.0
    loss_ssim_weight: float = 0.0

    # train/test split 比例。
    train_fraction: float = 0.8

    # 图像尺寸。如果 npz images 已提供，会用 images 的 H/W 覆盖。
    image_height: int = 256
    image_width: int = 256

    # 物理像素间距和深度范围，单位 mm。
    delta_lateral_mm: float = 0.2
    delta_axial_mm: float = 0.2
    delta_elevational_mm: float = 0.5
    near_mm: float = 0.0
    far_mm: float = 51.2
    width_mm: float = 51.2
    center_lateral: bool = True

    # 通用初始化尺度和 opacity。
    init_scale_mm: float = 0.5
    init_opacity: float = 0.1
    init_intensity: float = 0.5

    # GaussianPile / slice renderer 相关。
    sigma_z_mm: Optional[float] = None
    slice_thickness_mm: Optional[float] = None

    # UltraGauss 相关。
    ultragauss_chi2_threshold: float = 7.815
    ultragauss_diag_beta: float = 1.0e-4

    # UltraGray 相关。
    ultragray_k_sigma: float = 3.0
    ultragray_ops_max_offset_mm: float = 0.0
    ultragray_quadrature_window_radius_mm: float = 2.0

    # UltraGS baseline 相关。
    ultrags_tanfovx: float = 0.5
    ultrags_tanfovy: float = 0.5
    ultrags_active_sh_degree: int = 0


def load_config(path: Optional[str], cli_overrides: List[str]) -> GodConfig:
    data: Dict[str, Any] = {}
    if path:
        with open(path, "r", encoding="utf-8") as f:
            data.update(json.load(f))
    for item in cli_overrides:
        if "=" not in item:
            raise ValueError(f"override 必须是 key=value 格式，收到: {item}")
        k, v = item.split("=", 1)
        data[k] = _parse_scalar(v)
    cfg = GodConfig(**data)
    if cfg.sigma_z_mm is None:
        cfg.sigma_z_mm = cfg.delta_elevational_mm
    if cfg.slice_thickness_mm is None:
        cfg.slice_thickness_mm = cfg.delta_elevational_mm
    return cfg


def _parse_scalar(v: str) -> Any:
    if v.lower() in {"true", "false"}:
        return v.lower() == "true"
    try:
        if "." in v or "e" in v.lower():
            return float(v)
        return int(v)
    except ValueError:
        return v


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class NpzUltrasoundDataset:
    """极简 .npz dataloader。

    不继承原项目 Dataset/Scene；只把 npz 数组转成当前 pipeline 需要的 frame dict。
    """

    def __init__(self, cfg: GodConfig, split: str):
        self.cfg = cfg
        self.split = split
        self.raw = np.load(cfg.data_npz, allow_pickle=False)
        images = self.raw["images"].astype(np.float32)
        if images.max() > 1.5:
            # 兼容常见 uint8 超声图像。保存为 uint8 可显著减小示例数据体积。
            images = images / 255.0
        if images.ndim == 3:
            images = images[:, None, :, :]
        if images.ndim != 4:
            raise ValueError("images 必须是 (N,H,W) 或 (N,1,H,W)")
        self.images = images
        self.n, _, self.h, self.w = images.shape
        cfg.image_height = int(self.h)
        cfg.image_width = int(self.w)

        all_indices = np.arange(self.n)
        split_at = max(1, min(self.n - 1, int(round(self.n * float(cfg.train_fraction))))) if self.n > 1 else 1
        self.indices = all_indices[:split_at] if split == "train" else all_indices[split_at:]
        if len(self.indices) == 0:
            self.indices = all_indices[:1]

    def __len__(self) -> int:
        return int(len(self.indices))

    def random_frame(self, device: torch.device) -> Dict[str, torch.Tensor]:
        idx = int(np.random.choice(self.indices))
        return self.get_frame(idx, device)

    def iter_eval(self, device: torch.device, max_frames: int = 8) -> Iterable[Dict[str, torch.Tensor]]:
        for idx in self.indices[:max_frames]:
            yield self.get_frame(int(idx), device)

    def get_frame(self, idx: int, device: torch.device) -> Dict[str, torch.Tensor]:
        cfg = self.cfg
        image = torch.from_numpy(self.images[idx]).to(device=device)
        frame: Dict[str, torch.Tensor] = {"image": image, "frame_index": torch.tensor(idx, device=device)}

        origin = self._array_or_default("probe_origin_world", idx, np.zeros(3, dtype=np.float32))
        lat = self._array_or_default("axis_lateral_world", idx, np.array([1, 0, 0], dtype=np.float32))
        ax = self._array_or_default("axis_axial_world", idx, np.array([0, 1, 0], dtype=np.float32))
        elev = self._array_or_default("axis_elevational_world", idx, np.array([0, 0, 1], dtype=np.float32))
        origin_t = torch.from_numpy(origin).to(device=device, dtype=torch.float32)
        lat_t = _normalize(torch.from_numpy(lat).to(device=device, dtype=torch.float32))
        ax_t = _normalize(torch.from_numpy(ax).to(device=device, dtype=torch.float32))
        elev_t = _normalize(torch.from_numpy(elev).to(device=device, dtype=torch.float32))

        pix = self._pixel_positions(idx, origin_t, lat_t, ax_t, device)
        frame.update(
            {
                "pixel_positions_world": pix,
                "probe_origin_world": origin_t,
                "axis_lateral_world": lat_t,
                "axis_axial_world": ax_t,
                "axis_elevational_world": elev_t,
                "pose_world_to_probe": self._pose_world_to_probe(idx, origin_t, lat_t, ax_t, elev_t, device),
                "ray_origins": self._ray_origins(idx, origin_t, lat_t, device),
                "ray_dirs": self._ray_dirs(idx, ax_t, device),
                "viewmatrix": self._matrix_or_default("viewmatrix", idx, device),
                "projmatrix": self._matrix_or_default("projmatrix", idx, device),
                "camera_center": self._array_or_default_torch("camera_center", idx, origin_t.detach().cpu().numpy(), device),
                "tanfovx": self._scalar_or_default("tanfovx", idx, cfg.ultrags_tanfovx, device),
                "tanfovy": self._scalar_or_default("tanfovy", idx, cfg.ultrags_tanfovy, device),
            }
        )
        return frame

    def initial_points(self, device: torch.device, num_points: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if "init_means" in self.raw:
            means = torch.from_numpy(self.raw["init_means"].astype(np.float32)).to(device)
            if "init_intensities" in self.raw:
                inten = torch.from_numpy(self.raw["init_intensities"].astype(np.float32)).to(device).reshape(-1, 1)
            else:
                inten = torch.full((means.shape[0], 1), float(self.cfg.init_intensity), device=device)
            return means, inten

        # 从训练帧像素位置随机采样。这样至少和数据物理范围一致。
        samples: List[torch.Tensor] = []
        vals: List[torch.Tensor] = []
        while sum(x.shape[0] for x in samples) < num_points:
            frame = self.random_frame(device)
            pix = frame["pixel_positions_world"]
            img = frame["image"].reshape(-1, 1)
            take = min(num_points - sum(x.shape[0] for x in samples), pix.shape[0])
            ids = torch.randint(0, pix.shape[0], (take,), device=device)
            elev_noise = torch.randn((take, 1), device=device) * (0.25 * float(self.cfg.delta_elevational_mm))
            samples.append(pix.index_select(0, ids) + elev_noise * frame["axis_elevational_world"].view(1, 3))
            vals.append(img.index_select(0, ids).clamp(0.0, 1.0))
        return torch.cat(samples, dim=0), torch.cat(vals, dim=0)

    def _array_or_default(self, name: str, idx: int, default: np.ndarray) -> np.ndarray:
        if name not in self.raw:
            return default.astype(np.float32)
        arr = self.raw[name].astype(np.float32)
        return arr[idx] if arr.ndim >= 2 and arr.shape[0] == self.n else arr

    def _array_or_default_torch(self, name: str, idx: int, default: np.ndarray, device: torch.device) -> torch.Tensor:
        return torch.from_numpy(self._array_or_default(name, idx, default)).to(device=device, dtype=torch.float32)

    def _scalar_or_default(self, name: str, idx: int, default: float, device: torch.device) -> torch.Tensor:
        if name not in self.raw:
            return torch.tensor(float(default), device=device)
        arr = self.raw[name].astype(np.float32)
        val = arr[idx] if arr.ndim > 0 and arr.shape[0] == self.n else arr
        return torch.tensor(float(val), device=device)

    def _matrix_or_default(self, name: str, idx: int, device: torch.device) -> torch.Tensor:
        if name not in self.raw:
            return torch.eye(4, device=device, dtype=torch.float32)
        arr = self.raw[name].astype(np.float32)
        mat = arr[idx] if arr.ndim == 3 and arr.shape[0] == self.n else arr
        return torch.from_numpy(mat).to(device=device, dtype=torch.float32)

    def _pixel_positions(self, idx: int, origin: torch.Tensor, lat: torch.Tensor, ax: torch.Tensor, device: torch.device) -> torch.Tensor:
        if "pixel_positions_world" in self.raw:
            arr = self.raw["pixel_positions_world"].astype(np.float32)
            pix = arr[idx]
            pix = pix.reshape(-1, 3)
            return torch.from_numpy(pix).to(device=device, dtype=torch.float32)

        u = torch.arange(self.w, device=device, dtype=torch.float32)
        v = torch.arange(self.h, device=device, dtype=torch.float32)
        center = (self.w - 1) * 0.5 if self.cfg.center_lateral else 0.0
        lateral_mm = (u - center) * float(self.cfg.delta_lateral_mm)
        axial_mm = float(self.cfg.near_mm) + v * float(self.cfg.delta_axial_mm)
        yy, xx = torch.meshgrid(axial_mm, lateral_mm, indexing="ij")
        pix = origin.view(1, 1, 3) + xx[..., None] * lat.view(1, 1, 3) + yy[..., None] * ax.view(1, 1, 3)
        return pix.reshape(-1, 3).contiguous()

    def _pose_world_to_probe(self, idx: int, origin: torch.Tensor, lat: torch.Tensor, ax: torch.Tensor, elev: torch.Tensor, device: torch.device) -> torch.Tensor:
        if "pose_world_to_probe" in self.raw:
            arr = self.raw["pose_world_to_probe"].astype(np.float32)
            mat = arr[idx] if arr.ndim == 3 and arr.shape[0] == self.n else arr
            return torch.from_numpy(mat).to(device=device, dtype=torch.float32)
        r = torch.stack([lat, ax, elev], dim=0)
        t = -r.matmul(origin)
        pose = torch.eye(4, device=device, dtype=torch.float32)
        pose[:3, :3] = r
        pose[:3, 3] = t
        return pose

    def _ray_origins(self, idx: int, origin: torch.Tensor, lat: torch.Tensor, device: torch.device) -> torch.Tensor:
        if "ray_origins" in self.raw:
            arr = self.raw["ray_origins"].astype(np.float32)
            return torch.from_numpy(arr[idx]).to(device=device, dtype=torch.float32)
        u = torch.arange(self.w, device=device, dtype=torch.float32)
        center = (self.w - 1) * 0.5 if self.cfg.center_lateral else 0.0
        lateral_mm = (u - center) * float(self.cfg.delta_lateral_mm)
        return origin.view(1, 3) + lateral_mm[:, None] * lat.view(1, 3)

    def _ray_dirs(self, idx: int, ax: torch.Tensor, device: torch.device) -> torch.Tensor:
        if "ray_dirs" in self.raw:
            arr = self.raw["ray_dirs"].astype(np.float32)
            return torch.from_numpy(arr[idx]).to(device=device, dtype=torch.float32)
        return ax.view(1, 3).expand(self.w, 3).contiguous()


def _normalize(x: torch.Tensor) -> torch.Tensor:
    return x / torch.clamp(torch.linalg.norm(x), min=1.0e-8)


class GaussianParameters(nn.Module):
    """五个 baseline 共用的粗暴参数容器。"""

    def __init__(self, cfg: GodConfig, init_means: torch.Tensor, init_intensities: torch.Tensor):
        super().__init__()
        self.cfg = cfg
        n = init_means.shape[0]
        device = init_means.device
        self.means = nn.Parameter(init_means.float())
        self.raw_scales_3d = nn.Parameter(torch.full((n, 3), math.log(float(cfg.init_scale_mm)), device=device))
        self.raw_scales_2d = nn.Parameter(torch.full((n, 2), math.log(float(cfg.init_scale_mm)), device=device))
        quat = torch.zeros((n, 4), device=device)
        quat[:, 0] = 1.0
        self.raw_quats = nn.Parameter(quat)
        self.raw_opacities = nn.Parameter(_logit(torch.full((n, 1), float(cfg.init_opacity), device=device)))
        self.raw_intensities = nn.Parameter(_logit(init_intensities.float().reshape(n, 1).clamp(1.0e-4, 1.0 - 1.0e-4)))
        self.raw_precision_tril = nn.Parameter(_init_precision_tril(n, cfg, device))
        self.sh_gray4 = nn.Parameter(torch.cat([init_intensities.float().reshape(n, 1), torch.zeros((n, 3), device=device)], dim=1))
        self.sh_ultrags = nn.Parameter(_init_ultrags_sh(init_intensities.float().reshape(n, 1), device, degree=max(0, cfg.ultrags_active_sh_degree)))

    def scales_3d(self) -> torch.Tensor:
        return torch.exp(self.raw_scales_3d).clamp_min(1.0e-5)

    def scales_2d(self) -> torch.Tensor:
        return torch.exp(self.raw_scales_2d).clamp_min(1.0e-5)

    def quats(self) -> torch.Tensor:
        return F.normalize(self.raw_quats, dim=-1, eps=1.0e-8)

    def opacities(self) -> torch.Tensor:
        return torch.sigmoid(self.raw_opacities)

    def intensities(self) -> torch.Tensor:
        return torch.sigmoid(self.raw_intensities)


def _logit(x: torch.Tensor) -> torch.Tensor:
    x = x.clamp(1.0e-6, 1.0 - 1.0e-6)
    return torch.log(x / (1.0 - x))


def _init_precision_tril(n: int, cfg: GodConfig, device: torch.device) -> torch.Tensor:
    # 6 参数顺序沿用 UltraGauss: l11,l12,l13,l22,l23,l33。
    # 对角 raw 越大，precision 越大，Gaussian 越窄。
    inv_scale = 1.0 / max(float(cfg.init_scale_mm), 1.0e-4)
    raw_diag = math.sqrt(max(inv_scale - float(cfg.ultragauss_diag_beta), 1.0e-4))
    out = torch.zeros((n, 6), device=device)
    out[:, 0] = raw_diag
    out[:, 3] = raw_diag
    out[:, 5] = raw_diag
    return out


def _init_ultrags_sh(intensities: torch.Tensor, device: torch.device, degree: int) -> torch.Tensor:
    sh_c0 = 0.28209479177387814
    coeffs = (degree + 1) ** 2
    rgb = intensities.clamp(0.0, 1.0).expand(-1, 3)
    sh = torch.zeros((intensities.shape[0], coeffs, 3), device=device)
    sh[:, 0, :] = (rgb - 0.5) / sh_c0
    return sh


class RenderDispatcher:
    """把通用 GaussianParameters 转成每个 baseline 的 render dataclass。"""

    def __init__(self, cfg: GodConfig):
        self.cfg = cfg
        self.method = cfg.method
        self.mod = self._load_module(self.method)

    def _load_module(self, method: str) -> Any:
        if method == "ultragauss" and not torch.cuda.is_available():
            print("[pipeline] CUDA 不可用，UltraGauss 将使用纯 PyTorch fallback renderer。", flush=True)
            return None
        table = {
            "gaussianpile": (ROOT / "baselines" / "gaussianpile" / "render.py", "gp_render_delivery"),
            "slice_adapted_3dgs": (ROOT / "baselines" / "slice_adapted_3dgs" / "render.py", "sa3dgs_render_delivery"),
            "ultrags_baseline": (ROOT / "baselines" / "ultrags_baseline" / "render.py", "ultrags_base_render_delivery"),
            "ultragauss": (ROOT / "baselines" / "ultragauss" / "render.py", "ug_render_delivery"),
            "ultragray": (ROOT / "baselines" / "ultragray" / "render.py", "ugray_render_delivery"),
        }
        if method not in table:
            raise ValueError(f"未知 method: {method}")
        path, name = table[method]
        try:
            return _load_py(path, name)
        except ImportError as exc:
            if method == "ultragauss":
                print(f"[pipeline] UltraGauss CUDA renderer 未加载，改用 PyTorch fallback: {exc}", flush=True)
                return None
            raise

    def render(self, params: GaussianParameters, frame: Dict[str, torch.Tensor]):
        cfg = self.cfg
        m = self.mod
        if self.method == "gaussianpile":
            return m.render_gaussianpile(
                m.GaussianPileInput(params.means, params.scales_3d(), params.quats(), params.opacities().reshape(-1)),
                m.SliceGeometryInput(
                    frame["pixel_positions_world"],
                    frame["probe_origin_world"],
                    frame["axis_lateral_world"],
                    frame["axis_axial_world"],
                    frame["axis_elevational_world"],
                    frame["pose_world_to_probe"],
                ),
                m.GaussianPileRenderSettings(
                    image_height=cfg.image_height,
                    image_width=cfg.image_width,
                    lateral_spacing_mm_per_pix=cfg.delta_lateral_mm,
                    axial_spacing_mm_per_pix=cfg.delta_axial_mm,
                    axial_near_mm=cfg.near_mm,
                    lateral_col_offset=((cfg.image_width - 1) * 0.5 if cfg.center_lateral else 0.0),
                    sigma_z_mm=float(cfg.sigma_z_mm),
                ),
            )
        if self.method == "slice_adapted_3dgs":
            return m.render_slice_adapted_3dgs(
                m.SliceAdapted3DGSInput(params.means, params.scales_3d(), params.quats(), params.opacities(), params.intensities().expand(-1, 3)),
                m.SliceCameraInput(
                    image_height=cfg.image_height,
                    image_width=cfg.image_width,
                    half_width_mm=0.5 * float(cfg.width_mm),
                    half_height_mm=0.5 * max(float(cfg.far_mm - cfg.near_mm), cfg.image_height * cfg.delta_axial_mm),
                    focal_depth_mm=0.0,
                    world_view_transform=frame["pose_world_to_probe"].transpose(0, 1),
                ),
                m.SliceAdaptedRenderSettings(slice_thickness_mm=float(cfg.slice_thickness_mm)),
            )
        if self.method == "ultrags_baseline":
            return m.render_ultrags_baseline(
                m.UltraGSBaselineInput(
                    params.means,
                    params.scales_2d(),
                    params.quats(),
                    params.opacities(),
                    params.sh_ultrags,
                    active_sh_degree=int(cfg.ultrags_active_sh_degree),
                ),
                m.UltraGSCameraInput(
                    image_height=cfg.image_height,
                    image_width=cfg.image_width,
                    tanfovx=float(frame["tanfovx"].detach().item()),
                    tanfovy=float(frame["tanfovy"].detach().item()),
                    viewmatrix=frame["viewmatrix"],
                    projmatrix=frame["projmatrix"],
                    camera_center=frame["camera_center"],
                ),
                m.UltraGSRenderSettings(),
            )
        if self.method == "ultragauss":
            if m is None or params.means.device.type != "cuda":
                return _render_ultragauss_torch_fallback(params, frame, cfg)
            return m.render_ultragauss(
                m.UltraGaussInput(params.means, params.raw_precision_tril, params.opacities().reshape(-1), params.intensities().reshape(-1)),
                m.ProbeSliceInput(
                    frame["pixel_positions_world"],
                    frame["probe_origin_world"],
                    frame["axis_lateral_world"],
                    frame["axis_axial_world"],
                    frame["axis_elevational_world"],
                ),
                m.UltraGaussRenderSettings(
                    image_height=cfg.image_height,
                    image_width=cfg.image_width,
                    lateral_spacing_mm_per_pix=cfg.delta_lateral_mm,
                    axial_spacing_mm_per_pix=cfg.delta_axial_mm,
                    axial_near_mm=cfg.near_mm,
                    lateral_col_offset=((cfg.image_width - 1) * 0.5 if cfg.center_lateral else 0.0),
                    chi2_threshold=cfg.ultragauss_chi2_threshold,
                    diag_beta=cfg.ultragauss_diag_beta,
                ),
            )
        if self.method == "ultragray":
            return m.render_ultragray(
                m.UltraGrayInput(params.means, params.scales_3d(), params.quats(), params.raw_opacities.reshape(-1), params.sh_gray4),
                m.RaySliceInput(
                    frame["pixel_positions_world"].reshape(cfg.image_height, cfg.image_width, 3).permute(1, 0, 2).contiguous(),
                    frame["ray_origins"],
                    frame["ray_dirs"],
                    frame["probe_origin_world"],
                    frame["axis_lateral_world"],
                    frame["axis_axial_world"],
                    frame["axis_elevational_world"],
                ),
                m.UltraGrayRenderSettings(
                    image_width=cfg.image_width,
                    image_height=cfg.image_height,
                    width_mm=cfg.width_mm,
                    near_mm=cfg.near_mm,
                    lateral_col_offset=((cfg.image_width - 1) * 0.5 if cfg.center_lateral else 0.0),
                    k_sigma=cfg.ultragray_k_sigma,
                    ops_max_offset_mm=cfg.ultragray_ops_max_offset_mm,
                    quadrature_window_radius_mm=cfg.ultragray_quadrature_window_radius_mm,
                ),
            )
        raise ValueError(f"未知 method: {self.method}")


def _load_py(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载 {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


class _FallbackRenderOutput:
    def __init__(self, image: torch.Tensor, depth: Optional[torch.Tensor] = None, extras: Optional[Dict[str, torch.Tensor]] = None):
        self.image = image
        self.depth = depth
        self.extras = extras or {}


def _render_ultragauss_torch_fallback(params: GaussianParameters, frame: Dict[str, torch.Tensor], cfg: GodConfig) -> _FallbackRenderOutput:
    """UltraGauss 的纯 PyTorch fallback。

    这个 fallback 用于没有 CUDA 或没有编译 ultragauss_renderer._C 的环境。
    它保留 UltraGauss 的核心形式：3D Mahalanobis response + alpha-weighted intensity。
    为避免 CPU 内存爆炸，按像素 chunk 计算。
    """

    pixels = frame["pixel_positions_world"].to(params.means.device)
    means = params.means
    # 用 raw_precision_tril 构造 L，再得到 Lambda=L L^T。这里和 CUDA renderer 保持同一参数语义。
    raw = params.raw_precision_tril
    beta = float(cfg.ultragauss_diag_beta)
    L = torch.zeros((raw.shape[0], 3, 3), device=raw.device, dtype=raw.dtype)
    L[:, 0, 0] = raw[:, 0].square() + beta
    L[:, 1, 0] = raw[:, 1]
    L[:, 2, 0] = raw[:, 2]
    L[:, 1, 1] = raw[:, 3].square() + beta
    L[:, 2, 1] = raw[:, 4]
    L[:, 2, 2] = raw[:, 5].square() + beta
    precision = L.matmul(L.transpose(-1, -2))
    alpha0 = params.opacities().reshape(-1)
    inten = params.intensities().reshape(-1)

    h, w = int(cfg.image_height), int(cfg.image_width)
    out_chunks: List[torch.Tensor] = []
    alpha_chunks: List[torch.Tensor] = []
    pix_chunk = 2048 if pixels.device.type == "cpu" else 16384
    gauss_chunk = 512 if pixels.device.type == "cpu" else 4096
    bg_alpha = 1.0e-6
    for p0 in range(0, pixels.shape[0], pix_chunk):
        p = pixels[p0 : p0 + pix_chunk]
        numer = torch.zeros((p.shape[0],), device=pixels.device, dtype=pixels.dtype)
        denom = torch.zeros_like(numer)
        for g0 in range(0, means.shape[0], gauss_chunk):
            g1 = min(g0 + gauss_chunk, means.shape[0])
            d = p[:, None, :] - means[None, g0:g1, :]
            q = torch.einsum("pgi,gij,pgj->pg", d, precision[g0:g1], d)
            # 60 是数值下限，避免 exp underflow 造成无意义警告。
            weight = torch.exp(torch.clamp(-0.5 * q, min=-60.0, max=0.0)) * alpha0[g0:g1].view(1, -1)
            numer = numer + (weight * inten[g0:g1].view(1, -1)).sum(dim=1)
            denom = denom + weight.sum(dim=1)
        pred = numer / torch.clamp(denom + bg_alpha, min=1.0e-8)
        out_chunks.append(pred)
        alpha_chunks.append(denom)
    image = torch.cat(out_chunks, dim=0).view(h, w).unsqueeze(0).clamp(0.0, 1.0)
    alpha = torch.cat(alpha_chunks, dim=0).view(1, h, w)
    return _FallbackRenderOutput(image=image, depth=None, extras={"alpha_image": alpha, "renderer": torch.tensor(0)})


def image_loss(pred: torch.Tensor, gt: torch.Tensor, cfg: GodConfig) -> Tuple[torch.Tensor, Dict[str, float]]:
    pred = _canon_image(pred)
    gt = _canon_image(gt).to(dtype=pred.dtype, device=pred.device)
    if pred.shape != gt.shape:
        pred = pred.reshape_as(gt)
    l1 = torch.mean(torch.abs(pred - gt))
    total = float(cfg.loss_l1_weight) * l1
    ssim_loss = torch.tensor(0.0, device=pred.device)
    if cfg.loss_ssim_weight > 0:
        ssim_loss = 1.0 - simple_ssim(pred, gt)
        total = total + float(cfg.loss_ssim_weight) * ssim_loss
    return total, {"l1": float(l1.detach().item()), "ssim_loss": float(ssim_loss.detach().item()), "total": float(total.detach().item())}


def _canon_image(x: torch.Tensor) -> torch.Tensor:
    if x.ndim == 2:
        x = x.unsqueeze(0)
    if x.ndim == 3 and x.shape[0] != 1:
        x = x[:1]
    return x.float()


def simple_ssim(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    # 一个轻量全局 SSIM，避免额外依赖。正式论文评估建议用标准库重算。
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    mux = x.mean()
    muy = y.mean()
    vx = ((x - mux) ** 2).mean()
    vy = ((y - muy) ** 2).mean()
    vxy = ((x - mux) * (y - muy)).mean()
    return ((2 * mux * muy + c1) * (2 * vxy + c2)) / ((mux ** 2 + muy ** 2 + c1) * (vx + vy + c2))


class Callback:
    def on_start(self, ctx: Dict[str, Any]) -> None:
        pass

    def on_step_end(self, ctx: Dict[str, Any]) -> None:
        pass

    def on_finish(self, ctx: Dict[str, Any]) -> None:
        pass


class CsvLoggerCallback(Callback):
    def on_start(self, ctx: Dict[str, Any]) -> None:
        path = Path(ctx["cfg"].output_dir) / "scalars.csv"
        self.f = open(path, "w", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(self.f, fieldnames=["iteration", "split", "loss_total", "loss_l1", "loss_ssim", "num_gaussians", "seconds"])
        self.writer.writeheader()

    def on_step_end(self, ctx: Dict[str, Any]) -> None:
        cfg: GodConfig = ctx["cfg"]
        it = int(ctx["iteration"])
        if it % int(cfg.log_every) != 0:
            return
        row = {
            "iteration": it,
            "split": "train",
            "loss_total": ctx["losses"]["total"],
            "loss_l1": ctx["losses"]["l1"],
            "loss_ssim": ctx["losses"]["ssim_loss"],
            "num_gaussians": int(ctx["params"].means.shape[0]),
            "seconds": time.time() - ctx["start_time"],
        }
        self.writer.writerow(row)
        self.f.flush()
        print(f"[iter {it}] loss={row['loss_total']:.6f} l1={row['loss_l1']:.6f} n={row['num_gaussians']}", flush=True)

    def on_finish(self, ctx: Dict[str, Any]) -> None:
        self.f.close()


class EvalCallback(Callback):
    def on_step_end(self, ctx: Dict[str, Any]) -> None:
        cfg: GodConfig = ctx["cfg"]
        it = int(ctx["iteration"])
        if cfg.eval_every <= 0 or it % int(cfg.eval_every) != 0:
            return
        params: GaussianParameters = ctx["params"]
        renderer: RenderDispatcher = ctx["renderer"]
        test_data: NpzUltrasoundDataset = ctx["test_data"]
        vals: List[float] = []
        with torch.no_grad():
            for frame in test_data.iter_eval(ctx["device"], max_frames=4):
                out = renderer.render(params, frame)
                _, parts = image_loss(out.image, frame["image"], cfg)
                vals.append(parts["l1"])
        mean_l1 = float(np.mean(vals)) if vals else float("nan")
        with open(Path(cfg.output_dir) / "eval.csv", "a", encoding="utf-8") as f:
            f.write(f"{it},{mean_l1}\n")
        print(f"[eval {it}] mean_l1={mean_l1:.6f}", flush=True)


class CheckpointCallback(Callback):
    def on_start(self, ctx: Dict[str, Any]) -> None:
        self._save(ctx, "initial.pt")

    def on_step_end(self, ctx: Dict[str, Any]) -> None:
        cfg: GodConfig = ctx["cfg"]
        it = int(ctx["iteration"])
        if cfg.save_every > 0 and it % int(cfg.save_every) == 0:
            self._save(ctx, f"iter_{it:06d}.pt")

    def on_finish(self, ctx: Dict[str, Any]) -> None:
        self._save(ctx, "final.pt")

    def _save(self, ctx: Dict[str, Any], name: str) -> None:
        path = Path(ctx["cfg"].output_dir) / name
        torch.save(
            {
                "config": asdict(ctx["cfg"]),
                "iteration": int(ctx.get("iteration", 0)),
                "params": ctx["params"].state_dict(),
                "optimizer": ctx["optimizer"].state_dict(),
            },
            path,
        )


def train(cfg: GodConfig) -> None:
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "used_config.json", "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, ensure_ascii=False, indent=2)

    set_seed(cfg.seed)
    device = torch.device(cfg.device if cfg.device == "cpu" or torch.cuda.is_available() else "cpu")
    train_data = NpzUltrasoundDataset(cfg, "train")
    test_data = NpzUltrasoundDataset(cfg, "test")
    init_means, init_intensities = train_data.initial_points(device, int(cfg.num_init))
    params = GaussianParameters(cfg, init_means, init_intensities).to(device)
    renderer = RenderDispatcher(cfg)
    optimizer = torch.optim.Adam(params.parameters(), lr=float(cfg.learning_rate))
    callbacks: List[Callback] = [CsvLoggerCallback(), EvalCallback(), CheckpointCallback()]
    ctx: Dict[str, Any] = {
        "cfg": cfg,
        "device": device,
        "train_data": train_data,
        "test_data": test_data,
        "params": params,
        "renderer": renderer,
        "optimizer": optimizer,
        "iteration": 0,
        "losses": {},
        "start_time": time.time(),
    }
    for cb in callbacks:
        cb.on_start(ctx)

    for it in range(1, int(cfg.iterations) + 1):
        ctx["iteration"] = it
        frame = train_data.random_frame(device)
        optimizer.zero_grad(set_to_none=True)
        out = renderer.render(params, frame)
        loss, parts = image_loss(out.image, frame["image"], cfg)
        loss.backward()
        optimizer.step()
        ctx["losses"] = parts
        for cb in callbacks:
            cb.on_step_end(ctx)

    for cb in callbacks:
        cb.on_finish(ctx)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None, help="JSON config 路径")
    parser.add_argument("--set", nargs="*", default=[], help="覆盖项，例如 method=ultragauss iterations=10")
    args = parser.parse_args()
    cfg = load_config(args.config, args.set)
    train(cfg)


if __name__ == "__main__":
    main()
