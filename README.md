# 3D Ultrasound Gaussian Reconstruction

一个用于三维超声重建实验的最小可运行仓库，包含统一训练 pipeline、五种 Gaussian renderer 接口、合成烟雾测试数据，以及 UltraG-Ray benchmark 的下载与校验工具。

## 已验证环境

- Ubuntu 22.04
- NVIDIA RTX 4090（Compute Capability 8.9）
- Python 3.11
- PyTorch 2.2.2 + CUDA 12.1
- CUDA Toolkit / `nvcc` 12.1

## 快速开始

```bash
conda create -n ultra3d python=3.11 -y
conda activate ultra3d
pip install torch==2.2.2 torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r scripts/requirements-base.txt
```

运行内置合成数据的 GPU 烟雾测试：

```bash
python pipeline_god_file.py \
  --config configs/synthetic_liver_l1_ultragauss_smoke.json \
  --set device=cuda output_dir=runs/gpu_smoke iterations=1 num_init=16
```

已验证输出示例：`loss=0.194392`。运行会生成 `used_config.json`、`scalars.csv`、`initial.pt`、`iter_*.pt` 和 `final.pt`。

## Benchmark 数据

推荐快速测试使用 **UltraG-Ray Pig Shoulder v2**；更大规模实验使用 **Spine Phantom**。

```bash
python scripts/download_benchmarks.py
python scripts/validate_ultragray.py datasets/ultragray
```

原始数据不提交到 Git；目录、张量要求和转换注意事项见 [DATASET.md](DATASET.md)。Pipeline 的标准输入格式见 [DATA_FORMAT.md](DATA_FORMAT.md)。

## 支持的方法

- `gaussianpile`
- `slice_adapted_3dgs`
- `ultrags_baseline`
- `ultragauss`
- `ultragray`

可通过 `--set method=<name>` 切换方法。当前 UltraGauss 原生二进制来自旧 Python ABI，源码包还缺少部分公共头文件；无法加载时 pipeline 自动使用 PyTorch fallback。正式大规模训练前建议补齐源码并针对 CUDA 12.1 / `sm_89` 重编译。

## 数据安全

仓库不包含服务器地址、密码、私钥、训练 checkpoint 或原始 benchmark 数据。
