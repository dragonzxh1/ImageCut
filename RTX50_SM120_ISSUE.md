# RTX 50系列GPU兼容性问题

## 问题描述

使用RTX 5070 Ti等RTX 50系列GPU时，EasyOCR尝试使用GPU会出现以下错误：

```
CUDA error: no kernel image is available for execution on the device
```

## 根本原因

**RTX 50系列GPU使用CUDA计算能力12.0（sm_120）**，而当前稳定版PyTorch（2.5.1+cu121）**只支持到sm_90**。

- RTX 5070 Ti: sm_120 (计算能力 12.0)
- PyTorch 2.5.1+cu121: 支持 sm_50 到 sm_90

## 当前状态

✅ **程序仍然可以正常运行**，会自动回退到CPU模式：
- EasyOCR会使用CPU模式（功能正常，但速度较慢）
- rembg可能可以使用GPU（通过onnxruntime-gpu）
- 其他功能正常

## 解决方案

### 方案1：等待PyTorch官方更新（推荐）

PyTorch官方会逐步支持新的GPU架构。关注PyTorch发布说明，等待支持sm_120的版本。

### 方案2：尝试PyTorch Nightly版本

Nightly版本可能包含对新GPU架构的支持：

```bash
# 卸载当前版本
pip uninstall torch torchvision -y

# 安装nightly版本（CUDA 12.1）
pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu121
```

**注意：** Nightly版本可能不稳定，建议在测试环境使用。

### 方案3：使用CPU模式（当前方案）

程序已经自动回退到CPU模式，功能完全正常，只是处理速度较慢：
- 背景移除：使用CPU（较慢）
- 文字识别：使用CPU（较慢，但功能正常）

### 方案4：从源码编译PyTorch（不推荐）

从源码编译支持sm_120的PyTorch，但过程复杂且耗时。

## 验证GPU架构

运行以下命令检查GPU计算能力：

```python
import torch
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"计算能力: {torch.cuda.get_device_capability(0)}")
    print(f"PyTorch支持的架构: {torch.cuda.get_arch_list()}")
```

## 性能影响

使用CPU模式的影响：
- **背景移除**: 慢2-5倍
- **文字识别**: 慢3-10倍（取决于图片大小和文字数量）

对于批量处理，建议：
- 使用CPU模式（功能正常）
- 或等待PyTorch更新支持sm_120

## 相关链接

- [PyTorch官方文档](https://pytorch.org/get-started/locally/)
- [CUDA计算能力列表](https://developer.nvidia.com/cuda-gpus)
- [PyTorch GitHub Issues](https://github.com/pytorch/pytorch/issues) - 搜索 "sm_120" 或 "RTX 50"

## 更新日志

- 2024-12-11: 确认RTX 5070 Ti需要sm_120支持
- 2024-12-11: **问题已解决！** 安装PyTorch 2.10.0.dev20251210+cu128后，成功支持sm_120
- EasyOCR现在可以正常使用GPU加速

## ✅ 成功解决方案

**已验证可用的配置：**
- PyTorch: `2.10.0.dev20251210+cu128` (Nightly版本)
- CUDA: 12.8+
- GPU: RTX 5070 Ti (sm_120)
- EasyOCR: 成功使用GPU (`cuda:0`)

**安装命令：**
```bash
pip uninstall torch torchvision torchaudio -y
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
```

**验证脚本：**
运行 `check_easyocr_gpu.py` 验证GPU支持

