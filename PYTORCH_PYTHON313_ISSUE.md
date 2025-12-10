 # PyTorch GPU版本安装问题 - Python 3.13

## 问题描述

安装PyTorch GPU版本后，仍然显示 `2.9.1+cpu`，CUDA不可用。

## 根本原因

**Python 3.13 可能太新，PyTorch官方可能还没有为Python 3.13提供GPU版本的预编译包。**

PyTorch官方通常需要一些时间来为新版本的Python提供GPU预编译包。

## 解决方案

### 方案1：使用Python 3.11或3.12（推荐）

PyTorch对Python 3.11和3.12有完整的GPU支持。

**步骤：**

1. **创建新的虚拟环境（使用Python 3.11或3.12）**
   ```bash
   # 如果您有Python 3.11或3.12
   py -3.11 -m venv venv_gpu_py311
   # 或
   py -3.12 -m venv venv_gpu_py312
   ```

2. **激活新环境**
   ```bash
   venv_gpu_py311\Scripts\activate
   ```

3. **安装GPU版本的PyTorch**
   ```bash
   pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu121
   ```

4. **安装其他依赖**
   ```bash
   pip install -r requirements_gpu.txt
   ```

### 方案2：等待PyTorch官方支持

关注PyTorch官方发布，等待Python 3.13的GPU版本支持。

### 方案3：从源码编译（不推荐）

从源码编译PyTorch GPU版本，但这个过程非常复杂且耗时。

## 验证Python版本

```bash
python --version
```

如果显示Python 3.13.x，建议降级到3.11或3.12。

## 检查可用的Python版本

在Windows上，可以使用以下命令检查已安装的Python版本：

```bash
py --list
```

这会显示所有已安装的Python版本。

## 推荐配置

- **Python版本**: 3.11 或 3.12
- **CUDA版本**: 12.9（已安装）
- **PyTorch版本**: 2.9.1+cu121（GPU版本）

## 临时解决方案

如果暂时无法更换Python版本，程序仍然可以运行，但会使用CPU模式：
- rembg会使用CPU（较慢）
- EasyOCR会使用CPU（较慢）
- onnxruntime-gpu的CUDA执行提供者仍然可用（如果CUDA库正确）

性能影响：
- 背景移除：CPU模式慢2-5倍
- 文字识别：CPU模式慢3-10倍

