# GPU版本故障排除指南

## CUDA 版本兼容性

### onnxruntime-gpu 支持的 CUDA 版本

**当前支持：**
- ✅ **CUDA 11.x**：onnxruntime-gpu 使用 CUDA 11.8 构建，与任何 CUDA 11.x 版本兼容
- ✅ **CUDA 12.x**：onnxruntime-gpu 使用 CUDA 12.x 构建，与任何 CUDA 12.x 版本兼容
- ❌ **CUDA 13.x**：**目前不支持**（包括 CUDA 13.0、13.1 等）

**注意：** 如果您安装了 CUDA 13.1，onnxruntime-gpu 将无法使用 GPU，会自动回退到 CPU 模式。

### 解决方案

#### 如果您安装了 CUDA 13.1：

**方案1：降级到 CUDA 12.x（推荐）**
1. 卸载 CUDA 13.1
2. 安装 CUDA 12.4 或 12.6（最新稳定版本）
3. 重新运行程序

**方案2：使用 CPU 版本**
```bash
pip uninstall onnxruntime-gpu
pip install onnxruntime>=1.23.0
```

**方案3：等待官方支持**
- 关注 ONNX Runtime GitHub 仓库的更新
- 目前有用户请求支持 CUDA 13.0，但尚未实现

## 错误：缺少 cublasLt64_12.dll

### 问题描述
```
Error loading "onnxruntime_providers_cuda.dll" which depends on "cublasLt64_12.dll" which is missing.
```

这个错误表示 `onnxruntime-gpu` 需要 CUDA 12 的库文件，但系统中缺少这些文件。

**可能的原因：**
1. 未安装 CUDA Toolkit 12.x
2. 安装了 CUDA 13.x（不兼容）
3. CUDA 库文件路径未正确设置

### 解决方案

#### 方案1：安装 CUDA Toolkit 12.x（推荐，如果您的GPU支持）

**重要：** 如果您已安装 CUDA 13.1，需要先卸载，然后安装 CUDA 12.x。

1. **卸载 CUDA 13.x**（如果已安装）
   - 通过 Windows 控制面板卸载所有 CUDA 13.x 相关组件

2. **下载 CUDA Toolkit 12.x**
   - 访问：https://developer.nvidia.com/cuda-downloads
   - 选择适合您系统的版本（Windows/Linux）
   - **推荐版本：CUDA 12.4 或 12.6**（最新稳定版本）
   - **不要安装 CUDA 13.x**（目前不支持）

3. **安装后验证**
   ```bash
   nvcc --version
   ```
   应该显示 CUDA 12.x 版本（不是 13.x）

4. **设置环境变量**（Windows）
   - 确保 `CUDA_PATH` 环境变量指向 CUDA 安装目录
   - 将 `%CUDA_PATH%\bin` 添加到系统 PATH
   - 重启命令行或应用程序

#### 方案2：使用CPU版本（如果不需要GPU加速）

如果您的系统没有 NVIDIA GPU 或不想安装 CUDA，可以使用 CPU 版本：

1. **卸载 onnxruntime-gpu**
   ```bash
   pip uninstall onnxruntime-gpu
   ```

2. **安装 onnxruntime（CPU版本）**
   ```bash
   pip install onnxruntime>=1.23.0
   ```

   或者使用备用依赖文件：
   ```bash
   pip install -r requirements_gpu_fallback.txt
   ```

3. **程序会自动回退到CPU模式**
   - rembg 会使用 CPU
   - EasyOCR 会使用 CPU（代码已自动处理）

#### 方案3：使用混合模式（已实现）

代码已经实现了自动回退机制：
- 如果 GPU 可用，优先使用 GPU
- 如果 GPU 不可用或缺少依赖，自动回退到 CPU
- 程序仍然可以正常运行，只是速度较慢

### 验证GPU状态

运行以下Python代码检查GPU状态：

```python
import torch
print(f"CUDA可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU名称: {torch.cuda.get_device_name(0)}")
    print(f"CUDA版本: {torch.version.cuda}")
```

### 常见问题

#### Q: 为什么需要 CUDA 12？
A: 最新版本的 `onnxruntime-gpu` 需要 CUDA 12.x。如果您安装的是 CUDA 11.x，可能需要降级 `onnxruntime-gpu` 版本。

#### Q: 可以使用 CUDA 11 吗？
A: 可以，但需要安装对应版本的 `onnxruntime-gpu`：
```bash
pip install onnxruntime-gpu==1.16.0  # 支持 CUDA 11
```

#### Q: 可以使用 CUDA 13.1 吗？
A: **不可以**。onnxruntime-gpu 目前不支持 CUDA 13.x（包括 13.0、13.1 等）。
- 如果您安装了 CUDA 13.1，程序会自动回退到 CPU 模式
- 建议降级到 CUDA 12.x 以获得 GPU 加速
- 或者使用 CPU 版本的 onnxruntime

#### Q: 程序会完全无法运行吗？
A: 不会。代码已经实现了自动回退机制，即使 GPU 不可用，程序也会使用 CPU 模式运行，只是速度较慢。

#### Q: 如何确认当前使用的是GPU还是CPU？
A: 查看程序启动时的日志输出：
- `✓ GPU可用: [GPU名称]` - 使用GPU
- `⚠ GPU不可用，将使用CPU模式` - 使用CPU
- `✓ EasyOCR使用GPU模式` - EasyOCR使用GPU
- `✓ EasyOCR使用CPU模式` - EasyOCR使用CPU

### 性能对比

- **GPU模式**：背景移除快2-5倍，OCR快3-10倍
- **CPU模式**：功能正常，但速度较慢

### 推荐配置

- **有NVIDIA GPU + CUDA 12**：使用 `requirements_gpu.txt`（完整GPU支持）
- **有NVIDIA GPU但缺少CUDA库**：使用 `requirements_gpu_fallback.txt`（CPU模式，但EasyOCR仍可尝试GPU）
- **无NVIDIA GPU**：使用原版CPU版本（`requirements.txt`）

