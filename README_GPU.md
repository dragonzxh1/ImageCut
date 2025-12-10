# 卡片图像处理工具 - GPU版本

这是GPU加速版本的卡片图像处理工具，使用GPU加速rembg背景移除和EasyOCR文字识别。

## 环境要求

- Python 3.8+
- NVIDIA GPU（支持CUDA）
- **CUDA Toolkit 11.x 或 12.x**（**不支持 CUDA 13.x**）
- cuDNN

**重要：** onnxruntime-gpu 目前**不支持 CUDA 13.x**（包括 13.0、13.1 等）。如果您安装了 CUDA 13.1，程序会自动回退到 CPU 模式。

## 安装步骤

### 1. 创建虚拟环境

```bash
python -m venv venv_gpu
```

### 2. 激活虚拟环境

**Windows:**
```bash
venv_gpu\Scripts\activate
```

**Linux/Mac:**
```bash
source venv_gpu/bin/activate
```

### 3. 安装依赖

```bash
pip install -r requirements_gpu.txt
```

**注意：** 如果遇到CUDA相关错误，请确保：
- 已安装正确版本的CUDA Toolkit
- PyTorch版本与CUDA版本匹配
- 安装了对应版本的cuDNN

### 4. 验证GPU支持

```python
import torch
print(torch.cuda.is_available())  # 应该返回 True
print(torch.cuda.get_device_name(0))  # 显示GPU名称
```

## 使用方法

### 方法1：使用批处理文件（Windows）

双击 `run_app_gpu.bat` 启动应用

### 方法2：命令行启动

```bash
# 激活虚拟环境
venv_gpu\Scripts\activate  # Windows
# 或
source venv_gpu/bin/activate  # Linux/Mac

# 启动应用
streamlit run app_gpu.py
```

## GPU版本 vs CPU版本

### 主要区别

1. **rembg背景移除**
   - GPU版本：自动使用GPU加速（如果CUDA可用）
   - CPU版本：仅使用CPU

2. **EasyOCR文字识别**
   - GPU版本：`gpu=True`，使用GPU加速
   - CPU版本：`gpu=False`，仅使用CPU

3. **依赖包**
   - GPU版本：`onnxruntime-gpu`, `torch`, `torchvision`
   - CPU版本：`onnxruntime`

### 性能提升

- **背景移除**：GPU版本通常快2-5倍
- **文字识别**：GPU版本通常快3-10倍（取决于GPU性能）

## 文件说明

- `app_gpu.py` - GPU版本的Streamlit应用界面
- `cut_image_gpu.py` - GPU版本的核心处理逻辑
- `requirements_gpu.txt` - GPU版本的依赖包列表
- `run_app_gpu.bat` - Windows启动脚本
- `venv_gpu/` - GPU版本的虚拟环境目录

## 故障排除

### GPU不可用

如果GPU不可用，程序会自动回退到CPU模式，但性能会下降。

### CUDA版本不匹配

如果遇到CUDA相关错误：
1. 检查CUDA版本：`nvcc --version`
2. 检查PyTorch CUDA版本：`python -c "import torch; print(torch.version.cuda)"`
3. 确保两者版本兼容

### 内存不足

如果GPU内存不足：
- 减小批处理大小
- 使用较小的模型（如 `u2netp` 而不是 `birefnet-hrsod`）

## 注意事项

1. GPU版本需要NVIDIA GPU和CUDA支持
2. 首次运行EasyOCR时会下载模型文件（约500MB）
3. 确保GPU驱动已正确安装
4. 如果GPU不可用，程序会自动使用CPU，但性能会下降

