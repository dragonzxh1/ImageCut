# GPU分支说明

## 分支信息

- **分支名称**: `gpu`
- **GitHub地址**: https://github.com/dragonzxh1/ImageCut/tree/gpu
- **创建时间**: 2024-12-11

## 分支内容

GPU分支包含完整的GPU加速版本，支持：

### ✅ 已实现功能

1. **GPU加速的背景移除** (rembg)
   - 自动检测并使用GPU（如果可用）
   - 支持CUDA 12.x

2. **GPU加速的文字识别** (EasyOCR)
   - 支持RTX 50系列GPU (sm_120架构)
   - 使用PyTorch 2.10.0+cu128 (Nightly版本)

3. **完整的GPU诊断工具**
   - `check_gpu_status.py` - 检查GPU和CUDA配置
   - `check_easyocr_gpu.py` - 验证EasyOCR GPU支持

4. **详细的安装和配置文档**
   - RTX 50系列GPU支持说明
   - CUDA安装配置指南
   - 故障排除文档

## 快速开始

### 1. 切换到GPU分支

```bash
git checkout gpu
```

### 2. 创建虚拟环境

```bash
# 使用Python 3.11（推荐，GPU支持最好）
py -3.11 -m venv venv_gpu

# 或使用Python 3.12
py -3.12 -m venv venv_gpu
```

### 3. 激活虚拟环境

```bash
venv_gpu\Scripts\activate
```

### 4. 安装GPU版本的PyTorch

**对于RTX 50系列GPU（sm_120）：**
```bash
# 运行自动安装脚本
INSTALL_PYTORCH_RTX50.bat

# 或手动安装
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
```

**对于其他GPU（sm_50-sm_90）：**
```bash
# 运行自动安装脚本
install_pytorch_gpu.bat

# 或手动安装
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 5. 安装其他依赖

```bash
pip install -r requirements_gpu.txt
```

### 6. 验证GPU支持

```bash
python check_gpu_status.py
python check_easyocr_gpu.py
```

### 7. 运行GPU版本应用

```bash
# 方法1：使用批处理文件
run_app_gpu.bat

# 方法2：命令行
streamlit run app_gpu.py
```

## 系统要求

### 硬件要求
- NVIDIA GPU（支持CUDA）
- **RTX 50系列**: 需要PyTorch Nightly (cu128)
- **其他GPU**: 支持CUDA 11.x或12.x

### 软件要求
- Python 3.11 或 3.12（推荐3.11）
- CUDA Toolkit 12.x（推荐12.8+）
- cuDNN

## 文件说明

### 核心文件
- `app_gpu.py` - GPU版本的Streamlit应用
- `cut_image_gpu.py` - GPU版本的核心处理逻辑
- `requirements_gpu.txt` - GPU版本依赖
- `requirements_gpu_fallback.txt` - CPU回退版本依赖

### 安装脚本
- `INSTALL_PYTORCH_RTX50.bat` - RTX 50系列专用安装
- `install_pytorch_gpu.bat` - 通用GPU安装
- `recreate_venv_py311.bat` - Python 3.11环境重建
- `setup_cuda_env.bat` - CUDA环境变量设置

### 诊断工具
- `check_gpu_status.py` - GPU状态检查
- `check_easyocr_gpu.py` - EasyOCR GPU验证

### 文档
- `README_GPU.md` - GPU版本使用说明
- `GPU_TROUBLESHOOTING.md` - 故障排除指南
- `RTX50_SM120_ISSUE.md` - RTX 50系列兼容性说明
- `PYTORCH_PYTHON313_ISSUE.md` - Python 3.13兼容性说明
- `CUDA_SETUP_GUIDE.md` - CUDA配置指南

## 性能提升

使用GPU加速后：
- **背景移除**: 快2-5倍
- **文字识别**: 快3-10倍（取决于GPU性能）

## 已知问题

### RTX 50系列GPU
- ✅ **已解决**: 使用PyTorch Nightly (cu128)版本支持sm_120架构
- 需要安装: `pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128`

### Python 3.13
- ⚠️ PyTorch尚未提供Python 3.13的GPU预编译包
- 建议使用Python 3.11或3.12

## 故障排除

如果遇到GPU相关问题，请参考：
1. `GPU_TROUBLESHOOTING.md` - 通用故障排除
2. `RTX50_SM120_ISSUE.md` - RTX 50系列特定问题
3. `CUDA_SETUP_GUIDE.md` - CUDA配置问题

## 与主分支的区别

| 特性 | main分支 | gpu分支 |
|------|---------|--------|
| 背景移除 | CPU (rembg) | GPU加速 (rembg + onnxruntime-gpu) |
| 文字识别 | CPU (EasyOCR) | GPU加速 (EasyOCR + PyTorch GPU) |
| PyTorch | 不需要 | 必需（GPU版本） |
| CUDA | 不需要 | 必需 |
| 性能 | 标准 | 2-10倍提升 |

## 贡献

如果发现GPU相关的问题或改进建议，请：
1. 在GitHub Issues中报告
2. 或提交Pull Request到`gpu`分支

## 更新日志

- **2024-12-11**: 创建GPU分支
  - 添加完整的GPU加速支持
  - 支持RTX 50系列GPU (sm_120)
  - 添加详细的安装和配置文档
  - 添加GPU诊断工具

