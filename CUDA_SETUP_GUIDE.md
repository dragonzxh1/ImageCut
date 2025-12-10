# CUDA安装后配置指南

## 问题：安装CUDA后nvcc未找到

### 原因
安装CUDA Toolkit后，环境变量可能没有自动设置，或者需要重启系统才能生效。

## 解决方案

### 方案1：重启计算机（推荐）

**最简单的方法：**
1. 保存所有工作
2. 重启计算机
3. 重启后，环境变量会自动生效
4. 运行 `check_gpu_status.py` 验证

### 方案2：手动设置环境变量（无需重启）

#### Windows 方法1：使用系统设置

1. **打开系统环境变量设置**
   - 按 `Win + R`，输入 `sysdm.cpl`，回车
   - 点击"高级"选项卡
   - 点击"环境变量"按钮

2. **添加CUDA_PATH变量**
   - 在"系统变量"区域，点击"新建"
   - 变量名：`CUDA_PATH`
   - 变量值：`C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6`
     （根据您的实际安装路径修改版本号）

3. **添加到PATH**
   - 在"系统变量"中找到 `Path`，点击"编辑"
   - 点击"新建"，添加：`%CUDA_PATH%\bin`
   - 点击"新建"，添加：`%CUDA_PATH%\libnvvp`
   - 点击"确定"保存所有更改

4. **重新打开命令行窗口**
   - 关闭所有命令行窗口
   - 重新打开新的命令行窗口
   - 运行 `nvcc --version` 验证

#### Windows 方法2：使用批处理脚本

1. **运行自动设置脚本**
   ```bash
   setup_cuda_env.bat
   ```

2. **脚本会自动：**
   - 查找CUDA安装路径
   - 设置CUDA_PATH环境变量
   - 添加到PATH（当前会话）

3. **重新打开命令行窗口**
   - 关闭当前窗口
   - 打开新窗口测试

#### Windows 方法3：使用PowerShell（临时，仅当前会话）

```powershell
# 设置CUDA路径（根据实际安装路径修改）
$env:CUDA_PATH = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6"
$env:PATH = "$env:CUDA_PATH\bin;$env:PATH"

# 验证
nvcc --version
```

**注意：** 这种方法只在当前PowerShell会话有效，关闭窗口后失效。

### 方案3：使用批处理脚本（推荐）

我已经创建了 `setup_cuda_env.bat` 脚本，它会：
1. 自动查找CUDA安装路径
2. 设置环境变量
3. 验证nvcc是否可用

**使用方法：**
```bash
setup_cuda_env.bat
```

## 验证CUDA安装

### 1. 检查nvcc
```bash
nvcc --version
```
应该显示CUDA版本信息。

### 2. 检查环境变量
```bash
echo %CUDA_PATH%
```
应该显示CUDA安装路径。

### 3. 运行GPU状态检查
```bash
python check_gpu_status.py
```

## 常见CUDA安装路径

根据CUDA版本，常见路径为：
- CUDA 13.1: `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1`
- CUDA 12.6: `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6`
- CUDA 12.4: `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4`
- CUDA 11.8: `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8`

## 重要提示

### 关于CUDA 13.1
⚠️ **注意：** 如果您安装的是CUDA 13.1，onnxruntime-gpu目前不支持CUDA 13.x。

**建议：**
- 如果只需要使用程序：可以继续使用，程序会自动回退到CPU模式
- 如果需要GPU加速：降级到CUDA 12.6或12.4

### 重启 vs 不重启

| 方法 | 优点 | 缺点 |
|------|------|------|
| 重启计算机 | 确保所有程序都能访问环境变量 | 需要等待重启 |
| 手动设置+重新打开命令行 | 快速，无需重启 | 只对新的命令行窗口有效 |
| 使用setx命令 | 永久设置，无需重启 | 需要重新打开命令行窗口 |

**推荐：** 如果时间允许，重启计算机是最可靠的方法。

## 验证步骤

安装并配置后，按以下步骤验证：

1. **检查nvcc**
   ```bash
   nvcc --version
   ```

2. **检查PyTorch CUDA支持**
   ```bash
   python -c "import torch; print('CUDA可用:', torch.cuda.is_available())"
   ```

3. **运行完整检查**
   ```bash
   python check_gpu_status.py
   ```

4. **运行GPU版本程序**
   ```bash
   streamlit run app_gpu.py
   ```
   查看启动日志，应该显示 `✓ GPU可用: [GPU名称]`

## 如果仍然不可用

如果重启后仍然显示GPU不可用，可能的原因：

1. **PyTorch安装的是CPU版本**
   ```bash
   pip uninstall torch torchvision
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
   ```

2. **CUDA版本不匹配**
   - 检查CUDA版本：`nvcc --version`
   - 检查PyTorch CUDA版本：`python -c "import torch; print(torch.version.cuda)"`
   - 确保两者兼容

3. **GPU驱动问题**
   - 更新NVIDIA GPU驱动
   - 确保驱动版本与CUDA版本兼容


