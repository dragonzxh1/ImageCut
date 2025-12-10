@echo off
echo ========================================
echo 安装PyTorch GPU版本
echo ========================================
echo.
echo 当前状态：
echo - CUDA 12.9 已安装
echo - PyTorch CPU版本已安装，需要替换为GPU版本
echo.

REM 检查是否在虚拟环境中
python -c "import sys; print('虚拟环境:', sys.prefix)" 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo ✗ 无法检测Python环境
    pause
    exit /b 1
)

echo.
echo 步骤1: 卸载CPU版本的PyTorch
echo.
pause

pip uninstall torch torchvision torchaudio -y

echo.
echo 步骤2: 安装GPU版本的PyTorch
echo.
echo 注意：CUDA 12.9 兼容 cu121 和 cu124 版本的PyTorch
echo 将安装 cu121 版本（推荐）
echo.
pause

REM 安装PyTorch GPU版本（CUDA 12.1/12.9兼容）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ✗ 安装失败，尝试备用方案...
    echo.
    echo 尝试安装 cu124 版本...
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
)

echo.
echo 步骤3: 验证安装
echo.
python -c "import torch; print('PyTorch版本:', torch.__version__); print('CUDA可用:', torch.cuda.is_available()); print('CUDA版本:', torch.version.cuda if torch.cuda.is_available() else 'N/A')"

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ✓ 安装完成！
    echo.
    echo 请运行以下命令验证：
    echo   python check_gpu_status.py
) else (
    echo.
    echo ✗ 验证失败
)

echo.
pause

