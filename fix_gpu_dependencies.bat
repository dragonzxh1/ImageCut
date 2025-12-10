@echo off
echo ========================================
echo GPU版本依赖修复工具
echo ========================================
echo.
echo 此脚本将帮助您解决 CUDA 依赖问题
echo.
echo 选项：
echo 1. 安装 CPU 版本的 onnxruntime（推荐，如果遇到 CUDA DLL 错误）
echo 2. 检查当前 GPU 状态
echo 3. 退出
echo.
set /p choice="请选择 (1-3): "

if "%choice%"=="1" (
    echo.
    echo 正在卸载 onnxruntime-gpu...
    pip uninstall -y onnxruntime-gpu
    echo.
    echo 正在安装 onnxruntime (CPU版本)...
    pip install onnxruntime>=1.23.0
    echo.
    echo ✓ 已切换到 CPU 版本
    echo 程序将使用 CPU 模式运行，功能正常但速度较慢
    echo.
    pause
) else if "%choice%"=="2" (
    echo.
    echo 正在检查 GPU 状态...
    python -c "import torch; print('CUDA可用:', torch.cuda.is_available()); print('GPU名称:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
    echo.
    pause
) else if "%choice%"=="3" (
    exit
) else (
    echo 无效选择
    pause
)


