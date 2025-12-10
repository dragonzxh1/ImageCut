@echo off
echo ========================================
echo 尝试安装PyTorch Nightly版本
echo ========================================
echo.
echo 原因：RTX 5070 Ti需要sm_120支持
echo 当前PyTorch 2.5.1只支持到sm_90
echo Nightly版本可能包含对新GPU架构的支持
echo.
echo 注意：Nightly版本可能不稳定
echo.
pause

echo.
echo 步骤1: 卸载当前PyTorch
pip uninstall torch torchvision -y

echo.
echo 步骤2: 安装PyTorch Nightly (CUDA 12.1)
pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu121

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ✗ 安装失败
    pause
    exit /b 1
)

echo.
echo 步骤3: 验证安装
python -c "import torch; print('版本:', torch.__version__); print('CUDA可用:', torch.cuda.is_available()); print('计算能力:', torch.cuda.get_device_capability(0) if torch.cuda.is_available() else 'N/A')"

echo.
echo 步骤4: 测试GPU
python -c "import torch; x = torch.randn(3, 3).cuda(); print('GPU测试成功:', x.device)"

echo.
echo ========================================
echo 完成！
echo ========================================
echo.
pause

