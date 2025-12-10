@echo off
echo ========================================
echo 安装支持RTX 50系列的PyTorch版本
echo ========================================
echo.
echo 此脚本将安装PyTorch Nightly版本（cu128）
echo 该版本支持RTX 50系列的sm_120架构
echo.
echo 注意：这是Nightly（预览）版本，可能不稳定
echo.
pause

echo.
echo 步骤1: 卸载当前PyTorch
pip uninstall torch torchvision torchaudio -y

echo.
echo 步骤2: 安装PyTorch Nightly (CUDA 12.8)
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ✗ 安装失败
    echo.
    echo 可能的原因：
    echo 1. 网络连接问题
    echo 2. cu128版本尚未发布（尝试cu124或cu121）
    echo.
    pause
    exit /b 1
)

echo.
echo 步骤3: 验证安装
python -c "import torch; print('PyTorch版本:', torch.__version__); print('CUDA可用:', torch.cuda.is_available()); print('GPU计算能力:', torch.cuda.get_device_capability(0) if torch.cuda.is_available() else 'N/A')"

echo.
echo 步骤4: 测试GPU
python -c "import torch; x = torch.randn(3, 3).cuda(); print('GPU测试:', '成功' if x.device.type == 'cuda' else '失败')"

echo.
echo 步骤5: 验证EasyOCR GPU支持
echo 运行 check_easyocr_gpu.py 进行完整验证
python check_easyocr_gpu.py

echo.
echo ========================================
echo 安装完成！
echo ========================================
echo.
echo 如果看到"EasyOCR成功使用GPU加速"，说明配置成功！
echo.
pause

