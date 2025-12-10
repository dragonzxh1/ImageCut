@echo off
echo ========================================
echo 使用Python 3.11重新创建GPU虚拟环境
echo ========================================
echo.
echo 原因：Python 3.13太新，PyTorch还没有GPU预编译包
echo Python 3.11有完整的GPU支持
echo.

REM 检查Python 3.11是否可用
py -3.11 --version
if %ERRORLEVEL% NEQ 0 (
    echo ✗ Python 3.11未找到
    pause
    exit /b 1
)

echo.
echo 步骤1: 删除旧的虚拟环境（可选）
echo 按任意键继续，或Ctrl+C取消...
pause

if exist venv_gpu (
    echo 删除旧环境...
    rmdir /s /q venv_gpu
)

echo.
echo 步骤2: 使用Python 3.11创建新虚拟环境
py -3.11 -m venv venv_gpu

if %ERRORLEVEL% NEQ 0 (
    echo ✗ 创建虚拟环境失败
    pause
    exit /b 1
)

echo.
echo 步骤3: 激活虚拟环境并升级pip
call venv_gpu\Scripts\activate.bat
python -m pip install --upgrade pip

echo.
echo 步骤4: 安装GPU版本的PyTorch
python -m pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu121

if %ERRORLEVEL% NEQ 0 (
    echo ✗ PyTorch安装失败
    pause
    exit /b 1
)

echo.
echo 步骤5: 安装其他依赖
python -m pip install -r requirements_gpu.txt

echo.
echo 步骤6: 验证安装
python check_gpu_status.py

echo.
echo ========================================
echo 完成！
echo ========================================
echo.
echo 如果显示"CUDA可用: True"，说明安装成功！
echo.
pause

