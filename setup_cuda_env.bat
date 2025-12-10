@echo off
echo ========================================
echo CUDA环境变量设置工具
echo ========================================
echo.
echo 此脚本将帮助您设置CUDA环境变量
echo.

REM 检查常见的CUDA安装路径
set CUDA_PATHS[0]=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1
set CUDA_PATHS[1]=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0
set CUDA_PATHS[2]=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6
set CUDA_PATHS[3]=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4
set CUDA_PATHS[4]=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.3
set CUDA_PATHS[5]=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.2
set CUDA_PATHS[6]=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1
set CUDA_PATHS[7]=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.0
set CUDA_PATHS[8]=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8

echo 正在查找CUDA安装路径...
set FOUND=0

for /L %%i in (0,1,8) do (
    if exist "!CUDA_PATHS[%%i]!" (
        echo.
        echo 找到CUDA安装: !CUDA_PATHS[%%i]!
        set CUDA_PATH=!CUDA_PATHS[%%i]!
        set FOUND=1
        goto :found
    )
)

:found
if %FOUND%==0 (
    echo.
    echo ✗ 未找到CUDA安装路径
    echo.
    echo 请手动输入CUDA安装路径（例如: C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6）
    set /p CUDA_PATH="CUDA路径: "
)

if not exist "%CUDA_PATH%" (
    echo ✗ 路径不存在: %CUDA_PATH%
    pause
    exit /b 1
)

echo.
echo 找到CUDA路径: %CUDA_PATH%
echo.

REM 检查nvcc是否存在
if exist "%CUDA_PATH%\bin\nvcc.exe" (
    echo ✓ 找到nvcc.exe
) else (
    echo ✗ 未找到nvcc.exe，CUDA可能未正确安装
    pause
    exit /b 1
)

echo.
echo 正在设置环境变量（仅当前会话）...
setx CUDA_PATH "%CUDA_PATH%" >nul 2>&1
set CUDA_PATH=%CUDA_PATH%
set PATH=%CUDA_PATH%\bin;%PATH%

echo.
echo ✓ 环境变量已设置（仅当前命令行窗口）
echo.
echo 注意：
echo 1. 使用 setx 设置的环境变量需要重新打开命令行窗口才能生效
echo 2. 或者重启计算机以确保所有程序都能访问环境变量
echo 3. 当前窗口的PATH已临时添加，可以立即测试
echo.

REM 验证
echo 验证nvcc是否可用...
"%CUDA_PATH%\bin\nvcc.exe" --version >nul 2>&1
if %ERRORLEVEL%==0 (
    echo ✓ nvcc可用
    "%CUDA_PATH%\bin\nvcc.exe" --version
) else (
    echo ✗ nvcc不可用
)

echo.
echo 建议：重启计算机以确保所有程序都能访问CUDA
echo.
pause


