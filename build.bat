@echo off
echo 正在打包程序...
echo.

REM 检查是否安装了PyInstaller
python -c "import PyInstaller" 2>nul
if errorlevel 1 (
    echo 正在安装PyInstaller...
    pip install pyinstaller
)

REM 清理之前的打包文件
if exist build rmdir /s /q build
if exist dist rmdir /s /q dist
if exist CutImage.spec del /q CutImage.spec

echo 使用修复后的配置文件打包...
REM 使用修复后的spec文件打包
pyinstaller build_fixed.spec

if errorlevel 1 (
    echo.
    echo 打包失败！尝试使用标准配置...
    pyinstaller build.spec
)

echo.
if exist dist\CutImage.exe (
    echo 打包完成！可执行文件在 dist 目录中
    echo 文件路径: %CD%\dist\CutImage.exe
) else (
    echo 打包失败，请检查错误信息
)

pause

