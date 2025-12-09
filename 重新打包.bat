@echo off
echo ========================================
echo 重新打包程序（使用修复配置）
echo ========================================
echo.

REM 清理之前的打包文件
echo 清理之前的打包文件...
if exist build (
    echo 删除 build 目录...
    rmdir /s /q build
)
if exist dist (
    echo 删除 dist 目录...
    rmdir /s /q dist
)
if exist CutImage.spec (
    echo 删除旧的 spec 文件...
    del /q CutImage.spec
)

echo.
echo 开始打包...
echo.

REM 使用修复后的spec文件打包
pyinstaller build_fixed.spec

echo.
if exist dist\CutImage.exe (
    echo ========================================
    echo 打包成功！
    echo ========================================
    echo 可执行文件位置: %CD%\dist\CutImage.exe
    echo.
    echo 现在可以运行 dist\CutImage.exe 测试程序
) else (
    echo ========================================
    echo 打包失败！
    echo ========================================
    echo 请检查上面的错误信息
)

echo.
pause



