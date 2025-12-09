@echo off
echo 启动图片切割工具...
echo.

REM 激活虚拟环境（如果存在）
if exist venv\Scripts\activate.bat (
    call venv\Scripts\activate.bat
)

REM 运行Streamlit应用
streamlit run app.py

pause



