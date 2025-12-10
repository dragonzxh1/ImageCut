@echo off
echo 启动GPU版本的卡片图像处理工具...
echo.
echo 正在激活GPU虚拟环境...
call venv_gpu\Scripts\activate.bat
echo.
echo 启动Streamlit应用...
streamlit run app_gpu.py
pause

