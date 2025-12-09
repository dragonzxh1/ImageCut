"""
打包脚本 - 使用PyInstaller打包Streamlit应用
"""
import PyInstaller.__main__
import os
import sys

# PyInstaller参数
args = [
    'app.py',  # 主程序文件
    '--name=CutImage',  # 可执行文件名
    '--onefile',  # 打包成单个文件
    '--windowed',  # Windows下不显示控制台窗口（如果需要显示日志，改为--console）
    '--add-data=cut_image.py;.',  # 包含cut_image.py模块
    '--hidden-import=streamlit',  # 确保包含streamlit
    '--hidden-import=cv2',  # 确保包含opencv
    '--hidden-import=numpy',  # 确保包含numpy
    '--hidden-import=PIL',  # 确保包含PIL
    '--collect-all=streamlit',  # 收集streamlit的所有数据文件
    '--collect-all=cv2',  # 收集opencv的所有数据文件
    '--icon=NONE',  # 如果有图标文件可以指定
]

# 执行打包
PyInstaller.__main__.run(args)



