"""
修复脚本：在打包后的exe中修复importlib.metadata问题
这个文件需要在app.py开头导入
"""
import sys
import os

# 修复importlib.metadata在PyInstaller打包后的路径问题
if getattr(sys, 'frozen', False):
    # 如果是打包后的exe
    base_path = sys._MEIPASS
    
    # 添加streamlit的dist-info路径
    streamlit_path = os.path.join(base_path, 'streamlit')
    if os.path.exists(streamlit_path):
        # 尝试修复元数据路径
        try:
            import importlib.metadata
            # 强制重新加载元数据
            if hasattr(importlib.metadata, '_cache'):
                importlib.metadata._cache.clear()
        except:
            pass



