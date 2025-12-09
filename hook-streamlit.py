"""
PyInstaller hook for Streamlit
解决 importlib.metadata 找不到包元数据的问题
"""
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

# 收集所有streamlit相关的数据文件
datas = collect_data_files('streamlit')

# 收集所有streamlit的子模块
hiddenimports = collect_submodules('streamlit')

# 添加必要的隐藏导入
hiddenimports += [
    'importlib.metadata',
    'importlib_metadata',
    'pkg_resources',
    'streamlit.web.cli',
    'streamlit.runtime.scriptrunner.magic_funcs',
]



