# -*- mode: python ; coding: utf-8 -*-
"""
修复后的打包配置文件
解决 importlib.metadata 找不到包元数据的问题
"""

import os
from PyInstaller.utils.hooks import collect_data_files, collect_submodules, collect_all

block_cipher = None

# 收集streamlit的所有数据文件和子模块
streamlit_datas, streamlit_binaries, streamlit_hiddenimports = collect_all('streamlit')
cv2_datas, cv2_binaries, cv2_hiddenimports = collect_all('cv2')

a = Analysis(
    ['launcher.py'],  # 使用修复后的启动器作为入口
    pathex=[],
    binaries=streamlit_binaries + cv2_binaries,
    datas=[
        ('cut_image.py', '.'),
        ('app.py', '.'),  # 包含app.py
    ] + streamlit_datas + cv2_datas,
    hiddenimports=[
        'streamlit',
        'cv2',
        'numpy',
        'PIL',
        'streamlit.web.cli',
        'streamlit.runtime.scriptrunner.magic_funcs',
        'importlib.metadata',
        'importlib_metadata',
        'pkg_resources',
        'packaging',
        'packaging.version',
        'packaging.specifiers',
        'altair',
        'pandas',
        'pyarrow',
    ] + streamlit_hiddenimports + cv2_hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'langchain',  # streamlit的可选依赖，不需要
        'streamlit.external.langchain',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='CutImage',
    # 将app.py也包含进去
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,  # 显示控制台窗口以便查看日志
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,
)

