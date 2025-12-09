# -*- mode: python ; coding: utf-8 -*-

import os
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

block_cipher = None

# 收集streamlit的所有数据文件和子模块
streamlit_datas = collect_data_files('streamlit')
streamlit_hiddenimports = collect_submodules('streamlit')

a = Analysis(
    ['app.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('cut_image.py', '.'),
    ] + streamlit_datas,
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
        'pkg_resources.py2_warn',
        'packaging',
        'packaging.version',
        'packaging.specifiers',
    ] + streamlit_hiddenimports,
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

