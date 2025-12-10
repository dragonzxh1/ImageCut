# 修复PyInstaller打包后的元数据问题
import sys
import os
if getattr(sys, 'frozen', False):
    # 如果是打包后的exe，修复importlib.metadata路径
    try:
        import importlib.metadata
        if hasattr(importlib.metadata, '_cache'):
            importlib.metadata._cache.clear()
    except:
        pass

import streamlit as st
import cv2
import numpy as np
from pathlib import Path
import tempfile
import shutil
import zipfile
import io
from PIL import Image as PILImage
from cut_image import process_images
import os

# 页面配置
st.set_page_config(
    page_title="图片切割工具",
    page_icon="✂️",
    layout="wide"
)

# 标题
st.title("✂️ 图片切割工具")
st.markdown("---")

# 创建标签页切换功能
tab1, tab2 = st.tabs(["📷 单张处理", "📁 批量处理"])

# ==================== 标签页1：单张处理 ====================
with tab1:
    st.markdown("### 单张处理")
    st.markdown("""
    #### 使用说明
    1. 上传正面图片（包含4张子图的完整图片）
    2. 上传背面图片（包含4张子图的完整图片）
    3. 点击"开始处理"按钮
    4. 系统会自动检测并裁剪空白/深色背景，然后将图片切割成2x2的4个区域
    5. 处理完成后可以预览和下载结果
    """)
    
    # 创建两列布局
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📷 正面图片")
        front_file = st.file_uploader(
            "上传正面图片",
            type=['jpg', 'jpeg', 'png'],
            key='front'
        )
        
        if front_file is not None:
            # 显示上传的图片
            front_bytes = front_file.read()
            front_array = np.frombuffer(front_bytes, np.uint8)
            front_image = cv2.imdecode(front_array, cv2.IMREAD_COLOR)
            front_image_rgb = cv2.cvtColor(front_image, cv2.COLOR_BGR2RGB)
            st.image(front_image_rgb, caption="正面图片预览", use_container_width=True)
            st.info(f"图片尺寸: {front_image.shape[1]} x {front_image.shape[0]}")

    with col2:
        st.subheader("📷 背面图片")
        back_file = st.file_uploader(
            "上传背面图片",
            type=['jpg', 'jpeg', 'png'],
            key='back'
        )
        
        if back_file is not None:
            # 显示上传的图片
            back_bytes = back_file.read()
            back_array = np.frombuffer(back_bytes, np.uint8)
            back_image = cv2.imdecode(back_array, cv2.IMREAD_COLOR)
            back_image_rgb = cv2.cvtColor(back_image, cv2.COLOR_BGR2RGB)
            st.image(back_image_rgb, caption="背面图片预览", use_container_width=True)
            st.info(f"图片尺寸: {back_image.shape[1]} x {back_image.shape[0]}")

    # 处理按钮
    st.markdown("---")
    if st.button("🚀 开始处理", type="primary", use_container_width=True, key='process_single'):
        if front_file is None or back_file is None:
            st.error("❌ 请先上传正面和背面图片！")
        else:
            with st.spinner("正在处理图片，请稍候..."):
                try:
                    # 创建临时目录
                    with tempfile.TemporaryDirectory() as temp_dir:
                        output_dir = Path(temp_dir) / "output"
                        
                        # 保存上传的图片到临时文件
                        front_path = Path(temp_dir) / "front.jpg"
                        back_path = Path(temp_dir) / "back.jpg"
                        
                        front_file.seek(0)
                        with open(front_path, 'wb') as f:
                            f.write(front_file.read())
                        
                        back_file.seek(0)
                        with open(back_path, 'wb') as f:
                            f.write(back_file.read())
                        
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        # 使用新的处理流程
                        status_text.text("正在处理图片（新流程：rembg移除背景 -> 分割 -> 摆正）...")
                        progress_bar.progress(10)
                        
                        # 调用新的process_images函数
                        process_images(str(front_path), str(back_path), str(output_dir))
                        
                        progress_bar.progress(100)
                        status_text.text("处理完成！")
                        
                        # 读取处理结果用于显示
                        front_img = cv2.imread(str(front_path))
                        back_img = cv2.imread(str(back_path))
                        
                        # 显示中间过程图（新流程：白色背景JPG）
                        intermediate_dir = output_dir / 'intermediate'
                        if intermediate_dir.exists():
                            st.markdown("### 📸 中间过程图（背景移除后）")
                            # 查找JPG格式的中间过程图
                            front_no_bg = intermediate_dir / 'front_no_bg.jpg'
                            back_no_bg = intermediate_dir / 'back_no_bg.jpg'
                            
                            col_f, col_b = st.columns(2)
                            if front_no_bg.exists():
                                with col_f:
                                    st.image(str(front_no_bg), caption="正面（背景已移除）", use_container_width=True)
                            if back_no_bg.exists():
                                with col_b:
                                    st.image(str(back_no_bg), caption="背面（背景已移除）", use_container_width=True)
                        
                        # 显示最终结果（新流程：每张卡一个目录）
                        st.markdown("### ✅ 最终处理结果（已摆正的卡片）")
                        # 查找所有卡片目录（排除 intermediate 目录）
                        card_dirs = sorted([d for d in output_dir.iterdir() if d.is_dir() and d.name != 'intermediate'])
                        for card_dir in card_dirs:
                            # 查找目录中的 _A.jpg 和 _B.jpg 文件
                            front_files = list(card_dir.glob("*_A.jpg"))
                            back_files = list(card_dir.glob("*_B.jpg"))
                            
                            if front_files and back_files:
                                front_file = front_files[0]
                                back_file = back_files[0]
                                # 使用目录名作为卡号显示
                                card_number = card_dir.name
                                st.markdown(f"#### 卡号: {card_number}")
                                col_f, col_b = st.columns(2)
                                with col_f:
                                    st.image(str(front_file), caption="正面卡片", use_container_width=True)
                                with col_b:
                                    st.image(str(back_file), caption="背面卡片", use_container_width=True)
                        
                        # 创建ZIP文件供下载
                        zip_buffer = io.BytesIO()
                        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                            # 添加所有卡片目录（每张卡一个目录，包含卡号_A.jpg、卡号_B.jpg、卡号_label.txt）
                            card_dirs = sorted([d for d in output_dir.iterdir() if d.is_dir() and d.name != 'intermediate'])
                            for card_dir in card_dirs:
                                # 查找目录中的所有文件
                                front_files = list(card_dir.glob("*_A.jpg"))
                                back_files = list(card_dir.glob("*_B.jpg"))
                                label_files = list(card_dir.glob("*_label.txt"))
                                
                                # 添加A面
                                if front_files:
                                    zip_file.write(front_files[0], f'{card_dir.name}/{front_files[0].name}')
                                
                                # 添加B面
                                if back_files:
                                    zip_file.write(back_files[0], f'{card_dir.name}/{back_files[0].name}')
                                
                                # 添加标签文字文件
                                if label_files:
                                    zip_file.write(label_files[0], f'{card_dir.name}/{label_files[0].name}')
                            
                            # 添加中间过程图（JPG格式）
                            if intermediate_dir.exists():
                                for img_file in intermediate_dir.glob('*.jpg'):
                                    zip_file.write(img_file, f'intermediate/{img_file.name}')
                        
                        zip_buffer.seek(0)
                        
                        # 下载按钮
                        st.markdown("### 📥 下载结果")
                        st.download_button(
                            label="📦 下载所有结果 (ZIP)",
                            data=zip_buffer,
                            file_name="cut_images_result.zip",
                            mime="application/zip",
                            use_container_width=True
                        )
                        
                        st.success("✅ 处理完成！已移除背景、分割并摆正所有卡片。")
                    
                except Exception as e:
                    st.error(f"❌ 处理失败: {str(e)}")
                    st.exception(e)

# ==================== 标签页2：批量处理 ====================
with tab2:
    st.markdown("### 批量处理")
    st.markdown("""
    #### 使用说明
    1. 选择包含JPG图片的目录
    2. 系统会自动检测目录中的所有JPG文件
    3. 按照文件名顺序配对：第1个是正面，第2个是背面，第3个是正面，第4个是背面，以此类推
    4. 必须是双数文件才能配对处理
    5. 点击"开始批量处理"按钮
    6. 处理完成后可以下载所有结果
    """)
    
    # 目录选择
    st.markdown("#### 📁 选择图片目录")
    
    # 显示当前工作目录和示例
    current_dir = os.getcwd()
    st.info(f"💡 当前工作目录: `{current_dir}`")
    st.markdown("**提示：** 可以使用相对路径（如 `./images`）或绝对路径（如 `D:/Images` 或 `D:\\Images`）")
    
    input_dir = st.text_input(
        "输入图片目录路径",
        value="",
        help="例如: D:/Images 或 ./images 或 images",
        key='batch_input_dir'
    )
    
    if input_dir:
        # 处理相对路径和绝对路径
        input_path = Path(input_dir)
        if not input_path.is_absolute():
            # 如果是相对路径，相对于当前工作目录
            input_path = Path(current_dir) / input_path
        
        if input_path.exists() and input_path.is_dir():
            # 查找所有JPG文件（使用集合去重，避免大小写重复）
            jpg_files_set = set(input_path.glob("*.jpg")) | set(input_path.glob("*.JPG"))
            # 按文件名排序（不区分大小写）
            jpg_files = sorted(jpg_files_set, key=lambda x: x.name.lower())
            
            # 再次去重，确保没有重复文件（基于完整路径）
            seen_paths = set()
            unique_jpg_files = []
            for f in jpg_files:
                if str(f) not in seen_paths:
                    seen_paths.add(str(f))
                    unique_jpg_files.append(f)
            jpg_files = unique_jpg_files
            
            if jpg_files:
                st.info(f"找到 {len(jpg_files)} 个JPG文件")
                
                # 显示文件列表（调试用）
                with st.expander("📂 查看所有文件列表（按顺序）"):
                    for idx, f in enumerate(jpg_files):
                        st.text(f"索引 {idx}: {f.name}")
                
                # 检查是否为双数
                if len(jpg_files) % 2 != 0:
                    st.warning(f"⚠️ 警告：找到 {len(jpg_files)} 个文件，不是双数，无法完全配对。最后一个文件将被忽略。")
                    jpg_files = jpg_files[:-1]  # 移除最后一个文件
                
                if len(jpg_files) >= 2:
                    # 显示配对信息
                    st.markdown("#### 📋 文件配对列表")
                    pairs = []
                    num_pairs = len(jpg_files) // 2
                    for pair_idx in range(num_pairs):
                        front_idx = pair_idx * 2
                        back_idx = pair_idx * 2 + 1
                        
                        # 验证索引有效性
                        if front_idx >= len(jpg_files) or back_idx >= len(jpg_files):
                            st.error(f"❌ 配对 {pair_idx + 1} 索引错误: front_idx={front_idx}, back_idx={back_idx}, 总文件数={len(jpg_files)}")
                            continue
                        
                        front_file = jpg_files[front_idx]
                        back_file = jpg_files[back_idx]
                        
                        # 验证不是同一个文件
                        if front_file == back_file or str(front_file) == str(back_file):
                            st.error(f"❌ 配对 {pair_idx + 1} 错误：正面和背面是同一个文件: {front_file.name}")
                            continue
                        
                        pairs.append((front_file, back_file))
                        st.text(f"配对 {pair_idx + 1}: [{front_idx}] {front_file.name} (正面) ↔ [{back_idx}] {back_file.name} (背面)")
                    
                    # 批量处理按钮
                    st.markdown("---")
                    if st.button("🚀 开始批量处理", type="primary", use_container_width=True, key='process_batch'):
                        with st.spinner("正在批量处理图片，请稍候..."):
                            try:
                                # 创建临时目录存储所有结果
                                with tempfile.TemporaryDirectory() as temp_dir:
                                    all_results_dir = Path(temp_dir) / "all_results"
                                    all_results_dir.mkdir(exist_ok=True)
                                    
                                    progress_bar = st.progress(0)
                                    status_text = st.empty()
                                    
                                    successful_pairs = 0
                                    failed_pairs = 0
                                    
                                    # 处理每一对图片
                                    for idx, (front_file, back_file) in enumerate(pairs):
                                        pair_num = idx + 1
                                        status_text.text(f"正在处理第 {pair_num}/{len(pairs)} 对图片: {front_file.name} ↔ {back_file.name}")
                                        progress_bar.progress((idx) / len(pairs))
                                        
                                        try:
                                            # 为每对图片创建输出目录
                                            pair_output_dir = all_results_dir / f"pair_{pair_num:03d}"
                                            
                                            # 调用处理函数
                                            process_images(
                                                str(front_file),
                                                str(back_file),
                                                str(pair_output_dir)
                                            )
                                            
                                            successful_pairs += 1
                                            
                                        except Exception as e:
                                            st.error(f"❌ 处理第 {pair_num} 对图片失败: {str(e)}")
                                            failed_pairs += 1
                                    
                                    progress_bar.progress(1.0)
                                    status_text.text("批量处理完成！")
                                    
                                    # 创建ZIP文件包含所有结果
                                    zip_buffer = io.BytesIO()
                                    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                                        # 使用集合跟踪已添加的文件，避免重复
                                        added_files = set()
                                        
                                        # 添加所有配对的结果
                                        for pair_dir in sorted(all_results_dir.iterdir()):
                                            if pair_dir.is_dir():
                                                # 添加所有卡片目录
                                                for card_dir in pair_dir.iterdir():
                                                    if card_dir.is_dir() and card_dir.name != 'intermediate':
                                                        for file in card_dir.glob("*"):
                                                            if file.is_file():
                                                                # 保持目录结构：pair_001/卡号/文件名
                                                                zip_path = f'{pair_dir.name}/{card_dir.name}/{file.name}'
                                                                if zip_path not in added_files:
                                                                    zip_file.write(file, zip_path)
                                                                    added_files.add(zip_path)
                                                
                                                # 添加中间过程图
                                                intermediate_dir = pair_dir / 'intermediate'
                                                if intermediate_dir.exists():
                                                    for img_file in intermediate_dir.glob('*.jpg'):
                                                        zip_path = f'{pair_dir.name}/intermediate/{img_file.name}'
                                                        if zip_path not in added_files:
                                                            zip_file.write(img_file, zip_path)
                                                            added_files.add(zip_path)
                                    
                                    zip_buffer.seek(0)
                                    
                                    # 显示处理结果统计
                                    st.markdown("### 📊 处理结果统计")
                                    st.success(f"✅ 成功处理: {successful_pairs} 对")
                                    if failed_pairs > 0:
                                        st.error(f"❌ 失败: {failed_pairs} 对")
                                    
                                    # 下载按钮
                                    st.markdown("### 📥 下载所有结果")
                                    st.download_button(
                                        label="📦 下载所有结果 (ZIP)",
                                        data=zip_buffer,
                                        file_name="batch_cut_images_result.zip",
                                        mime="application/zip",
                                        use_container_width=True
                                    )
                                    
                            except Exception as e:
                                st.error(f"❌ 批量处理失败: {str(e)}")
                                st.exception(e)
                else:
                    st.warning("⚠️ 需要至少2个JPG文件才能进行配对处理")
            else:
                st.warning("⚠️ 目录中没有找到JPG文件")
        else:
            st.error("❌ 目录不存在或不是有效目录")

# 页脚
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>图片切割工具 - 自动检测背景并切割图片</p>
</div>
""", unsafe_allow_html=True)

