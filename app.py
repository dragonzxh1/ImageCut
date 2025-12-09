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
from cut_image import detect_and_crop_background, split_image_2x2, save_paired_images, process_single_subimage

# 页面配置
st.set_page_config(
    page_title="图片切割工具",
    page_icon="✂️",
    layout="wide"
)

# 标题
st.title("✂️ 图片切割工具")
st.markdown("---")

# 说明
st.markdown("""
### 使用说明
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
if st.button("🚀 开始处理", type="primary", use_container_width=True):
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
                    
                    # 读取图片
                    front_img = cv2.imread(str(front_path))
                    back_img = cv2.imread(str(back_path))
                    
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # 第一步：先切割图片成子图
                    status_text.text("正在切割图片...")
                    progress_bar.progress(20)
                    front_parts_raw = split_image_2x2(front_img)
                    back_parts_raw = split_image_2x2(back_img)
                    
                    # 验证切割结果数量是否一致
                    front_count = len(front_parts_raw)
                    back_count = len(back_parts_raw)
                    front_keys = set(front_parts_raw.keys())
                    back_keys = set(back_parts_raw.keys())
                    
                    # 检查数量是否一致
                    if front_count != back_count:
                        st.error(f"""
                        ❌ **错误：切割结果数量不一致！**
                        
                        - 正面图片切割后得到 **{front_count}** 个部分
                        - 背面图片切割后得到 **{back_count}** 个部分
                        
                        **请检查：**
                        1. 正面和背面图片是否都包含相同数量的子图
                        2. 图片布局是否一致
                        
                        **请重新上传正确的图片！**
                        """)
                        st.stop()
                    
                    # 检查位置是否一致
                    if front_keys != back_keys:
                        missing_in_back = front_keys - back_keys
                        missing_in_front = back_keys - front_keys
                        error_msg = "❌ **错误：切割结果位置不一致！**\n\n"
                        if missing_in_back:
                            error_msg += f"- 正面图片包含但背面图片缺失的位置: {', '.join(missing_in_back)}\n"
                        if missing_in_front:
                            error_msg += f"- 背面图片包含但正面图片缺失的位置: {', '.join(missing_in_front)}\n"
                        error_msg += "\n**请重新上传正确的图片！**"
                        st.error(error_msg)
                        st.stop()
                    
                    # 第二步：对每个子图分别进行顶点检测和裁剪
                    status_text.text("正在处理每个子图的背景...")
                    progress_bar.progress(40)
                    
                    front_parts = {}
                    back_parts = {}
                    
                    positions = ['top-left', 'top-right', 'bottom-left', 'bottom-right']
                    position_names = {
                        'top-left': '左上',
                        'top-right': '右上',
                        'bottom-left': '左下',
                        'bottom-right': '右下'
                    }
                    
                    for i, position in enumerate(positions):
                        if position in front_parts_raw:
                            progress = 40 + int((i + 1) / len(positions) * 40)
                            progress_bar.progress(progress)
                            status_text.text(f"正在处理 {position_names[position]} ({position})...")
                            
                            front_parts[position] = process_single_subimage(front_parts_raw[position])
                            back_parts[position] = process_single_subimage(back_parts_raw[position])
                    
                    # 显示原始切割的子图
                    st.markdown("### ✂️ 原始切割结果")
                    for position in positions:
                        if position in front_parts_raw:
                            st.markdown(f"#### {position_names[position]} ({position}) - 原始切割")
                            col_f, col_b = st.columns(2)
                            with col_f:
                                st.image(cv2.cvtColor(front_parts_raw[position], cv2.COLOR_BGR2RGB), 
                                        caption="正面原始", use_container_width=True)
                            with col_b:
                                st.image(cv2.cvtColor(back_parts_raw[position], cv2.COLOR_BGR2RGB), 
                                        caption="背面原始", use_container_width=True)
                    
                    # 验证切割结果数量是否一致
                    front_count = len(front_parts)
                    back_count = len(back_parts)
                    front_keys = set(front_parts.keys())
                    back_keys = set(back_parts.keys())
                    
                    # 检查数量是否一致
                    if front_count != back_count:
                        st.error(f"""
                        ❌ **错误：切割结果数量不一致！**
                        
                        - 正面图片切割后得到 **{front_count}** 个部分
                        - 背面图片切割后得到 **{back_count}** 个部分
                        
                        **请检查：**
                        1. 正面和背面图片是否都包含相同数量的子图
                        2. 图片布局是否一致
                        3. 背景裁剪是否正确
                        
                        **请重新上传正确的图片！**
                        """)
                        st.stop()
                    
                    # 检查位置是否一致
                    if front_keys != back_keys:
                        missing_in_back = front_keys - back_keys
                        missing_in_front = back_keys - front_keys
                        error_msg = "❌ **错误：切割结果位置不一致！**\n\n"
                        if missing_in_back:
                            error_msg += f"- 正面图片包含但背面图片缺失的位置: {', '.join(missing_in_back)}\n"
                        if missing_in_front:
                            error_msg += f"- 背面图片包含但正面图片缺失的位置: {', '.join(missing_in_front)}\n"
                        error_msg += "\n**请重新上传正确的图片！**"
                        st.error(error_msg)
                        st.stop()
                    
                    # 检查每个部分是否有效（不为空）
                    invalid_parts = []
                    for key in front_keys:
                        if key in front_parts and front_parts[key].size == 0:
                            invalid_parts.append(f"正面-{key}")
                        if key in back_parts and back_parts[key].size == 0:
                            invalid_parts.append(f"背面-{key}")
                    
                    if invalid_parts:
                        st.error(f"""
                        ❌ **错误：部分处理结果为空！**
                        
                        以下位置处理失败：
                        {', '.join(invalid_parts)}
                        
                        **可能的原因：**
                        1. 图片尺寸太小
                        2. 背景裁剪过度
                        3. 图片格式不正确
                        
                        **请重新上传正确的图片！**
                        """)
                        st.stop()
                    
                    # 保存配对图片
                    status_text.text("正在保存结果...")
                    progress_bar.progress(90)
                    save_paired_images(front_parts, back_parts, str(output_dir))
                    
                    progress_bar.progress(100)
                    status_text.text("处理完成！")
                    
                    # 显示最终处理结果预览
                    st.markdown("### ✅ 最终处理结果（已裁剪背景）")
                    for position in positions:
                        if position in front_parts:
                            st.markdown(f"#### {position_names[position]} ({position})")
                            col_f, col_b = st.columns(2)
                            with col_f:
                                st.image(cv2.cvtColor(front_parts[position], cv2.COLOR_BGR2RGB), 
                                        caption="正面（已处理）", use_container_width=True)
                            with col_b:
                                st.image(cv2.cvtColor(back_parts[position], cv2.COLOR_BGR2RGB), 
                                        caption="背面（已处理）", use_container_width=True)
                    
                    # 创建ZIP文件供下载
                    zip_buffer = io.BytesIO()
                    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                        for position in positions:
                            position_dir = output_dir / position
                            zip_file.write(position_dir / 'front.jpg', 
                                         f'{position}/front.jpg')
                            zip_file.write(position_dir / 'back.jpg', 
                                         f'{position}/back.jpg')
                    
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
                    
                    st.success("✅ 处理完成！所有图片已按位置配对保存。")
                    
            except Exception as e:
                st.error(f"❌ 处理失败: {str(e)}")
                st.exception(e)

# 页脚
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>图片切割工具 - 自动检测背景并切割图片</p>
</div>
""", unsafe_allow_html=True)

