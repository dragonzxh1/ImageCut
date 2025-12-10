import cv2
import numpy as np
import os
from pathlib import Path
import argparse
from typing import Optional, Tuple, List, Dict
try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False
    print("警告: easyocr未安装，文字提取功能将不可用。请安装: pip install easyocr")

try:
    from rembg import remove, new_session
    REMBG_AVAILABLE = True
except ImportError:
    REMBG_AVAILABLE = False
    print("警告: rembg未安装，背景移除功能将不可用。请安装: pip install rembg")

# 检查GPU是否可用
GPU_AVAILABLE = False
try:
    import torch
    GPU_AVAILABLE = torch.cuda.is_available()
    if GPU_AVAILABLE:
        print(f"✓ GPU可用: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠ GPU不可用，将使用CPU模式")
except ImportError:
    print("⚠ PyTorch未安装，无法检测GPU状态")
except Exception as e:
    print(f"⚠ GPU检测失败: {e}，将使用CPU模式")

# ============================================================================
# 图像预处理函数 - 用于提高OCR准确性，处理噪点问题
# ============================================================================

def denoise_image_for_ocr(image: np.ndarray, method='adaptive') -> np.ndarray:
    """
    对图像进行去噪处理，提高OCR准确性
    
    Args:
        image: 输入图像（BGR格式）
        method: 去噪方法
            - 'adaptive': 自适应去噪（推荐）
            - 'gaussian': 高斯模糊
            - 'median': 中值滤波
            - 'morphology': 形态学操作
            - 'combined': 组合方法
    
    Returns:
        去噪后的图像（BGR格式）
    """
    if method == 'adaptive':
        # 自适应去噪：根据图像特征选择最佳方法
        # 转换为灰度图进行分析
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 计算图像对比度
        contrast = np.std(gray)
        
        # 如果对比度较低，使用更强的去噪
        if contrast < 30:
            # 低对比度：使用形态学操作 + 高斯模糊
            denoised = apply_morphology_denoise(image)
            denoised = cv2.GaussianBlur(denoised, (3, 3), 0)
        elif contrast < 60:
            # 中等对比度：使用中值滤波
            denoised = cv2.medianBlur(image, 3)
        else:
            # 高对比度：轻微高斯模糊
            denoised = cv2.GaussianBlur(image, (3, 3), 0.5)
        
        return denoised
    
    elif method == 'gaussian':
        return cv2.GaussianBlur(image, (3, 3), 0)
    
    elif method == 'median':
        return cv2.medianBlur(image, 3)
    
    elif method == 'morphology':
        return apply_morphology_denoise(image)
    
    elif method == 'combined':
        # 组合方法：先中值滤波，再形态学操作
        denoised = cv2.medianBlur(image, 3)
        denoised = apply_morphology_denoise(denoised)
        return denoised
    
    else:
        return image


def apply_morphology_denoise(image: np.ndarray) -> np.ndarray:
    """
    使用形态学操作去噪
    适用于去除小的噪点，同时保持文字清晰
    
    Args:
        image: 输入图像（BGR格式）
    
    Returns:
        去噪后的图像
    """
    # 转换为灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 创建小的结构元素（用于去除小噪点）
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    
    # 开运算：先腐蚀后膨胀，去除小噪点
    opened = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # 闭运算：先膨胀后腐蚀，连接断开的文字
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    # 转换回BGR
    result = cv2.cvtColor(closed, cv2.COLOR_GRAY2BGR)
    
    return result


def enhance_contrast_for_ocr(image: np.ndarray) -> np.ndarray:
    """
    增强图像对比度，提高OCR准确性
    
    Args:
        image: 输入图像（BGR格式）
    
    Returns:
        增强后的图像
    """
    # 转换为LAB色彩空间
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # 对L通道应用CLAHE（对比度受限的自适应直方图均衡化）
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    
    # 合并通道并转换回BGR
    enhanced = cv2.merge([l, a, b])
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
    
    return enhanced


# ============================================================================
# 关键调整参数 - 如果运行结果不完美，请调整以下参数
# ============================================================================

# REMOVE_BORDER_PIXELS: 检测到边缘后，向内收缩的像素数（用于去除灰色纸片边缘）
# 现象：如果"灰色边缘"还在
# 调整：把这个值设大一点（比如 20 或 30），程序会在检测到边缘后自动向内收缩裁剪
REMOVE_BORDER_PIXELS = 20

# BORDER_THRESHOLD: 黑边检测阈值（用于 remove_black_border 函数）
# 现象：如果程序把"背景"也当成了卡片切进去了（切出来的图黑边巨大）
# 调整：把 30 调大（比如 50 或 60）。这意味着"只有更亮的东西才算卡片"
# 现象：如果卡片透明边缘没切到，只切到了里面的彩图
# 调整：把 30 调小（比如 10 或 15），或者增强灯光
BORDER_THRESHOLD = 30


def remove_black_border(image: np.ndarray, border_threshold: int = None) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """
    去除图像边缘的黑边
    
    关键调整参数：
    - border_threshold: 如果程序把"背景"也当成了卡片切进去了（切出来的图黑边巨大），
                        把 30 调大（比如 50 或 60）。这意味着"只有更亮的东西才算卡片"。
                        如果卡片透明边缘没切到，只切到了里面的彩图，把 30 调小（比如 10 或 15）。
    
    Args:
        image: 输入图像（BGR格式）
        border_threshold: 黑边检测阈值（默认 30）
        
    Returns:
        (去除黑边后的图像, (x_min, y_min, x_max, y_max) 边界框)
    """
    # 使用全局配置参数，如果未指定则使用默认值
    if border_threshold is None:
        border_threshold = BORDER_THRESHOLD
    
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 检测黑边：边缘区域如果大部分像素值 < border_threshold，认为是黑边
    border_margin = max(10, min(h, w) // 20)  # 边缘检测区域
    
    # 从上边缘开始检测
    top_border = 0
    for y in range(border_margin):
        if np.mean(gray[y, :]) > border_threshold:
            top_border = max(0, y - 2)  # 留一点余量
            break
    
    # 从下边缘开始检测
    bottom_border = h
    for y in range(h - 1, h - border_margin - 1, -1):
        if np.mean(gray[y, :]) > border_threshold:
            bottom_border = min(h, y + 3)  # 留一点余量
            break
    
    # 从左边缘开始检测
    left_border = 0
    for x in range(border_margin):
        if np.mean(gray[:, x]) > border_threshold:
            left_border = max(0, x - 2)  # 留一点余量
            break
    
    # 从右边缘开始检测
    right_border = w
    for x in range(w - 1, w - border_margin - 1, -1):
        if np.mean(gray[:, x]) > border_threshold:
            right_border = min(w, x + 3)  # 留一点余量
            break
    
    # 裁剪去除黑边
    if top_border > 0 or bottom_border < h or left_border > 0 or right_border < w:
        cropped = image[top_border:bottom_border, left_border:right_border]
        return cropped, (left_border, top_border, right_border, bottom_border)
    
    return image, (0, 0, w, h)


def detect_inner_card_contour(image: np.ndarray) -> Optional[np.ndarray]:
    """
    检测内部卡片的外边缘轮廓（去除塑料外壳，只保留卡片本身）
    核心思路：检测卡片本身的边缘，而不是塑料外壳
    卡片特征：
    1. 有白色边框
    2. 内部有丰富的彩色内容（高饱和度）
    3. 比塑料外壳小
    4. 与塑料外壳之间有浅灰色间隙
    
    Args:
        image: 输入图像（BGR格式）
        
    Returns:
        卡片轮廓（4个顶点的四边形），如果找到，否则返回None
    """
    h, w = image.shape[:2]
    
    # 第一步：去除黑边
    image_no_border, border_box = remove_black_border(image)
    h_clean, w_clean = image_no_border.shape[:2]
    
    # 第二步：转换到不同颜色空间
    gray = cv2.cvtColor(image_no_border, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    
    # LAB颜色空间
    lab = cv2.cvtColor(image_no_border, cv2.COLOR_BGR2LAB)
    lab_l = lab[:, :, 0]  # 亮度通道
    lab_a = lab[:, :, 1]  # 绿-红通道
    lab_b = lab[:, :, 2]  # 蓝-黄通道
    blurred_lab_l = cv2.GaussianBlur(lab_l, (7, 7), 0)
    blurred_lab_a = cv2.GaussianBlur(lab_a, (7, 7), 0)
    blurred_lab_b = cv2.GaussianBlur(lab_b, (7, 7), 0)
    
    # HSV颜色空间
    hsv = cv2.cvtColor(image_no_border, cv2.COLOR_BGR2HSV)
    hsv_s = hsv[:, :, 1]  # 饱和度通道（卡片内容通常高饱和度）
    hsv_v = hsv[:, :, 2]  # 明度通道
    blurred_hsv_s = cv2.GaussianBlur(hsv_s, (7, 7), 0)
    blurred_hsv_v = cv2.GaussianBlur(hsv_v, (7, 7), 0)
    
    # 第三步：检测卡片特征（使用更简单直接的方法）
    # 方法1：检测高饱和度区域（卡片内部有丰富的彩色内容）
    # 降低阈值，确保能检测到卡片内容
    _, high_saturation = cv2.threshold(blurred_hsv_s, 50, 255, cv2.THRESH_BINARY)
    
    # 方法2：检测白色边框（卡片的外边缘）
    # 降低阈值，确保能检测到白色边框
    _, white_border = cv2.threshold(blurred_lab_l, 180, 255, cv2.THRESH_BINARY)
    
    # 方法3：检测高对比度区域（卡片内部有丰富的颜色变化）
    # 使用LAB A和B通道检测颜色变化
    sobelx_a = cv2.Sobel(blurred_lab_a, cv2.CV_64F, 1, 0, ksize=3)
    sobely_a = cv2.Sobel(blurred_lab_a, cv2.CV_64F, 0, 1, ksize=3)
    gradient_a = np.sqrt(sobelx_a**2 + sobely_a**2)
    
    sobelx_b = cv2.Sobel(blurred_lab_b, cv2.CV_64F, 1, 0, ksize=3)
    sobely_b = cv2.Sobel(blurred_lab_b, cv2.CV_64F, 0, 1, ksize=3)
    gradient_b = np.sqrt(sobelx_b**2 + sobely_b**2)
    
    color_gradient = gradient_a + gradient_b
    if color_gradient.max() > 0:
        color_gradient = np.uint8(np.clip(color_gradient * 255 / color_gradient.max(), 0, 255))
        _, color_edges = cv2.threshold(color_gradient, 30, 255, cv2.THRESH_BINARY)
    else:
        color_edges = np.zeros_like(blurred, dtype=np.uint8)
    
    # 方法4：使用OTSU自适应阈值（检测卡片区域）
    _, otsu_binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # 方法5：Canny边缘检测（检测卡片边缘）
    canny_edges = cv2.Canny(blurred, 20, 80)  # 降低阈值，更敏感
    
    # 合并所有检测结果（不使用掩码排除，让轮廓筛选来处理）
    combined = cv2.bitwise_or(high_saturation, white_border)
    combined = cv2.bitwise_or(combined, color_edges)
    combined = cv2.bitwise_or(combined, otsu_binary)
    combined = cv2.bitwise_or(combined, canny_edges)
    
    # 第四步：形态学操作（连接边缘，填充小孔）
    kernel = np.ones((5, 5), np.uint8)
    closed = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=2)
    dilated = cv2.dilate(closed, kernel, iterations=1)
    
    # 第五步：查找轮廓
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None
    
    # 第六步：筛选轮廓
    # 卡片通常占据图像的10%-80%（比塑料外壳小，但比细节大）
    # 先尝试较严格的范围
    min_area = h_clean * w_clean * 0.10  # 最小10%
    max_area = h_clean * w_clean * 0.80  # 最大80%（排除塑料外壳）
    
    valid_contours = [c for c in contours if min_area < cv2.contourArea(c) < max_area]
    
    if not valid_contours:
        # 如果没找到，放宽条件
        min_area = h_clean * w_clean * 0.05  # 最小5%
        max_area = h_clean * w_clean * 0.90  # 最大90%
        valid_contours = [c for c in contours if min_area < cv2.contourArea(c) < max_area]
    
    if not valid_contours:
        # 如果还是没找到，选择所有轮廓中面积最大的几个，然后筛选
        if contours:
            # 按面积排序，选择前5个最大的
            sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
            # 从中选择面积在合理范围内的
            valid_contours = [c for c in sorted_contours 
                            if cv2.contourArea(c) > h_clean * w_clean * 0.05]
    
    if not valid_contours:
        return None
    
    # 第七步：选择最合适的轮廓
    # 优先选择面积适中的轮廓（不是最大的，也不是最小的）
    # 同时考虑轮廓的紧凑性（矩形度）
    best_contour = None
    best_score = -1
    
    for contour in valid_contours:
        area = cv2.contourArea(contour)
        # 计算矩形度（轮廓面积 / 最小外接矩形面积）
        rect = cv2.minAreaRect(contour)
        rect_area = rect[1][0] * rect[1][1]
        if rect_area > 0:
            rectangularity = area / rect_area
        else:
            rectangularity = 0
        
        # 计算长宽比（接近1更好，即更接近正方形）
        if rect[1][0] > 0 and rect[1][1] > 0:
            aspect_ratio = max(rect[1][0], rect[1][1]) / min(rect[1][0], rect[1][1])
        else:
            aspect_ratio = 10  # 惩罚极端的长宽比
        
        # 综合评分：面积适中 + 矩形度高 + 长宽比合理
        # 面积权重：20%-60%之间得分最高（卡片通常在这个范围）
        area_ratio = area / (h_clean * w_clean)
        if 0.20 <= area_ratio <= 0.60:
            area_score = 1.0
        elif 0.15 <= area_ratio < 0.20 or 0.60 < area_ratio <= 0.70:
            area_score = 0.8
        elif 0.10 <= area_ratio < 0.15 or 0.70 < area_ratio <= 0.75:
            area_score = 0.6
        else:
            area_score = 0.3
        
        # 长宽比评分（卡片通常是竖着的，长宽比在1.3-1.8之间比较合理）
        if 1.0 <= aspect_ratio <= 2.0:
            aspect_score = 1.0
        elif 0.8 <= aspect_ratio < 1.0 or 2.0 < aspect_ratio <= 2.5:
            aspect_score = 0.7
        else:
            aspect_score = 0.4
        
        # 综合评分（矩形度权重最高，因为卡片应该是矩形）
        score = area_score * 0.3 + rectangularity * 0.5 + aspect_score * 0.2
        
        if score > best_score:
            best_score = score
            best_contour = contour
    
    if best_contour is None:
        # 如果评分失败，选择面积最大的有效轮廓
        best_contour = max(valid_contours, key=cv2.contourArea)
    
    # 第八步：多边形拟合 - 将轮廓拟合为四边形
    epsilon_start = 0.01
    epsilon_end = 0.1
    epsilon_step = 0.01
    
    best_approx = None
    
    for epsilon in np.arange(epsilon_start, epsilon_end, epsilon_step):
        epsilon_val = epsilon * cv2.arcLength(best_contour, True)
        approx = cv2.approxPolyDP(best_contour, epsilon_val, True)
        
        if len(approx) == 4:
            best_approx = approx
            break
        elif len(approx) > 4 and best_approx is None:
            best_approx = approx
    
    if best_approx is not None and len(best_approx) == 4:
        # 将坐标映射回原始图像（加上黑边偏移）
        x_offset, y_offset = border_box[0], border_box[1]
        best_approx = best_approx.reshape(-1, 2)
        best_approx[:, 0] += x_offset
        best_approx[:, 1] += y_offset
        return best_approx.reshape(-1, 1, 2)
    
    # 如果拟合后不是4个顶点，使用最小外接矩形
    rect = cv2.minAreaRect(best_contour)
    box = cv2.boxPoints(rect)
    box = np.int32(box)
    
    # 将坐标映射回原始图像（加上黑边偏移）
    x_offset, y_offset = border_box[0], border_box[1]
    box[:, 0] += x_offset
    box[:, 1] += y_offset
    
    return box.reshape(-1, 1, 2)


def detect_card_contour(image: np.ndarray) -> Optional[np.ndarray]:
    """
    检测内部卡片的外边缘轮廓（新方法：只检测卡片，不检测塑料外壳）
    这是新的主函数，调用detect_inner_card_contour
    
    Args:
        image: 输入图像（BGR格式）
        
    Returns:
        卡片轮廓（4个顶点的四边形），如果找到，否则返回None
    """
    return detect_inner_card_contour(image)
    h, w = image.shape[:2]
    
    # 第一步：先去除黑边
    image_no_border, border_box = remove_black_border(image)
    h_clean, w_clean = image_no_border.shape[:2]
    
    # 第二步：图像预处理 - 使用多种颜色空间增强颜色差异检测
    gray = cv2.cvtColor(image_no_border, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 0)
    
    # 转换到LAB颜色空间（对颜色差异更敏感）
    lab = cv2.cvtColor(image_no_border, cv2.COLOR_BGR2LAB)
    lab_l = lab[:, :, 0]  # 亮度通道
    lab_a = lab[:, :, 1]  # 绿-红通道
    lab_b = lab[:, :, 2]  # 蓝-黄通道
    blurred_lab_l = cv2.GaussianBlur(lab_l, (9, 9), 0)
    blurred_lab_a = cv2.GaussianBlur(lab_a, (9, 9), 0)
    blurred_lab_b = cv2.GaussianBlur(lab_b, (9, 9), 0)
    
    # 转换到HSV颜色空间（对色调差异更敏感）
    hsv = cv2.cvtColor(image_no_border, cv2.COLOR_BGR2HSV)
    hsv_h = hsv[:, :, 0]  # 色调通道
    hsv_s = hsv[:, :, 1]  # 饱和度通道
    hsv_v = hsv[:, :, 2]  # 明度通道
    blurred_hsv_h = cv2.GaussianBlur(hsv_h, (9, 9), 0)
    blurred_hsv_s = cv2.GaussianBlur(hsv_s, (9, 9), 0)
    blurred_hsv_v = cv2.GaussianBlur(hsv_v, (9, 9), 0)
    
    # 第三步：分析图像特征，判断背景类型
    edge_margin = 15
    edge_pixels = np.concatenate([
        gray[:edge_margin, :].flatten(),  # 上边缘
        gray[-edge_margin:, :].flatten(),  # 下边缘
        gray[:, :edge_margin].flatten(),  # 左边缘
        gray[:, -edge_margin:].flatten(),  # 右边缘
    ])
    
    edge_mean = np.mean(edge_pixels)
    edge_std = np.std(edge_pixels)
    
    # 判断背景类型
    is_white_background = edge_mean > 200  # 白色背景
    is_dark_background = edge_mean < 50    # 深色背景（黑色背景）
    is_transparent = edge_std > 30  # 透明材质（反光导致标准差大）
    
    # 检测浅灰色塑料材质（针对PSA评级卡片的塑料外壳）
    # 浅灰色通常在LAB L通道中值在100-180之间
    lab_l_mean = np.mean(blurred_lab_l)
    lab_l_std = np.std(blurred_lab_l)
    has_light_gray_plastic = (lab_l_mean > 80 and lab_l_mean < 200) and (lab_l_std > 20)
    
    # 第四步：根据背景类型选择不同的处理方法
    # 对于塑料材质卡片，需要检测塑料材质的外边缘，而不是卡片本身
    # 重点：利用颜色差异（LAB和HSV颜色空间）来检测塑料材质和纸质的边界
    
    # 通用方法：基于颜色差异的检测（对所有背景类型都有效）
    # 方法1：LAB颜色空间的A和B通道梯度（检测颜色变化）
    sobelx_a = cv2.Sobel(blurred_lab_a, cv2.CV_64F, 1, 0, ksize=5)
    sobely_a = cv2.Sobel(blurred_lab_a, cv2.CV_64F, 0, 1, ksize=5)
    gradient_a = np.sqrt(sobelx_a**2 + sobely_a**2)
    
    sobelx_b = cv2.Sobel(blurred_lab_b, cv2.CV_64F, 1, 0, ksize=5)
    sobely_b = cv2.Sobel(blurred_lab_b, cv2.CV_64F, 0, 1, ksize=5)
    gradient_b = np.sqrt(sobelx_b**2 + sobely_b**2)
    
    # 合并LAB颜色梯度
    color_gradient = gradient_a + gradient_b
    if color_gradient.max() > 0:
        color_gradient = np.uint8(np.clip(color_gradient * 255 / color_gradient.max(), 0, 255))
        _, color_binary = cv2.threshold(color_gradient, 30, 255, cv2.THRESH_BINARY)
    else:
        color_binary = np.zeros_like(blurred, dtype=np.uint8)
    
    # 方法2：HSV颜色空间的H通道梯度（检测色调变化）
    sobelx_h = cv2.Sobel(blurred_hsv_h.astype(np.float32), cv2.CV_64F, 1, 0, ksize=5)
    sobely_h = cv2.Sobel(blurred_hsv_h.astype(np.float32), cv2.CV_64F, 0, 1, ksize=5)
    gradient_h = np.sqrt(sobelx_h**2 + sobely_h**2)
    if gradient_h.max() > 0:
        gradient_h = np.uint8(np.clip(gradient_h * 255 / gradient_h.max(), 0, 255))
        _, h_binary = cv2.threshold(gradient_h, 30, 255, cv2.THRESH_BINARY)
    else:
        h_binary = np.zeros_like(blurred, dtype=np.uint8)
    
    # 方法3：LAB颜色空间的阈值检测（检测颜色区域）
    _, lab_a_binary = cv2.threshold(blurred_lab_a, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, lab_b_binary = cv2.threshold(blurred_lab_b, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    lab_color_diff = cv2.bitwise_xor(lab_a_binary, lab_b_binary)  # 颜色差异区域
    
    # 方法4：LAB颜色空间的Canny边缘检测（检测颜色变化边缘）
    # LAB A通道的Canny（检测绿-红颜色变化）
    canny_lab_a = cv2.Canny(blurred_lab_a, 30, 100)
    # LAB B通道的Canny（检测蓝-黄颜色变化）
    canny_lab_b = cv2.Canny(blurred_lab_b, 30, 100)
    # LAB L通道的Canny（检测亮度变化）
    canny_lab_l = cv2.Canny(blurred_lab_l, 30, 100)
    
    # 方法5：HSV颜色空间的Canny边缘检测（检测色调和饱和度变化）
    # HSV H通道的Canny（检测色调变化，对颜色差异敏感）
    canny_hsv_h = cv2.Canny(blurred_hsv_h, 20, 80)  # 降低阈值，因为H通道值域较小
    # HSV S通道的Canny（检测饱和度变化）
    canny_hsv_s = cv2.Canny(blurred_hsv_s, 30, 100)
    # HSV V通道的Canny（检测明度变化）
    canny_hsv_v = cv2.Canny(blurred_hsv_v, 30, 100)
    
    # 方法6：灰度图的传统边缘检测
    sobelx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=5)
    gradient = np.sqrt(sobelx**2 + sobely**2)
    if gradient.max() > 0:
        gradient = np.uint8(np.clip(gradient * 255 / gradient.max(), 0, 255))
    else:
        gradient = np.zeros_like(blurred, dtype=np.uint8)
    
    # 方法7：灰度图的Canny边缘检测
    canny_gray = cv2.Canny(blurred, 30, 100)
    
    # 方法8：专门检测浅灰色塑料边框与黑色背景的边界
    # 使用LAB L通道（亮度通道）来检测浅灰色（高亮度）与黑色（低亮度）的边界
    # 浅灰色塑料通常在L通道中值较高（100-180），黑色背景值很低（0-50）
    # 使用自适应阈值来检测这个边界
    _, lab_l_plastic = cv2.threshold(blurred_lab_l, 80, 255, cv2.THRESH_BINARY)  # 检测浅灰色区域
    # 使用Canny在LAB L通道上检测亮度变化（浅灰色到黑色的边界）
    canny_lab_l_plastic = cv2.Canny(blurred_lab_l, 20, 60)  # 较低阈值以检测浅灰色边界
    
    # 方法9：从边缘向内扫描，检测第一个明显的亮度变化（塑料边缘）
    # 专门针对浅灰色塑料与黑色背景的边界检测
    edge_mask = np.zeros_like(blurred_lab_l, dtype=np.uint8)
    
    if has_light_gray_plastic and is_dark_background:
        # 针对浅灰色塑料 + 黑色背景的情况
        scan_margin = 3  # 扫描边距
        brightness_threshold = 60  # 亮度阈值：黑色背景 < 60，浅灰色塑料 > 60
        jump_threshold = 40  # 亮度跳变阈值
        
        # 从上边缘向内扫描，找到第一个从黑色到浅灰色的跳变
        for y in range(scan_margin, min(scan_margin + 100, h_clean // 2)):
            row = blurred_lab_l[y, :]
            # 计算每行的平均亮度
            row_mean = np.mean(row)
            # 如果从低亮度（黑色）跳变到高亮度（浅灰色）
            if row_mean > brightness_threshold:
                # 检查是否有明显的跳变（从边缘的黑色到这里的浅灰色）
                edge_row = blurred_lab_l[scan_margin, :]
                if np.mean(edge_row) < brightness_threshold - 20:  # 边缘是黑色
                    edge_mask[y, :] = 255
                    break
        
        # 从下边缘向内扫描
        for y in range(max(0, h_clean - scan_margin - 100), h_clean - scan_margin):
            row = blurred_lab_l[y, :]
            row_mean = np.mean(row)
            if row_mean > brightness_threshold:
                edge_row = blurred_lab_l[h_clean - scan_margin - 1, :]
                if np.mean(edge_row) < brightness_threshold - 20:
                    edge_mask[y, :] = 255
                    break
        
        # 从左边缘向内扫描
        for x in range(scan_margin, min(scan_margin + 100, w_clean // 2)):
            col = blurred_lab_l[:, x]
            col_mean = np.mean(col)
            if col_mean > brightness_threshold:
                edge_col = blurred_lab_l[:, scan_margin]
                if np.mean(edge_col) < brightness_threshold - 20:
                    edge_mask[:, x] = 255
                    break
        
        # 从右边缘向内扫描
        for x in range(max(0, w_clean - scan_margin - 100), w_clean - scan_margin):
            col = blurred_lab_l[:, x]
            col_mean = np.mean(col)
            if col_mean > brightness_threshold:
                edge_col = blurred_lab_l[:, w_clean - scan_margin - 1]
                if np.mean(edge_col) < brightness_threshold - 20:
                    edge_mask[:, x] = 255
                    break
    
    # 合并所有颜色差异检测结果（包括Canny）
    combined = cv2.bitwise_or(color_binary, h_binary)
    combined = cv2.bitwise_or(combined, lab_color_diff)
    combined = cv2.bitwise_or(combined, gradient)
    combined = cv2.bitwise_or(combined, canny_gray)
    # 合并LAB颜色空间的Canny结果
    combined = cv2.bitwise_or(combined, canny_lab_a)
    combined = cv2.bitwise_or(combined, canny_lab_b)
    combined = cv2.bitwise_or(combined, canny_lab_l)
    # 合并HSV颜色空间的Canny结果
    combined = cv2.bitwise_or(combined, canny_hsv_h)
    combined = cv2.bitwise_or(combined, canny_hsv_s)
    combined = cv2.bitwise_or(combined, canny_hsv_v)
    # 合并浅灰色塑料边框检测结果
    if has_light_gray_plastic:
        combined = cv2.bitwise_or(combined, lab_l_plastic)
        combined = cv2.bitwise_or(combined, canny_lab_l_plastic)
        combined = cv2.bitwise_or(combined, edge_mask)
    
    if is_white_background:
        # 白色背景：使用梯度检测和边缘检测
        # 方法5：自适应阈值（针对白色背景）
        binary = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 11, 2
        )
        
        # 方法6：OTSU阈值（反转，因为背景是白色）
        _, binary_otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # 结合所有方法
        combined = cv2.bitwise_or(combined, binary)
        combined = cv2.bitwise_or(combined, binary_otsu)
        
        # 注意：Canny边缘检测已经在通用方法中添加，这里不再重复
        # 如果需要，可以添加额外的Canny检测（使用不同参数）
        edges_extra = cv2.Canny(blurred, 20, 80)  # 更敏感的阈值
        combined = cv2.bitwise_or(combined, edges_extra)
        
    elif is_transparent:
        # 透明塑料：重点检测塑料材质的外边缘
        # 使用多种方法组合检测，特别利用颜色差异
        
        # 方法5：OTSU阈值
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 方法6：检测高光区域（透明塑料的反光）
        _, binary_high = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY)
        
        # 方法7：检测低光区域（塑料边缘的阴影）
        _, binary_low = cv2.threshold(blurred, 50, 255, cv2.THRESH_BINARY_INV)
        
        # 方法8：LAB亮度通道的阈值（检测透明塑料的亮度变化）
        _, lab_l_binary = cv2.threshold(blurred_lab_l, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 方法9：HSV饱和度通道（透明塑料和纸质的饱和度可能不同）
        _, hsv_s_binary = cv2.threshold(blurred_hsv_s, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 结合所有方法（颜色差异检测已经在combined中）
        combined = cv2.bitwise_or(combined, binary)
        combined = cv2.bitwise_or(combined, binary_high)
        combined = cv2.bitwise_or(combined, binary_low)
        combined = cv2.bitwise_or(combined, lab_l_binary)
        combined = cv2.bitwise_or(combined, hsv_s_binary)
        
        # 注意：Canny边缘检测已经在通用方法中添加
        # 添加额外的Canny检测（针对透明塑料，使用更敏感的阈值）
        edges_extra = cv2.Canny(blurred, 30, 90)  # 更敏感的阈值以检测透明边缘
        combined = cv2.bitwise_or(combined, edges_extra)
        # LAB和HSV颜色空间的Canny对透明塑料特别有效，已在通用方法中合并
        
    else:
        # 深色背景：使用原有方法 + 颜色差异检测
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # 注意：Canny边缘检测已经在通用方法中添加
        # 添加额外的Canny检测（针对深色背景）
        edges_extra = cv2.Canny(blurred, 40, 120)
        combined = cv2.bitwise_or(combined, binary)
        combined = cv2.bitwise_or(combined, edges_extra)
    
    # 第五步：形态学操作（连接边缘，填充小孔）
    kernel = np.ones((5, 5), np.uint8)
    closed = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=3)
    dilated = cv2.dilate(closed, kernel, iterations=2)
    
    # 第六步：查找轮廓
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        # 备用方法：使用更激进的边缘检测
        edges = cv2.Canny(blurred, 20, 80)
        dilated_edges = cv2.dilate(edges, kernel, iterations=5)
        contours, _ = cv2.findContours(dilated_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None
    
    # 第七步：筛选轮廓（在去除黑边后的图像中筛选）
    # 对于浅灰色塑料外壳，应该选择最大的轮廓（塑料外壳的外边缘）
    # 塑料外壳通常占据图像的大部分区域（60%-95%）
    if has_light_gray_plastic:
        # 浅灰色塑料外壳：选择最大的轮廓，面积应该在60%-95%之间
        min_area = h_clean * w_clean * 0.50  # 提高最小面积，确保是塑料外壳而不是卡片
        max_area = h_clean * w_clean * 0.98
    else:
        # 其他情况：使用原来的阈值
        min_area = h_clean * w_clean * 0.15
        max_area = h_clean * w_clean * 0.98
    
    valid_contours = [c for c in contours if min_area < cv2.contourArea(c) < max_area]
    
    if not valid_contours:
        # 如果没找到，降低要求再试一次
        if has_light_gray_plastic:
            min_area = h_clean * w_clean * 0.30  # 降低但仍然是较大的面积
        else:
            min_area = h_clean * w_clean * 0.10
        valid_contours = [c for c in contours if min_area < cv2.contourArea(c) < max_area]
    
    if not valid_contours:
        # 最后尝试：选择所有轮廓中最大的（可能是塑料外壳）
        if contours:
            valid_contours = [max(contours, key=cv2.contourArea)]
        else:
            return None
    
    # 选择最大的轮廓（应该是塑料材质的外边缘）
    # 对于浅灰色塑料，确保选择的是最大的轮廓（塑料外壳）
    largest_contour = max(valid_contours, key=cv2.contourArea)
    
    # 验证：如果是浅灰色塑料，确保轮廓面积足够大（至少是图像的40%）
    if has_light_gray_plastic:
        contour_area = cv2.contourArea(largest_contour)
        if contour_area < h_clean * w_clean * 0.40:
            # 如果面积太小，可能是检测到了卡片而不是塑料外壳
            # 尝试选择更大的轮廓
            all_large_contours = [c for c in contours if cv2.contourArea(c) > h_clean * w_clean * 0.40]
            if all_large_contours:
                largest_contour = max(all_large_contours, key=cv2.contourArea)
    
    # 第八步：多边形拟合 - 将轮廓拟合为四边形
    epsilon_start = 0.01
    epsilon_end = 0.1
    epsilon_step = 0.01
    
    best_approx = None
    
    for epsilon in np.arange(epsilon_start, epsilon_end, epsilon_step):
        epsilon_val = epsilon * cv2.arcLength(largest_contour, True)
        approx = cv2.approxPolyDP(largest_contour, epsilon_val, True)
        
        if len(approx) == 4:
            best_approx = approx
            break
        elif len(approx) > 4 and best_approx is None:
            best_approx = approx
    
    if best_approx is not None and len(best_approx) == 4:
        # 将坐标映射回原始图像（加上黑边偏移）
        x_offset, y_offset = border_box[0], border_box[1]
        best_approx = best_approx.reshape(-1, 2)
        best_approx[:, 0] += x_offset
        best_approx[:, 1] += y_offset
        return best_approx.reshape(-1, 1, 2)
    
    # 如果拟合后不是4个顶点，使用最小外接矩形
    rect = cv2.minAreaRect(largest_contour)
    box = cv2.boxPoints(rect)
    box = np.int32(box)
    
    # 将坐标映射回原始图像（加上黑边偏移）
    x_offset, y_offset = border_box[0], border_box[1]
    box[:, 0] += x_offset
    box[:, 1] += y_offset
    
    return box.reshape(-1, 1, 2)


def get_card_corners(contour: np.ndarray) -> np.ndarray:
    """
    从轮廓中提取卡牌的四个角点
    
    Args:
        contour: 卡牌轮廓
        
    Returns:
        四个角点的坐标数组（float32格式，用于透视变换）
    """
    # 如果轮廓已经有4个点，直接使用
    if len(contour) == 4:
        corners = contour.reshape(4, 2).astype(np.float32)
    else:
        # 否则，找到最小外接矩形
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        corners = box.astype(np.float32)
    
    # 对点进行排序：左上、右上、右下、左下
    return order_points(corners)


def order_points(pts: np.ndarray) -> np.ndarray:
    """
    对四个点进行排序：左上、右上、右下、左下
    
    Args:
        pts: 四个点的坐标（float32格式）
        
    Returns:
        排序后的点（float32格式）
    """
    # 确保输入是float32格式
    pts = pts.astype(np.float32)
    
    # 初始化结果数组
    rect = np.zeros((4, 2), dtype=np.float32)
    
    # 左上角点：x+y最小
    # 右下角点：x+y最大
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # 左上
    rect[2] = pts[np.argmax(s)]  # 右下
    
    # 右上角点：x-y最小（或x最大且y较小）
    # 左下角点：x-y最大（或x最小且y较大）
    diff = np.diff(pts, axis=1).flatten()
    rect[1] = pts[np.argmin(diff)]  # 右上
    rect[3] = pts[np.argmax(diff)]  # 左下
    
    # 验证排序：确保左上和右下不是同一个点
    if np.array_equal(rect[0], rect[2]):
        # 如果相同，使用x坐标重新排序
        sorted_pts = pts[np.argsort(pts[:, 0])]
        rect[0] = sorted_pts[0]  # 最左
        rect[2] = sorted_pts[-1]  # 最右
        sorted_pts_y = pts[np.argsort(pts[:, 1])]
        rect[1] = sorted_pts_y[0]  # 最上
        rect[3] = sorted_pts_y[-1]  # 最下
    
    return rect


def calculate_card_dimensions(corners: np.ndarray) -> tuple:
    """
    计算卡牌的目标尺寸（确保是正长方形）
    
    Args:
        corners: 四个角点（已排序：左上、右上、右下、左下）
        
    Returns:
        (宽度, 高度) - 使用平均尺寸确保是正长方形
    """
    # 计算宽度（取上下两边的平均值）
    width_a = np.sqrt(((corners[1][0] - corners[0][0]) ** 2) + 
                      ((corners[1][1] - corners[0][1]) ** 2))
    width_b = np.sqrt(((corners[2][0] - corners[3][0]) ** 2) + 
                      ((corners[2][1] - corners[3][1]) ** 2))
    width_avg = (width_a + width_b) / 2
    
    # 计算高度（取左右两边的平均值）
    height_a = np.sqrt(((corners[3][0] - corners[0][0]) ** 2) + 
                       ((corners[3][1] - corners[0][1]) ** 2))
    height_b = np.sqrt(((corners[2][0] - corners[1][0]) ** 2) + 
                       ((corners[2][1] - corners[1][1]) ** 2))
    height_avg = (height_a + height_b) / 2
    
    # 使用平均尺寸，确保是正长方形
    width = int(width_avg)
    height = int(height_avg)
    
    return width, height


def perspective_transform(image: np.ndarray, corners: np.ndarray) -> np.ndarray:
    """
    对图像进行透视变换，矫正方向并裁剪
    变换后会自动去除残留的黑色边缘和灰色纸片
    
    Args:
        image: 输入图像（BGR格式）
        corners: 四个角点（已排序：左上、右上、右下、左下）
        
    Returns:
        矫正后的图像（已去除黑色边缘和灰色纸片）
    """
    # 计算目标尺寸
    width, height = calculate_card_dimensions(corners)
    
    # 确保尺寸合理
    if width <= 0 or height <= 0:
        print(f"警告：计算出的尺寸无效 ({width}x{height})，使用原始图像")
        return image
    
    # 定义目标点（输出图像的四个角）
    dst = np.array([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1]
    ], dtype=np.float32)
    
    # 计算透视变换矩阵
    M = cv2.getPerspectiveTransform(corners, dst)
    
    # 应用透视变换
    warped = cv2.warpPerspective(image, M, (width, height), flags=cv2.INTER_LINEAR)
    
    # 再次去除残留的黑色边缘和灰色纸片（使用LAB L通道严格过滤）
    cleaned, _ = remove_black_border(warped)
    
    # 进一步清理：使用LAB L通道去除灰色纸片边缘
    lab = cv2.cvtColor(cleaned, cv2.COLOR_BGR2LAB)
    lab_l = lab[:, :, 0]
    
    # 定义阈值：黑色背景 L < 40，灰色纸片 L < 70，塑料 L >= 80
    black_threshold = 40
    gray_threshold = 70
    plastic_min = 80
    
    # 从四个边缘向内扫描，找到第一个塑料边缘
    h_clean, w_clean = cleaned.shape[:2]
    margin = 3
    scan_depth = min(50, h_clean // 4, w_clean // 4)
    
    # 从上边缘向下扫描
    top = margin
    for y in range(margin, margin + scan_depth):
        row = lab_l[y, :]
        if np.mean(row) >= plastic_min and np.sum(row >= plastic_min) >= len(row) * 0.4:
            top = y
            break
    
    # 从下边缘向上扫描
    bottom = h_clean - margin
    for y in range(h_clean - margin - 1, max(margin, h_clean - margin - scan_depth - 1), -1):
        row = lab_l[y, :]
        if np.mean(row) >= plastic_min and np.sum(row >= plastic_min) >= len(row) * 0.4:
            bottom = y + 1
            break
    
    # 从左边缘向右扫描
    left = margin
    for x in range(margin, margin + scan_depth):
        col = lab_l[:, x]
        if np.mean(col) >= plastic_min and np.sum(col >= plastic_min) >= len(col) * 0.4:
            left = x
            break
    
    # 从右边缘向左扫描
    right = w_clean - margin
    for x in range(w_clean - margin - 1, max(margin, w_clean - margin - scan_depth - 1), -1):
        col = lab_l[:, x]
        if np.mean(col) >= plastic_min and np.sum(col >= plastic_min) >= len(col) * 0.4:
            right = x + 1
            break
    
    # 确保边界合理
    if top < bottom and left < right and bottom - top > h_clean * 0.5 and right - left > w_clean * 0.5:
        final = cleaned[top:bottom, left:right]
        return final
    
    return cleaned


def remove_background_rembg(image: np.ndarray, model_name: str = 'u2net') -> Optional[np.ndarray]:
    """
    使用 rembg 库移除图像背景
    
    Args:
        image: 输入图像（BGR格式，OpenCV格式）
        model_name: 使用的模型名称，可选值：
                   - 'u2net': 通用模型（默认）
                   - 'u2netp': 轻量级模型
                   - 'u2net_human_seg': 人像分割
                   - 'isnet-general-use': 通用高精度模型
                   - 'birefnet-general': 通用模型
                   - 'birefnet-hrsod': 高分辨率显著对象检测模型（推荐用于卡片）
    
    Returns:
        移除背景后的图像（BGRA格式，保持原始BGR颜色空间），如果 rembg 不可用则返回 None
    """
    if not REMBG_AVAILABLE:
        print("  警告: rembg 未安装，跳过背景移除")
        return None
    
    try:
        # rembg 需要 RGB 格式，OpenCV 是 BGR
        # 注意：确保使用正确的数据类型（uint8）
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 转换为 PIL Image（rembg 支持 PIL Image）
        # 确保图像是 uint8 类型，范围 0-255
        if image_rgb.dtype != np.uint8:
            image_rgb = np.clip(image_rgb, 0, 255).astype(np.uint8)
        
        from PIL import Image
        pil_image = Image.fromarray(image_rgb)
        
        # 创建session（尝试使用GPU，如果不可用则回退到CPU）
        # rembg会自动检测GPU，如果CUDA可用则使用GPU
        # 如果CUDA不可用或缺少依赖，会自动回退到CPU
        # 如果遇到CUDA DLL错误，rembg会自动使用CPU，但会显示警告
        try:
            session = new_session(model_name)
        except Exception as e:
            print(f"  警告: 创建session失败: {e}")
            print("  尝试使用CPU模式...")
            # 如果GPU失败，rembg应该会自动回退到CPU
            # 如果仍然失败，可能是其他问题
            raise
        
        # 使用 rembg 移除背景（返回 RGBA 格式，RGB 颜色空间）
        output = remove(pil_image, session=session)
        
        # 转换回 numpy array (RGBA, RGB 颜色空间)
        output_array = np.array(output)
        
        # 如果需要 BGR 格式（保持 alpha 通道），转换 RGB -> BGR
        # 提取 RGB 通道并转换为 BGR，然后重新组合 alpha 通道
        rgb_channels = output_array[:, :, :3]  # RGB 通道
        alpha_channel = output_array[:, :, 3:4]  # Alpha 通道
        bgr_channels = cv2.cvtColor(rgb_channels, cv2.COLOR_RGB2BGR)
        output_bgra = np.concatenate([bgr_channels, alpha_channel], axis=2)
        
        return output_bgra  # 返回 BGRA 格式，保持原始颜色
    except Exception as e:
        print(f"  警告: rembg 背景移除失败: {e}")
        return None


def detect_and_crop_background(image):
    """
    检测并裁剪图片中的空白/深色背景
    使用新的塑料外壳检测方法（CLAHE + 自适应阈值）
    
    Args:
        image: OpenCV图像对象
        
    Returns:
        裁剪后的图像
    """
    # 使用新的塑料外壳检测方法
    contour = detect_plastic_slab_contour(image)
    
    if contour is not None:
        # 提取并排序角点
        corners = get_card_corners(contour)
        
        # 确保角点坐标在图像范围内
        h, w = image.shape[:2]
        corners[:, 0] = np.clip(corners[:, 0], 0, w - 1)
        corners[:, 1] = np.clip(corners[:, 1], 0, h - 1)
        
        # 使用透视变换进行裁剪和矫正
        cropped = perspective_transform(image, corners)
        return cropped
    
    # 如果轮廓检测失败，使用备用方法
    # 转换为灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 使用OTSU阈值
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # 形态学操作
    kernel = np.ones((5, 5), np.uint8)
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=3)
    dilated = cv2.dilate(closed, kernel, iterations=2)
    
    # 查找轮廓
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        # 如果还是找不到，尝试Canny边缘检测
        edges = cv2.Canny(gray, 50, 150)
        dilated_edges = cv2.dilate(edges, kernel, iterations=3)
        contours, _ = cv2.findContours(dilated_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        # 如果还是找不到，返回原图
        print("警告: 无法检测到内容区域，返回原图")
        return image
    
    # 找到所有轮廓的边界框
    x_min = image.shape[1]
    y_min = image.shape[0]
    x_max = 0
    y_max = 0
    
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        x_min = min(x_min, x)
        y_min = min(y_min, y)
        x_max = max(x_max, x + w)
        y_max = max(y_max, y + h)
    
    # 添加一些边距（5%的容差）
    margin_x = int((x_max - x_min) * 0.05)
    margin_y = int((y_max - y_min) * 0.05)
    
    x_min = max(0, x_min - margin_x)
    y_min = max(0, y_min - margin_y)
    x_max = min(image.shape[1], x_max + margin_x)
    y_max = min(image.shape[0], y_max + margin_y)
    
    # 裁剪图像
    cropped = image[y_min:y_max, x_min:x_max]
    
    return cropped


def detect_card_count(image: np.ndarray) -> int:
    """
    检测图片中有多少张卡片（通过检测塑料外壳数量）
    
    Args:
        image: 输入图像
        
    Returns:
        卡片数量
    """
    # 去除黑边
    image_no_border, _ = remove_black_border(image)
    h, w = image_no_border.shape[:2]
    
    # 转换为灰度图
    gray = cv2.cvtColor(image_no_border, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 0)
    
    # 检测边缘
    edges = cv2.Canny(blurred, 30, 100)
    
    # 形态学操作
    kernel = np.ones((5, 5), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=2)
    closed = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel, iterations=3)
    
    # 查找轮廓
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return 1  # 默认1张
    
    # 筛选大轮廓（塑料外壳通常占图像较大比例）
    min_area = h * w * 0.20  # 至少占20%
    large_contours = [c for c in contours if cv2.contourArea(c) > min_area]
    
    # 按面积排序，选择前几个最大的
    large_contours = sorted(large_contours, key=cv2.contourArea, reverse=True)
    
    # 计算卡片数量（通常每个塑料外壳是一个卡片）
    card_count = len(large_contours)
    
    # 如果检测到的数量不合理，返回默认值
    if card_count == 0:
        return 1
    elif card_count > 4:
        return 4  # 最多4张
    
    return card_count


def split_image_2x2(image):
    """
    将图片切割成2x2的4个区域（保持向后兼容）
    
    Args:
        image: OpenCV图像对象
        
    Returns:
        包含4个区域的字典: {'top-left': image, 'top-right': image, 
                           'bottom-left': image, 'bottom-right': image}
    """
    height, width = image.shape[:2]
    
    # 计算切割点
    mid_x = width // 2
    mid_y = height // 2
    
    # 切割成4个区域
    top_left = image[0:mid_y, 0:mid_x]
    top_right = image[0:mid_y, mid_x:width]
    bottom_left = image[mid_y:height, 0:mid_x]
    bottom_right = image[mid_y:height, mid_x:width]
    
    return {
        'top-left': top_left,
        'top-right': top_right,
        'bottom-left': bottom_left,
        'bottom-right': bottom_right
    }


def split_image_by_count(image: np.ndarray, count: int) -> Dict[str, np.ndarray]:
    """
    根据卡片数量动态切割图片
    
    Args:
        image: 输入图像
        count: 卡片数量
        
    Returns:
        包含切割区域的字典
    """
    height, width = image.shape[:2]
    parts = {}
    
    if count == 1:
        # 1张卡片：不切割
        parts['card-0'] = image
    elif count == 2:
        # 2张卡片：水平切割
        mid_x = width // 2
        parts['card-0'] = image[:, 0:mid_x]
        parts['card-1'] = image[:, mid_x:width]
    elif count == 3:
        # 3张卡片：2行布局（2+1）
        mid_x = width // 2
        mid_y = height // 2
        parts['card-0'] = image[0:mid_y, 0:mid_x]
        parts['card-1'] = image[0:mid_y, mid_x:width]
        parts['card-2'] = image[mid_y:height, width//4:3*width//4]
    elif count >= 4:
        # 4张或更多：2x2布局
        mid_x = width // 2
        mid_y = height // 2
        parts['card-0'] = image[0:mid_y, 0:mid_x]
        parts['card-1'] = image[0:mid_y, mid_x:width]
        parts['card-2'] = image[mid_y:height, 0:mid_x]
        parts['card-3'] = image[mid_y:height, mid_x:width]
    
    return parts


def detect_plastic_slab_contour(image: np.ndarray, remove_border_pixels: int = None) -> Optional[np.ndarray]:
    """
    检测塑料外壳的边缘轮廓（排除外部灰色纸质边缘和黑色背景）
    核心思路：从图像边缘向内扫描，找到从深灰色纸片到浅灰色塑料的第二个跳变
    
    层次结构（从外到内）：
    - 黑色背景：LAB L < 35
    - 深灰色纸片：LAB L 35-65
    - 浅灰色塑料：LAB L >= 90（这是我们要找的边缘）
    
    策略：找到从深灰色纸片（L 35-65）到浅灰色塑料（L >= 90）的跳变
    
    Args:
        image: 输入图像（BGR格式）
        remove_border_pixels: 检测到边缘后，向内收缩的像素数（用于去除灰色纸片边缘）
                            如果灰色边缘还在，调大这个值（比如 20 或 30）
        
    Returns:
        塑料外壳轮廓（4个顶点的轴对齐矩形），如果找到，否则返回None
    """
    h, w = image.shape[:2]
    
    # 第一步：去除黑边
    image_no_border, border_box = remove_black_border(image)
    h_clean, w_clean = image_no_border.shape[:2]
    
    # 第二步：转换到LAB颜色空间，使用L通道（亮度）
    lab = cv2.cvtColor(image_no_border, cv2.COLOR_BGR2LAB)
    lab_l = lab[:, :, 0]  # 亮度通道
    blurred_l = cv2.GaussianBlur(lab_l, (9, 9), 0)  # 增加模糊半径，减少噪点
    
    # 定义更精确的阈值
    black_max = 35        # 黑色背景的上限
    gray_paper_min = 35   # 深灰色纸片的最小值
    gray_paper_max = 65   # 深灰色纸片的最大值
    plastic_min = 90      # 浅灰色塑料的最小亮度（提高阈值，排除深灰色纸片）
    jump_threshold = 25   # 从纸片到塑料的最小亮度跳变
    
    # 辅助函数：检查一行/一列是否主要是深灰色纸片
    def is_gray_paper(values):
        """检查是否主要是深灰色纸片（L 35-65）"""
        gray_count = np.sum((values >= gray_paper_min) & (values <= gray_paper_max))
        return gray_count >= len(values) * 0.5  # 至少50%是纸片
    
    # 辅助函数：检查一行/一列是否主要是浅灰色塑料
    def is_plastic(values, min_ratio=0.5):
        """检查是否主要是浅灰色塑料（L >= 90）"""
        plastic_count = np.sum(values >= plastic_min)
        return plastic_count >= len(values) * min_ratio
    
    # 辅助函数：计算平均亮度
    def get_mean_brightness(values):
        return np.mean(values)
    
    # 第三步：从四个边缘向内扫描，找到第一个塑料边缘
    # 改进策略：跳过所有长边区域（黑色背景、灰色纸片），找到真正的塑料边缘
    # 每一条边都可能有多层：黑色背景 -> 灰色纸片 -> 塑料
    scan_margin = 5  # 扫描边距
    scan_depth = min(400, h_clean // 2, w_clean // 2)  # 增加扫描深度，确保能跳过长边
    
    top_edge = None
    bottom_edge = None
    left_edge = None
    right_edge = None
    
    # 辅助函数：检查是否是背景或纸片区域（需要跳过的长边）
    def is_background_or_paper(values):
        """检查是否是背景或纸片区域（需要跳过的长边）"""
        mean_val = get_mean_brightness(values)
        # 黑色背景或深灰色纸片
        return mean_val < gray_paper_max
    
    # 从上边缘向下扫描
    # 策略：跳过所有背景/纸片长边，找到第一个稳定的塑料区域
    edge_row_mean = get_mean_brightness(blurred_l[scan_margin, :])
    in_background = is_background_or_paper(blurred_l[scan_margin, :])
    plastic_start_y = None
    
    for y in range(scan_margin, scan_margin + scan_depth):
        row = blurred_l[y, :]
        row_mean = get_mean_brightness(row)
        is_plastic_row = is_plastic(row, min_ratio=0.5)
        is_bg_row = is_background_or_paper(row)
        
        # 如果当前行是塑料，且之前是背景/纸片，说明找到了塑料边缘
        if is_plastic_row and in_background:
            # 验证：检查前几行确实是背景/纸片
            prev_y = max(scan_margin, y - 10)
            prev_row = blurred_l[prev_y, :]
            if is_background_or_paper(prev_row):
                # 检查是否有明显的跳变
                if row_mean - get_mean_brightness(prev_row) >= 30:  # 明显的亮度跳变
                    top_edge = y
                    break
        
        # 更新状态：如果进入塑料区域，记录开始位置
        if is_plastic_row and plastic_start_y is None:
            plastic_start_y = y
        
        # 如果已经在塑料区域，且连续多行都是塑料，说明找到了稳定的塑料边缘
        if plastic_start_y is not None and y - plastic_start_y >= 5:
            if is_plastic(row, min_ratio=0.6):  # 要求更高，确保是稳定的塑料
                top_edge = plastic_start_y
                break
        
        # 更新背景状态
        if is_bg_row:
            in_background = True
            plastic_start_y = None
        else:
            in_background = False
    
    # 从下边缘向上扫描
    edge_row_mean = get_mean_brightness(blurred_l[h_clean - scan_margin - 1, :])
    in_background = is_background_or_paper(blurred_l[h_clean - scan_margin - 1, :])
    plastic_start_y = None
    
    for y in range(h_clean - scan_margin - 1, max(scan_margin, h_clean - scan_margin - scan_depth - 1), -1):
        row = blurred_l[y, :]
        row_mean = get_mean_brightness(row)
        is_plastic_row = is_plastic(row, min_ratio=0.5)
        is_bg_row = is_background_or_paper(row)
        
        if is_plastic_row and in_background:
            prev_y = min(h_clean - scan_margin - 1, y + 10)
            prev_row = blurred_l[prev_y, :]
            if is_background_or_paper(prev_row):
                if row_mean - get_mean_brightness(prev_row) >= 30:
                    bottom_edge = y
                    break
        
        if is_plastic_row and plastic_start_y is None:
            plastic_start_y = y
        
        if plastic_start_y is not None and plastic_start_y - y >= 5:
            if is_plastic(row, min_ratio=0.6):
                bottom_edge = plastic_start_y
                break
        
        if is_bg_row:
            in_background = True
            plastic_start_y = None
        else:
            in_background = False
    
    # 从左边缘向右扫描
    edge_col_mean = get_mean_brightness(blurred_l[:, scan_margin])
    in_background = is_background_or_paper(blurred_l[:, scan_margin])
    plastic_start_x = None
    
    for x in range(scan_margin, scan_margin + scan_depth):
        col = blurred_l[:, x]
        col_mean = get_mean_brightness(col)
        is_plastic_col = is_plastic(col, min_ratio=0.5)
        is_bg_col = is_background_or_paper(col)
        
        if is_plastic_col and in_background:
            prev_x = max(scan_margin, x - 10)
            prev_col = blurred_l[:, prev_x]
            if is_background_or_paper(prev_col):
                if col_mean - get_mean_brightness(prev_col) >= 30:
                    left_edge = x
                    break
        
        if is_plastic_col and plastic_start_x is None:
            plastic_start_x = x
        
        if plastic_start_x is not None and x - plastic_start_x >= 5:
            if is_plastic(col, min_ratio=0.6):
                left_edge = plastic_start_x
                break
        
        if is_bg_col:
            in_background = True
            plastic_start_x = None
        else:
            in_background = False
    
    # 从右边缘向左扫描
    edge_col_mean = get_mean_brightness(blurred_l[:, w_clean - scan_margin - 1])
    in_background = is_background_or_paper(blurred_l[:, w_clean - scan_margin - 1])
    plastic_start_x = None
    
    for x in range(w_clean - scan_margin - 1, max(scan_margin, w_clean - scan_margin - scan_depth - 1), -1):
        col = blurred_l[:, x]
        col_mean = get_mean_brightness(col)
        is_plastic_col = is_plastic(col, min_ratio=0.5)
        is_bg_col = is_background_or_paper(col)
        
        if is_plastic_col and in_background:
            prev_x = min(w_clean - scan_margin - 1, x + 10)
            prev_col = blurred_l[:, prev_x]
            if is_background_or_paper(prev_col):
                if col_mean - get_mean_brightness(prev_col) >= 30:
                    right_edge = x
                    break
        
        if is_plastic_col and plastic_start_x is None:
            plastic_start_x = x
        
        if plastic_start_x is not None and plastic_start_x - x >= 5:
            if is_plastic(col, min_ratio=0.6):
                right_edge = plastic_start_x
                break
        
        if is_bg_col:
            in_background = True
            plastic_start_x = None
        else:
            in_background = False
    
    # 如果扫描失败，使用备用方法：基于LAB L通道的掩码（使用更严格的阈值）
    if (top_edge is None or bottom_edge is None or left_edge is None or right_edge is None or
        top_edge >= bottom_edge - 10 or left_edge >= right_edge - 10 or
        bottom_edge - top_edge < h_clean * 0.2 or right_edge - left_edge < w_clean * 0.2):
        print("  边缘扫描失败，使用备用方法...")
        # 创建掩码：严格排除黑色背景（L < 35）和深灰色纸质（L < 90），只保留浅灰色塑料（L >= 90）
        plastic_mask = np.zeros((h_clean, w_clean), dtype=np.uint8)
        plastic_mask[blurred_l >= plastic_min] = 255  # 只保留 L >= 90 的区域
        
        # 形态学操作：填补空洞，去除小噪点
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
        plastic_mask = cv2.morphologyEx(plastic_mask, cv2.MORPH_CLOSE, kernel, iterations=3)
        plastic_mask = cv2.morphologyEx(plastic_mask, cv2.MORPH_OPEN, kernel, iterations=4)  # 去除小噪点
        
        # 查找轮廓
        contours, _ = cv2.findContours(plastic_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        # 选择最大的轮廓
        largest_contour = max(contours, key=cv2.contourArea)
        area_ratio = cv2.contourArea(largest_contour) / (h_clean * w_clean)
        
        if area_ratio < 0.15:  # 面积太小，可能不是塑料外壳
            return None
        
        # 使用边界框（轴对齐矩形）
        x, y, w, h = cv2.boundingRect(largest_contour)
        top_edge = y
        bottom_edge = y + h
        left_edge = x
        right_edge = x + w
        
        # 验证边界框内的区域确实主要是塑料（L >= 90）
        bbox_region = blurred_l[top_edge:bottom_edge, left_edge:right_edge]
        if bbox_region.size > 0:
            plastic_ratio = np.sum(bbox_region >= plastic_min) / bbox_region.size
            if plastic_ratio < 0.5:  # 如果塑料比例太低，可能包含了纸片
                print(f"  警告：检测区域塑料比例过低 ({plastic_ratio:.2%})，可能包含纸片")
    else:
        print(f"  边缘扫描成功: top={top_edge}, bottom={bottom_edge}, left={left_edge}, right={right_edge}")
    
    # 第四步：强制向内收缩，去除灰色纸片边缘
    # 这是关键步骤：无论检测结果如何，都向内收缩固定像素数，确保切掉灰色纸片边缘
    # 根据用户指南，如果灰色边缘还在，应该调大 remove_border_pixels（默认 20）
    
    # 确保 remove_border_pixels 不为 None
    if remove_border_pixels is None:
        remove_border_pixels = REMOVE_BORDER_PIXELS
    
    # 确保收缩后不会导致区域太小
    min_height = h_clean * 0.15
    min_width = w_clean * 0.15
    
    # 计算可收缩的最大值
    max_shrink_top = max(0, (bottom_edge - top_edge) - int(min_height))
    max_shrink_bottom = max(0, (bottom_edge - top_edge) - int(min_height))
    max_shrink_left = max(0, (right_edge - left_edge) - int(min_width))
    max_shrink_right = max(0, (right_edge - left_edge) - int(min_width))
    
    # 向内收缩（物理外挂：直接切掉边缘）
    actual_shrink = min(remove_border_pixels, max_shrink_top, max_shrink_bottom, 
                        max_shrink_left, max_shrink_right)
    
    if actual_shrink > 0:
        top_edge += actual_shrink
        bottom_edge -= actual_shrink
        left_edge += actual_shrink
        right_edge -= actual_shrink
        print(f"  向内收缩 {actual_shrink} 像素以去除灰色纸片边缘")
    
    # 第五步：二次验证和清理 - 检查边界区域是否还有深灰色纸片
    detected_region = image_no_border[top_edge:bottom_edge, left_edge:right_edge]
    if detected_region.size == 0:
        return None
    
    lab_region = cv2.cvtColor(detected_region, cv2.COLOR_BGR2LAB)
    lab_l_region = lab_region[:, :, 0]
    
    # 如果边界区域还有深灰色纸片，进一步收缩
    additional_shrink = 3  # 额外收缩边距
    
    # 检查上边界：如果平均亮度低于塑料阈值，说明可能包含纸片
    if top_edge + additional_shrink < bottom_edge:
        top_row = lab_l_region[0, :]
        if np.mean(top_row) < plastic_min:  # 使用塑料阈值，确保边界是塑料
            top_edge += additional_shrink
    
    # 检查下边界
    if bottom_edge - additional_shrink > top_edge:
        bottom_row = lab_l_region[-1, :]
        if np.mean(bottom_row) < plastic_min:
            bottom_edge -= additional_shrink
    
    # 检查左边界
    if left_edge + additional_shrink < right_edge:
        left_col = lab_l_region[:, 0]
        if np.mean(left_col) < plastic_min:
            left_edge += additional_shrink
    
    # 检查右边界
    if right_edge - additional_shrink > left_edge:
        right_col = lab_l_region[:, -1]
        if np.mean(right_col) < plastic_min:
            right_edge -= additional_shrink
    
    # 最终验证
    if (top_edge >= bottom_edge - 5 or left_edge >= right_edge - 5 or
        bottom_edge - top_edge < h_clean * 0.15 or right_edge - left_edge < w_clean * 0.15):
        print("  警告：检测到的区域太小，可能不正确")
        return None
    
    # 第五步：构建轴对齐矩形（确保边垂直/平行）
    # 将坐标映射回原始图像（加上黑边偏移）
    x_offset, y_offset = border_box[0], border_box[1]
    
    box = np.array([
        [left_edge + x_offset, top_edge + y_offset],           # 左上
        [right_edge + x_offset, top_edge + y_offset],          # 右上
        [right_edge + x_offset, bottom_edge + y_offset],       # 右下
        [left_edge + x_offset, bottom_edge + y_offset]          # 左下
    ], dtype=np.float32)
    
    return box.reshape(-1, 1, 2).astype(np.int32)


def detect_inner_card(image: np.ndarray) -> Optional[np.ndarray]:
    """
    检测内部卡片轮廓（使用test_card_detection_standalone.py的方法）
    
    Args:
        image: 输入图像
        
    Returns:
        卡片轮廓（4个顶点的四边形），如果找到，否则返回None
    """
    h, w = image.shape[:2]
    
    # 去除黑边
    image_no_border, border_box = remove_black_border(image)
    h_clean, w_clean = image_no_border.shape[:2]
    
    # 转换颜色空间
    gray = cv2.cvtColor(image_no_border, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    
    lab = cv2.cvtColor(image_no_border, cv2.COLOR_BGR2LAB)
    lab_l = lab[:, :, 0]
    blurred_lab_l = cv2.GaussianBlur(lab_l, (7, 7), 0)
    
    hsv = cv2.cvtColor(image_no_border, cv2.COLOR_BGR2HSV)
    hsv_s = hsv[:, :, 1]
    blurred_hsv_s = cv2.GaussianBlur(hsv_s, (7, 7), 0)
    
    # 检测高饱和度区域
    _, high_sat = cv2.threshold(blurred_hsv_s, 50, 255, cv2.THRESH_BINARY)
    
    # 检测白色边框
    _, white_border = cv2.threshold(blurred_lab_l, 180, 255, cv2.THRESH_BINARY)
    
    # OTSU阈值
    _, otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Canny边缘
    canny = cv2.Canny(blurred, 20, 80)
    
    # 合并
    combined = cv2.bitwise_or(high_sat, white_border)
    combined = cv2.bitwise_or(combined, otsu)
    combined = cv2.bitwise_or(combined, canny)
    
    # 形态学操作
    kernel = np.ones((5, 5), np.uint8)
    closed = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=2)
    dilated = cv2.dilate(closed, kernel, iterations=1)
    
    # 查找轮廓
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None
    
    # 按面积排序
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    
    # 排除最大的轮廓（通常是塑料外壳）
    largest_area_ratio = cv2.contourArea(contours[0]) / (h_clean * w_clean)
    if largest_area_ratio > 0.85:
        remaining_contours = contours[1:] if len(contours) > 1 else []
    else:
        remaining_contours = contours
    
    # 筛选有效轮廓
    valid_contours = []
    for c in remaining_contours:
        area_ratio = cv2.contourArea(c) / (h_clean * w_clean)
        if 0.10 <= area_ratio <= 0.80:
            valid_contours.append(c)
    
    if not valid_contours:
        # 放宽条件
        for c in remaining_contours:
            area_ratio = cv2.contourArea(c) / (h_clean * w_clean)
            if 0.05 <= area_ratio <= 0.75:
                valid_contours.append(c)
    
    if not valid_contours:
        # 使用内部内容检测
        outer_contour = contours[0] if contours and largest_area_ratio > 0.85 else None
        return detect_card_by_inner_content(image_no_border, border_box, outer_contour)
    
    # 选择最佳轮廓
    best_contour = max(valid_contours, key=cv2.contourArea)
    
    # 拟合为四边形
    epsilon = 0.02 * cv2.arcLength(best_contour, True)
    approx = cv2.approxPolyDP(best_contour, epsilon, True)
    
    if len(approx) >= 4:
        x_offset, y_offset = border_box[0], border_box[1]
        approx = approx.reshape(-1, 2)
        approx[:, 0] += x_offset
        approx[:, 1] += y_offset
        return approx.reshape(-1, 1, 2)
    
    # 使用最小外接矩形
    rect = cv2.minAreaRect(best_contour)
    box = cv2.boxPoints(rect)
    box = np.int32(box)
    x_offset, y_offset = border_box[0], border_box[1]
    box[:, 0] += x_offset
    box[:, 1] += y_offset
    return box.reshape(-1, 1, 2)


def detect_card_by_inner_content(image_no_border: np.ndarray, border_box: Tuple, 
                                  outer_contour: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
    """
    通过检测卡片内部内容来定位卡片
    在塑料外壳内部寻找高饱和度、高对比度的区域
    """
    h, w = image_no_border.shape[:2]
    
    # 创建掩码
    mask = np.ones((h, w), dtype=np.uint8) * 255
    if outer_contour is not None:
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [outer_contour], 255)
    
    # 转换颜色空间
    hsv = cv2.cvtColor(image_no_border, cv2.COLOR_BGR2HSV)
    hsv_s = hsv[:, :, 1]
    blurred_hsv_s = cv2.GaussianBlur(hsv_s, (7, 7), 0)
    
    lab = cv2.cvtColor(image_no_border, cv2.COLOR_BGR2LAB)
    lab_l = lab[:, :, 0]
    blurred_lab_l = cv2.GaussianBlur(lab_l, (7, 7), 0)
    
    # 检测高饱和度区域（卡片内容）
    _, high_sat = cv2.threshold(blurred_hsv_s, 60, 255, cv2.THRESH_BINARY)
    high_sat = cv2.bitwise_and(high_sat, mask)
    
    # 检测白色边框（卡片边缘）
    _, white_border = cv2.threshold(blurred_lab_l, 190, 255, cv2.THRESH_BINARY)
    white_border = cv2.bitwise_and(white_border, mask)
    
    # 合并
    combined = cv2.bitwise_or(high_sat, white_border)
    
    # 形态学操作
    kernel = np.ones((7, 7), np.uint8)
    closed = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=3)
    dilated = cv2.dilate(closed, kernel, iterations=2)
    
    # 查找轮廓
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None
    
    # 选择最大的轮廓（应该是卡片）
    largest = max(contours, key=cv2.contourArea)
    area_ratio = cv2.contourArea(largest) / (h * w)
    
    if area_ratio < 0.05:
        return None
    
    # 拟合为四边形
    epsilon = 0.02 * cv2.arcLength(largest, True)
    approx = cv2.approxPolyDP(largest, epsilon, True)
    
    if len(approx) >= 4:
        x_offset, y_offset = border_box[0], border_box[1]
        approx = approx.reshape(-1, 2)
        approx[:, 0] += x_offset
        approx[:, 1] += y_offset
        return approx.reshape(-1, 1, 2)
    
    # 使用最小外接矩形
    rect = cv2.minAreaRect(largest)
    box = cv2.boxPoints(rect)
    box = np.int32(box)
    x_offset, y_offset = border_box[0], border_box[1]
    box[:, 0] += x_offset
    box[:, 1] += y_offset
    return box.reshape(-1, 1, 2)


def detect_card_by_saturation(image: np.ndarray) -> Optional[np.ndarray]:
    """
    通过检测高饱和度区域来识别卡片（卡片通常是高饱和度的长方形）
    
    Args:
        image: 输入图像（BGR格式）
        
    Returns:
        卡片轮廓（4个顶点的四边形），如果找到，否则返回None
    """
    h, w = image.shape[:2]
    
    # 转换到HSV颜色空间
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv_s = hsv[:, :, 1]  # 饱和度通道
    hsv_v = hsv[:, :, 2]  # 明度通道
    
    # 高斯模糊
    blurred_s = cv2.GaussianBlur(hsv_s, (7, 7), 0)
    blurred_v = cv2.GaussianBlur(hsv_v, (7, 7), 0)
    
    # 检测高饱和度区域（卡片特征）
    # 阈值可以根据实际情况调整
    _, high_sat = cv2.threshold(blurred_s, 50, 255, cv2.THRESH_BINARY)
    
    # 同时检测高亮度区域（卡片通常也比较亮）
    _, high_bright = cv2.threshold(blurred_v, 100, 255, cv2.THRESH_BINARY)
    
    # 合并：高饱和度且高亮度
    combined = cv2.bitwise_and(high_sat, high_bright)
    
    # 形态学操作：连接断开的区域
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=3)
    dilated = cv2.dilate(closed, kernel, iterations=2)
    
    # 查找轮廓
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None
    
    # 按面积排序
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    
    # 选择最大的轮廓（应该是卡片）
    largest = contours[0]
    area_ratio = cv2.contourArea(largest) / (h * w)
    
    # 卡片应该占据图像的合理比例（10%-90%）
    if area_ratio < 0.10 or area_ratio > 0.90:
        return None
    
    # 拟合为四边形
    epsilon = 0.02 * cv2.arcLength(largest, True)
    approx = cv2.approxPolyDP(largest, epsilon, True)
    
    if len(approx) >= 4:
        # 重新排序为：左上、右上、右下、左下
        from cut_image import order_points
        card_contour = order_points(approx.reshape(-1, 2)).reshape(-1, 1, 2).astype(np.int32)
        return card_contour
    
    return None


def detect_label_region(image: np.ndarray, card_contour: np.ndarray, reader, ocr_results=None) -> Tuple[Optional[np.ndarray], Optional[List]]:
    """
    精确检测label区域（返回一个连续的长方形区域）
    策略：
    1. 卡片通常是高饱和度的长方形区域
    2. label是除了卡片之外的所有区域
    3. label区域应该是一个连续的长方形，包含所有label文字
    
    Args:
        image: 原始图像（BGR格式）
        card_contour: 卡片轮廓（相对于image的坐标，如果为None则自动检测）
        reader: EasyOCR读取器对象（已初始化，可选）
        ocr_results: 已执行的OCR结果（可选，如果提供则复用，避免重复OCR）
        
    Returns:
        (label区域的掩码, OCR结果)
        label区域的掩码（255表示label区域，0表示其他区域），如果未找到则返回None
        返回的区域是一个连续的长方形
        OCR结果：如果执行了OCR，返回结果列表，否则返回None
    """
    h, w = image.shape[:2]
    
    # 如果没有提供卡片轮廓，尝试自动检测
    if card_contour is None:
        card_contour = detect_card_by_saturation(image)
        if card_contour is None:
            # 如果无法检测到卡片，返回整张图片作为label区域
            return np.ones((h, w), dtype=np.uint8) * 255, None
    
    # 获取卡片边界框
    x_card, y_card, w_card, h_card = cv2.boundingRect(card_contour)
    
    # 策略：label区域通常是卡片下方或侧边的长方形区域
    # 找到包含所有label文字的最小外接矩形
    
    label_rect = None
    
    # 如果提供了reader，使用文字位置来确定label区域
    ocr_results_used = None
    if reader is not None:
        try:
            # 如果提供了OCR结果，直接使用；否则执行OCR
            if ocr_results is not None:
                results = ocr_results
                print(f"    复用已执行的OCR结果（{len(results)} 个文字区域）")
            else:
                # EasyOCR需要RGB格式
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # 检测所有文字区域
                print(f"    执行OCR检测文字区域...")
                results = reader.readtext(image_rgb)
                ocr_results_used = results
            
            if results:
                # 收集所有label文字的位置（不在卡片内的文字）
                label_bboxes = []
                
                # 调试：统计信息
                total_texts = len(results)
                in_card_count = 0
                label_text_count = 0
                
                for (bbox, text, confidence) in results:
                    if confidence < 0.3:  # 过滤低置信度的文字
                        continue
                    
                    bbox_array = np.array(bbox, dtype=np.int32)
                    
                    # 检查文字是否在卡片内
                    center_x = int(np.mean(bbox_array[:, 0]))
                    center_y = int(np.mean(bbox_array[:, 1]))
                    
                    # 如果文字在卡片内，跳过
                    if cv2.pointPolygonTest(card_contour, (center_x, center_y), False) >= 0:
                        in_card_count += 1
                        # 调试：检查是否是数字（可能是卡号）
                        import re
                        if re.search(r'\d{6,}', text):
                            print(f"    警告: 发现6位以上数字 '{text}' 在卡片内，可能被误判")
                        continue
                    
                    # 添加到label文字列表
                    label_bboxes.append(bbox_array)
                    label_text_count += 1
                    
                    # 调试：如果找到数字，特别标记
                    import re
                    if re.search(r'\d{6,}', text):
                        print(f"    发现label区域的数字: '{text}' 位置: ({center_x}, {center_y})")
                
                print(f"    文字统计: 总共 {total_texts} 个，卡片内 {in_card_count} 个，label区域 {label_text_count} 个")
                
                # 如果找到了label文字，计算包含所有文字的最小外接矩形
                if label_bboxes:
                    # 找到所有label文字的最小外接矩形
                    all_points = np.concatenate(label_bboxes, axis=0)
                    x_min = int(np.min(all_points[:, 0]))
                    y_min = int(np.min(all_points[:, 1]))
                    x_max = int(np.max(all_points[:, 0]))
                    y_max = int(np.max(all_points[:, 1]))
                    
                    # 调试：显示找到的label文字数量
                    print(f"    找到 {len(label_bboxes)} 个label文字区域")
                    print(f"    Label文字区域: ({x_min}, {y_min}) 到 ({x_max}, {y_max})")
                    
                    # 添加更大的边距（20像素），确保包含所有文字
                    margin = 20
                    x_min = max(0, x_min - margin)
                    y_min = max(0, y_min - margin)
                    x_max = min(w, x_max + margin)
                    y_max = min(h, y_max + margin)
                    
                    label_rect = (x_min, y_min, x_max - x_min, y_max - y_min)
                    print(f"    扩展后的Label区域: ({x_min}, {y_min}) 到 ({x_max}, {y_max})")
                else:
                    print(f"    警告: 未找到任何label文字区域（所有文字都在卡片内？）")
        except Exception as e:
            print(f"    警告: 使用文字优化label区域失败: {e}")
    
    # 如果没有找到label文字区域，使用默认策略：
    # label通常在卡片下方或侧边，创建一个长方形区域
    if label_rect is None:
        # 默认策略：假设label在卡片下方
        # 创建一个从卡片底部到图片底部的长方形区域
        margin = 5
        x_min = max(0, x_card - margin)
        y_min = min(h, y_card + h_card + margin)
        x_max = min(w, x_card + w_card + margin)
        y_max = h
        
        # 如果下方区域太小，尝试右侧
        if (y_max - y_min) < h * 0.1:
            x_min = min(w, x_card + w_card + margin)
            y_min = max(0, y_card - margin)
            x_max = w
            y_max = min(h, y_card + h_card + margin)
        
        label_rect = (x_min, y_min, x_max - x_min, y_max - y_min)
    
    # 创建label区域掩码（长方形）
    label_mask = np.zeros((h, w), dtype=np.uint8)
    x, y, rect_w, rect_h = label_rect
    
    # 确保矩形在图像范围内
    x = max(0, min(x, w - 1))
    y = max(0, min(y, h - 1))
    rect_w = min(rect_w, w - x)
    rect_h = min(rect_h, h - y)
    
    if rect_w > 0 and rect_h > 0:
        label_mask[y:y+rect_h, x:x+rect_w] = 255
        
        # 排除卡片区域（如果label区域与卡片重叠）
        label_area_before = np.sum(label_mask > 0)
        cv2.fillPoly(label_mask, [card_contour], 0)
        label_area_after = np.sum(label_mask > 0)
        
        # 调试：显示label区域大小
        print(f"    Label区域（排除卡片前）: {label_area_before} 像素")
        print(f"    Label区域（排除卡片后）: {label_area_after} 像素")
    
    # 检查label区域是否足够大
    if np.sum(label_mask > 0) < 100:
        print(f"    警告: Label区域太小（{np.sum(label_mask > 0)} 像素），返回None")
        return None, ocr_results_used
    
    return label_mask, ocr_results_used


def extract_card_number(text: str) -> Optional[str]:
    """
    从文本中提取至少7位的连续数字作为卡号
    支持处理被标点符号（如逗号、句号、空格）分隔的数字
    
    Args:
        text: 输入的文本内容
    
    Returns:
        提取到的卡号（字符串），如果未找到则返回 None
    """
    import re
    
    if not text:
        print(f"    调试: extract_card_number 收到空文本")
        return None
    
    # 调试：打印输入的文本（限制长度避免过长）
    text_preview = text[:200] if len(text) > 200 else text
    print(f"    调试: 提取卡号的文本: {text_preview}")
    
    # 方法1：查找所有连续的数字（至少7位）
    pattern = r'\d{7,}'
    matches = re.findall(pattern, text)
    
    # 方法2：处理被标点符号分隔的数字（如 "551,041,454,3" 或 "551.041.454.3"）
    # 注意：只移除数字之间的分隔符，避免将不同数字错误合并
    # 使用更智能的方法：查找被分隔符分隔的数字段，然后合并
    # 例如："551,041,454,3" -> 合并为 "5510414543"
    # 但 "8.5 5510414543" 不应该合并为 "85510414543"
    
    # 先尝试方法2a：查找被分隔符分隔的连续数字段
    # 匹配模式：数字 + 分隔符 + 数字 + 分隔符 + ... + 数字
    # 要求：至少3段，每段至少3位数字（避免将 "8.5" 这样的短数字段包含进去）
    # 这样可以匹配 "551,041,454,3" 但不会匹配 "8.5 5510414543"
    pattern_separated = r'(?:\d{3,}[,.\s\-_]){2,}\d{2,}'
    separated_matches = re.findall(pattern_separated, text)
    
    # 清理这些匹配：移除分隔符
    cleaned_separated = []
    for match in separated_matches:
        cleaned = re.sub(r'[,.\s\-_]', '', match)
        if len(cleaned) >= 7:
            cleaned_separated.append(cleaned)
    
    # 方法2b：简单去除分隔符（但只用于没有明显分隔的数字段）
    # 只在没有找到方法2a的匹配时使用
    # 注意：这个方法可能会错误合并相邻的不同数字（如 "8.5 5510414543"）
    # 所以只在方法2a失败时使用
    if not cleaned_separated:
        # 更保守的方法：只移除数字之间的分隔符，不移除数字和其他字符之间的分隔符
        # 查找被分隔符分隔的数字段，但要求至少3段，每段至少2位
        # 这样可以匹配 "551,041,454,3" 但不会匹配 "8.5 5510414543"
        pattern_conservative = r'(?:\d{2,}[,.\-]){2,}\d{2,}'  # 只匹配逗号、句号、连字符，不包括空格
        conservative_matches = re.findall(pattern_conservative, text)
        if conservative_matches:
            for match in conservative_matches:
                cleaned = re.sub(r'[,.\-]', '', match)
                if len(cleaned) >= 7:
                    cleaned_separated.append(cleaned)
        
        # 如果仍然没有找到，才使用简单方法（但会有误合并的风险）
        if not cleaned_separated:
            text_cleaned = re.sub(r'[,.\s\-_]', '', text)
            pattern_cleaned = r'\d{7,}'
            matches_cleaned = re.findall(pattern_cleaned, text_cleaned)
            cleaned_separated = matches_cleaned
    else:
        matches_cleaned = []  # 如果方法2a找到了，方法2b不需要
    
    # 合并所有方法的结果
    all_matches = list(set(matches + cleaned_separated))
    
    print(f"    调试: 找到的数字匹配（方法1）: {matches}")
    print(f"    调试: 找到的数字匹配（方法2，去除分隔符后）: {cleaned_separated}")
    print(f"    调试: 合并后的匹配: {all_matches}")
    
    if all_matches:
        # 智能选择策略：
        # 1. 优先选择方法1（直接匹配）的结果，因为它们更可靠
        # 2. 如果方法1有结果，优先从方法1中选择
        # 3. 如果方法1没有结果，才使用方法2的结果
        method1_valid = [m for m in matches if 7 <= len(m) <= 12]
        method2_valid = [m for m in cleaned_separated if 7 <= len(m) <= 12]
        
        if method1_valid:
            # 方法1有有效结果，优先使用
            card_number = max(method1_valid, key=len)
            print(f"    调试: 选择的卡号: {card_number} (从方法1中选择，长度 {len(card_number)} 位)")
        elif method2_valid:
            # 方法1没有，使用方法2
            card_number = max(method2_valid, key=len)
            print(f"    调试: 选择的卡号: {card_number} (从方法2中选择，长度 {len(card_number)} 位)")
        else:
            # 都不在常见范围，选择最长的
            valid_lengths = [m for m in all_matches if 7 <= len(m) <= 12]
            if valid_lengths:
                card_number = max(valid_lengths, key=len)
                print(f"    调试: 选择的卡号: {card_number} (从 {len(all_matches)} 个匹配中选择，长度 {len(card_number)} 位)")
            else:
                card_number = max(all_matches, key=len)
                print(f"    调试: 选择的卡号: {card_number} (从 {len(all_matches)} 个匹配中选择，长度 {len(card_number)} 位，超出常见范围)")
        return card_number
    
    # 方法3：如果仍然没找到，尝试更宽松的匹配
    # 查找被分隔的数字段，然后合并（至少7位）
    # 例如："551,041,454,3" -> "5510414543" 或 "551 041 454 3" -> "5510414543"
    # 先移除所有非数字字符，然后查找连续数字
    text_digits_only = re.sub(r'[^\d]', '', text)
    if len(text_digits_only) >= 7:
        # 查找所有至少7位的连续数字段
        long_digits = re.findall(r'\d{7,}', text_digits_only)
        if long_digits:
            card_number = max(long_digits, key=len)
            print(f"    调试: 使用方法3（移除所有非数字字符后），找到卡号: {card_number}")
            return card_number
        
        # 如果整个文本去除非数字后就是一个长数字，直接使用
        if len(text_digits_only) >= 7:
            print(f"    调试: 使用方法3（整个文本去除非数字后），找到卡号: {text_digits_only}")
            return text_digits_only
    
    print(f"    调试: 未找到符合条件的卡号（至少7位的数字）")
    return None


def extract_label_text_with_easyocr(image: np.ndarray, card_contour: Optional[np.ndarray], reader) -> str:
    """
    提取卡片label区域的文字内容（使用EasyOCR，精确检测label区域）
    
    策略：
    1. 卡片通常是高饱和度的长方形区域
    2. label是除了卡片之外的所有区域
    
    Args:
        image: 原始图像（BGRA格式，带透明背景）
        card_contour: 卡片轮廓（相对于image的坐标），如果为None则自动检测
        reader: EasyOCR读取器对象（已初始化）
        
    Returns:
        提取的文字内容
    """
    if not EASYOCR_AVAILABLE:
        return ""
    
    h, w = image.shape[:2]
    
    # 如果图像是BGRA格式，转换为BGR
    if image.shape[2] == 4:
        # 创建白色背景
        bgr_image = np.ones((h, w, 3), dtype=np.uint8) * 255
        # 将BGRA合成到白色背景上
        alpha = image[:, :, 3:4] / 255.0
        bgr_image = (bgr_image * (1 - alpha) + image[:, :, :3] * alpha).astype(np.uint8)
    else:
        bgr_image = image
    
    # 图像预处理：去噪和增强对比度，提高OCR准确性
    # 这对于有噪点的label特别重要
    print(f"    对label区域进行图像预处理（去噪和对比度增强）...")
    try:
        # 使用自适应去噪
        bgr_image = denoise_image_for_ocr(bgr_image, method='adaptive')
        # 增强对比度
        bgr_image = enhance_contrast_for_ocr(bgr_image)
        print(f"    ✓ 图像预处理完成")
    except Exception as e:
        print(f"    ⚠ 图像预处理失败: {e}，继续使用原始图像")
    
    # 重要：split_cards_from_transparent 返回的 card_contour 是整个卡片+label的外轮廓
    # 我们需要重新检测卡片本身的轮廓（高饱和度区域），而不是使用传入的轮廓
    # 因为传入的轮廓包含了label区域，会导致所有文字都被判断为在卡片内
    print(f"    调试: 重新检测卡片轮廓（忽略传入的轮廓，因为它可能包含label区域）")
    card_contour = detect_card_by_saturation(bgr_image)
    if card_contour is None:
        print("    警告: 无法自动检测卡片轮廓，将提取整张图片的文字")
        # 如果无法检测到卡片，提取整张图片的文字
        card_contour = None
    else:
        # 调试：显示检测到的卡片轮廓大小
        x_card, y_card, w_card, h_card = cv2.boundingRect(card_contour)
        card_area = w_card * h_card
        total_area = h * w
        print(f"    调试: 检测到卡片轮廓，覆盖 {card_area/total_area*100:.1f}% 的图像")
    
    # 精确检测label区域（排除卡片区域）
    # 注意：detect_label_region 会执行OCR，我们复用结果避免重复OCR
    label_mask, ocr_results = detect_label_region(bgr_image, card_contour, reader)
    
    if label_mask is None:
        return ""
    
    label_area = np.sum(label_mask > 0)
    if label_area < 50:
        return ""
    
    try:
        # detect_label_region 应该总是返回OCR结果（如果提供了reader）
        # 如果返回None，说明没有执行OCR，这种情况不应该发生
        if ocr_results is not None:
            results = ocr_results
            print(f"    复用OCR结果，避免重复检测（{len(results)} 个文字区域）")
        else:
            # 这种情况不应该发生，但如果发生了，执行OCR作为后备
            print(f"    警告: detect_label_region 未返回OCR结果，执行OCR作为后备...")
            image_rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
            results = reader.readtext(image_rgb)
        
        # 调试信息：显示label区域大小
        label_area_pixels = np.sum(label_mask > 0)
        total_pixels = h * w
        print(f"    Label区域: {label_area_pixels} 像素 ({label_area_pixels/total_pixels*100:.1f}% 的图像)")
        print(f"    检测到 {len(results)} 个文字区域")
        
        # 合并所有检测到的文字（只保留在label区域内的）
        texts = []
        filtered_count = 0
        for (bbox, text, confidence) in results:
            if confidence < 0.5:  # 过滤低置信度的文字
                continue
            
            # 计算文字区域的边界框
            bbox_array = np.array(bbox, dtype=np.int32)
            center_x = int(np.mean(bbox_array[:, 0]))
            center_y = int(np.mean(bbox_array[:, 1]))
            
            # 检查中心点是否在label掩码内
            if not (0 <= center_y < h and 0 <= center_x < w):
                continue
            
            if label_mask[center_y, center_x] == 0:
                # 中心点不在label区域内，跳过
                filtered_count += 1
                continue
            
            # 进一步验证：检查文字区域与label区域的重叠度
            # 创建文字区域的掩码
            text_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(text_mask, [bbox_array], 255)
            
            # 计算文字区域与label区域的重叠
            overlap = cv2.bitwise_and(text_mask, label_mask)
            overlap_pixels = np.sum(overlap > 0)
            text_pixels = np.sum(text_mask > 0)
            
            if text_pixels == 0:
                filtered_count += 1
                continue
            
            overlap_ratio = overlap_pixels / text_pixels
            
            # 如果文字区域与label区域重叠超过50%，认为是label文字
            # 这样可以确保文字主要在label区域内
            if overlap_ratio > 0.5:
                texts.append(text)
            else:
                filtered_count += 1
        
        print(f"    过滤了 {filtered_count} 个文字区域，保留了 {len(texts)} 个label文字")
        result_text = " ".join(texts).strip()
        
        # 调试：显示提取到的文字内容（限制长度）
        if result_text:
            preview = result_text[:200] if len(result_text) > 200 else result_text
            print(f"    提取到的label文字预览: {preview}")
            # 检查是否包含6位以上数字
            import re
            numbers = re.findall(r'\d{6,}', result_text)
            if numbers:
                print(f"    发现6位以上数字: {numbers}")
            else:
                print(f"    警告: 提取的文字中没有6位以上的数字")
        else:
            print(f"    警告: 提取到的label文字为空")
        
        return result_text
    except Exception as e:
        print(f"    警告: 文字提取失败: {e}")
        import traceback
        traceback.print_exc()
        return ""


def process_single_subimage(subimage: np.ndarray) -> np.ndarray:
    """
    处理单个子图：检测顶点并裁剪背景
    
    Args:
        subimage: 单个子图（OpenCV图像对象）
        
    Returns:
        处理后的图像（已裁剪背景）
    """
    # 对每个子图进行顶点检测和裁剪
    processed = detect_and_crop_background(subimage)
    return processed


def save_paired_images(front_parts, back_parts, output_dir='output'):
    """
    按位置配对保存正面和背面的图片
    
    Args:
        front_parts: 正面图片的4个部分（字典，已处理）
        back_parts: 背面图片的4个部分（字典，已处理）
        output_dir: 输出目录
    """
    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # 定义位置列表
    positions = ['top-left', 'top-right', 'bottom-left', 'bottom-right']
    
    # 为每个位置创建目录并保存图片
    for position in positions:
        position_dir = output_path / position
        position_dir.mkdir(exist_ok=True)
        
        # 保存正面图片
        front_path = position_dir / 'front.jpg'
        cv2.imwrite(str(front_path), front_parts[position])
        print(f"已保存: {front_path}")
        
        # 保存背面图片
        back_path = position_dir / 'back.jpg'
        cv2.imwrite(str(back_path), back_parts[position])
        print(f"已保存: {back_path}")


def split_cards_from_transparent(image_bgra: np.ndarray, min_area_ratio: float = 0.05) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    从透明背景图片中分割出每张卡片
    
    策略：卡片通常排列在四个角上（2x2布局），先按水平和垂直中线分割成4个象限，
    然后在每个象限中检测卡片，避免卡片之间的连接问题。
    
    Args:
        image_bgra: 输入图像（BGRA格式，带透明背景）
        min_area_ratio: 最小区域面积比例（相对于整个图像）
    
    Returns:
        列表，每个元素是 (card_image, card_contour)，card_image 是 BGRA 格式
    """
    if image_bgra.shape[2] != 4:
        raise ValueError("输入图像必须是 BGRA 格式（4通道）")
    
    h, w = image_bgra.shape[:2]
    total_area = h * w
    min_area = total_area * min_area_ratio
    
    # 提取 alpha 通道
    alpha = image_bgra[:, :, 3]
    
    # 创建二值掩码：有像素的区域（alpha > 0）
    mask = (alpha > 0).astype(np.uint8) * 255
    
    # 形态学操作：填补小孔，但不要过度连接卡片
    # 使用较小的kernel和较少的迭代次数，避免将多张卡片连接在一起
    kernel_small = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    # 先开运算去除小噪点
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_small, iterations=1)
    # 再闭运算填补小孔，但迭代次数减少
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_small, iterations=1)
    
    # 策略：先按水平和垂直中线分割成4个象限
    # 这样可以避免不同象限的卡片被连接在一起
    mid_x = w // 2
    mid_y = h // 2
    
    print(f"    图像尺寸: {w} x {h}, 中心点: ({mid_x}, {mid_y})")
    print(f"    按2x2布局分割为4个象限...")
    
    # 定义4个象限：左上、右上、左下、右下
    quadrants = [
        (0, 0, mid_x, mid_y, "左上"),           # 左上
        (mid_x, 0, w - mid_x, mid_y, "右上"),  # 右上
        (0, mid_y, mid_x, h - mid_y, "左下"),  # 左下
        (mid_x, mid_y, w - mid_x, h - mid_y, "右下"),  # 右下
    ]
    
    all_cards = []
    
    # 在每个象限中检测卡片
    for quad_x, quad_y, quad_w, quad_h, quad_name in quadrants:
        print(f"    处理{quad_name}象限: ({quad_x}, {quad_y}) 尺寸: {quad_w} x {quad_h}")
        
        # 提取象限区域
        quad_mask = mask[quad_y:quad_y+quad_h, quad_x:quad_x+quad_w]
        quad_alpha = alpha[quad_y:quad_y+quad_h, quad_x:quad_x+quad_w]
        
        # 如果象限中没有内容，跳过
        if np.sum(quad_mask > 0) < min_area * 0.3:
            print(f"      {quad_name}象限内容太少，跳过")
            continue
        
        # 在象限内查找轮廓
        quad_contours, _ = cv2.findContours(quad_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not quad_contours:
            continue
        
        # 筛选象限内的有效轮廓
        quad_min_area = (quad_w * quad_h) * min_area_ratio
        for contour in quad_contours:
            area = cv2.contourArea(contour)
            if area < quad_min_area:
                continue
            
            # 计算边界框（相对于象限）
            x_local, y_local, card_w, card_h = cv2.boundingRect(contour)
            
            # 检查宽高比（卡片通常是矩形）
            aspect_ratio = max(card_w, card_h) / min(card_w, card_h) if min(card_w, card_h) > 0 else 0
            if aspect_ratio > 3.0:  # 太细长，可能是误检
                continue
            
            # 转换为全局坐标
            x_global = quad_x + x_local
            y_global = quad_y + y_local
            
            # 创建全局轮廓
            contour_global = contour.copy()
            contour_global[:, 0, 0] += quad_x
            contour_global[:, 0, 1] += quad_y
            
            all_cards.append((contour_global, area, quad_name))
            print(f"      {quad_name}象限: 找到卡片，面积 {area:.0f} 像素 ({area/total_area*100:.2f}%)")
    
    # 如果象限分割成功，使用象限结果
    if all_cards:
        print(f"    象限分割成功，找到 {len(all_cards)} 张卡片")
        # 按面积排序（从大到小）
        all_cards.sort(key=lambda x: x[1], reverse=True)
        valid_cards = [(contour, area) for contour, area, _ in all_cards]
    else:
        print(f"    象限分割未找到卡片，回退到全局检测...")
        # 回退到全局检测（简化版本）
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return []
        
        # 筛选轮廓
        valid_cards = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area:
                continue
            
            x, y, card_w, card_h = cv2.boundingRect(contour)
            aspect_ratio = max(card_w, card_h) / min(card_w, card_h) if min(card_w, card_h) > 0 else 0
            if aspect_ratio > 3.0:
                continue
            
            valid_cards.append((contour, area))
        
        valid_cards.sort(key=lambda x: x[1], reverse=True)
    
    # 如果象限分割成功，使用象限结果
    if all_cards:
        print(f"    象限分割成功，找到 {len(all_cards)} 张卡片")
        # 按面积排序（从大到小）
        all_cards.sort(key=lambda x: x[1], reverse=True)
        valid_cards = [(contour, area) for contour, area, _ in all_cards]
    else:
        print(f"    象限分割未找到卡片，回退到全局检测...")
        # 回退到全局检测（使用旧的逻辑）
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return []
        
        # 筛选轮廓
        valid_cards = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area:
                continue
            
            x, y, card_w, card_h = cv2.boundingRect(contour)
            aspect_ratio = max(card_w, card_h) / min(card_w, card_h) if min(card_w, card_h) > 0 else 0
            if aspect_ratio > 3.0:
                continue
            
            valid_cards.append((contour, area))
        
        valid_cards.sort(key=lambda x: x[1], reverse=True)
    
    # 提取每张卡片
    cards = []
    for contour, _ in valid_cards:
        # 获取边界框
        x, y, card_w, card_h = cv2.boundingRect(contour)
        
        # 提取卡片区域（包含透明背景）
        card_image = image_bgra[y:y+card_h, x:x+card_w].copy()
        
        # 调整轮廓坐标（相对于卡片区域）
        card_contour = contour.copy()
        card_contour[:, 0, 0] -= x
        card_contour[:, 0, 1] -= y
        
        cards.append((card_image, card_contour))
    
    return cards


def convert_bgra_to_bgr_white_background(image_bgra: np.ndarray) -> np.ndarray:
    """
    将 BGRA 格式（透明背景）转换为 BGR 格式（白色背景）
    
    Args:
        image_bgra: 输入图像（BGRA格式，带透明背景）
    
    Returns:
        输出图像（BGR格式，白色背景）
    """
    h, w = image_bgra.shape[:2]
    
    # 创建白色背景
    bgr_image = np.ones((h, w, 3), dtype=np.uint8) * 255
    
    # 提取 alpha 通道
    if image_bgra.shape[2] == 4:
        alpha = image_bgra[:, :, 3:4] / 255.0
        # 将 BGRA 合成到白色背景上
        bgr_image = (bgr_image * (1 - alpha) + image_bgra[:, :, :3] * alpha).astype(np.uint8)
    else:
        # 如果已经是 BGR，直接返回
        bgr_image = image_bgra
    
    return bgr_image


def straighten_card(image_bgra: np.ndarray, contour: np.ndarray) -> np.ndarray:
    """
    摆正卡片（使用透视变换）
    
    Args:
        image_bgra: 卡片图像（BGRA格式）
        contour: 卡片轮廓（相对于 image_bgra 的坐标）
    
    Returns:
        摆正后的卡片图像（BGRA格式）
    """
    # 获取四个角点
    corners = get_card_corners(contour)
    
    if corners is None or len(corners) != 4:
        # 如果无法获取角点，返回原图
        return image_bgra
    
    # 计算目标尺寸
    width, height = calculate_card_dimensions(corners)
    
    if width <= 0 or height <= 0:
        return image_bgra
    
    # 定义目标点
    dst = np.array([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1]
    ], dtype=np.float32)
    
    # 计算透视变换矩阵
    M = cv2.getPerspectiveTransform(corners, dst)
    
    # 应用透视变换（BGRA 格式）
    warped = cv2.warpPerspective(image_bgra, M, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_TRANSPARENT)
    
    return warped


def process_images(front_image_path, back_image_path, output_dir='output'):
    """
    处理正面和背面图片的主函数
    新流程：
    1. 使用 rembg 移除背景（得到透明背景 PNG）
    2. 基于透明背景分割图片（找到有像素的部分）
    3. 摆正每张卡片
    
    Args:
        front_image_path: 正面图片路径
        back_image_path: 背面图片路径
        output_dir: 输出目录
    """
    if not REMBG_AVAILABLE:
        raise ValueError("rembg 未安装，无法使用新流程。请安装: pip install rembg onnxruntime")
    
    # 读取图片
    print(f"正在读取正面图片: {front_image_path}")
    front_image = cv2.imread(front_image_path)
    if front_image is None:
        raise ValueError(f"无法读取正面图片: {front_image_path}")
    
    print(f"正在读取背面图片: {back_image_path}")
    back_image = cv2.imread(back_image_path)
    if back_image is None:
        raise ValueError(f"无法读取背面图片: {back_image_path}")
    
    # 第一步：使用 rembg 移除背景
    # 使用 birefnet-hrsod 模型（高分辨率显著对象检测，适合卡片）
    rembg_model = 'birefnet-hrsod'
    print(f"\n=== 第一步：使用 rembg 移除背景（模型: {rembg_model}） ===")
    print("  处理正面图片...")
    front_bgra = remove_background_rembg(front_image, model_name=rembg_model)
    if front_bgra is None:
        raise ValueError("正面图片背景移除失败")
    
    print("  处理背面图片...")
    back_bgra = remove_background_rembg(back_image, model_name=rembg_model)
    if back_bgra is None:
        raise ValueError("背面图片背景移除失败")
    
    # 保存中间过程图（转换为白色背景的 JPG）
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    intermediate_dir = output_path / 'intermediate'
    intermediate_dir.mkdir(exist_ok=True, parents=True)
    
    # 将透明背景转换为白色背景
    front_bgr = convert_bgra_to_bgr_white_background(front_bgra)
    back_bgr = convert_bgra_to_bgr_white_background(back_bgra)
    
    cv2.imwrite(str(intermediate_dir / 'front_no_bg.jpg'), front_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
    cv2.imwrite(str(intermediate_dir / 'back_no_bg.jpg'), back_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
    print(f"  中间过程图已保存: front_no_bg.jpg, back_no_bg.jpg")
    
    # 第二步：基于透明背景分割图片
    print("\n=== 第二步：基于透明背景分割图片 ===")
    print("  分割正面图片...")
    front_cards = split_cards_from_transparent(front_bgra)
    print(f"  找到 {len(front_cards)} 张正面卡片")
    
    print("  分割背面图片...")
    back_cards = split_cards_from_transparent(back_bgra)
    print(f"  找到 {len(back_cards)} 张背面卡片")
    
    if len(front_cards) != len(back_cards):
        print(f"  警告：正面和背面卡片数量不一致（正面: {len(front_cards)}, 背面: {len(back_cards)}）")
    
    # 第三步：摆正每张卡片
    print(f"\n=== 第三步：摆正每张卡片 ===")
    front_results = {}
    back_results = {}
    
    num_cards = min(len(front_cards), len(back_cards))
    for i in range(num_cards):
        card_key = f'card-{i}'
        print(f"\n处理 {card_key}...")
        
        # 摆正正面卡片
        front_card_image, front_contour = front_cards[i]
        front_straightened = straighten_card(front_card_image, front_contour)
        front_results[card_key] = front_straightened
        print(f"  ✓ 正面卡片已摆正，尺寸: {front_straightened.shape[1]} x {front_straightened.shape[0]}")
        
        # 摆正背面卡片
        back_card_image, back_contour = back_cards[i]
        back_straightened = straighten_card(back_card_image, back_contour)
        back_results[card_key] = back_straightened
        print(f"  ✓ 背面卡片已摆正，尺寸: {back_straightened.shape[1]} x {back_straightened.shape[0]}")
    
    # 第四步：提取标签文字（只提取标签区域，不提取卡片上的文字）
    print(f"\n=== 第四步：提取标签文字 ===")
    label_texts = {}
    
    if EASYOCR_AVAILABLE:
        # 初始化EasyOCR读取器（只初始化一次，提高效率）
        print("  正在初始化EasyOCR（首次运行会下载模型，可能需要一些时间）...")
        try:
            # 使用GPU（如果可用），否则使用CPU
            if GPU_AVAILABLE:
                # 检查GPU架构兼容性
                try:
                    import torch
                    if torch.cuda.is_available():
                        device_capability = torch.cuda.get_device_capability(0)
                        device_name = torch.cuda.get_device_name(0)
                        # RTX 50系列需要sm_120，但PyTorch 2.5.1只支持到sm_90
                        if device_capability[0] >= 12:
                            print(f"  ⚠ 检测到GPU: {device_name} (计算能力: {device_capability[0]}.{device_capability[1]})")
                            print(f"  ⚠ 当前PyTorch版本可能不支持此GPU架构，将尝试使用GPU，如果失败会自动回退到CPU")
                except:
                    pass
                
                try:
                    reader = easyocr.Reader(['en'], gpu=True)
                    print("  ✓ EasyOCR使用GPU模式")
                except Exception as e:
                    error_msg = str(e)
                    if "no kernel image" in error_msg.lower() or "cuda capability" in error_msg.lower():
                        print(f"  ⚠ GPU架构不兼容: {device_name if 'device_name' in locals() else '您的GPU'}需要更新的PyTorch版本")
                        print(f"  ⚠ 当前PyTorch版本: {torch.__version__} 不支持此GPU架构")
                        print(f"  ⚠ 建议: 等待PyTorch官方更新，或尝试安装PyTorch nightly版本")
                        print(f"  ⚠ 回退到CPU模式（功能正常，但速度较慢）")
                    else:
                        print(f"  ⚠ GPU模式失败: {e}")
                        print(f"  ⚠ 回退到CPU模式")
                    reader = easyocr.Reader(['en'], gpu=False)
            else:
                reader = easyocr.Reader(['en'], gpu=False)
                print("  ✓ EasyOCR使用CPU模式")
            print("  ✓ EasyOCR初始化完成")
            
            for i in range(num_cards):
                card_key = f'card-{i}'
                print(f"\n  提取 {card_key} 的标签文字...")
                
                # 获取原始分割后的卡片图像（包含标签区域）
                front_card_image, front_contour = front_cards[i]
                back_card_image, back_contour = back_cards[i]
                
                # 先提取正面标签文字
                front_label_text = extract_label_text_with_easyocr(front_card_image, front_contour, reader)
                
                # 检查正面是否找到卡号
                card_number = None
                if front_label_text:
                    card_number = extract_card_number(front_label_text)
                    if card_number:
                        print(f"    ✓ 已在正面提取到卡号: {card_number}，跳过背面OCR")
                
                # 如果正面没有找到卡号，才提取背面标签文字
                back_label_text = ""
                if not card_number:
                    back_label_text = extract_label_text_with_easyocr(back_card_image, back_contour, reader)
                else:
                    print(f"    跳过背面OCR（正面已找到卡号）")
                
                label_texts[card_key] = {
                    'front': front_label_text,
                    'back': back_label_text
                }
                
                if front_label_text:
                    print(f"    正面标签: {front_label_text}")
                if back_label_text:
                    print(f"    背面标签: {back_label_text}")
        except Exception as e:
            print(f"  警告: EasyOCR初始化失败: {e}")
            print("  跳过标签文字提取")
    else:
        print("  警告: EasyOCR未安装，跳过标签文字提取")
    
    # 第五步：提取卡号并保存结果
    print(f"\n=== 第五步：提取卡号并保存结果 ===")
    for card_key in front_results.keys():
        # 提取卡号（从正面或背面的标签文字中）
        # 注意：在第四步中已经提取了标签文字，并且如果正面找到卡号，背面OCR已被跳过
        card_number = None
        if card_key in label_texts:
            # 优先从正面标签提取卡号
            front_text = label_texts[card_key]['front']
            back_text = label_texts[card_key]['back']
            
            print(f"\n  处理 {card_key} 的卡号提取:")
            print(f"    正面标签文字长度: {len(front_text)} 字符")
            print(f"    背面标签文字长度: {len(back_text)} 字符")
            
            # 尝试从正面提取
            if front_text:
                print(f"    尝试从正面标签提取卡号...")
                card_number = extract_card_number(front_text)
                if card_number:
                    print(f"    ✓ 已在正面提取到卡号: {card_number}")
            
            # 如果正面没有提取到卡号，才尝试从背面提取
            if not card_number and back_text:
                print(f"    正面未找到卡号，尝试从背面标签提取...")
                card_number = extract_card_number(back_text)
                if card_number:
                    print(f"    ✓ 在背面提取到卡号: {card_number}")
            
            if card_number:
                print(f"  ✓ {card_key} 提取到卡号: {card_number}")
            else:
                print(f"  ✗ 警告: {card_key} 未找到卡号（至少7位的连续数字），使用备用命名")
        else:
            print(f"  ✗ 警告: {card_key} 没有标签文字数据，使用备用命名")
        
        # 确定目录名和文件名
        if card_number:
            # 使用卡号作为目录名
            card_dir_name = card_number
            # 文件名使用 卡号_A.jpg 和 卡号_B.jpg
            front_filename = f"{card_number}_A.jpg"
            back_filename = f"{card_number}_B.jpg"
            label_filename = f"{card_number}_label.txt"
        else:
            # 如果找不到卡号，使用备用命名
            card_dir_name = card_key
            front_filename = f"{card_key}_A.jpg"
            back_filename = f"{card_key}_B.jpg"
            label_filename = f"{card_key}_label.txt"
        
        # 为每张卡创建单独的目录
        card_dir = output_path / card_dir_name
        card_dir.mkdir(exist_ok=True)
        
        # 将透明背景转换为白色背景
        front_bgr = convert_bgra_to_bgr_white_background(front_results[card_key])
        back_bgr = convert_bgra_to_bgr_white_background(back_results[card_key])
        
        # 保存到卡片目录中（使用卡号_A、卡号_B的命名）
        front_path = card_dir / front_filename
        back_path = card_dir / back_filename
        
        cv2.imwrite(str(front_path), front_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
        print(f"  已保存正面: {front_path}")
        
        cv2.imwrite(str(back_path), back_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
        print(f"  已保存背面: {back_path}")
        
        # 保存标签文字（如果提取成功）
        if card_key in label_texts:
            label_file = card_dir / label_filename
            
            with open(label_file, 'w', encoding='utf-8') as f:
                f.write(f"正面标签: {label_texts[card_key]['front']}\n")
                f.write(f"背面标签: {label_texts[card_key]['back']}\n")
                if card_number:
                    f.write(f"提取的卡号: {card_number}\n")
            print(f"  已保存标签文字: {label_file}")
    
    print(f"\n=== 处理完成 ===")
    print(f"共处理 {num_cards} 张卡片")
    print(f"结果保存在: {output_dir}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='切割图片：去除背景并分割成2x2区域')
    parser.add_argument('--front', '-f', type=str, required=True,
                       help='正面图片路径')
    parser.add_argument('--back', '-b', type=str, required=True,
                       help='背面图片路径')
    parser.add_argument('--output', '-o', type=str, default='output',
                       help='输出目录（默认: output）')
    
    args = parser.parse_args()
    
    try:
        process_images(args.front, args.back, args.output)
    except Exception as e:
        print(f"错误: {e}")
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())

