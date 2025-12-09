import cv2
import numpy as np
import os
from pathlib import Path
import argparse
from typing import Optional


def detect_card_contour(image: np.ndarray) -> Optional[np.ndarray]:
    """
    检测卡牌/图片轮廓（支持深色背景、白色背景、透明塑料、白色纸张）
    核心思路：根据背景类型选择不同的检测策略
    
    Args:
        image: 输入图像（BGR格式）
        
    Returns:
        卡牌轮廓（4个顶点的四边形），如果找到，否则返回None
    """
    h, w = image.shape[:2]
    
    # 第一步：图像预处理
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 0)
    
    # 第二步：分析图像特征，判断背景类型
    edge_margin = 20
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
    is_dark_background = edge_mean < 50    # 深色背景
    is_transparent = edge_std > 30  # 透明材质（反光导致标准差大）
    
    # 第三步：根据背景类型选择不同的处理方法
    if is_white_background:
        # 白色背景：使用梯度检测和边缘检测
        # 方法1：Sobel梯度检测
        sobelx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
        gradient = np.sqrt(sobelx**2 + sobely**2)
        if gradient.max() > 0:
            gradient = np.uint8(np.clip(gradient * 255 / gradient.max(), 0, 255))
        else:
            gradient = np.zeros_like(blurred, dtype=np.uint8)
        
        # 方法2：自适应阈值（针对白色背景）
        binary = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 11, 2
        )
        
        # 方法3：OTSU阈值（反转，因为背景是白色）
        _, binary_otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # 结合梯度图和阈值图
        combined = cv2.bitwise_or(gradient, binary)
        combined = cv2.bitwise_or(combined, binary_otsu)
        
        # 方法4：Canny边缘检测（对白色背景更敏感）
        edges = cv2.Canny(blurred, 30, 100)  # 降低阈值以检测更多边缘
        combined = cv2.bitwise_or(combined, edges)
        
    elif is_transparent:
        # 透明塑料：检测反光和边缘
        # 使用OTSU
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 检测高光区域（透明塑料的反光）
        _, binary_high = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY)
        
        # 结合两种二值化结果
        combined = cv2.bitwise_or(binary, binary_high)
        
        # Canny边缘检测
        edges = cv2.Canny(blurred, 50, 150)
        combined = cv2.bitwise_or(combined, edges)
        
    else:
        # 深色背景：使用原有方法
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        edges = cv2.Canny(blurred, 50, 150)
        combined = cv2.bitwise_or(binary, edges)
    
    # 第四步：形态学操作
    kernel = np.ones((5, 5), np.uint8)
    closed = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=3)
    dilated = cv2.dilate(closed, kernel, iterations=2)
    
    # 第五步：查找轮廓
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        # 备用方法：使用更激进的边缘检测
        edges = cv2.Canny(blurred, 20, 80)
        dilated_edges = cv2.dilate(edges, kernel, iterations=5)
        contours, _ = cv2.findContours(dilated_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None
    
    # 第六步：筛选轮廓
    min_area = h * w * 0.10
    max_area = h * w * 0.95
    
    valid_contours = [c for c in contours if min_area < cv2.contourArea(c) < max_area]
    
    if not valid_contours:
        min_area = h * w * 0.05
        valid_contours = [c for c in contours if min_area < cv2.contourArea(c) < max_area]
    
    if not valid_contours:
        return None
    
    # 选择最大的轮廓
    largest_contour = max(valid_contours, key=cv2.contourArea)
    
    # 第七步：多边形拟合 - 将轮廓拟合为四边形
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
        return best_approx
    
    # 如果拟合后不是4个顶点，使用最小外接矩形
    rect = cv2.minAreaRect(largest_contour)
    box = cv2.boxPoints(rect)
    box = np.int32(box)
    
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
    计算卡牌的目标尺寸
    
    Args:
        corners: 四个角点
        
    Returns:
        (宽度, 高度)
    """
    # 计算宽度（取上下两边的平均值）
    width_a = np.sqrt(((corners[1][0] - corners[0][0]) ** 2) + 
                      ((corners[1][1] - corners[0][1]) ** 2))
    width_b = np.sqrt(((corners[2][0] - corners[3][0]) ** 2) + 
                      ((corners[2][1] - corners[3][1]) ** 2))
    width = max(int(width_a), int(width_b))
    
    # 计算高度（取左右两边的平均值）
    height_a = np.sqrt(((corners[3][0] - corners[0][0]) ** 2) + 
                       ((corners[3][1] - corners[0][1]) ** 2))
    height_b = np.sqrt(((corners[2][0] - corners[1][0]) ** 2) + 
                       ((corners[2][1] - corners[1][1]) ** 2))
    height = max(int(height_a), int(height_b))
    
    return width, height


def perspective_transform(image: np.ndarray, corners: np.ndarray) -> np.ndarray:
    """
    对图像进行透视变换，矫正方向并裁剪
    
    Args:
        image: 输入图像（BGR格式）
        corners: 四个角点（已排序：左上、右上、右下、左下）
        
    Returns:
        矫正后的图像
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
    
    return warped


def detect_and_crop_background(image):
    """
    检测并裁剪图片中的空白/深色背景
    使用改进的轮廓检测方法和透视变换
    
    Args:
        image: OpenCV图像对象
        
    Returns:
        裁剪后的图像
    """
    # 首先尝试使用改进的轮廓检测方法
    contour = detect_card_contour(image)
    
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


def split_image_2x2(image):
    """
    将图片切割成2x2的4个区域
    
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


def process_images(front_image_path, back_image_path, output_dir='output'):
    """
    处理正面和背面图片的主函数
    新流程：先切割成子图，再对每个子图进行顶点检测和裁剪
    
    Args:
        front_image_path: 正面图片路径
        back_image_path: 背面图片路径
        output_dir: 输出目录
    """
    # 读取图片
    print(f"正在读取正面图片: {front_image_path}")
    front_image = cv2.imread(front_image_path)
    if front_image is None:
        raise ValueError(f"无法读取正面图片: {front_image_path}")
    
    print(f"正在读取背面图片: {back_image_path}")
    back_image = cv2.imread(back_image_path)
    if back_image is None:
        raise ValueError(f"无法读取背面图片: {back_image_path}")
    
    # 第一步：先切割图片成子图
    print("正在切割正面图片...")
    front_parts_raw = split_image_2x2(front_image)
    
    print("正在切割背面图片...")
    back_parts_raw = split_image_2x2(back_image)
    
    # 验证切割结果数量是否一致
    front_count = len(front_parts_raw)
    back_count = len(back_parts_raw)
    front_keys = set(front_parts_raw.keys())
    back_keys = set(back_parts_raw.keys())
    
    if front_count != back_count:
        raise ValueError(f"切割结果数量不一致：正面{front_count}个，背面{back_count}个")
    
    if front_keys != back_keys:
        raise ValueError(f"切割结果位置不一致：正面{front_keys}，背面{back_keys}")
    
    # 第二步：对每个子图分别进行顶点检测和裁剪
    print("正在处理每个子图的背景...")
    front_parts_processed = {}
    back_parts_processed = {}
    
    positions = ['top-left', 'top-right', 'bottom-left', 'bottom-right']
    for position in positions:
        if position in front_parts_raw:
            print(f"  处理正面-{position}...")
            front_parts_processed[position] = process_single_subimage(front_parts_raw[position])
            
            print(f"  处理背面-{position}...")
            back_parts_processed[position] = process_single_subimage(back_parts_raw[position])
    
    # 可选：保存原始切割的子图用于调试
    debug_dir = Path(output_dir) / 'debug'
    debug_dir.mkdir(exist_ok=True, parents=True)
    for position in positions:
        if position in front_parts_raw:
            cv2.imwrite(str(debug_dir / f'front_{position}_raw.jpg'), front_parts_raw[position])
            cv2.imwrite(str(debug_dir / f'back_{position}_raw.jpg'), back_parts_raw[position])
            cv2.imwrite(str(debug_dir / f'front_{position}_processed.jpg'), front_parts_processed[position])
            cv2.imwrite(str(debug_dir / f'back_{position}_processed.jpg'), back_parts_processed[position])
    print(f"调试图片已保存到: {debug_dir}")
    
    # 第三步：保存处理后的配对图片
    print("正在保存配对图片...")
    save_paired_images(front_parts_processed, back_parts_processed, output_dir)
    
    print(f"\n处理完成！所有图片已保存到: {output_dir}")


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

