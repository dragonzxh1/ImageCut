# OCR噪点处理和卡号提取改进

## 问题描述

label区域上的噪点导致OCR提取文字和数字时不连续，产生了类似逗号的文字，导致提取卡号不正确。

例如：
- 原始卡号：`5510414543`
- OCR识别结果：`551,041,454,3` 或 `551.041.454.3`
- 导致无法正确提取卡号

## 解决方案

### 1. 图像预处理（去噪和对比度增强）

在OCR之前对label区域图像进行预处理：

#### 功能
- **自适应去噪**：根据图像对比度自动选择最佳去噪方法
  - 低对比度：形态学操作 + 高斯模糊
  - 中等对比度：中值滤波
  - 高对比度：轻微高斯模糊
- **对比度增强**：使用CLAHE（对比度受限的自适应直方图均衡化）提高文字清晰度

#### 实现位置
- `denoise_image_for_ocr()` - 去噪函数
- `apply_morphology_denoise()` - 形态学去噪
- `enhance_contrast_for_ocr()` - 对比度增强

#### 使用
在 `extract_label_text_with_easyocr()` 函数中自动应用：
```python
# 图像预处理：去噪和增强对比度，提高OCR准确性
bgr_image = denoise_image_for_ocr(bgr_image, method='adaptive')
bgr_image = enhance_contrast_for_ocr(bgr_image)
```

### 2. 改进的卡号提取逻辑

`extract_card_number()` 函数现在支持多种方法提取卡号：

#### 方法1：直接匹配连续数字
- 查找至少7位的连续数字：`\d{7,}`
- 例如：`5510414543` ✓

#### 方法2：去除分隔符后匹配
- 移除常见分隔符（逗号、句号、空格、连字符、下划线）
- 然后查找连续数字
- 例如：`551,041,454,3` → `5510414543` ✓

#### 方法3：完全去除非数字字符
- 移除所有非数字字符
- 如果结果长度≥7位，直接使用
- 例如：`551.041.454.3` → `5510414543` ✓

#### 代码示例
```python
def extract_card_number(text: str) -> Optional[str]:
    # 方法1：直接匹配
    matches = re.findall(r'\d{7,}', text)
    
    # 方法2：去除分隔符后匹配
    text_cleaned = re.sub(r'[,.\s\-_]', '', text)
    matches_cleaned = re.findall(r'\d{7,}', text_cleaned)
    
    # 方法3：完全去除非数字字符
    text_digits_only = re.sub(r'[^\d]', '', text)
    if len(text_digits_only) >= 7:
        return text_digits_only
    
    # 返回最长的匹配
    all_matches = list(set(matches + matches_cleaned))
    if all_matches:
        return max(all_matches, key=len)
    
    return None
```

## 效果

### 改进前
- 输入：`551,041,454,3`
- 提取结果：`None`（无法识别）

### 改进后
- 输入：`551,041,454,3`
- 提取结果：`5510414543` ✓

## 配置选项

### 去噪方法
可以通过修改 `denoise_image_for_ocr()` 的 `method` 参数选择：
- `'adaptive'` - 自适应（推荐，默认）
- `'gaussian'` - 高斯模糊
- `'median'` - 中值滤波
- `'morphology'` - 形态学操作
- `'combined'` - 组合方法

### 对比度增强
可以通过 `enhance_contrast_for_ocr()` 的参数调整：
- `clipLimit=2.0` - 对比度限制（默认2.0）
- `tileGridSize=(8, 8)` - 网格大小（默认8x8）

## 测试

可以使用以下测试用例验证改进：

```python
# 测试用例
test_cases = [
    "5510414543",           # 正常情况
    "551,041,454,3",        # 逗号分隔
    "551.041.454.3",        # 句号分隔
    "551 041 454 3",        # 空格分隔
    "551-041-454-3",        # 连字符分隔
    "Card: 551,041,454,3",  # 带其他文字
]

for text in test_cases:
    result = extract_card_number(text)
    print(f"输入: {text} -> 输出: {result}")
```

## 相关文件

- `cut_image_gpu.py` - GPU版本（已更新）
- `cut_image.py` - CPU版本（建议同步更新）

## 更新日期

2024-12-11

