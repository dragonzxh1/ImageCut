# 多张卡片检测修复

## 问题描述

在处理包含多张卡片的图片（如TESTBACK.jpg，包含4张卡片）时，程序只检测到2张卡片。

## 问题原因

1. **形态学操作过度连接**：原始的形态学操作（`MORPH_CLOSE`迭代3次，`MORPH_OPEN`迭代2次，使用5x5的kernel）将多张卡片连接成一个大的轮廓。

2. **缺少分离机制**：当多张卡片被连接在一起时，没有机制来分离它们。

## 解决方案

### 1. 减少形态学操作强度

```python
# 改进前
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)

# 改进后
kernel_small = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_small, iterations=1)
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_small, iterations=1)
```

- 使用更小的kernel（3x3而不是5x5）
- 减少迭代次数（1次而不是2-3次）
- 先开运算去除噪点，再闭运算填补小孔

### 2. 智能分离连接的卡片

对于面积超过图像15%的大轮廓（可能是多张卡片连接），使用距离变换和连通组件分析来分离：

```python
# 使用距离变换找到卡片中心
dist_transform = cv2.distanceTransform(roi_mask, cv2.DIST_L2, 5)

# 找到局部最大值（卡片中心）
_, sure_fg = cv2.threshold(dist_transform, 0.3 * dist_transform.max(), 255, 0)

# 查找连通组件（每张卡片应该是一个组件）
num_labels, labels = cv2.connectedComponents(sure_fg)

# 对每个组件，膨胀以包含整个卡片
for label_id in range(1, num_labels):
    component_mask = (labels == label_id).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (50, 50))
    component_mask = cv2.dilate(component_mask, kernel, iterations=1)
    # 查找轮廓并分离
```

### 3. 处理流程

1. **初始检测**：使用轻量级形态学操作检测轮廓
2. **分类轮廓**：
   - 大轮廓（面积>15%）：可能是多张卡片连接，需要分离
   - 小轮廓（面积5%-15%）：可能是单张卡片
3. **分离大轮廓**：
   - 使用距离变换找到卡片中心
   - 使用连通组件分析识别每张卡片
   - 膨胀每个组件以包含整个卡片
   - 提取分离后的轮廓
4. **合并结果**：将所有轮廓合并，按面积排序

## 测试结果

### 测试图像：TESTBACK.jpg
- **原始问题**：只检测到2张卡片
- **修复后**：成功检测到4张卡片 ✓

### 检测到的卡片信息

| 卡片 | 尺寸 | 面积比例 |
|------|------|----------|
| Card 1 | 1732 x 2798 | 18.44% |
| Card 2 | 1734 x 2711 | 18.07% |
| Card 3 | 1233 x 2609 | 12.14% |
| Card 4 | 1186 x 2369 | 11.18% |

## 代码位置

修改的文件：`cut_image_gpu.py`

函数：`split_cards_from_transparent()`

主要改进：
- 第2366-2372行：减少形态学操作强度
- 第2380-2450行：添加智能分离逻辑

## 注意事项

1. **性能影响**：距离变换和连通组件分析会增加处理时间，但通常可以接受
2. **参数调整**：
   - `min_area_ratio=0.05`：最小卡片面积比例
   - `area_ratio > 0.15`：触发分离的阈值
   - `0.3 * dist_transform.max()`：距离变换阈值
   - `(50, 50)`：膨胀kernel大小
3. **适用场景**：适用于卡片之间有明显间距的情况，如果卡片完全重叠可能无法分离

## 更新日期

2024-12-11

