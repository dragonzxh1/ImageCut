# 图片切割工具

这个工具可以自动检测并裁剪图片中的空白/深色背景，然后将图片切割成2x2的4个区域，并将正面和背面的对应位置图片配对保存。

## 功能特点

- ✅ 自动检测并裁剪空白/深色背景
- ✅ 将图片切割成2x2的4个区域（左上、右上、左下、右下）
- ✅ 按位置配对保存正面和背面图片
- ✅ 自动创建目录结构

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

### 命令行方式

```bash
python cut_image.py --front 正面图片.jpg --back 背面图片.jpg --output output
```

参数说明：
- `--front` 或 `-f`: 正面图片路径（必需）
- `--back` 或 `-b`: 背面图片路径（必需）
- `--output` 或 `-o`: 输出目录（可选，默认为 `output`）

### 示例

```bash
python cut_image.py -f front.jpg -b back.jpg -o result
```

## 输出结构

处理完成后，会在输出目录下创建以下结构：

```
output/
  ├── top-left/
  │   ├── front.jpg    (正面的左上)
  │   └── back.jpg     (背面的左上)
  ├── top-right/
  │   ├── front.jpg    (正面的右上)
  │   └── back.jpg     (背面的右上)
  ├── bottom-left/
  │   ├── front.jpg    (正面的左下)
  │   └── back.jpg     (背面的左下)
  ├── bottom-right/
  │   ├── front.jpg    (正面的右下)
  │   └── back.jpg     (背面的右下)
  └── debug/
      ├── front_cropped.jpg  (裁剪后的正面完整图)
      └── back_cropped.jpg   (裁剪后的背面完整图)
```

## 工作原理

1. **背景检测**: 使用边缘检测和轮廓分析自动识别图片中的有效内容区域
2. **背景裁剪**: 去除四周的空白/深色背景，只保留有效内容
3. **图片切割**: 将裁剪后的图片按中心点切割成2x2的4个区域
4. **配对保存**: 将正面和背面的对应位置图片保存到同一目录

## 注意事项

- 确保正面和背面图片的尺寸和布局相似
- 如果背景检测不准确，可以调整代码中的阈值参数
- 调试图片会保存在 `debug` 目录中，可用于检查裁剪效果

## 技术栈

- OpenCV: 图像处理和计算机视觉
- NumPy: 数值计算
- Python 3.7+

