"""
使用示例：如何在代码中直接调用图片切割功能
"""
from cut_image import process_images

# 示例1: 基本使用
if __name__ == '__main__':
    # 指定正面和背面图片路径
    front_path = 'front.jpg'  # 替换为你的正面图片路径
    back_path = 'back.jpg'    # 替换为你的背面图片路径
    output_directory = 'output'
    
    try:
        process_images(front_path, back_path, output_directory)
        print("处理成功！")
    except Exception as e:
        print(f"处理失败: {e}")

