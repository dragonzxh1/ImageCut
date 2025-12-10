"""
验证EasyOCR是否能使用GPU
针对RTX 50系列（sm_120）的GPU支持测试
"""
import torch
import easyocr

print("=" * 60)
print("EasyOCR GPU支持验证")
print("=" * 60)
print()

# 1. 首先检查PyTorch层面的GPU状态
print("1. PyTorch GPU状态检查")
print("-" * 60)
print(f"PyTorch版本: {torch.__version__}")
print(f"CUDA可用: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"GPU设备名称: {torch.cuda.get_device_name(0)}")
    capability = torch.cuda.get_device_capability(0)
    print(f"CUDA计算能力: {capability[0]}.{capability[1]} (sm_{capability[0]}{capability[1]})")
    
    # 检查支持的架构
    if hasattr(torch.cuda, 'get_arch_list'):
        arch_list = torch.cuda.get_arch_list()
        print(f"PyTorch支持的架构: {arch_list}")
        if f'sm_{capability[0]}{capability[1]}' in arch_list or f'sm_{capability[0]}{capability[1]:02d}' in arch_list:
            print("✓ GPU架构在支持列表中")
        else:
            print("⚠ GPU架构不在支持列表中，但可能仍可使用")
    
    # 测试GPU计算
    try:
        x = torch.randn(3, 3).cuda()
        y = x @ x.T
        print(f"✓ GPU计算测试成功: {y.device}")
    except Exception as e:
        print(f"✗ GPU计算测试失败: {e}")
else:
    print("✗ CUDA不可用")

print()
print("2. EasyOCR初始化")
print("-" * 60)

try:
    # 初始化EasyOCR Reader
    # gpu=True是默认值，但在显卡无法识别时会自动变为False
    print("正在初始化EasyOCR（首次运行会下载模型，可能需要一些时间）...")
    reader = easyocr.Reader(['en'], gpu=True, verbose=False)
    print("✓ EasyOCR初始化完成")
    
    print()
    print("3. EasyOCR设备检查")
    print("-" * 60)
    
    # 检查EasyOCR实际使用的设备
    # EasyOCR的detector和recognizer都是PyTorch模型
    try:
        detector_device = next(reader.detector.parameters()).device
        print(f"Detector设备: {detector_device}")
        
        recognizer_device = next(reader.recognizer.parameters()).device
        print(f"Recognizer设备: {recognizer_device}")
        
        if detector_device.type == 'cuda' and recognizer_device.type == 'cuda':
            print("✓ EasyOCR成功使用GPU！")
            print(f"  GPU设备: {detector_device}")
        elif detector_device.type == 'cuda' or recognizer_device.type == 'cuda':
            print("⚠ EasyOCR部分使用GPU")
            print(f"  Detector: {detector_device}")
            print(f"  Recognizer: {recognizer_device}")
        else:
            print("✗ EasyOCR使用CPU模式")
            print("  原因: PyTorch可能不支持此GPU架构，或GPU初始化失败")
    except Exception as e:
        print(f"⚠ 无法检查设备信息: {e}")
        print("  这可能是因为模型尚未加载到设备上")
    
    print()
    print("4. 简单OCR测试")
    print("-" * 60)
    
    # 创建一个简单的测试图像（白色背景，黑色文字）
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont
    
    # 创建测试图像
    img = Image.new('RGB', (200, 50), color='white')
    draw = ImageDraw.Draw(img)
    try:
        # 尝试使用系统字体
        font = ImageFont.truetype("arial.ttf", 24)
    except:
        font = ImageFont.load_default()
    draw.text((10, 10), "TEST 123", fill='black', font=font)
    
    # 转换为numpy数组
    test_image = np.array(img)
    
    # 运行OCR
    print("运行OCR测试...")
    results = reader.readtext(test_image)
    
    if results:
        print(f"✓ OCR测试成功，识别到 {len(results)} 个文本区域")
        for result in results:
            print(f"  文本: {result[1]}, 置信度: {result[2]:.2f}")
    else:
        print("⚠ OCR测试未识别到文本（可能是测试图像问题）")
    
except Exception as e:
    print(f"✗ EasyOCR初始化或测试失败: {e}")
    import traceback
    traceback.print_exc()

print()
print("=" * 60)
print("验证完成")
print("=" * 60)
print()
print("总结:")
if torch.cuda.is_available():
    capability = torch.cuda.get_device_capability(0)
    if capability[0] >= 12:
        print("✓ GPU计算能力12.0+，应该支持RTX 50系列")
    else:
        print("⚠ GPU计算能力较低，可能不支持RTX 50系列")
    
    try:
        reader = easyocr.Reader(['en'], gpu=True, verbose=False)
        detector_device = next(reader.detector.parameters()).device
        if detector_device.type == 'cuda':
            print("✓ EasyOCR成功使用GPU加速")
        else:
            print("✗ EasyOCR仍在使用CPU模式")
    except:
        print("⚠ 无法验证EasyOCR设备状态")
else:
    print("✗ CUDA不可用，无法使用GPU")

