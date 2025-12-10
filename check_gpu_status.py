"""
GPU状态检查脚本
用于诊断GPU和CUDA配置问题
"""
import sys

print("=" * 60)
print("GPU和CUDA状态检查")
print("=" * 60)
print()

# 1. 检查PyTorch
print("1. 检查PyTorch安装...")
try:
    import torch
    print(f"   ✓ PyTorch版本: {torch.__version__}")
    
    # 检查CUDA是否可用
    print(f"   CUDA可用: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"   ✓ CUDA版本: {torch.version.cuda}")
        print(f"   ✓ cuDNN版本: {torch.backends.cudnn.version()}")
        print(f"   ✓ GPU数量: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"   ✓ GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"     显存: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")
    else:
        print("   ✗ CUDA不可用")
        print("   可能的原因：")
        print("   - PyTorch安装的是CPU版本")
        print("   - CUDA驱动未正确安装")
        print("   - CUDA版本与PyTorch不匹配")
        
except ImportError:
    print("   ✗ PyTorch未安装")
    print("   请运行: pip install torch torchvision")

print()

# 2. 检查onnxruntime
print("2. 检查onnxruntime...")
try:
    import onnxruntime as ort
    print(f"   ✓ onnxruntime版本: {ort.__version__}")
    
    # 检查可用的执行提供者
    available_providers = ort.get_available_providers()
    print(f"   ✓ 可用执行提供者: {available_providers}")
    
    if 'CUDAExecutionProvider' in available_providers:
        print("   ✓ CUDA执行提供者可用")
    else:
        print("   ✗ CUDA执行提供者不可用")
        print("   可能的原因：")
        print("   - 安装的是onnxruntime（CPU版本）而不是onnxruntime-gpu")
        print("   - CUDA库文件缺失或不兼容")
        
except ImportError:
    print("   ✗ onnxruntime未安装")

print()

# 3. 检查系统CUDA
print("3. 检查系统CUDA安装...")
import subprocess
import os

# 检查nvcc
try:
    result = subprocess.run(['nvcc', '--version'], 
                          capture_output=True, text=True, timeout=5)
    if result.returncode == 0:
        lines = result.stdout.split('\n')
        for line in lines:
            if 'release' in line.lower():
                print(f"   ✓ {line.strip()}")
    else:
        print("   ✗ nvcc未找到或无法运行")
except FileNotFoundError:
    print("   ✗ nvcc未找到（CUDA Toolkit可能未安装或未添加到PATH）")
except Exception as e:
    print(f"   ✗ 检查nvcc时出错: {e}")

# 检查CUDA环境变量
cuda_path = os.environ.get('CUDA_PATH')
if cuda_path:
    print(f"   ✓ CUDA_PATH环境变量: {cuda_path}")
    cuda_bin = os.path.join(cuda_path, 'bin')
    if os.path.exists(cuda_bin):
        print(f"   ✓ CUDA bin目录存在: {cuda_bin}")
    else:
        print(f"   ✗ CUDA bin目录不存在: {cuda_bin}")
else:
    print("   ⚠ CUDA_PATH环境变量未设置")

print()

# 4. 检查rembg
print("4. 检查rembg...")
try:
    from rembg import new_session
    print("   ✓ rembg已安装")
    print("   注意: rembg会自动检测GPU，如果CUDA不可用会使用CPU")
except ImportError:
    print("   ✗ rembg未安装")

print()

# 5. 检查EasyOCR
print("5. 检查EasyOCR...")
try:
    import easyocr
    print("   ✓ EasyOCR已安装")
    print("   注意: EasyOCR需要PyTorch GPU支持才能使用GPU")
except ImportError:
    print("   ✗ EasyOCR未安装")

print()
print("=" * 60)
print("诊断完成")
print("=" * 60)
print()
print("建议：")
print("1. 如果PyTorch显示CUDA不可用，请安装GPU版本的PyTorch：")
print("   对于RTX 50系列（sm_120）GPU：")
print("     运行: INSTALL_PYTORCH_RTX50.bat")
print("     或手动安装: pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128")
print()
print("   对于其他GPU（sm_50-sm_90）：")
print("     运行: install_pytorch_gpu.bat")
print("     或手动安装: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
print()
print("2. 如果onnxruntime-gpu的CUDA不可用，请检查：")
print("   - 是否安装了onnxruntime-gpu（不是onnxruntime）")
print("   - CUDA版本是否匹配（需要CUDA 11.x或12.x，不支持13.x）")
print()
print("3. 确保CUDA Toolkit已正确安装并添加到PATH")
print()
print("4. 验证EasyOCR GPU支持：")
print("     运行: python check_easyocr_gpu.py")


