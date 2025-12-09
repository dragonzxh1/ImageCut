"""
启动器脚本 - 修复版
解决打包后Streamlit无法运行和重复打开浏览器的问题
"""
import sys
import os
import subprocess
import webbrowser
import time
import socket

def check_port(host, port, timeout=1):
    """检查端口是否可用"""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        result = sock.connect_ex((host, port))
        sock.close()
        return result == 0
    except:
        return False

def main():
    # 获取当前exe的目录
    if getattr(sys, 'frozen', False):
        # 打包后的exe
        base_path = sys._MEIPASS
        app_path = os.path.join(base_path, 'app.py')
        
        # 关键修复：在打包环境中，需要找到Python解释器
        # 由于exe本身就是Python解释器，我们需要直接运行streamlit
        # 但更好的方法是：检查是否有独立的Python环境
        
        # 尝试方法1：使用pythonw.exe（如果存在）
        python_exe = None
        exe_dir = os.path.dirname(sys.executable)
        
        # 查找可能的Python解释器
        possible_paths = [
            os.path.join(exe_dir, 'python.exe'),
            os.path.join(exe_dir, 'pythonw.exe'),
            sys.executable,  # 最后尝试使用exe本身
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                python_exe = path
                break
        
        if not python_exe:
            python_exe = sys.executable
    else:
        # 开发模式
        base_path = os.path.dirname(os.path.abspath(__file__))
        app_path = os.path.join(base_path, 'app.py')
        python_exe = sys.executable
    
    print("正在启动图片切割工具...")
    print(f"应用路径: {app_path}")
    print(f"Python解释器: {python_exe}")
    
    # 检查app.py是否存在
    if not os.path.exists(app_path):
        print(f"错误：找不到app.py文件: {app_path}")
        input("按回车键退出...")
        sys.exit(1)
    
    # 启动streamlit服务器
    try:
        # 关键修复：使用正确的参数启动streamlit
        # 禁用自动打开浏览器，我们手动控制
        cmd = [
            python_exe, '-m', 'streamlit', 'run', app_path,
            '--server.headless', 'true',
            '--server.port', '8501',
            '--browser.gatherUsageStats', 'false',
            '--server.enableCORS', 'false',
            '--server.enableXsrfProtection', 'false',
            '--server.runOnSave', 'false',
            '--browser.serverAddress', 'localhost',
            '--server.address', 'localhost',
        ]
        
        print(f"执行命令: {' '.join(cmd)}")
        
        # 启动进程，不捕获输出以便看到错误信息
        process = subprocess.Popen(
            cmd,
            stdout=sys.stdout,
            stderr=sys.stderr,
            cwd=base_path  # 设置工作目录
        )
        
        # 等待服务器启动
        print("等待服务器启动...")
        max_wait = 15  # 最多等待15秒
        waited = 0
        opened = False
        
        while waited < max_wait:
            time.sleep(0.5)
            waited += 0.5
            
            # 检查进程是否还在运行
            if process.poll() is not None:
                # 进程已结束，可能启动失败
                print(f"\n错误：Streamlit进程已退出，退出码: {process.returncode}")
                print("请检查上面的错误信息")
                input("按回车键退出...")
                sys.exit(1)
            
            # 检查端口是否可用
            if check_port('localhost', 8501, timeout=0.5):
                # 服务器已启动，只打开一次浏览器
                if not opened:
                    url = 'http://localhost:8501'
                    print(f"\n服务器已启动！正在打开浏览器: {url}")
                    # 只打开一次
                    webbrowser.open(url)
                    opened = True
                    print("浏览器已打开，请不要重复打开")
                    break
        
        if not opened:
            print(f"\n警告：等待{max_wait}秒后服务器可能还未完全启动")
            print("请手动访问: http://localhost:8501")
            # 即使没检测到，也尝试打开一次（可能检测有延迟）
            if not opened:
                url = 'http://localhost:8501'
                print(f"尝试打开浏览器: {url}")
                webbrowser.open(url)
                opened = True
        
        if opened:
            print("\n" + "="*50)
            print("服务器运行中...")
            print("按 Ctrl+C 停止服务器")
            print("="*50)
        
        # 等待进程结束
        try:
            process.wait()
        except KeyboardInterrupt:
            print("\n正在关闭服务器...")
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
            print("服务器已关闭")
            
    except FileNotFoundError:
        print(f"错误：找不到Python解释器: {python_exe}")
        print("请确保Python已正确安装")
        input("按回车键退出...")
        sys.exit(1)
    except Exception as e:
        print(f"启动失败: {e}")
        import traceback
        traceback.print_exc()
        input("按回车键退出...")
        sys.exit(1)

if __name__ == "__main__":
    main()



