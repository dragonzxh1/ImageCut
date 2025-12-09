"""
启动器脚本 - 用于打包后的exe文件
直接导入streamlit运行，避免subprocess的问题
"""
import sys
import os
import webbrowser
import time
import threading

def main():
    # 修复PyInstaller打包后的元数据问题
    if getattr(sys, 'frozen', False):
        try:
            import importlib.metadata
            if hasattr(importlib.metadata, '_cache'):
                importlib.metadata._cache.clear()
        except:
            pass
    
    # 获取app.py路径
    if getattr(sys, 'frozen', False):
        base_path = sys._MEIPASS
        app_path = os.path.join(base_path, 'app.py')
    else:
        base_path = os.path.dirname(os.path.abspath(__file__))
        app_path = os.path.join(base_path, 'app.py')
    
    print("正在启动图片切割工具...")
    print(f"应用路径: {app_path}")
    
    # 检查app.py是否存在
    if not os.path.exists(app_path):
        print(f"错误：找不到app.py文件: {app_path}")
        input("按回车键退出...")
        sys.exit(1)
    
    # 设置工作目录
    os.chdir(base_path)
    
    # 设置环境变量
    os.environ['STREAMLIT_SERVER_HEADLESS'] = 'true'
    os.environ['STREAMLIT_BROWSER_GATHER_USAGE_STATS'] = 'false'
    os.environ['STREAMLIT_SERVER_ENABLE_CORS'] = 'false'
    os.environ['STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION'] = 'false'
    # 关键修复：禁用开发模式，这样才能设置server.port
    os.environ['STREAMLIT_GLOBAL_DEVELOPMENT_MODE'] = 'false'
    
    # 设置sys.argv来模拟命令行参数
    # 注意：在非开发模式下才能设置server.port
    sys.argv = [
        'streamlit',
        'run',
        app_path,
        '--server.headless', 'true',
        '--server.port', '8501',
        '--browser.gatherUsageStats', 'false',
        '--server.enableCORS', 'false',
        '--server.enableXsrfProtection', 'false',
        '--server.runOnSave', 'false',
        '--server.address', 'localhost',
        '--global.developmentMode', 'false',  # 禁用开发模式
    ]
    
    # 在新线程中打开浏览器（只打开一次）
    opened = {'value': False}  # 使用字典以便在函数中修改
    
    def open_browser():
        # 等待服务器启动
        max_wait = 15
        waited = 0
        
        while waited < max_wait:
            time.sleep(0.5)
            waited += 0.5
            
            # 检查端口是否可用
            try:
                import socket
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(0.5)
                result = sock.connect_ex(('localhost', 8501))
                sock.close()
                if result == 0:
                    # 服务器已启动，只打开一次
                    if not opened['value']:
                        url = 'http://localhost:8501'
                        print(f"\n✓ 服务器已启动！正在打开浏览器: {url}")
                        webbrowser.open(url)
                        opened['value'] = True
                    return
            except:
                pass
        
        # 如果15秒后还没检测到，也尝试打开（可能检测有延迟）
        if not opened['value']:
            url = 'http://localhost:8501'
            print(f"\n尝试打开浏览器: {url}")
            webbrowser.open(url)
            opened['value'] = True
    
    browser_thread = threading.Thread(target=open_browser, daemon=True)
    browser_thread.start()
    
    # 直接导入并运行streamlit
    try:
        print("正在启动Streamlit服务器...")
        import streamlit.web.cli as stcli
        stcli.main()
    except KeyboardInterrupt:
        print("\n正在关闭服务器...")
    except Exception as e:
        print(f"\n启动失败: {e}")
        import traceback
        traceback.print_exc()
        input("按回车键退出...")
        sys.exit(1)

if __name__ == "__main__":
    main()

