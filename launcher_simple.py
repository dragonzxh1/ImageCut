"""
简化版启动器 - 直接导入streamlit运行
避免subprocess的问题
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
    
    # 设置环境变量
    os.environ['STREAMLIT_SERVER_HEADLESS'] = 'true'
    os.environ['STREAMLIT_BROWSER_GATHER_USAGE_STATS'] = 'false'
    os.environ['STREAMLIT_SERVER_ENABLE_CORS'] = 'false'
    os.environ['STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION'] = 'false'
    
    # 导入streamlit并运行
    try:
        import streamlit.web.cli as stcli
        import streamlit.runtime.scriptrunner.magic_funcs
        
        # 设置sys.argv来模拟命令行参数
        sys.argv = [
            'streamlit',
            'run',
            app_path,
            '--server.headless', 'true',
            '--server.port', '8501',
            '--browser.gatherUsageStats', 'false',
            '--server.enableCORS', 'false',
            '--server.enableXsrfProtection', 'false',
        ]
        
        # 在新线程中打开浏览器
        def open_browser():
            time.sleep(5)  # 等待服务器启动
            webbrowser.open('http://localhost:8501')
        
        browser_thread = threading.Thread(target=open_browser, daemon=True)
        browser_thread.start()
        
        # 运行streamlit
        stcli.main()
        
    except Exception as e:
        print(f"启动失败: {e}")
        import traceback
        traceback.print_exc()
        input("按回车键退出...")
        sys.exit(1)

if __name__ == "__main__":
    main()



