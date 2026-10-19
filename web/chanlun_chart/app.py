import pathlib
import sys

# 将项目中的 src 目录，添加到 sys.path 中
src_path = pathlib.Path(__file__).parent.parent / ".." / "src"
sys.path.append(str(src_path))
web_server_path = pathlib.Path(__file__).parent
sys.path.append(str(web_server_path))


is_wpf_launcher = False
try:
    # WPF 启动，每次 print 都 flush，并且将字符编码转为 GBK（避免乱码）
    if "wpf_launcher" in sys.argv:
        is_wpf_launcher = True

        class filter:
            def __init__(self, target):
                self.target = target

            def write(self, s):
                # errors="replace"：遇到 GBK 无法编码的字符（如 emoji）时替换而不是抛异常
                self.target.buffer.write(s.encode("gbk", errors="replace"))
                self.target.flush()

            def flush(self):
                self.target.flush()

            def close(self):
                self.target.close()

        # 注意：stdin 是只读流，不能包成只写的 filter 类
        # （否则 input()/读标准输入会 AttributeError）
        sys.stdout = filter(sys.stdout)
        sys.stderr = filter(sys.stderr)

except Exception:
    pass

import traceback
import webbrowser
from concurrent.futures import ThreadPoolExecutor

from tornado.httpserver import HTTPServer
from tornado.ioloop import IOLoop
from tornado.wsgi import WSGIContainer

import chanlun.encodefix  # Fix Windows print 乱码问题  # noqa: F401
from chanlun import config

try:
    from cl_app import create_app
except Exception as e:
    print(e)
    traceback.print_exc()

    if is_wpf_launcher is False:
        input("出现异常，按回车键退出")

# ... 前面的所有导入和定义保持不变 ...

# 在文件末尾，__main__ 之前，创建全局 app 对象供 Gunicorn 使用
app = create_app()

if __name__ == "__main__":
    # 原有的 Tornado 启动代码，保留以便直接运行
    try:
        # app = create_app()   # 这行可以删掉，因为上面已经创建了
        s = HTTPServer(WSGIContainer(app, executor=ThreadPoolExecutor(10)))
        s.bind(9900, config.WEB_HOST)
        print("启动成功")
        s.start(1)
        if len(sys.argv) >= 2 and sys.argv[1] == "nobrowser":
            pass
        else:
            webbrowser.open("http://127.0.0.1:9900")
        IOLoop.instance().start()
    except Exception as e:
        print(e)
        traceback.print_exc()
        if is_wpf_launcher is False:
            input("出现异常，按回车键退出")
