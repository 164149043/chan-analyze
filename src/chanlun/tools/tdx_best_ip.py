# coding: utf-8
# see https://github.com/rainx/pytdx/issues/38 IP寻优的简单办法
# by yutianst
# 修改：增加综合测速（ping + 下载速度），缓存30天

import datetime
import os
import pickle
import time
from pytdx.exhq import TdxExHq_API
from pytdx.hq import TdxHq_API

stock_ip = [
    {"ip": "180.153.18.170", "port": 7709},
    {"ip": "218.75.126.9", "port": 7709},
    {"ip": "60.12.136.250", "port": 7709},
    {"ip": "60.191.117.167", "port": 7709},
    {"ip": "shtdx.gtjas.com", "port": 7709},
    {"ip": "sztdx.gtjas.com", "port": 7709},
    {"ip": "110.41.147.114", "port": 7709, "name": "通达信深圳双线主站1"},
    {"ip": "110.41.2.72", "port": 7709, "name": "通达信深圳双线主站2"},
    {"ip": "110.41.4.4", "port": 7709, "name": "通达信深圳双线主站3"},
    {"ip": "175.178.112.197", "port": 7709, "name": "通达信深圳双线主站4"},
    {"ip": "175.178.128.227", "port": 7709, "name": "通达信深圳双线主站5"},
    {"ip": "110.41.154.219", "port": 7709, "name": "通达信深圳双线主站6"},
    {"ip": "124.70.176.52", "port": 7709, "name": "通达信上海双线主站1"},
    {"ip": "122.51.120.217", "port": 7709, "name": "通达信上海双线主站2"},
    {"ip": "123.60.186.45", "port": 7709, "name": "通达信上海双线主站3"},
    {"ip": "123.60.164.122", "port": 7709, "name": "通达信上海双线主站4"},
    {"ip": "111.229.247.189", "port": 7709, "name": "通达信上海双线主站5"},
    {"ip": "124.70.199.56", "port": 7709, "name": "通达信上海双线主站6"},
    {"ip": "121.36.54.217", "port": 7709, "name": "通达信北京双线主站1"},
    {"ip": "121.36.81.195", "port": 7709, "name": "通达信北京双线主站2"},
    {"ip": "123.249.15.60", "port": 7709, "name": "通达信北京双线主站3"},
    {"ip": "124.71.85.110", "port": 7709, "name": "通达信广州双线主站1"},
    {"ip": "139.9.51.18", "port": 7709, "name": "通达信广州双线主站2"},
    {"ip": "139.159.239.163", "port": 7709, "name": "通达信广州双线主站3"},
    {"ip": "122.51.232.182", "port": 7709, "name": "通达信上海双线主站7"},
    {"ip": "118.25.98.114", "port": 7709, "name": "通达信上海双线主站8"},
    {"ip": "121.36.225.169", "port": 7709, "name": "通达信上海双线主站9"},
    {"ip": "123.60.70.228", "port": 7709, "name": "通达信上海双线主站10"},
    {"ip": "123.60.73.44", "port": 7709, "name": "通达信上海双线主站11"},
    {"ip": "124.70.133.119", "port": 7709, "name": "通达信上海双线主站12"},
    {"ip": "124.71.187.72", "port": 7709, "name": "通达信上海双线主站13"},
    {"ip": "124.71.187.122", "port": 7709, "name": "通达信上海双线主站14"},
    {"ip": "129.204.230.128", "port": 7709, "name": "通达信深圳双线主站7"},
    {"ip": "124.70.75.113", "port": 7709, "name": "通达信北京双线主站4"},
    {"ip": "124.71.9.153", "port": 7709, "name": "通达信广州双线主站4"},
    {"ip": "123.60.84.66", "port": 7709, "name": "通达信上海双线主站15"},
    {"ip": "111.230.186.52", "port": 7709, "name": "通达信深圳双线主站8"},
    {"ip": "120.46.186.223", "port": 7709, "name": "通达信北京双线主站5"},
    {"ip": "124.70.22.210", "port": 7709, "name": "通达信北京双线主站6"},
    {"ip": "139.9.133.247", "port": 7709, "name": "通达信北京双线主站7"},
    {"ip": "116.205.163.254", "port": 7709, "name": "通达信广州双线主站5"},
    {"ip": "116.205.171.132", "port": 7709, "name": "通达信广州双线主站6"},
    {"ip": "116.205.183.150", "port": 7709, "name": "通达信广州双线主站7"},
]

future_ip = [
    {"ip": "112.74.214.43", "port": 7727, "name": "扩展市场深圳双线1"},
    {"ip": "120.25.218.6", "port": 7727, "name": "扩展市场深圳双线2"},
    {"ip": "43.139.173.246", "port": 7727, "name": "扩展市场深圳双线3"},
    {"ip": "159.75.90.107", "port": 7727, "name": "扩展市场深圳双线4"},
    {"ip": "106.52.170.195", "port": 7727, "name": "扩展市场深圳双线5"},
    {"ip": "139.9.191.175", "port": 7727, "name": "扩展市场广州双线3"},
    {"ip": "175.24.47.69", "port": 7727, "name": "扩展市场上海双线7"},
    {"ip": "150.158.9.199", "port": 7727, "name": "扩展市场上海双线1"},
    {"ip": "150.158.20.127", "port": 7727, "name": "扩展市场上海双线2"},
    {"ip": "49.235.119.116", "port": 7727, "name": "扩展市场上海双线3"},
    {"ip": "49.234.13.160", "port": 7727, "name": "扩展市场上海双线4"},
    {"ip": "116.205.143.214", "port": 7727, "name": "扩展市场广州双线1"},
    {"ip": "124.71.223.19", "port": 7727, "name": "扩展市场广州双线2"},
    {"ip": "113.45.175.47", "port": 7727, "name": "扩展市场广州双线4"},
    {"ip": "123.60.173.210", "port": 7727, "name": "扩展市场上海双线5"},
    {"ip": "118.89.69.202", "port": 7727, "name": "扩展市场上海双线6"},
]

CACHE_FILE = "best_stock_ip_cache.pkl"   # 独立缓存文件，不与 tdx_kline_fetcher 冲突
CACHE_EXPIRE_DAYS = 30


def ping(ip, port=7709, type_="stock"):
    """保持原有函数不变，仅测试连接（ping）"""
    api = TdxHq_API()
    apix = TdxExHq_API()
    __time1 = datetime.datetime.now()
    try:
        if type_ in ["stock"]:
            with api.connect(ip, port, time_out=0.7):
                res = api.get_security_list(0, 1)
                if res is not None:
                    if len(res) > 800:
                        return datetime.datetime.now() - __time1
                    else:
                        return datetime.timedelta(9, 9, 0)
                else:
                    return datetime.timedelta(9, 9, 0)
        elif type_ in ["future"]:
            with apix.connect(ip, port, time_out=0.7):
                res = apix.get_instrument_count()
                if res is not None:
                    if res > 20000:
                        return datetime.datetime.now() - __time1
                    else:
                        return datetime.timedelta(9, 9, 0)
                else:
                    return datetime.timedelta(9, 9, 0)
    except Exception:
        return datetime.timedelta(9, 9, 0)


def _test_download_speed(ip, port, timeout=3, type_="stock"):
    """测试实际下载速度（获取100条日K线）"""
    api = None
    try:
        if type_ == "future":
            # 扩展行情（期货）协议，用上期所螺纹连续日K测速
            api = TdxExHq_API()
            if not api.connect(ip, port, time_out=timeout):
                return None
            start = time.perf_counter()
            data = api.get_instrument_bars(4, 30, "RBL8", 0, 100)
        else:
            api = TdxHq_API()
            if not api.connect(ip, port, time_out=timeout):
                return None
            start = time.perf_counter()
            data = api.get_security_bars(4, 1, '000001', 0, 100)
        if data is None or len(data) == 0:
            return None
        elapsed = time.perf_counter() - start
        return elapsed
    except Exception:
        return None
    finally:
        try:
            api.close()
        except:
            pass


def _get_cached_best_ip(_type):
    cache_file = f"{CACHE_FILE}.{_type}"
    if not os.path.exists(cache_file):
        return None
    try:
        with open(cache_file, 'rb') as f:
            data = pickle.load(f)
        ip_info = data.get('ip_info')
        timestamp = data.get('timestamp')
        if ip_info and timestamp:
            age = (datetime.datetime.now() - timestamp).total_seconds() / (3600 * 24)
            if age < CACHE_EXPIRE_DAYS:
                print(f"使用缓存的 {_type} 服务器: {ip_info['ip']}:{ip_info['port']} (缓存时间 {age:.1f} 天前)")
                return ip_info
            else:
                print(f"缓存IP已过期（{age:.1f}天前），重新探测...")
                return None
    except Exception as e:
        print(f"读取缓存失败: {e}")
        return None


def _save_best_ip(ip_info, _type):
    cache_file = f"{CACHE_FILE}.{_type}"
    try:
        data = {'ip_info': ip_info, 'timestamp': datetime.datetime.now()}
        with open(cache_file, 'wb') as f:
            pickle.dump(data, f)
        print(f"已缓存 {_type} 最优服务器至 {cache_file}")
    except Exception as e:
        print(f"保存缓存失败: {e}")


def select_best_ip(_type="stock"):
    """
    选择最优服务器（综合 ping + 下载速度）
    :param _type: "stock" 或 "future"
    :return: 字典，包含 ip, port, name（如果有）
    """
    # 1. 尝试从缓存读取
    cached = _get_cached_best_ip(_type)
    if cached is not None:
        return cached

    ip_list = stock_ip if _type == "stock" else future_ip
    results = []

    print(f"正在探测 {_type} 服务器（ping + 下载速度综合测试）...")
    for item in ip_list:
        ip = item['ip']
        port = item['port']
        # 先测 ping（使用原有 ping 函数，但返回 timedelta，我们转换为毫秒）
        ping_delta = ping(ip, port, _type)
        if ping_delta >= datetime.timedelta(seconds=9):
            # ping 不通
            print(f"  {ip}:{port} ping 不通")
            continue
        ping_ms = ping_delta.total_seconds() * 1000  # 转毫秒

        # 测下载速度
        download_sec = _test_download_speed(ip, port, type_=_type)
        if download_sec is None:
            print(f"  {ip}:{port} 下载失败")
            continue

        results.append({
            'ip': ip,
            'port': port,
            'ping_ms': ping_ms,
            'download_sec': download_sec,
            'name': item.get('name', '')
        })
        print(f"  {ip}:{port} ping {ping_ms:.2f}ms, 下载 {download_sec:.3f}s")

    if not results:
        raise RuntimeError("所有IP均无法连接或下载数据，请检查网络。")

    # 排序：优先下载速度，其次 ping
    results.sort(key=lambda x: (x['download_sec'], x['ping_ms']))
    best = results[0]
    print(f"最优服务器: {best['ip']}:{best['port']}, "
          f"ping {best['ping_ms']:.2f}ms, 下载 {best['download_sec']:.3f}s")

    # 构造返回字典（兼容原格式）
    result_dict = {'ip': best['ip'], 'port': best['port']}
    if best.get('name'):
        result_dict['name'] = best['name']

    # 缓存
    _save_best_ip(result_dict, _type)

    return result_dict


if __name__ == "__main__":
    print("股票服务器测试")
    ip = select_best_ip("stock")
    print(ip)

    print("\n期货服务器测试")
    ip = select_best_ip("future")
    print(ip)