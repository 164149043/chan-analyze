# -*- coding: utf-8 -*-
"""临时测试：欧意 OKX K线获取（直连 vs 走本机 Clash 代理）"""
import datetime
import sys

import ccxt

PROXY = "http://127.0.0.1:7897"


def fetch_once(label, params):
    print(f"\n===== {label} =====")
    ex = ccxt.okx(params)
    try:
        ohlcv = ex.fetch_ohlcv("BTC/USDT", "1h", limit=10)
        print(f"成功，共 {len(ohlcv)} 根 1h K线（BTC/USDT 现货）:")
        for ts, o, h, l, c, v in ohlcv[-5:]:
            dt = datetime.datetime.fromtimestamp(ts / 1000)
            print(f"  {dt:%Y-%m-%d %H:%M}  O:{o:<10} H:{h:<10} L:{l:<10} C:{c:<10} V:{v:.4f}")
        return True
    except Exception as e:
        print(f"失败: {type(e).__name__}: {e}")
        return False


def main():
    # 1) 直连（不带代理）
    direct_ok = fetch_once("直连（无代理）", {"timeout": 10000})

    # 2) 走本机 Clash 代理
    proxy_ok = fetch_once(
        "走代理 127.0.0.1:7897",
        {
            "timeout": 15000,
            "proxies": {"http": PROXY, "https": PROXY},
        },
    )

    print("\n===== 结论 =====")
    print(f"直连: {'OK' if direct_ok else '不可用'}")
    print(f"代理: {'OK' if proxy_ok else '不可用'}")
    if not direct_ok and not proxy_ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
