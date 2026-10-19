from chanlun import config
from chanlun.base import Market
from chanlun.exchange.exchange import Exchange

# 全局保存交易所对象，避免创建多个交易所对象
g_exchange_obj = {}


def get_exchange(market: Market) -> Exchange:
    """
    获取市场的交易所对象，根据config配置中设置的进行获取
    """
    global g_exchange_obj
    if market.value in g_exchange_obj.keys():
        return g_exchange_obj[market.value]

    if market == Market.A:
        # 沪深 A股 交易所
        if config.EXCHANGE_A == "tdx":
            from chanlun.exchange.exchange_tdx import ExchangeTDX

            g_exchange_obj[market.value] = ExchangeTDX()
        elif config.EXCHANGE_A == "db":
            from chanlun.exchange.exchange_db import ExchangeDB

            g_exchange_obj[market.value] = ExchangeDB(Market.A.value)
        else:
            raise Exception(f"不支持的沪深交易所 {config.EXCHANGE_A}")

    elif market == Market.FUTURES:
        # 期货 交易所
        if config.EXCHANGE_FUTURES == "tdx_futures":
            from chanlun.exchange.exchange_tdx_futures import ExchangeTDXFutures

            g_exchange_obj[market.value] = ExchangeTDXFutures()
        elif config.EXCHANGE_FUTURES == "db":
            from chanlun.exchange.exchange_db import ExchangeDB

            g_exchange_obj[market.value] = ExchangeDB(Market.FUTURES.value)
        else:
            raise Exception(f"不支持的期货交易所 {config.EXCHANGE_FUTURES}")

    elif market == Market.CURRENCY:
        # 数字货币 交易所
        if config.EXCHANGE_CURRENCY == "binance":
            from chanlun.exchange.exchange_binance import ExchangeBinance

            g_exchange_obj[market.value] = ExchangeBinance()
        elif config.EXCHANGE_CURRENCY == "okx":
            from chanlun.exchange.exchange_okx import ExchangeOKX

            g_exchange_obj[market.value] = ExchangeOKX()
        elif config.EXCHANGE_CURRENCY == "db":
            from chanlun.exchange.exchange_db import ExchangeDB

            g_exchange_obj[market.value] = ExchangeDB(Market.CURRENCY.value)
        else:
            raise Exception(f"不支持的数字货币交易所 {config.EXCHANGE_CURRENCY}")
    elif market == Market.CURRENCY_SPOT:
        # 数字货币 交易所
        if config.EXCHANGE_CURRENCY_SPOT == "binance_spot":
            from chanlun.exchange.exchange_binance_spot import ExchangeBinanceSpot

            g_exchange_obj[market.value] = ExchangeBinanceSpot()
        elif config.EXCHANGE_CURRENCY_SPOT == "okx_spot":
            from chanlun.exchange.exchange_okx_spot import ExchangeOKXSpot

            g_exchange_obj[market.value] = ExchangeOKXSpot()
        elif config.EXCHANGE_CURRENCY_SPOT == "db":
            from chanlun.exchange.exchange_db import ExchangeDB

            g_exchange_obj[market.value] = ExchangeDB(Market.CURRENCY_SPOT.value)
        else:
            raise Exception(f"不支持的数字货币交易所 {config.EXCHANGE_CURRENCY_SPOT}")
    else:
        raise Exception(f"不支持的市场 {market}")

    return g_exchange_obj[market.value]
