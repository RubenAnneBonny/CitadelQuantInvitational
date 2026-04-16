import json
import time
import logging
from RotmanInteractiveTraderApi import (
    RotmanInteractiveTraderApi,
    OrderType,
    OrderAction,
)
from settings import settings

#logging.basicConfig(level=logging.INFO)


if __name__ == "__main__":
    client = RotmanInteractiveTraderApi(
        api_key=settings["api_key"], api_host=settings["api_host"]
    )

    # verify connection
    trader = client.get_trader()
    #logging.info(f"Connected as trader {json.dumps(trader, indent=2)}")

    # get portfolio and available securities
    portfolio = client.get_portfolio()

    for ticker, security in portfolio.items():
        logging.info(f"Security {ticker}: {json.dumps(security, indent=2)}")

    # check case status
    case = client.get_case()
    logging.info(json.dumps(case, indent=2))

    # wait for case to start
    while case["status"] != "ACTIVE":
        logging.info("Case not active yet, waiting...")
        time.sleep(1)
        case = client.get_case()

    security1="ETF"
    security2="BBB"
    portfolio = client.get_portfolio()
    porto=dict(portfolio.items())
    tot_ETF=porto[security1]["position"]
    tot_IND=porto[security2]["position"]

    if(tot_ETF>0):
        client.place_order(
            security2, OrderType.MARKET, tot_ETF, OrderAction.SELL
        )
    else:
        client.place_order(
            security2, OrderType.MARKET, tot_ETF, OrderAction.BUY
        )

    if(tot_IND>0):
        client.place_order(
            security1, OrderType.MARKET, tot_IND, OrderAction.SELL
        )
    else:
        client.place_order(
            security1, OrderType.MARKET, tot_IND, OrderAction.BUY
        )