import json
import time
import logging
from RotmanInteractiveTraderApi import (
    RotmanInteractiveTraderApi,
    OrderType,
    OrderAction,
)
from settings import settings

logging.basicConfig(level=logging.INFO)


if __name__ == "__main__":
    client = RotmanInteractiveTraderApi(
        api_key=settings["api_key"], api_host=settings["api_host"]
    )

    # verify connection
    trader = client.get_trader()
    logging.info(f"Connected as trader {json.dumps(trader, indent=2)}")

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

    # place a test order for the first tradable security
    first_tradeable_ticker = next(
        (k for k, v in portfolio.items() if v["is_tradeable"])
    )
    order = client.place_order(
        first_tradeable_ticker, OrderType.MARKET, 100, OrderAction.BUY
    )

    
    logging.info(f"Placed order: {json.dumps(order, indent=2)}")
