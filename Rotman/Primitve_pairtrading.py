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

start_capital=1000000
total=start_capital
per=total/2
tot_ETF=0
tot_IND=0
sd=8.615719760068005

buy_in=2.5555555555555554
back=0.3111111111111111



if(diff[i]>=buy_in*sd and diff[i-1]<buy_in*sd and start_capital>0):
    amount_etf=per//test_data[security2][i]
    amount_ind=per//test_data[security1][i]

    tot_IND+=amount_ind
    tot_ETF-=amount_etf

    total+=amount_etf*test_data[security2][i]
    total-=amount_ind*test_data[security1][i]

    #print(tot_ETF,i,total)

if(diff[i]<=back*sd and diff[i-1]>back*sd and tot_ETF<0):
    total-=tot_ETF*test_data[security2][i]
    total+=tot_IND*test_data[security1][i]
    tot_ETF=0
    tot_IND=0

    #print(tot_ETF,i,total)


profit.append(total/start_capital)