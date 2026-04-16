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


    bought=False
    security_prev=0
    start_capital=1000000
    total=start_capital
    per=total/2
    tot_ETF=0
    tot_IND=0
    intercept,coef=-15.801328357255073,2.37548408
    sd=3.6381906636177406

    buy_in=1.2222222222222223
    back=0.5

    security1="IND"
    security2="ETF"
    
    porto=dict(portfolio.items())
    y_fit=porto["IND"]["last"]*coef+intercept
    #print(y_fit-porto["ETF"]["last"])

    ETF_MAX=256.70
    ETF_MIN=256.70
    
    while(case["status"]=="ACTIVE"):
        porto=dict(portfolio.items())
        y_fit=porto["IND"]["last"]*coef+intercept
        diff=y_fit-porto["ETF"]["last"]

        ETF_MAX=max(ETF_MAX,porto["ETF"]["last"])
        ETF_MIN=min(ETF_MIN)

        #when diff high positiv
        if(diff>=buy_in*sd and not bought):
            amount_etf=per//porto[security2]["bid"]
            amount_ind=per//porto[security1]["ask"]

            tot_IND+=amount_ind
            tot_ETF-=amount_etf

            client.place_order(
                "ETF", OrderType.MARKET, amount_etf, OrderAction.SELL
            )

            client.place_order(
                "IND", OrderType.MARKET, amount_ind, OrderAction.BUY
            )

            bought=1
        #when diff high negative
        if(diff<=buy_in*sd and not bought):
            amount_etf=per//porto[security2]["ask"]
            amount_ind=per//porto[security1]["bid"]

            tot_IND-=amount_ind
            tot_ETF+=amount_etf

            client.place_order(
                "ETF", OrderType.MARKET, amount_etf, OrderAction.BUY
            )

            client.place_order(
                "IND", OrderType.MARKET, amount_ind, OrderAction.SELL
            )

            bought=1

        #when going back
        if bought:
            if tot_ETF>0:
                if((abs(diff)<=back*sd or abs(1-porto["ETF"]["last"]/ETF_MAX)>0.05)):
                    if(tot_ETF>0):
                        client.place_order(
                            "ETF", OrderType.MARKET, tot_ETF, OrderAction.SELL
                        )
                    else:
                        client.place_order(
                            "ETF", OrderType.MARKET, tot_ETF, OrderAction.BUY
                        )

                    if(tot_IND>0):
                        client.place_order(
                            "IND", OrderType.MARKET, tot_IND, OrderAction.SELL
                        )
                    else:
                        client.place_order(
                            "IND", OrderType.MARKET, tot_IND, OrderAction.BUY
                        )
                    tot_ETF=0
                    tot_IND=0

                    bought=False
            else:
                if((abs(diff)<=back*sd or abs(1-porto["ETF"]["last"]/ETF_MIN)>0.05)):
                    if(tot_ETF>0):
                        client.place_order(
                            "ETF", OrderType.MARKET, tot_ETF, OrderAction.SELL
                        )
                    else:
                        client.place_order(
                            "ETF", OrderType.MARKET, tot_ETF, OrderAction.BUY
                        )

                    if(tot_IND>0):
                        client.place_order(
                            "IND", OrderType.MARKET, tot_IND, OrderAction.SELL
                        )
                    else:
                        client.place_order(
                            "IND", OrderType.MARKET, tot_IND, OrderAction.BUY
                        )
                    tot_ETF=0
                    tot_IND=0

                    bought=False

    #Risk management as well
    """
    # place a test order for the first tradable security
    first_tradeable_ticker = next(
        (k for k, v in portfolio.items() if v["is_tradeable"])
    )
    order = client.place_order(
        first_tradeable_ticker, OrderType.MARKET, 100, OrderAction.BUY
    )
    
    logging.info(f"Placed order: {json.dumps(order, indent=2)}")
"""