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
    start_capital=1000000
    mean=-1.0595512300584884
    total=start_capital
    intercept,coef=95.46931822547003,1.05311287
    sd=4.31792969543484

    buy_in=1.2222222222222223
    back=0.9444444444444444

    security1="ETF"
    security2="BBB"
    
    porto=dict(portfolio.items())
    y_fit=porto[security1]["last"]*coef+intercept
    #print(y_fit-porto["ETF"]["last"])

    ETF_MAX=256.70
    ETF_MIN=256.70
    
    while(case["status"]=="ACTIVE"):
        portfolio = client.get_portfolio()
        porto=dict(portfolio.items())

        y_fit=porto[security1]["last"]*coef+intercept
        diff=y_fit-porto[security2]["last"]

        ETF_MAX=max(ETF_MAX,porto[security2]["last"])
        ETF_MIN=min(ETF_MIN,porto[security2]["last"])

        #print(diff,buy_in*sd,back*sd)

        #when diff high positiv
        if(diff>=buy_in*sd and not bought):
            ETF_MAX=porto[security1]["last"]
            ETF_MIN=porto[security1]["last"]
            amount_etf=total//2//porto[security2]["bid"]
            amount_ind=total//2//porto[security1]["ask"]

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
        """
        if(-diff<=buy_in*sd and not bought):
            ETF_MAX=porto[security2]["last"]
            ETF_MIN=porto[security2]["last"]
            amount_etf=total//2//porto[security2]["ask"]
            amount_ind=total//2//porto[security1]["bid"]

            tot_IND-=amount_ind
            tot_ETF+=amount_etf

            client.place_order(
                "ETF", OrderType.MARKET, amount_etf, OrderAction.BUY
            )

            client.place_order(
                "IND", OrderType.MARKET, amount_ind, OrderAction.SELL
            )

            bought=1"""

        #print(ETF_MAX,ETF_MIN,abs(diff),back*sd)
        
        #when going back
        if bought:
            if tot_ETF>0:
                if((abs(diff)<=back*sd or abs(1-porto[security2]["last"]/ETF_MAX)>0.05)):
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
                    tot_ETF=0
                    tot_IND=0

                    bought=False
                    """
            else:
                if((abs(diff)<=back*sd or abs(1-porto["ETF"]["last"]/ETF_MIN)>0.05)):
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
                    tot_ETF=0
                    tot_IND=0

                    bought=False
    """
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