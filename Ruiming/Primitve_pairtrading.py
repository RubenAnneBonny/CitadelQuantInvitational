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
    #print(y_fit-porto["Sec1"]["last"])

    tot_Sec2=0
    tot_Sec1=0

    highstoploss=999999999
    lowstoploss=0
    calmtime=time.time()
    last=time.time()
    
    while(case["status"]=="ACTIVE"):
        
        if(time.time()-calmtime<1):
            continue

        if(time.time()-last>5):
            last=time.time()
            print("still working")

        portfolio = client.get_portfolio()
        porto=dict(portfolio.items())

        y_fit=porto[security1]["last"]*coef+intercept
        diff=porto[security2]["last"]-y_fit

        #print(diff,buy_in*sd,back*sd)

        #when diff high positiv
        if(diff>=buy_in*sd and not bought):
            amount_Sec1=total//2//porto[security1]["bid"]
            amount_Sec2=total//2//porto[security2]["ask"]

            tot_Sec2+=amount_Sec2
            tot_Sec1-=amount_Sec1

            lowstoploss=porto[security2]["ask"]-0.5
            highstoploss=porto[security1]["bid"]+0.5

            client.place_order(
                security1, OrderType.MARKET, amount_Sec1, OrderAction.SELL
            )

            client.place_order(
                security2, OrderType.MARKET, amount_Sec2, OrderAction.BUY
            )
            print("Bought",tot_Sec1,tot_Sec2)

            bought=1

        #when diff high negative


        #print(Sec1_MAX,Sec1_MIN,abs(diff),back*sd)
        
        #when going back
        if bought:
            if porto[security2]["ask"]<lowstoploss or porto[security1]["bid"]>highstoploss:
                tot_Sec1=porto[security1]["position"]
                tot_Sec2=porto[security2]["position"]

                if(tot_Sec1>0):
                    client.place_order(
                        security1, OrderType.MARKET, tot_Sec1, OrderAction.SELL
                    )
                else:
                    client.place_order(
                        security1, OrderType.MARKET, abs(tot_Sec1), OrderAction.BUY
                    )

                if(tot_Sec2>0):
                    client.place_order(
                        security2, OrderType.MARKET, tot_Sec2, OrderAction.SELL
                    )
                else:
                    client.place_order(
                        security2, OrderType.MARKET, abs(tot_Sec2), OrderAction.BUY
                    )

                lowstoploss=0
                highstoploss=9999999
                print("Stop loss",tot_Sec1,tot_Sec2)

                calmtime=time.time()

            if tot_Sec1>0:
                if((diff)<=back*sd):
                    if(tot_Sec1>0):
                        client.place_order(
                            security2, OrderType.MARKET, abs(tot_Sec1), OrderAction.SELL
                        )
                    else:
                        client.place_order(
                            security2, OrderType.MARKET, abs(tot_Sec1), OrderAction.BUY
                        )

                    if(tot_Sec2>0):
                        client.place_order(
                            security1, OrderType.MARKET, abs(tot_Sec2), OrderAction.SELL
                        )
                    else:
                        client.place_order(
                            security1, OrderType.MARKET, abs(tot_Sec2), OrderAction.BUY
                        )
                    tot_Sec1=0
                    tot_Sec2=0

                    bought=False

                    print("Sold out",tot_Sec1,tot_Sec2)