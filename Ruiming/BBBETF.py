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
    mean=-2.0595512300584884
    total=start_capital
    intercept,coef=95.46931822547003,2.05311287
    sd=4.31792969543484

    buy_in=2.075
    back=0.825

    security1="ETF"
    security2="BBB"
    
    porto=dict(portfolio.items())
    y_fit=porto[security1]["last"]*coef+intercept

    tot_Sec2=0
    tot_Sec1=0

    stoploss=0
    stoploss_ratio=3.5
    calmtime=time.time()
    last=time.time()
    
    while(case["status"]=="ACTIVE"):
        if(time.time()-calmtime<3):
            continue

        portfolio = client.get_portfolio()
        porto=dict(portfolio.items())

        y_fit=porto[security1]["last"]*coef+intercept
        diff=porto[security2]["last"]-y_fit

        if(time.time()-last>5):
            last=time.time()
            print(total,tot_Sec1,tot_Sec2)
            print(f"diff{diff}")
            print("still working")

        #when diff high positiv
        if(diff>=mean+buy_in*sd and not bought):
            amount_Sec1=total//2//porto[security1]["bid"]
            amount_Sec2=total//2//porto[security2]["ask"]

            tot_Sec2+=amount_Sec2
            tot_Sec1-=amount_Sec1

            total-=amount_Sec2*porto[security2]["ask"]
            total+=amount_Sec1*porto[security1]["bid"]

            stoploss=mean+stoploss_ratio*sd

            client.place_order(
                security1, OrderType.MARKET, amount_Sec1, OrderAction.SELL
            )

            client.place_order(
                security2, OrderType.MARKET, amount_Sec2, OrderAction.BUY
            )
            print("Bought",tot_Sec1,tot_Sec2)
            bought=1

        #when diff high negative
        elif(diff<=mean-buy_in*sd and not bought):
            amount_Sec2=total//2//porto[security2]["bid"]
            amount_Sec1=total//2//porto[security1]["ask"]

            tot_Sec2-=amount_Sec2
            tot_Sec1+=amount_Sec1

            total+=amount_Sec2*porto[security2]["bid"]
            total-=amount_Sec1*porto[security1]["ask"]

            stoploss=mean-stoploss_ratio*sd

            client.place_order(
                security2, OrderType.MARKET, amount_Sec2, OrderAction.SELL
            )

            client.place_order(
                security1, OrderType.MARKET, amount_Sec1, OrderAction.BUY
            )
            print("Bought",tot_Sec1,tot_Sec2)

            bought=1


        #print(Sec1_MAX,Sec1_MIN,abs(diff),back*sd)
        
        #when going back

        elif bought and ((diff<stoploss and porto[security1]["position"]>0) or (diff>stoploss and porto[security2]["position"]>0)):
            
            
            total+=porto[security2]["position"]*porto[security2]["bid"]
            total+=porto[security1]["position"]*porto[security1]["ask"]
            
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

            tot_Sec1=0
            tot_Sec2=0
            print("Stop loss",tot_Sec1,tot_Sec2)
            bought=0

            calmtime=time.time()

        elif(diff<=mean+back*sd and tot_Sec1<0 and bought==1):
            total+=porto[security2]["position"]*porto[security2]["bid"]
            total+=porto[security1]["position"]*porto[security1]["ask"]
            client.place_order(
                security2, OrderType.MARKET, porto[security2]["position"], OrderAction.SELL
            )

            client.place_order(
                security1, OrderType.MARKET, porto[security1]["position"], OrderAction.BUY
            )

            tot_Sec2=0
            tot_Sec1=0
            print("Pulled out",tot_Sec1,tot_Sec2)
            bought=0
            #print(tot_ETF,i,total)

        elif(diff>=mean-back*sd and tot_Sec1>0 and bought==1):
            total+=porto[security1]["position"]*porto[security1]["bid"]
            total-=porto[security2]["position"]*porto[security2]["ask"]

            client.place_order(
                security1, OrderType.MARKET, porto[security1]["position"], OrderAction.SELL
            )

            client.place_order(
                security2, OrderType.MARKET, porto[security2]["position"], OrderAction.BUY
            )

            tot_Sec2=0
            tot_Sec1=0
            lowstoploss=0
            highstoploss=9999999
            print("Pulled out",tot_Sec1,tot_Sec2)
            bought=0
            #print(tot_ETF,i,total)