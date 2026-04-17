import json
import time
import logging
import pandas as pd
from RotmanInteractiveTraderApi import (
    RotmanInteractiveTraderApi,
    OrderType,
    OrderAction,
)
from settings import settings
from sklearn.linear_model import LinearRegression

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

    data=[]
    securities=["AAA","BBB","CCC","DDD","ETF","IND"]
    print("Started")
    bought=False
    start_capital=200000
    mean=0
    total=start_capital
    intercept,coef=95.46931822547003,2.05311287
    sd=1.5

    buy_in=1
    back=0.5

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
        time.sleep(0.99)
        portfolio = client.get_portfolio()
        porto=dict(portfolio.items())

        dic={}
        for i in securities:
            dic[i]=porto[i]["last"]
        data.append(dic)

        if(len(data)<20):
            continue
        
        if bought==0:
            data_df=pd.DataFrame(data)
            limdf1=data_df[security1]
            limdf2=data_df[security2]
            linear_model = LinearRegression(fit_intercept=True)
            linear_model.fit(limdf1.values.reshape(-1,1), limdf2.values)
            intercept,coef=linear_model.intercept_,linear_model.coef_

        if(time.time()-calmtime<3):
            continue

        y_fit=porto[security1]["last"]*coef+intercept
        diff=porto[security2]["last"]-y_fit

        if(time.time()-last>5):
            last=time.time()
            print(total,tot_Sec1,tot_Sec2)
            print(f"diff{diff}, intercept {intercept}")
            print(f"coef {coef},sd {sd}")
            print("still working")

        #when diff high positiv
        if(diff>=mean+buy_in*sd and bought==0):
            amount_Sec1=total//2//porto[security1]["bid"]
            amount_Sec2=total//2//porto[security2]["ask"]

            tot_Sec2+=amount_Sec2
            tot_Sec1-=amount_Sec1

            total-=amount_Sec2*porto[security2]["ask"]
            total+=amount_Sec1*porto[security1]["bid"]

            stoploss=mean+stoploss_ratio*sd

            client.place_order(
                security1, OrderType.MARKET, amount_Sec1, OrderAction.BUY
            )

            client.place_order(
                security2, OrderType.MARKET, amount_Sec2, OrderAction.SELL
            )
            print("Bought {security1} short",porto[security1]["position"],porto[security2]["position"])
            print(f"diff {diff}, condition {mean+buy_in*sd}")
            print(f"price {porto[security2]['ask']} and {porto[security1]['bid']}")
            bought=1
            print(f"boought {bought}")

        #when diff high negative
        elif(diff<=mean-buy_in*sd and bought==0):
            amount_Sec2=total//2//porto[security2]["bid"]
            amount_Sec1=total//2//porto[security1]["ask"]

            tot_Sec2-=amount_Sec2
            tot_Sec1+=amount_Sec1

            total+=amount_Sec2*porto[security2]["bid"]
            total-=amount_Sec1*porto[security1]["ask"]

            stoploss=mean-stoploss_ratio*sd

            client.place_order(
                security2, OrderType.MARKET, abs(amount_Sec2), OrderAction.BUY
            )

            client.place_order(
                security1, OrderType.MARKET, abs(amount_Sec1), OrderAction.SELL
            )
            print("Bought {security1} long",porto[security1]["position"],porto[security2]["position"])
            print(f"diff {diff}, condition {mean-buy_in*sd}")
            print(f"price {porto[security2]['bid'],porto[security1]['ask']}")
            bought=1
            print(f"boought {bought}")

        #print(Sec1_MAX,Sec1_MIN,abs(diff),back*sd)
        
        #when going back

        elif bought==1 and ((diff<stoploss and porto[security2]["position"]>0) or (diff>stoploss and porto[security1]["position"]>0)):
            
            
            total+=porto[security2]["position"]*porto[security2]["bid"]
            total+=porto[security1]["position"]*porto[security1]["ask"]
            
            tot_Sec1=porto[security1]["position"]
            tot_Sec2=porto[security2]["position"]

            if(tot_Sec1>0):
                client.place_order(
                    security1, OrderType.MARKET, tot_Sec1, OrderAction.BUY
                )
            else:
                client.place_order(
                    security1, OrderType.MARKET, abs(tot_Sec1), OrderAction.SELL
                )

            if(tot_Sec2>0):
                client.place_order(
                    security2, OrderType.MARKET, tot_Sec2, OrderAction.BUY
                )
            else:
                client.place_order(
                    security2, OrderType.MARKET, abs(tot_Sec2), OrderAction.SELL
                )

            tot_Sec1=0
            tot_Sec2=0

            print(f"stoploss {stoploss}")
            print("Stoploss",porto[security1]["position"],porto[security2]["position"])
            print(f"diff {diff}, condition {stoploss}")
            print(f"price {porto[security2]['bid'],porto[security1]['ask']}")
            bought=0
            print(f"boought {bought}")
            calmtime=time.time()

        elif(diff<=mean+back*sd and tot_Sec1<0 and bought==1):
            total+=porto[security2]["position"]*porto[security2]["bid"]
            total+=porto[security1]["position"]*porto[security1]["ask"]
            client.place_order(
                security2, OrderType.MARKET, abs(porto[security2]["position"]), OrderAction.BUY
            )

            client.place_order(
                security1, OrderType.MARKET, abs(porto[security1]["position"]), OrderAction.SELL
            )

            tot_Sec2=0
            tot_Sec1=0
            print("Bought back",porto[security1]["position"],porto[security2]["position"])
            print(f"diff {diff}, condition {mean+back*sd}")
            print(f"price {porto[security2]['bid'],porto[security1]['ask']}")
            bought=0
            print(f"boought {bought}")
            #print(tot_ETF,i,total)

        elif(diff>=mean-back*sd and tot_Sec1>0 and bought==1):
            total+=porto[security1]["position"]*porto[security1]["bid"]
            total-=porto[security2]["position"]*porto[security2]["ask"]

            client.place_order(
                security1, OrderType.MARKET, abs(porto[security1]["position"]), OrderAction.BUY
            )

            client.place_order(
                security2, OrderType.MARKET, abs(porto[security2]["position"]), OrderAction.SELL
            )

            tot_Sec2=0
            tot_Sec1=0
            print("Bought back",porto[security1]["position"],porto[security2]["position"])
            print(f"diff {diff}, condition {mean-back*sd}")
            print(f"price {porto[security2]['ask'],porto[security1]['bid']}")
            bought=0
            print(f"boought {bought}")
            #print(tot_ETF,i,total)