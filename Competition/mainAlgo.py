from rit_client import RITClient

## Add functions here
def spread(securities, ritClient: RITClient) -> bool:
    security = securities["CRZY"]

    diff = (security["Ask"] - security["Bid"]) * 100
    
    ritClient.buy_market("CRZY", diff)

    return False

class function:
    def __init__(self, func):
        self.func = func
        self.on = True
        self.off_ticks = 0

functions = []

## Appen all functions here
functions.append(function(spread))

client = RITClient()

TICKS_OFF = 1

while True:
    securities = client.get_security()

    ## Loop throug all functions
    for func in functions:
        if not func.on:
            func.off_ticks += 1

            if func.off_ticks >= TICKS_OFF:
                func.on = True
            else:
                continue            

        func.on = func.func(securities, client)