from rit_client import RITClient

## Add functions here
def spread(securities, ritClient: RITClient) -> bool:
    security = securities["CRZY"]

    diff = (security["ask"] - security["bid"]) * 100
    
    ritClient.buy_market("CRZY", diff)

    return False

def tame_spread(securities, ritClient: RITClient) -> bool:
    security = securities["TAME"]

    diff = (security["ask"] - security["bid"]) * 100
    
    ritClient.sell_market("TAME", diff)

    return False

class function:
    def __init__(self, func):
        self.func = func
        self.on = True
        self.off_ticks = 0
        self.no_ticks = False

functions = []

## Appen all functions here
functions.append(function(spread))
functions.append(function(tame_spread))

client = RITClient()

TICKS_OFF = 5

pre_tick = -1

while True:
    tick = client.get_case()["tick"]
    if pre_tick == tick:
        continue

    pre_tick = tick

    securities = client.get_securities()
    securities = {s["ticker"]: s for s in client.get_securities()}

    ## Loop throug all functions
    for func in functions:
        if not func.on:
            if not func.no_ticks:
                func.off_ticks += 1

            if func.off_ticks >= TICKS_OFF:
                func.on = True
                func.off_ticks = 0
            else:
                continue            

        func.on = func.func(securities, client)