from rit_client import RITClient

## Add functions here
def spread(security, ritClient: RITClient) -> bool:
    diff = (security["Ask"] - security["Bid"]) * 100
    
    ritClient.buy_market("CRZY", diff)

    return False

class function:
    def __init__(self, func):
        self.func = func
        self.on = True

functions = []

functions.append(function(spread))

client = RITClient()

while True:
    security = client.get_security()

    for func in functions:
        if not func.on:
            continue

        func.on = func.func(security, client)