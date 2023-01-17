import sys

sys.path.append('../input')

from flight_revenue_simulator import simulate_revenue, score_me

def pricing_function(days_left, tickets_left, demand_level):

    """Sample pricing function"""

    

    # Manual solution. There's likely an optimized / automated solution that optimizes over the manual demand_level tiers created here

    

    # Base case: if there's no more days left, we need to sell all the remaining tickets we have left, at whatever price the last day's

        # demand_level price is

    if days_left == 1:

        return demand_level - tickets_left

    

    # General case: based on the level of demand, we sell a proportion of our unsold tickets accordingly. The higher the demand, the

        # more tickets we sell. We still choose to sell some tickets during moderate levels of demand, primarily so that we don't take

        # a big volume discount when trying to sell at a high-level demand day, and also to lessen the risk of hoarding all our tickets

        # until the last day if a high-level demand day doesn't get generated

    sell_proportion = 0.0

    

    if demand_level > 100 and demand_level < 125:

        sell_proportion = 0.0

    elif demand_level >= 125 and demand_level < 150:

        sell_proportion = 0.1

    elif demand_level >=150 and demand_level < 175:

        sell_proportion = 0.3

    elif demand_level >=175:

        sell_proportion = 0.5

        

    # Few days left momentum boost:

    if days_left < 6:

        sell_proportion = sell_proportion * 2

    

    # Output

    price = demand_level - int(tickets_left * sell_proportion)

    

    return price
simulate_revenue(days_left=7, tickets_left=50, pricing_function=pricing_function, verbose=True)
score_me(pricing_function)