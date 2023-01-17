import sys
sys.path.append('../input')
from flight_revenue_simulator import simulate_revenue, score_me
print("Import Done!")
demand_list = []
avrg_demand = demand_list

def avrg_calc(demand_level):
    avrg_demand.append(demand_level)
    return np.mean(avrg_demand)

def std_demand(demand_level):
    return np.std(avrg_demand)
def pricing_function(days_left, tickets_left, demand_level):
    """Sample pricing function"""
    average_demand = avrg_calc(demand_level)
    STD_demand = std_demand(demand_level)
    price = demand_level - round((tickets_left / days_left)) # makes sure every seat is sold at the highest (currently possible) price . Basically it's a linear d/dx
    if(average_demand > demand_level and (len(avrg_demand) != 0 ) and days_left > 1 ): # if current demands are less than the average demand, and it's not the last day
        price = demand_level - (tickets_left / days_left) + 3        # sell 3 less seats
    if( (average_demand - STD_demand) > demand_level and (len(avrg_demand) != 0 ) and days_left > 1 ):              # if current demands are below the STD for demands
        price = demand_level - (tickets_left / days_left) + 6
    if( (demand_level  >= average_demand + (1.35* STD_demand)) and (len(avrg_demand) >= 5 )):  # if current demands are above 1.5 times the STD, then you better sell more seats
        price = demand_level - (tickets_left / days_left) - 11
    if( (demand_level  <= average_demand - (2* STD_demand)) and (len(avrg_demand) >= 5 )):
        price = demand_level - (tickets_left / days_left) + 9
    return price
simulate_revenue(days_left=7, tickets_left=50, pricing_function=pricing_function, verbose=True)
score_me(pricing_function)