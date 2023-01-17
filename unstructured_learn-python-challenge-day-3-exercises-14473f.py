# SETUP. You don't need to worry for now about what this code does or how it works. If you're ever curious about the 
# code behind these exercises, it's available under an open source license here: https://github.com/Kaggle/learntools/
import sys; sys.path.insert(0, '../input/learntools/pseudo_learntools')
from learntools.python import binder; binder.bind(globals())
from learntools.python.ex3 import *
print('Setup complete.')
# Your code goes here. Define a function called 'sign'
def sign(num):
    if not (isinstance(num, int) or isinstance(num, float)):
        raise ValueError('num must be of int or float type')
        
    if num < 0:
        return -1
    elif num > 0:
        return 1
    else:
        return 0


q1.check()
#q1.solution()
def to_smash(total_candies):
    """Return the number of leftover candies that must be smashed after distributing
    the given number of candies evenly between 3 friends.
    
    >>> to_smash(91)
    1
    """
    print("Splitting", total_candies, "candies")
    return total_candies % 3

to_smash(91)
to_smash(1)
def to_smash(total_candies):
    """Return the number of leftover candies that must be smashed after distributing
    the given number of candies evenly between 3 friends.
    
    >>> to_smash(91)
    1
    """
    print("Splitting", total_candies, ("candies" if total_candies > 1 else "candy"))
    return total_candies % 3

to_smash(91)
to_smash(1)
#q2.hint()
#q2.solution()
def prepared_for_weather(have_umbrella, rain_level, have_hood, is_workday):
    # Don't change this code. Our goal is just to find the bug, not fix it!
    return have_umbrella or rain_level < 5 and have_hood or not rain_level > 0 and is_workday

# Change the values of these inputs so they represent a case where prepared_for_weather
# returns the wrong answer.
have_umbrella = False
rain_level = 0.0
have_hood = False
is_workday = False

# Check what the function returns given the current values of the variables above
actual = prepared_for_weather(have_umbrella, rain_level, have_hood, is_workday)
print(actual)

q3.check()
#q3.hint()
#q3.solution()
def is_negative(number):
    if number < 0:
        return True
    else:
        return False

def concise_is_negative(number):
    # pass # Your code goes here (try to keep it to one line!)
    return True if number < 0 else False

q4.check()
#q4.hint()
#q4.solution()
def onionless(ketchup, mustard, onion):
    """Return whether the customer doesn't want onions.
    """
    return not onion
def wants_all_toppings(ketchup, mustard, onion):
    """Return whether the customer wants "the works" (all 3 toppings)
    """
    return ketchup and mustard and onion

q5.a.check()
#q5.a.hint()
#q5.a.solution()
def wants_plain_hotdog(ketchup, mustard, onion):
    """Return whether the customer wants a plain hot dog with no toppings.
    """
    return not (ketchup or mustard or onion)

q5.b.check()
#q5.b.hint()
#q5.b.solution()
def exactly_one_sauce(ketchup, mustard, onion):
    """Return whether the customer wants either ketchup or mustard, but not both.
    (You may be familiar with this operation under the name "exclusive or")
    """
    return bool(ketchup) ^ bool(mustard)

q5.c.check()
#q5.c.hint()
#q5.c.solution()
def exactly_one_topping(ketchup, mustard, onion):
    """Return whether the customer wants exactly one of the three available toppings
    on their hot dog.
    """
    return True if (ketchup + mustard + onion) == 1 else False

q6.check()
#q6.hint()
#q6.solution()
def should_hit(player_total, dealer_total, player_aces):
    """Return True if the player should hit (request another card) given the current game
    state, or False if the player should stay. player_aces is the number of aces the player has.
    """
    return False
q7.simulate_one_game()
q7.simulate(n_games=100000)
def should_hit(player_total, dealer_total, player_aces):
    """Return True if the player should hit (request another card) given the current game
    state, or False if the player should stay. player_aces is the number of aces the player has.
    """
    # Hit till total is at least total_threshold
#     total_threshold = 17
#     return player_total < total_threshold
    
    # If difference between player_total and dealer_total > strategey_cutoff 
    # then play defensively otherwise play aggressively
#     strategy_cutoff = 9
#     defensive_threshold = 13
#     aggressive_threshold = 18
#     if (dealer_total - player_total) > strategy_cutoff:
#         return player_total < aggressive_threshold  # Aim aggressively for threshold if far behind dealer hand
#     elif (player_total - dealer_total) > strategy_cutoff:
#         return player_total < defensive_threshold  # Aim defensively for threshold if dealer is far behind us
#     else:
#         return player_total < aggressive_threshold  # Player and Dealer scores very close. Play aggressively.
    
    # Consider player_aces in the decision. If score while considering Ace as 1 is low, then hit
    ace_low = -11 + 1 #  -11 as Ace high is worth 11 points and + 1
    ace_threshold = 9
    aceless_threshold = 17
    if player_aces != 0:
        return (player_total + ace_low*player_aces) < ace_threshold
    else:
        return (player_total < aceless_threshold)
    

q7.simulate(n_games=300000)