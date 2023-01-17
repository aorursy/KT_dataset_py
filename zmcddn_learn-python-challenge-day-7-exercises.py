# SETUP. You don't need to worry for now about what this code does or how it works. If you're ever curious about the 
# code behind these exercises, it's available under an open source license here: https://github.com/Kaggle/learntools/
import sys; sys.path.insert(0, '../input/learntools/pseudo_learntools')
from learntools.python import binder; binder.bind(globals())
from learntools.python.ex7 import *
print('Setup complete.')
# Import the jimmy_slots submodule
from learntools.python import jimmy_slots
# Call the get_graph() function to get Jimmy's graph
graph = jimmy_slots.get_graph()
graph
def prettify_graph(graph):
    """Modify the given graph according to Jimmy's requests: add a title, make the y-axis
    start at 0, label the y-axis. (And, if you're feeling ambitious, format the tick marks
    as dollar amounts using the "$" symbol.)
    """
    graph.set_title("Results of 500 slot machine pulls")
    # Complete steps 2 and 3 here
    graph.set_xlim(0, 500)
    graph.set_ylim(0, 350)
    graph.set_ylabel("Balance")
    
    ticks = graph.get_yticks()
    # Format those values into strings beginning with dollar sign
    new_labels = ['${}'.format(int(amt)) for amt in ticks]
    # Set the new labels
    graph.set_yticklabels(new_labels)

graph = jimmy_slots.get_graph()
prettify_graph(graph)
graph
q1.solution()
def best_items(racers):
    """Given a list of racer dictionaries, return a dictionary mapping items to the number
    of times those items were picked up by racers who finished in first place.
    """
    winner_item_counts = {}
    for i in range(len(racers)):
        # The i'th racer dictionary
        racer = racers[i]
        # We're only interested in racers who finished in first
        if racer['finish'] == 1:
            for i in racer['items']:
                # Add one to the count for this item (adding it to the dict if necessary)
                if i not in winner_item_counts:
                    winner_item_counts[i] = 0
                winner_item_counts[i] += 1

        # Data quality issues :/ Print a warning about racers with no name set. We'll take care of it later.
        if racer['name'] is None:
            print("WARNING: Encountered racer with unknown name on iteration {}/{} (racer = {})".format(
                i+1, len(racers), racer['name'])
                 )
    return winner_item_counts
sample = [
    {'name': 'Peach', 'items': ['green shell', 'banana', 'green shell',], 'finish': 3},
    {'name': 'Bowser', 'items': ['green shell',], 'finish': 1},
    {'name': None, 'items': ['mushroom',], 'finish': 2},
    {'name': 'Toad', 'items': ['green shell', 'mushroom'], 'finish': 1},
]
best_items(sample)
# Import luigi's full dataset of race data
from learntools.python.luigi_analysis import full_dataset

# Fix me!
def best_items(racers):
    winner_item_counts = {}
    for i in range(len(racers)):
        # The i'th racer dictionary
        racer = racers[i]
        # We're only interested in racers who finished in first
        if racer['finish'] == 1:
            for item in racer['items']:
                # Add one to the count for this item (adding it to the dict if necessary)
                if item not in winner_item_counts:
                    winner_item_counts[item] = 0
                winner_item_counts[item] += 1

        # Data quality issues :/ Print a warning about racers with no name set. We'll take care of it later.
        if racer['name'] is None:
            print("WARNING: Encountered racer with unknown name on iteration {}/{} (racer = {})".format(
                i+1, len(racers), racer['name'])
                 )
    return winner_item_counts

# Try analyzing the imported full dataset
best_items(full_dataset)
q2.hint()
q2.solution()
def blackjack_hand_greater_than(hand_1, hand_2):
    """
    Return True if hand_1 beats hand_2, and False otherwise.
    
    In order for hand_1 to beat hand_2 the following must be true:
    - The total of hand_1 must not exceed 21
    - The total of hand_1 must exceed the total of hand_2 OR hand_2's total must exceed 21
    
    Hands are represented as a list of cards. Each card is represented by a string.
    
    When adding up a hand's total, cards with numbers count for that many points. Face
    cards ('J', 'Q', and 'K') are worth 10 points. 'A' can count for 1 or 11.
    
    When determining a hand's total, you should try to count aces in the way that 
    maximizes the hand's total without going over 21. e.g. the total of ['A', 'A', '9'] is 21,
    the total of ['A', 'A', '9', '3'] is 14.
    
    Examples:
    >>> blackjack_hand_greater_than(['K'], ['3', '4'])
    True
    >>> blackjack_hand_greater_than(['K'], ['10'])
    False
    >>> blackjack_hand_greater_than(['K', 'K', '2'], ['3'])
    False
    """
    ten_points = ['J', 'Q', 'K']
                 
    def get_total(hand):
        total = 0
        if hand:
            number_of_A = 0
            for item in range(len(hand)):
                if hand[item] != 'A':
                    if hand[item] in ten_points:
                        total += 10
                    else:
                        total += int(hand[item])
                else:
                    number_of_A += 1
            if number_of_A != 0:
                if number_of_A == 1:
                    if total <= 10:
                        total += 11
                    else:
                        total += 1
                else:
                    if total == 0:
                        total = 21
                    elif total < 10:
                        total += 12
                    else:
                        total += 2
            if total > 21:
                total = 0
        return total

    hand_1_total = get_total(hand_1)
    hand_2_total = get_total(hand_2)
    
    return hand_1_total > hand_2_total

q3.check()
q3.hint()
q3.solution()
from learntools.python import roulette
import random

def random_and_superstitious(wheel):
    """Interact with the given wheel over 100 spins with the following strategy:
    - if the wheel lands on 4, don't bet on the next spin
    - otherwise, bet on a random number on the wheel (from 0 to 10)
    """
    last_number = 0
    while wheel.num_remaining_spins() > 0:
        if last_number == 4:
            # Unlucky! Don't bet anything.
            guess = None
        else:
            guess = random.randint(0, 10)
        last_number = wheel.spin(number_to_bet_on=guess)

roulette.evaluate_roulette_strategy(random_and_superstitious)
help(roulette.RouletteSession)
help(roulette.evaluate_roulette_strategy)
def conditional_roulette_probs(history):
    """

    Example: 
    conditional_roulette_probs([1, 3, 1, 5, 1])
    > {1: {3: 0.5, 5: 0.5}, 
       3: {1: 1.0},
       5: {1: 1.0}
      }
    """
    prob = {}
    digits = []
    for i in range(1, len(history)):
        pre_digit = str(history[i-1])
        next_digit = str(history[i])
        if pre_digit not in digits:
            digits.append(pre_digit)
            digits.append([next_digit, 1])
        else:
            place = digits.index(pre_digit)
            if next_digit not in digits[place+1]:
                digits[place+1].append(next_digit)
                digits[place+1].append(1)
            else:
                next_place = digits[place+1].index(next_digit)
                digits[place+1][next_place+1] += 1

    for item_index in range(0, len(digits), 2):
#         print(digits[item_index])
        current_digit = int(digits[item_index])
        prob[current_digit] = {}
        for next_item_index in range(0, len(digits[item_index+1]), 2):
            next_digit = int(digits[item_index+1][next_item_index])
            total_times = 0
            for i in range(1, len(digits[item_index+1]), 2):
                total_times += digits[item_index+1][i]
            prob[current_digit][next_digit] = digits[item_index+1][next_item_index+1] / total_times
    return prob
    
def my_agent(wheel):
    history = []
    total_num = wheel.num_remaining_spins()
    train_percentage = 0.75
    for i in range(int(train_percentage*total_num)):
        history.append(wheel.spin())
        
    pair_prob = conditional_roulette_probs(history)
    highest_prob = 0
    pair = [0, 0]
    for pre_num, next_num in pair_prob.items():
        for roll, prob in next_num.items():
            if prob > highest_prob:
                highest_prob = prob
                pair[0] = pre_num
                pair[1] = roll

    last_number=0
    while wheel.num_remaining_spins() > 0:
        if last_number == pair[0]:
            guess = pair[1]
        else:
            guess = None
        last_number = wheel.spin(number_to_bet_on=guess)

roulette.evaluate_roulette_strategy(my_agent)