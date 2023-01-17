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
    graph.set_ylim(0)
    graph.set_ylabel("Blance")
    # solution
    ticks = graph.get_yticks()
    new_labels = ['${}'.format(int(amt)) for amt in ticks]
    graph.set_yticklabels(new_labels)

graph = jimmy_slots.get_graph()
prettify_graph(graph)
graph
#q1.solution()
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
    for j in range(len(racers)):
        # The j'th racer dictionary
        racer = racers[j]
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
                j+1, len(racers), racer['name'])
                 )
    return winner_item_counts

# Try analyzing the imported full dataset
best_items(full_dataset)
#q2.hint()
#q2.solution()
def calculate_point(hand):
    p = []
    aces = 0
    face_cards = ['J','Q','K']
    for card in hand:
        if card == 'A':
            p.append(11)
            aces += 1
        elif card in face_cards:
            p.append(10)
        else:
            p.append(int(card))
    pre_total = sum(p)
    while (pre_total > 21) and (aces > 0):
        pre_total -= 10
        aces -= 1
    total = pre_total
        
    return total

calculate_point(['K', 'K', '2'])
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
    p1 = calculate_point(hand_1)
    p2 = calculate_point(hand_2)
    if p1 > 21:
        res = False
    elif p2 > 21:
        res = True
    else:
        res = p1 > p2
    return res

q3.check()
#q3.hint()
#q3.solution()
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
from learntools.python import roulette
import random

def conditional_roulette_probs(history):
    counts = {}
    for i in range(1,len(history)):
        pre = history[i-1]
        if pre not in counts:
            counts[pre] = {}
        if history[i] not in counts[pre]:
            counts[pre][history[i]] = 0
        counts[pre][history[i]] += 1
    prob = {}
    for pre, roll in counts.items():
        prob[pre] = {}
        total = sum(roll.values())
        for entry in roll:
            sub_prob = roll[entry]/total
            prob[pre][entry] = sub_prob
    return prob

def on_bet_num(prob,last_num):
    """This function can choose the number which has biggest probability
    according to the dictionary prob that is obtained from conditional_roulette_probs func
    """
    if last_num not in prob:
        num = 0
    else:
        pre_num_prob = prob[last_num]
        for n,p in pre_num_prob.items():
            if p == max(pre_num_prob.values()):
                num = n
    return num

def my_agent(wheel):
    """Interact with the given wheel over 100 spins with the following strategy:
    - if the wheel lands on 4, don't bet on the next spin
    - otherwise, bet on a random number on the wheel (from 0 to 10)
    """
    last_number = 0
    history = []
    while wheel.num_remaining_spins() > 0:
        if last_number in history:
            history_prob = conditional_roulette_probs(history)
            guess = on_bet_num(history_prob,last_number)
        else:
            guess = None
        last_number = wheel.spin(number_to_bet_on=guess)
        history.append(last_number)
#     print(history)

roulette.evaluate_roulette_strategy(my_agent, num_simulations=2000, num_spins_per_simulation=100)