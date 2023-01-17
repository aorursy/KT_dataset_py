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
help(graph.__format__)
def prettify_graph(graph):
    """Modify the given graph according to Jimmy's requests: add a title, make the y-axis
    start at 0, label the y-axis. (And, if you're feeling ambitious, format the tick marks
    as dollar amounts using the "$" symbol.)
    """
    graph.set_title("Results of 500 slot machine pulls")
    graph.set_ylim(0)
    graph.set_ylabel("Balance")
    graph.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    # Complete steps 2 and 3 here

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
            for j in racer['items']:
                # Add one to the count for this item (adding it to the dict if necessary)
                if i not in winner_item_counts:
                    winner_item_counts[j] = 0
                winner_item_counts[j] += 1

        # Data quality issues :/ Print a warning about racers with no name set. We'll take care of it later.
        if racer['name'] is None:
            print("WARNING: Encountered racer with unknown name on iteration {}/{} (racer = {})".format(
                i+1, len(racers), racer['name'])
                 )
    return winner_item_counts

#print(full_dataset)
# Try analyzing the imported full dataset
best_items(full_dataset)
q2.hint()
q2.solution()
def score_hand(hand):
    score = 0
    aces = 0
    for card in hand:
        if card == 'A':
            aces+=1
        elif (card == "K" or card == "Q" or card == "J"):
            score+=10
        else:
            score+=int(card)
    
    #print(aces)
    while aces > 0:
        if aces == 1:
            if 21-score >= 11:
                score+= 11
            else:
                score+=1
        else:
            if 21-score >= 12:
                score+=11
            else:
                score+=1
        aces-=1
    return score

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
    pass
    return(score_hand(hand_1) <= 21 and (score_hand(hand_1) > score_hand(hand_2) or score_hand(hand_2)>21))


#print(score_hand(['A','3','Q','Q'])    )

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
from learntools.python import roulette
import random

def my_agent(wheel):
    counts = {}
    sequels = {}
    best_sequels = {}
    
    guess = None
    last_num = wheel.spin(number_to_bet_on=guess)
    
    while wheel.num_remaining_spins() > 0:
        new_num = wheel.spin(number_to_bet_on=guess)
        
        #count how many times we saw the last number
        if last_num not in sequels:
            counts[last_num] = 1
            sequels[last_num]={}
        else:
            counts[last_num] += 1

        #count how many times we've seen teh new number after the old number
        if new_num not in sequels[last_num]:
            sequels[last_num][new_num] = 1
        else:
            sequels[last_num][new_num] +=1    

        #keep a record of the new number we've seen most often after the old number
        if last_num not in best_sequels:
            best_sequels[last_num] = [new_num,1]
        else:
            if(best_sequels[last_num][1] < sequels[last_num][new_num] ):
                best_sequels[last_num] = [new_num,sequels[last_num][new_num]]
        
        #if our most frequent number to show up after the new number has probability of more than 34%, bet on it
        if (new_num in best_sequels and best_sequels[new_num][1]/counts[new_num] > 0.34):
            guess = best_sequels[new_num][0]
        else:
            guess = None
            
        last_num = new_num

roulette.evaluate_roulette_strategy(my_agent)