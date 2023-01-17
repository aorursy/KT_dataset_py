# SETUP. You don't need to worry for now about what this code does or how it works. If you're ever curious about the 
# code behind these exercises, it's available under an open source license here: https://github.com/Kaggle/learntools/
import sys; sys.path.insert(0, '../input/learntools/pseudo_learntools')
from learntools.python import binder; binder.bind(globals())
from learntools.python.ex5 import *
print('Setup complete.')
def has_lucky_number(nums):
    """Return whether the given list of numbers is lucky. A lucky list contains
    at least one number divisible by 7.
    """
    for num in nums:
        if num % 7 == 0:
            return True
        else:
            return False
def has_lucky_number(nums):
    """Return whether the given list of numbers is lucky. A lucky list contains
    at least one number divisible by 7.
    """
    for num in nums:
        if num % 7 == 0:
            return True
#         else:
#             return False
        
    return False

q1.check()
#q1.hint()
#q1.solution()
#[1, 2, 3, 4] > 2
def elementwise_greater_than(L, thresh):
    """Return a list with the same length as L, where the value at index i is 
    True if L[i] is greater than thresh, and False otherwise.
    
    >>> elementwise_greater_than([1, 2, 3, 4], 2)
    [False, False, True, True]
    """
    result_list = []
    for elem in L:
        if elem <= thresh:
            result_list.append(False)
        else:
            result_list.append(True)
    
    return result_list

q2.check()
#q2.solution()
def menu_is_boring(meals):
    """Given a list of meals served over some period of time, return True if the
    same meal has ever been served two days in a row, and False otherwise.
    """
    for i in range(len(meals) - 1): # Not including the last element
        if meals[i] == meals[i+1]:
            return True
    
    return False

q3.check()
#q3.hint()
#q3.solution()
play_slot_machine()
def estimate_average_slot_payout(n_runs):
    """Run the slot machine n_runs times and return the average payout collected
    """
    winnings = 0
    for i in range(n_runs):
        winnings += play_slot_machine()
    
    return (winnings/n_runs) - 1  # $1 to play each time

test_runs = 10000000
print("Estimating for", test_runs, "runs...")
print(estimate_average_slot_payout(test_runs))
#q4.solution()
def slots_survival_probability(start_balance, n_spins, n_simulations):
    """Return the approximate probability (as a number between 0 and 1) that we can complete the 
    given number of spins of the slot machine before running out of money, assuming we start 
    with the given balance. Estimate the probability by running the scenario the specified number of times.
    
    >>> slots_survival_probability(10.00, 10, 1000)
    1.0
    >>> slots_survival_probability(1.00, 2, 1000)
    .25
    """
    def survived(start_balance, n_spins):
        bank_balance = start_balance
        for i in range(n_spins):
            if bank_balance < 1.0:
                return False
            
            bank_balance += (play_slot_machine() - 1)  # Cost of play is $1
            
        return True
    
    
    survival_count = 0
    for i in range(n_simulations):
        if survived(start_balance, n_spins):
            survival_count += 1
        
    return (survival_count/n_simulations)

q5.check()
q5.solution()