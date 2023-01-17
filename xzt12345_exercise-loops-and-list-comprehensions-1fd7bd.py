# SETUP. You don't need to worry for now about what this code does or how it works. If you're ever curious about the 

# code behind these exercises, it's available under an open source license here: https://github.com/Kaggle/learntools/

from learntools.core import binder; binder.bind(globals())

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

    lucky = []

    for num in nums:

        if num % 7 == 0:

            lucky.append(num)

    return lucky

nums =[7,1,3,14,5,35]

has_lucky_number(nums)



    

        
def has_lucky_number(nums):

    """Return whether the given list of numbers is lucky. A lucky list contains

    at least one number divisible by 7.

    """

    return any([num % 7 == 0 for num in nums])

nums = [1,7,5,14]

has_lucky_number(nums)



q1.check()
# q1.hint()

q1.solution()
# [1, 2, 3, 4] > 2
def elementwise_greater_than(L, thresh):

    """Return a list with the same length as L, where the value at index i is 

    True if L[i] is greater than thresh, and False otherwise.

    

    >>> elementwise_greater_than([1, 2, 3, 4], 2)

    [False, False, True, True]

    """

    pass



q2.check()
def elementwise_greater_than(L, thresh):

    """Return a list with the same length as L, where the value at index i is 

    True if L[i] is greater than thresh, and False otherwise.

    

    >>> elementwise_greater_than([1, 2, 3, 4], 2)

    [False, False, True, True]

    """

    return [L[i] > thresh for i in range(len(L))]



q2.check()
#q2.solution()
def menu_is_boring(meals):

    """Given a list of meals served over some period of time, return True if the

    same meal has ever been served two days in a row, and False otherwise.

    """

    for i in range(len(meals) - 1):

        if meals[i] == meals[i+1]:

            return True

    return False        



q3.check()
# q3.hint()

q3.solution()
play_slot_machine()

help(play_slot_machine)
def estimate_average_slot_payout(n_runs):

    """Run the slot machine n_runs times and return the average net profit per run.

    Example calls (note that return value is nondeterministic!):

    >>> estimate_average_slot_payout(1)

    -1

    >>> estimate_average_slot_payout(1)

    0.5

    """

    pass
def estimate_average_slot_payout(n_runs):

    """Run the slot machine n_runs times and return the average net profit per run.

    Example calls (note that return value is nondeterministic!):

    >>> estimate_average_slot_payout(1)

    -1

    >>> estimate_average_slot_payout(1)

    0.5

    """

    Sum =0

    for i in range(n_runs):

        Sum = Sum + play_slot_machine() -1

    return Sum/n_runs

print(estimate_average_slot_payout(10000000))



    

   
q4.solution()
def slots_survival_probability(start_balance, n_spins, n_simulations):

    """Return the approximate probability (as a number between 0 and 1) that we can complete the 

    given number of spins of the slot machine before running out of money, assuming we start 

    with the given balance. Estimate the probability by running the scenario the specified number of times.

    

    >>> slots_survival_probability(10.00, 10, 1000)

    1.0

    >>> slots_survival_probability(1.00, 2, 1000)

    .25

    """

    pass



q5.check()
def slots_survival_probability(start_balance, n_spins, n_simulations):

    # How many times did we last the given number of spins?

    successes = 0

    # A convention in Python is to use '_' to name variables we won't use

    for _ in range(n_simulations):

        balance = start_balance

        spins_left = n_spins

        while balance >= 1 and spins_left:

            # subtract the cost of playing

            balance = balance + play_slot_machine() - 1

            spins_left -= 1

        # did we make it to the end?

        if spins_left == 0:

            successes += 1

    return successes / n_simulations

q5.check()
def slots_survival_probability(start_balance, n_spins, n_simulations):

    """Return the approximate probability (as a number between 0 and 1) that we can complete the 

    given number of spins of the slot machine before running out of money, assuming we start 

    with the given balance. Estimate the probability by running the scenario the specified number of times.

    

    >>> slots_survival_probability(10.00, 10, 1000)

    1.0

    >>> slots_survival_probability(1.00, 2, 1000)

    .25

    """

    Sum = 0

    number = 0

    if start_balance >= 2:

        for num in range(n_simulations):

            for i in range(0,start_balance):

                Sum +=play_slot_machine()

            if (Sum >= (n_spins - start_balance)):

                number = number + 1

        return (number/n_simulations)

    else:

        for num in range(n_simulations):

            if (play_slot_machine() > 1):

                number = number + 1

        return number/n_simulations

            



slots_survival_probability(1, 2, 10000)

q5.check()







            
def slots_survival_probability(start_balance, n_spins, n_simulations):

    """Return the approximate probability (as a number between 0 and 1) that we can complete the 

    given number of spins of the slot machine before running out of money, assuming we start 

    with the given balance. Estimate the probability by running the scenario the specified number of times.

    

    >>> slots_survival_probability(10.00, 10, 1000)

    1.0

    >>> slots_survival_probability(1.00, 2, 1000)

    .25

    """

    Sum = 0

    number = 0

    while start_balance >= 1:

        for num in range(n_simulations):

            for i in range(0,start_balance):

                Sum +=play_slot_machine()

            if (int(Sum) >= (n_spins - start_balance)):

                number = number + 1

        return (number/n_simulations)

q5.check()

q5.solution()