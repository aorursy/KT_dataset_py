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
nums=[1,4]
ans=[]
def has_lucky_number(nums):
    """Return whether the given list of numbers is lucky. A lucky list contains
    at least one number divisible by 7.
    """
    
    for num in (nums):
        #print (num)
        if num % 7 == 0:
            #return True
            ans.append('T')
        else:
            #return False
            ans.append('F')
    #print ('Ans is',ans)
has_lucky_number(nums)
if 'T' in ans:
    print ("Lucky list")
else:
    print ("unlucky list")

q1.check()
q1.hint()
q1.solution()
[1, 2, 3, 4] > 2
def elementwise_greater_than(L, thresh):
    """Return a list with the same length as L, where the value at index i is 
    True if L[i] is greater than thresh, and False otherwise.
    
    >>> elementwise_greater_than([1, 2, 3, 4], 2)
    [False, False, True, True]
    """
    ans=[]
    for num in L:
        if num > thresh:
            ans.append(True)
        else:
            ans.append(False)
    return ans
 
    #return [L>thresh for num in L]
    #pass
    #return (ans)

q2.check()
q2.solution()
def menu_is_boring(meals):
    """Given a list of meals served over some period of time, return True if the
    same meal has ever been served two days in a row, and False otherwise.
    """
    for i in range(len(meals)-1):
        #print (dish)
        if meals[i] == meals[i+1]:
            print (meals[i],meals[i+1])
            return True
        #else:
    return False

q3.check()


q3.hint()
q3.solution()
play_slot_machine()
from random import *
def estimate_average_slot_payout(n_runs):
    """Run the slot machine n_runs times and return the average net profit per run.
    Example calls (note that return value is nondeterministic!):
    >>> estimate_average_slot_payout(1)
    -1
    >>> estimate_average_slot_payout(1)
    0.5
    """
    avg=0
    for i in n_runs:
       sum=sum+(randint(0,100)-1)
    avg=sum/n_runs
    print ("Hi",avg)
#q4.solution()
from random import *
def slots_survival_probability(start_balance, n_spins, n_simulations):
    """Return the approximate probability (as a number between 0 and 1) that we can complete the 
    given number of spins of the slot machine before running out of money, assuming we start 
    with the given balance. Estimate the probability by running the scenario the specified number of times.
    
    >>> slots_survival_probability(10.00, 10, 1000)
    1.0
    >>> slots_survival_probability(1.00, 2, 1000)
    .25
    """
    counter_check=0
    for j in range(n_simulations):
        for i in range(n_spins):
            if(start_balance>0):
                counter_check=counter_check+1
                start_balance=start_balance-1
                start_balance=start_balance+randint(0,100)
            if(start_balance<0):
                counter_check=counter_check-1
    prob=counter_check/n_simulations
    return prob
    #pass

q5.check()
q5.solution()