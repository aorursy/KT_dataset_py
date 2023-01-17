from learntools.core import binder; binder.bind(globals())

from learntools.python.ex5 import *

print('Setup complete.')
def has_lucky_number(nums):

    for num in nums:

        if num % 7 == 0:

            return True

        else:

            return False
def has_lucky_number(nums):

    a=0

    for num in nums:

        if num % 7 != 0:

            a=0

        else:

            a=a+1

        print(a)

    if a>0 :

        return True

    else:

        return False
def has_lucky_number(nums):

    for num in nums:

        if num % 7 == 0:

            return True

    return False
nums=[1,7,14,58,21]

print(has_lucky_number(nums))



nums_2=[3,21,4]

print(has_lucky_number(nums_2))
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

    pass



q2.check()
def elementwise_greater_than(L, thresh):

    x=0

    new_list=[]

    for index in L:

        if L[x]>thresh:

            new_list.append(True)

        else:

            new_list.append(False)

        x=x+1

    return new_list
q2.check()
#q2.solution()
def menu_is_boring(meals):

    """Given a list of meals served over some period of time, return True if the

    same meal has ever been served two days in a row, and False otherwise.

    """

    pass



q3.check()
def menu_is_boring(meals):

    x=0

    for meal in meals:

        if meals[x]==meals[x+1]:

            return True

        x=x+1

    return False
def menu_is_boring(meals):

    x=0

    while meals[x]!=meals[-1]:

        print(meals[x])

        if meals[x]==meals[x+1]:

            return True

        x=x+1

    return False



#Aqui eu estou comparando os elementos meals[x] e não os índices



def menu_is_boring(meals):

    x=0

    while index.meals[x]!=meals[-1]:

        print(meals[x])

        if meals[x]==meals[x+1]:

            return True

        x=x+1

    return False



#index procura o index do argumento então ele precisa de um argumento que seja um elemento e não uma posição
def menu_is_boring(meals):

    # Iterate over all indices of the list, except the last one

    for i in range(len(meals)-1):

        if meals[i] == meals[i+1]:

            return True

    return False
meals=['Spam','Eggs','Eggs','Bacon']

print(menu_is_boring(meals))
q3.check()
q3.hint()

q3.solution()
play_slot_machine()
def estimate_average_slot_payout(n_runs):

    """Run the slot machine n_runs times and return the average net profit per run.

    Example calls (note that return value is nondeterministic!):

    >>> estimate_average_slot_payout(1)

    -1

    >>> estimate_average_slot_payout(1)

    0.5

    """

    pass
import statistics 



def estimate_average_slot_payout(n_runs):

    list=[]

    for i in range(n_runs):

        if i%500==0:

            list.append(20)

        else:

            list.append(0)

    expected_value=statistics.mean(list)

    return expected_value



estimate_average_slot_payout(10000)
q4.solution()