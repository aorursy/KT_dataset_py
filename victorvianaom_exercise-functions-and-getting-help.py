# SETUP. You don't need to worry for now about what this code does or how it works.

from learntools.core import binder; binder.bind(globals())

from learntools.python.ex2 import *

print('Setup complete.')
def round_to_two_places(num):

    """Return the given number rounded to two decimal places. 

    

    >>> round_to_two_places(3.14159)

    3.14

    """

    return round(num, ndigits=2)



# Check your answer

q1.check()
# Uncomment the following for a hint

#q1.hint()

# Or uncomment the following to peek at the solution

#q1.solution()
# Put your test code here

round(6.123456789, ndigits=-1)
# Check your answer (Run this code cell to receive credit!)

q2.solution()
def to_smash(total_candies, num_friends=3):

    """Return the number of leftover candies that must be smashed after distributing

    the given number of candies evenly between 3 friends(or you can pass as an argument 

    the number of friends the candies will be split between).

    

    >>> to_smash(91)

    1

    >>> to_smash(91, 4)

    3

    """

    return total_candies % num_friends



# Check your answer

q3.check()
#q3.hint()
#q3.solution()
round_to_two_places(9.9999)
x = -10

y = 15

# Which of the two variables above has the smallest absolute value?

smallest_abs = min(abs(x), abs(y))

print(smallest_abs)
def f(x):

    y = abs(x)

    return y



print(f(-5))