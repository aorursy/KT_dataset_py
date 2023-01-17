# SETUP. You don't need to worry for now about what this code does or how it works.

from learntools.core import binder; binder.bind(globals())

from learntools.python.ex2 import *

print('Setup complete.')
def round_to_two_places(num):

    """Return the given number rounded to two decimal places. 

    

    >>> round_to_two_places(3.14159)

    3.14

    """

    # Replace this body with your own code.

    return round(num, 2)

    # ("pass" is a keyword that does literally nothing. We used it as a placeholder

    # because after we begin a code block, Python requires at least one line of code)

    pass



q1.check()
# Uncomment the following for a hint

#q1.hint()

# Or uncomment the following to peek at the solution

#q1.solution()
# Put your test code here

round(239584, -2)
q2.solution()
def to_smash(total_candies, n_friends = 3):

    """Return the number of leftover candies that must be smashed after distributing

    the given number of candies evenly between the number of friends. 

    If number of friends is not given, default is 3. 

    

    >>> to_smash(91)

    1

    """

    return total_candies % n_friends



q3.check()
#q3.hint()
#q3.solution()
round_to_two_places(9.9999)
x = -10

y = 5

# Which of the two variables above has the smallest absolute value?

smallest_abs = min(abs(x) , abs(y))

print(smallest_abs)
def f(x):

    y = abs(x)

    return y



print(f(5))
# Importing the function 'time' from the module of the same name. 

# (We'll discuss imports in more depth later)

from time import time

t = time()

print(t, "seconds since the Epoch")
from time import sleep

duration = 5

print("Getting sleepy. See you in", duration, "seconds")

sleep(duration)

print("I'm back. What did I miss?")
def time_call(fn, arg):

    """Return the amount of time the given function takes (in seconds) when called with the given argument.

    """

    

    t0 = time()

    fn(arg)

    t1 = time()

    

    return t1-t0

    

    pass



time_call(sleep, 2)
#q5.hint()

q5.solution()
def slowest_call(fn, arg1, arg2, arg3):

    """Return the amount of time taken by the slowest of the following function

    calls: fn(arg1), fn(arg2), fn(arg3)

    """

    

    return max(time_call(fn, arg1), time_call(fn, arg2), time_call(fn, arg3))

    

    pass



slowest_call(sleep, 4,6,2)





"""

 t1 = time()

    fn(arg1)

    t2 = time()

    fn(arg2)

    t3 = time()

    fn(arg3)

    t4 = time()

    

    c1 = t2 - t1

    c2 = t3 - t2

    c3 = t4 - t3

    """
#q6.hint()
q6.solution()