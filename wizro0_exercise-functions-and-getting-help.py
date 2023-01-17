# SETUP. You don't need to worry for now about what this code does or how it works. If you're ever curious about the 
# code behind these exercises, it's available under an open source license here: https://github.com/Kaggle/learntools/
from learntools.core import binder; binder.bind(globals())
from learntools.python.ex2 import *
print('Setup complete.')
help(round)
def round_to_two_places(num):
    """Return the given number rounded to two decimal places. 
    
    >>> round_to_two_places(3.14159)
    3.14
    """
    # Replace this body with your own code.
    # ("pass" is a keyword that does literally nothing. We used it as a placeholder
    # because after we begin a code block, Python requires at least one line of code)
    pass
    return  round(num,2)

q1.check()
# Uncomment the following for a hint
#q1.hint()
# Or uncomment the following to peek at the solution
#q1.solution()
# Put your test code here
#q2.solution()
91 % 4
def to_smash(total_candies, n=3):
    """Return the number of leftover candies that must be smashed after distributing
    the given number of candies evenly between n number of friends. The default value
    of n is 3
    
    >>> to_smash(91, 4)
    3
    """
    return total_candies % n

q3.check()
#q3.hint()
#q3.solution()
round_to_two_places(9.9999)
help(min)
x = -10
y = 5
# Which of the two variables above has the smallest absolute value?
smallest_abs = min(x, y, key=abs)
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
    pass
    time_1 = time()
    fn(arg)
    return (time() - time_1)

# Verifying the time_call function
print(time_call(sleep, 4))
#q5.hint()
#q5.solution()
def slowest_call(fn, arg1, arg2, arg3):
    """Return the amount of time taken by the slowest of the following function
    calls: fn(arg1), fn(arg2), fn(arg3)
    """
    pass
    time_1 = time()
    fn(arg1)
    time_2 = time()
    fn(arg2)
    time_3 = time()
    fn(arg3)
    time_4 = time()
    return min((time_2 - time_1), (time_3 - time_2), (time_4 - time_3),)
#Verifying

print(slowest_call(sleep, 3,2,4))
q6.hint()
q6.solution()