# SETUP. You don't need to worry for now about what this code does or how it works.

from learntools.core import binder; binder.bind(globals())

from learntools.python.ex2 import *

print('Setup complete.')


#Return the given number rounded to two decimal places. 

def round_to_two_places(num):

    round2=round(num,2)

    return round2



print(round_to_two_places(3.14159))



q1.check()
# Uncomment the following for a hint

q1.hint()

# Or uncomment the following to peek at the solution

q1.solution()
a=1.12345

print(round(a,2),round(a,-1),round(a,-2),round(a,-3))



q2.solution()
def to_smash(total_candies,num_friends=3):

    return(total_candies%num_friends)

to_smash(91)



q3.check()
q3.hint()
q3.solution()
round_to_two_places(9.9999)
x = -10

y = 5

# # Which of the two variables above has the smallest absolute value?

smallest_abs = min(abs(x),abs(y))

smallest_abs
def f(x):

    y = abs(x)

    return y



print(f(5))
q1.check()

q2.check()

q3.check()

q4.check()