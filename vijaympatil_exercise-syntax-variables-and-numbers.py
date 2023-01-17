print("You've successfully run some Python code")

print("Congratulations!")
from learntools.core import binder

binder.bind(globals())

from learntools.python.ex1 import binder

print("Setup complete! You're ready to start question 0.")
color='red'

print(color)
color='blue'

print(color)
pi = 3.14159 

diameter = 3

radius=diameter/2

area=pi*radius*radius

print('Area of Circle')

print(area)
# Uncomment and run the lines below if you need help.

#q1.hint()

#q1.solution()


a = [1, 2, 3]

b = [3, 2, 1]

c=a

a=b

b=c

print(b)
a=5 - 3

print(a)
b=((8 - 3) * (2 - (1 + 1)))

print(a)
#q3.b.hint()
# Check your answer (Run this code cell to receive credit!)

#q3.b.solution()
# Variables representing the number of candies collected by alice, bob, and carol

alice_candies = 121

bob_candies = 77

carol_candies = 109

smash=(alice_candies+bob_candies+carol_candies)%3

print(smash)