# Define the variable as follow 

my_lucky_number = 10

my_name = "Saurabh"
# let's print my lucky number

print(my_lucky_number)
# let's check whether lucky number is positive or NOT

if my_lucky_number > 0:

    print("Lucky number is positive.")

elif my_lucky_number == 0:

    print("Lucky number is zero.")

else:

    print("Lucky number is Negative.")

    

# Notice the colon at the end of the expression in the if statement.
type(my_lucky_number)
type(my_name)
type(3.14)
type(True)
# True Division

5/2
# Floor Division

# Gives a quotient

5//2
# Modulus 

# Return Remindor

5%2
x = 2

x += 3

print(x)
x = "spam"

print(x)

x += "eggs"

print(x)
# Left-right associativity

# Output: 3

print(5 * 2 // 3)
# Shows left-right associativity

# Output: 0

print(5 * (2 // 3))
# Exponent operator ** has right-to-left associativity in Python.

# Right-left associativity of ** exponent operator

print(2 ** 3 ** 2)
print((2 ** 3) ** 2)
help(round)
help(round(2.9))



# it is same as help(int)
help(int)
def least_difference(a, b, c):

    diff1 = abs(a - b)

    diff2 = abs(b - c)

    diff3 = abs(a - c)

    return min(diff1, diff2, diff3)
print(

    least_difference(1, 10, 100),

    least_difference(1, 10, 10),

    least_difference(5, 6, 7), # Python allows trailing commas in argument lists. How nice is that?

)
help(least_difference)
# Pure Function : Pure functions have no side effects, and return a value that depends only on their arguments.

def least_difference(a, b, c):

    """Return the smallest difference between any two numbers

    among a, b and c.

    

    >>> least_difference(1, 5, -5)

    4

    """

    diff1 = abs(a - b)

    diff2 = abs(b - c)

    diff3 = abs(a - c)

    return min(diff1, diff2, diff3)
help(least_difference)
print(1, 2, 3, sep=' < ')
print(1, 2, 3)
# Impure Function : Not return a value that depends only on their arguments.

def add_value_in_thousand(a=1):

    print(1000+a)
add_value_in_thousand()
add_value_in_thousand(134)
def mult_by_five(x):

    return 5 * x



def call(fn, arg):

    """Call fn on arg"""

    return fn(arg)



# Higher order functions.

def squared_call(fn, arg):

    """Call fn on the result of calling fn on arg"""

    return fn(fn(arg))



print(

    call(mult_by_five, 1),

    squared_call(mult_by_five, 1), 

    sep='\n', # '\n' is the newline character - it starts a new line

)
def mod_5(x):

    """Return the remainder of x after dividing by 5"""

    return x % 5



print(

    'Which number is biggest?',

    max(100, 51, 14),

    'Which number is the biggest modulo 5?',

    max(100, 51, 14, key=mod_5),

    sep='\n',

)
def func():

    pass
help(round)

round(12345.54321,ndigits=2)
round(12345.54321,-2)
# Comparisons are a little bit clever...

3.0 == 3
# But not too clever...

'3' == 3
True or True and False
def inspect(x):

    if x == 0:

        print(x, "is zero")

    elif x > 0:

        print(x, "is positive")

    elif x < 0:

        print(x, "is negative")

    else:

        print(x, "is unlike anything I've ever seen...")



inspect(0)

inspect(-15)
print(bool(1)) # all numbers are treated as true, except 0

print(bool(0))

print(bool("asf")) # all strings are treated as true, except the empty string ""

print(bool(""))

# Generally empty sequences (strings, lists, and other types we've yet to see like lists and tuples)

# are "falsey" and the rest are "truthy"
if 0:

    print(0)

elif "spam":

    print("spam")
# NOn- optimizated



def quiz_message(grade):

    if grade < 50:

        outcome = 'failed'

    else:

        outcome = 'passed'

    print('You', outcome, 'the quiz with a grade of', grade)

    

quiz_message(80)
# Optimized



def quiz_message(grade):

    outcome = 'failed' if grade < 50 else 'passed'

    print('You', outcome, 'the quiz with a grade of', grade)

    

quiz_message(45)
primes = [2, 3, 5, 7]

planets = ['Mercury', 'Venus', 'Earth', 'Mars', 'Jupiter', 'Saturn', 'Uranus', 'Neptune']

hands = [['J', 'Q', 'K'], ['2', '2', '2'], ['6', 'A', 'K']]



# A list can contain a mix of different types of variables:

my_favourite_things = [32, 'raindrops on roses', hands]
print(primes[0])

print(primes[1])

print(primes[3])
print(primes[-1])
planets[0:3]
# till 2nd planet

planets[:3]
# from 3rd planet onward

planets[3:]
# Same as all list

planets[:]
# The last 2 planets

planets[-2:]
# reversing planets order



numbers = [0,1,2,3,4,5,6,7,8,9]

direction = -1 

# -1 show direction from end_index to start_index

# 1 show opposite of that

numbers[::direction]
planets[3] = 'Ztrimus'

planets
c = 12 + 3j

print("Real Value of C : ",c.real)

print("Imaginary Value of C : ",c.imag)
# in this some function are in place function

dir(planets)
# let's add elon musk's beloved planet back in solar system



planets.append('Mars')

planets
planets.pop()
planets.index('Ztrimus')
'qwery' in planets
'Earth' in planets
# We cant just copy list by simple doing below

list_a = [1,2,3,4,5]

list_b = list_a

print("list_a : ",list_a)

print("list_b : ",list_b)



# Now if i make change into b changes also relfect into both "a" and "b".



list_b[2] = 3333333

print("\nAfter Conversion in only list_b\n")

print("list_a : ",list_a)

print("list_b : ",list_b)

t = (1,2,3,4,5,6)

t
# Uncomment below link to get error

# t[0] = 4534
0.125.as_integer_ratio()
a = 1

b = 0

a, b = b, a

print(a, b)
planets = ['Mercury', 'Venus', 'Earth', 'Mars', 'Jupiter', 'Saturn', 'Uranus', 'Neptune']

for planet in planets:

    print(planet, end='_')
dice_rolls = (1,2,3,4,5,6)

for number in dice_rolls:

    print(number, end = '\n')
print("saurus")

print("ztrimus")

print("saurus", end='')

print("ztrimus", end='')
help(print)

print("========Output========\n")

print("Saurus","Ztrimus",end="!",sep="-")
a_list = [11,22,33,44,55,66,77,88,99]

for i in range(5):

    print("Doing important work. i=", i, "list_value : ", a_list[i])
i = 0

while i < 6:

    print(i, end = ' -> ')

    i += 1
# Comprehension Version code

cubes = [i**3 for i in range(10)]

cubes
# Non-comprehension Version code

cubes = []

for i in range(10):

    cubes.append(i**3)

cubes
a_list
short_list = [i for i in a_list if i < 50]

short_list
# str.upper() returns an all-caps version of a string

loud_short_planets = [planet.upper() + '!' for planet in planets if len(planet) < 6]

loud_short_planets
[55 for planet in planets]
# Another Long Code



def count_negatives(nums):

    """Return the number of negative numbers in the given list.

    

    >>> count_negatives([5, -1, -2, 0, 3])

    2

    """

    n_negative = 0

    for num in nums:

        if num < 0:

            n_negative = n_negative + 1

    return n_negative
# Comprehension Code

def count_negatives(nums):

    return len([num for num in nums if num < 0])



# Much better, right?
# More Comprehension Code

def count_negatives(nums):

    # Reminder: in the "booleans and conditionals" exercises, we learned about a quirk of 

    # Python where it calculates something like True + True + False + True to be equal to 3.

    return sum([num < 0 for num in nums])
# The Zen of Python

import this
# Checking for Lucky number

nums = [7,14,21,28,70,84,44]



def has_lucky_number(nums):

    """Return whether the given list of numbers is lucky. A lucky list contains

    at least one number divisible by 7.

    """

    for num in nums:

        if num % 7 == 0:

            return True

    # We've exhausted the list without finding a lucky number

    return False



print("Is this Lucky list", nums, " : ",has_lucky_number(nums))

print("Is this Lucky list", a_list, " : ",has_lucky_number(a_list))
# List Comprehension

def has_lucky_number(nums):

    return any([num%7==0 for num in nums])



print("Is this Lucky list", nums, " : ",has_lucky_number(nums))

print("Is this Lucky list", a_list, " : ",has_lucky_number(a_list))
help(any)
# [1, 2, 3, 4] > 2
a = [1,2,3,4,5,6]

for i in range(len(a)):

    b = a[:i]+a[i+1:]

    print(b)

    if a[i] in b:

        print("True")
a = ''

len(a)
# Old

'%s %s' % ('one', 'two')
# New

'{} {}'.format('one', 'two')
# Old

'%d %d' % (1, 2)
# New

'{} {}'.format(1, 2)
x = 'May python with you'

y = "May python with you"

x == y
print("I'm a ztrimus")

print('She said,"Machine learning is new to me !"')
# Uncomment below line and run it.

# print('I'm a ztrimus')
print('I\'m a Ztrimus')

print('What\'s up?')

print("That's \"cool\"")

print("Look, a mountain: /\\")

print("1\n2 3") # \n : newline character
len("I'm a ztrimus")
len('I\'m a ztrimus')
hello = "hello\nworld"

print(hello)
"""Lately I been, I been losing sleep

Dreaming about the things that we could be

But baby I been, I been prayin' hard

Said no more counting dollars

We'll be counting stars

Yeah, we'll be counting stars

"""
len("\n")

# The newline character is just a single character! 

# (Even though we represent it to Python using a combination of two characters.)
triplequoted_hello = """hello

world"""

print(triplequoted_hello)

triplequoted_hello == hello
print('Saurus' + '_Ztrimus')
# Uncomment and run it

# print('who is '+2+'nd person on moon?')
print('who is '+str(2)+'nd person on moon?')
song = "La.." * 10

print(song)
# Try to multiply a string by 0 (zero) and see what happens.

'how many extra legs do you have'*0
# Indexing



name = "Ztrimus"



print(name[0])

print(name[-1])
# Slicing



print(name[2:])

print(name[-2:])
# How long is this string?

len(name)
# Yes, we can even loop over them

[char.upper()+"!" for char in name]
# name[0] = 'B'



# name.append doesn't work either
fact = "Sun rises in the East"
fact.upper()
fact.lower()
# Searching for the first index of a substring

fact.index('Eas')
star = "Sun"

fact.startswith(star)
fact.endswith("In the East")
fact.endswith("in the East")
words = fact.split()

words
' '.join(words)
# Split : Occasionally you'll want to split on something other than whitespace

birthdate = '1997-12-18'

year, month, day = birthdate.split('-')
year
month
day
# Join : str.join() takes us in the other direction, sewing a list of strings up into one long string, 

# using the string it was called on as a separator.

'/'.join([day, month, year])
# Yes, we can put unicode characters right in our string literals :)

' 👏 '.join([word.upper() for word in words])
star + ', is bright.'
position = 1

# star + ' is ' + position + "st " + "star in our solar system."
star + ' is ' + str(position) + "st " + "star in our solar system."
# Use of format

'{} is {}st star in our solar system.'.format(star, position)
sun_mass = 1.303 * 10**22

earth_mass = 5.9722 * 10**24

population = 52910390

#           2 decimal points   3 decimal points, format as percent  separate with commas

"{} weighs about {:.2} kilograms ({:.3%} of Earth's mass). It is home to {:,} human.".format(

    star, sun_mass, sun_mass / earth_mass, population)
# Referring to format() arguments by index, starting from 0

s = """Pluto's a {0}.

No, it's a {1}.

{0}!

{1}!""".format('planet', 'dwarf planet')

print(s)
# Use of `f`

f'{star} is {position}st star in our solar system.'
numbers = {'one':1, 'two':2, 'three':3}
numbers['one']
numbers['two']
numbers['twelve'] = 12

numbers
numbers['one'] = 'Sun'

numbers
planets = ['Mercury', 'Venus', 'Earth', 'Mars', 'Jupiter', 'Saturn', 'Uranus', 'Neptune']

planet_to_initial = {planet: planet[0] for planet in planets}

planet_to_initial
'Saturn' in planet_to_initial
'Betelgeuse' in planet_to_initial
for k in numbers:

    print("{} = {}".format(k, numbers[k]))
# Get all the initials, sort them alphabetically, and put them in a space-separated string.

' '.join(sorted(planet_to_initial.values()))
for planet, initial in planet_to_initial.items():

    print("{} begins with \"{}\"".format(planet.rjust(10), initial))
import math



print("It's math! It has type {}".format(type(math)))
print(dir(math))
# Variable

math.pi
# Function

math.log(64,2)
# We can also call help() on the module itself.

help(math)
# Short name or Alias

import math as mt

mt.pi
mt.pi
# Short name equivalent following process

import pandas

pd = pandas
pd.Series([1,2,3,])
from math import *

pi
from math import *

from numpy import *



# print(pi, log(64,2))
from math import log, pi

from numpy import asarray
import numpy

print("numpy.random is a", type(numpy.random))

print("it contains names such as...",

      dir(numpy.random)[-15:]

     )
# Roll 10 dice

rolls = numpy.random.randint(low=1, high=6, size=10)

rolls
type(rolls)
print(dir(rolls))
# What am I trying to do with this dice roll data? Maybe I want the average roll, in which case the "mean"

# method looks promising...

rolls.mean()
# Or maybe I just want to get back on familiar ground, in which case I might want to check out "tolist"

rolls.tolist()
# That "ravel" attribute sounds interesting. I'm a big classical music fan.

help(rolls.ravel)
# Okay, just tell me everything there is to know about numpy.ndarray

# (Click the "output" button to see the novel-length output)

help(rolls)
# rolls + 10
# [3, 4, 1, 2, 2, 1] + 10
# At which indices are the dice less than or equal to 3?

rolls <= 3
xlist = [[1,2,3],[2,4,6],]

# Create a 2-dimensional array

x = numpy.asarray(xlist)

print("xlist = {}\nx =\n{}".format(xlist, x))
# Get the last element of the second row of our numpy array

x[1,-1]
# Get the last element of the second sublist of our nested list?

# uncomment below code and see



# xlist[1,-1]