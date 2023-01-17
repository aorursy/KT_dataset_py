# Single line comments
"""

Multi line comments in python , they can span 

over a lot of lines.

"""
print("This is through a print statement.")
import math



print(math.pi)
# getting help on a function is as easy as:

help(print)
help(math) # help on library
help(math.pi) # help on a variable from library
import math

print(dir(math))
# variable assignment

var_str = "some random string."

var_int = 123

var_float = 12.345

var_bool = True

var_complex = 2 + 3j
# Finding type of variables in python

print(

    type(var_str),

    type(var_int),

    type(var_float),

    type(var_bool),

    type(var_complex)

     )
a = print('')

print(a)

a == None
type(None)
# Basic Operations in python

print("addition",12+3)

print("subtraction",4-1)

print("multiplication",3*7)

print("division",12/5)

print("Quotient : True Division", 12 // 5)

print("Remainder : Modulus", 12 % 5)

print("exponentation", 3**5)

print("negation", -5)
# Functions

def func_name(param_1,param_2, param_3):

    return ( param_1 + param_2 ) / param_3;



func_name(8,6,5)
def area_of_circle(rad, pi = 3.14):

    """

    Calculates the area of a circle given the radius.

    pi is optional parameter, if not provided pi is taken as 3.14.

    

    >>> area_of_circle(1)

    3.14

    

    >>> area_of_circle(1, 22/7)

    3.142857142857143

    """

    return pi * (rad ** 2);



print(area_of_circle(1))

print(area_of_circle(1, 22/7))
help(area_of_circle)
def mult_by_five(x):

    return 5 * x



def call(fn, arg):

    """Call fn on arg"""

    return fn(arg)



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
a = 5

b = 9

print(

    a == a, #True

    a != a, #False

    a > b,  #False

    a >= b, #False

    a < a,  #False

    a <= a  #True

)
print(3.0 == 3, '3' == 3)
def xor_fnc(a,b):

    return (a and not b) or (not a and b) # Combining and , or , not



print(

xor_fnc(True,True),

xor_fnc(True,False),

xor_fnc(False,True),

xor_fnc(False,False),

)
score = 65

if score > 80:

    print('Distinction')

elif score > 60:

    print('A Grade')

elif score > 40:

    print('B Grade')

else:

    print('Fail')
primes = [2, 3, 5, 7]



planets = ['Mercury', 'Venus', 'Earth', 'Mars', 'Jupiter', 'Saturn', 'Uranus', 'Neptune']



hands = [

    ['J', 'Q', 'K'],

    ['2', '2', '2'],

    ['6', 'A', 'K'], # (Comma after the last element is optional)

]



my_favourite_things = [32, 'raindrops on roses', help]

type(primes)
type(hands)
planets[0] # List index starts from zero
planets[-1] # Negative number counts from last
planets[:2] # select n elements (start index inclusive, end index not inclusive)
planets[:-1]
print(

    planets[3:1],

    planets[-1:-3]

)
planets[3] = 'Malacandra'

planets
planets[:3] = ['Mur', 'Vee', 'Ur']

planets
planets[:4] = ['Mercury', 'Venus', 'Earth', 'Mars',]

planets
print(

    len(planets),

    sorted(planets),

    sum(primes),

    max(primes),

    sep = '\n'

)
planets = ['Mercury', 'Venus', 'Earth', 'Mars', 'Jupiter', 'Saturn', 'Uranus', 'Neptune']

planets.append('Pluto')

planets
planets.pop()
planets.index('Venus')
planets.append('X360')

"Venus" in planets
a = (1,2,3)
type(a)
try: 

    a[0]=111

    

except TypeError as e:

    print(e)
days = ['Mon', 'Tue' ,'Wed']

for day in days:

    print(day, end = ' ')
i=0

while i < 11:

    print(i, end=',')

    i+=1
squares = [i**2 for i in range(10)]

squares
planets = ['Mercury', 'Venus', 'Earth', 'Mars', 'Jupyter', 'Saturn', 'Uranus', 'Neptune', 'Pluto']

big_planets = [planet.upper() + '!' for planet in planets if len(planet) > 5]

big_planets
var_str_a = "Some str"

var_str_with_escape_seq = 'Children\'s Academy\n We the people\n'

print(var_str_with_escape_seq)



hello = "hello\nworld"

triplequoted_hello = """hello

world"""



print(triplequoted_hello)



triplequoted_hello == hello
some_str = "Lorem ipsum"

some_str.index('re')
some_str.startswith('Lo')
try: 

    some_str[2] = 'K'

except BaseException as e:

    print(e)

    
some_str.split()
datestr = '1956-01-31'

year, month, day = datestr.split('-')

print(year, month, day)
'/'.join([month, day, year])
planet = "Pluto"

position = 9



"{}, you'll always be the {}th planet to me.".format(planet, position)
var_dict = {'one':1, 'two':2, 'three':3}

var_dict['one']
var_dict['eleven'] = 11

var_dict
planets = ['Mercury', 'Venus', 'Earth', 'Mars', 'Jupiter', 'Saturn', 'Uranus', 'Neptune']

planet_to_initial = {planet: planet[0] for planet in planets}

planet_to_initial
# in operator works on keys

'Saturn' in planet_to_initial
numbers = {'one':1, 'two':2, 'three':3}

for k in numbers:

    print("{} = {}".format(k, numbers[k]))
numbers.keys()
numbers.values()
for planet, initial in planet_to_initial.items():

    print("{} begins with \"{}\"".format(planet.rjust(10), initial))
for planet, initial in planet_to_initial.items():

    print("{}".format(planet.rjust(50), initial))
'...,,,..,,.asda..,.,.sdfsd..,.,.,.,,'.rstrip('.,')
# Roll 10 dice

import numpy

rolls = numpy.random.randint(low=1, high=6, size=10)

rolls
try:

    [1, 4, 5, 4, 2, 4, 3, 1, 1, 1] + 10



except BaseException as e:

    print(e)
type(rolls)
rolls + 10