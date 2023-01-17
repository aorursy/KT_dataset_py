x = 5

y = 2

print(x+y)

print(x-y)

print(x*y)

print(x/y)

print(x**y)

print(x%y)

print(x//y) # // gives the the whole number of a division



x = 1

x *= 2

print(x)



y = 1

y = y * 2

print(y)
print(10 / 3 )
t, f = True, False

print(t, f)
print(t and t)

print(t or f)

print(not t)
print(1 and True)
z = 'hello world'

print(z)



z2 = 'hello' + 'world'

print(z2)



print("%s %s has %d characters" % ('hello', 'world', len('hello world')))

s = "helloa"

print(s.capitalize())

print(s.upper())

print(s.rjust(7))      # Right-justify a string, padding with spaces; prints "  hello"

print(s.center(7))     # Center a string, padding with spaces; prints " hello "

print(s.replace('l', '(ell)'))  # Replace all instances of one substring with another;

                               # prints "he(ell)(ell)o"

print('  world '.strip())
x = ['a', 'b', 'c', 'd', 'e']

print(x[0])

print(x[:1])

print(x[0:3])

print(x[1:])



print(x[-1])

print(x[:-1])

print(x[-1:])

print(x[-3:-1])
x.append('f')

print(x)



print(x.pop())

print(x)



x.extend(['f', 'g'])

print(x)
animals = ['cat', 'dog', 'monkey']

for animal in animals:

    print(animal)
nums = [0, 1, 2, 3, 4]

squares = []

for x in nums:

    squares.append(x ** 2)

print(squares)
nums = [0, 1, 2, 3, 4]

squares = [x ** 2 for x in nums]

print(squares)
nums = [0, 1, 2, 3, 4]

even_squares = [x ** 2 for x in nums if x % 2 == 0]

print(even_squares)
d = {

    'nigeria': 'abuja'

}

print(d)

print(d['nigeria'])



d['congo'] = 'congo'

d['ghana'] = 'accra'

d['nigeria'] = 'lagos'



print(d)
for k, v in d.items():

    print(k,v)
d.keys()
s = set(['a', 'b', 'c', 'c'])

s.add('tome')

s.remove('b')

print([x for x in enumerate(s)])

print(s)



print(s - set(['a']))

print(s.intersection(set(['a', 'q'])))
x = tuple(range(5))

print(x)

print(type(x))

print(x[0])

print(x[0: 2])
def sign(x):

    if x > 0:

        return 'positive'

    elif x < 0:

        return 'negative'

    else:

        return 'zero'



for x in [-1, 0, 1]:

    print(sign(x))
class Car:



    # Constructor

    def __init__(self, name, color):

        self.name = name  # Create an instance variable

        self.color = color  # Create an instance variable



    # Instance method

    def identify(self, loud=False):

        if loud:

            print('HELLO, %s %s!' % (self.name.upper(), self.color.upper()))

        else:

            print('Hello, %s %s!' % (self.name, self.color))



benz = Car('Mercedes', 'red')  # Construct an instance of the Greeter class

toyota = Car('Toyota', 'blue')

benz.identify()            # Call an instance method; prints "Hello, Fred"

toyota.identify(loud=True) 
import numpy as np
a = np.array([[1,1], [2.3, 4], [3,1]])
c = np.full((2,2), 7)
a = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12], range(5, 9)])



# Use slicing to pull out the subarray consisting of the first 2 rows

# and columns 1 and 2; b is the following array of shape (2, 2):

# [[2 3]

#  [6 7]]

b = a[:2, :]

print(a)

print(b)
a[[1], :]

b = np.array([0, 2, 0, 1])

a[np.arange(4), b]
a
a[a>4].reshape(3,4)
import matplotlib.pyplot as plt
%matplotlib inline
# Compute the x and y coordinates for points on sine and cosine curves

x = np.arange(0, 3 * np.pi, 0.1)

y_sin = np.sin(x)

y_cos = np.cos(x)



# Set up a subplot grid that has height 2 and width 1,

# and set the first such subplot as active.

plt.subplot(2, 1, 1)



# Make the first plot

plt.plot(x, y_sin)

plt.title('Sine')



# Set the second subplot as active, and make the second plot.

plt.subplot(2, 1, 2)

plt.plot(x, y_cos)

plt.title('Cosine')



# Show the figure.

plt.show()