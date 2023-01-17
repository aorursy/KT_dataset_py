friends = ['Joseph', 'Glenn', 'Sally']
for friend in friends:
    print('Happy New Year:', friend)
print('Goodbye!')
n = 5
while n > 0:
    print(n)
    n = n - 1
print('Blastoff!')
while True:
    line = input('I will remember whatever you tell me. Type quit when you are done.\n')
    if line == 'quit':
        break
    print('I will remember this:', line)
print('Peace!')
while True:
    line = input('Enter anything except #. Enter done to stop.')
    if line[0] == '#':
        continue
    if line == 'done':
        break
    print(line)
print('Done!')
#Counting loop

count = 0
for itervar in [3, 41, 12, 9, 74, 15]:
    count = count + 1
print('Count: ', count)
#Summing loop

total = 0
for itervar in [3, 41, 12, 9, 74, 15]:
    total = total + itervar
print('Total: ', total)
num_list = [10,20,30,40,50]

largest = max(num_list)

smallest = min(num_list)

print('Big freakin number is', largest)
print('The smallest one in list is', smallest)

print('Length of the list is', len(num_list))

int('32')
int(3.99999)
#Trying to convert text to integer value is foolish! Don't you think?

int('Hello')
float(32)
str(32)
import random
for i in range(10):
    x = random.random()
    print(x)
random.randint(5, 10)
import math

math.sqrt(4)
degrees = 45
radians = degrees / 360.0 * 2 * math.pi
math.sin(radians)
#Defining a function

def print_something():
    print("Okay! 'Something'")
    
#Calling the function
print_something()
#Defining a function with arguments
def sum(a,b):
    return a+b

a = int(input("Enter first number \n"))
b = int(input("Enter second number \n"))


added = sum(a,b)
print("Sum of the two numbers", added)

