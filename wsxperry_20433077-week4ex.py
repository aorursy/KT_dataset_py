'''
Finger Exercise 1
'''
x = float(input('Please enter the number of miles:'))
y = x/0.62137
z = y * 1000
print(x, 'miles is equivalent to')
print('{:.2f}'.format(y),'km/', '{:.2f}'.format(z), 'meters')


'''
Finger Exercise 2
'''
x = str(input('Please enter your name:'))
y = int(input('Please enter your age:'))
z = 2047-2020+y
print('Hi', x,'! In 2047 you will be', z, '!')


'''
Finger Exercise 3
'''
x = int(input('Please enter the first variable:'))
y = int(input('Please enter the second variable:'))
z = int(input('Please enter the third variable:'))
if x % 2 == 1 and y % 2 == 1 and z % 2 == 1:
    if x > y and x > z:
        print(x, 'is the largest odd number')
    elif y > x and y > z:
        print(y, 'is the largest odd number')
    else:
        print(z, 'is the largest odd number')
elif x % 2 == 1 and y % 2 == 1 and z % 2 == 0:
    if y > x:
        print(y, 'is the largest odd number')
    else:
        print(x, 'is the largest odd number')
elif x % 2 == 1 and y % 2 == 0 and z % 2 == 1:
    if x > z:
        print(x, 'is the largest odd number')
    else:
        print(z, 'is the largest odd number')
elif x % 2 == 0 and y % 2 == 1 and z % 2 == 1:
    if y > z:
        print(y, 'is the largest odd number')
    else:
        print(z, 'is the largest odd number')
elif x % 2 == 1 and y % 2 == 0 and z % 2 == 0:
    print(x, 'is the largest odd number')
elif x % 2 == 0 and y % 2 == 1 and z % 2 == 0:
    print(y, 'is the largest odd number')
elif x % 2 == 0 and y % 2 == 0 and z % 2 == 1:
    print(z, 'is the largest odd number')
else:
    print('Sorry, none of them are odd numbers')

'''
Finger Exercise 4
'''
numXs = int(input('How many times should I print the letter X?'))
toPrint = ''
i = 0
while i < numXs:
    print(str(X) * i)
    i += 1



'''
Finger Exercise 5
'''
a = int(input('Please enter the first integer'))
b = int(input('Please enter the second integer'))
c = int(input('Please enter the third integer'))
d = int(input('Please enter the fourth integer'))
e = int(input('Please enter the fifth integer'))
f = int(input('Please enter the sixth integer'))
g = int(input('Please enter the seventh integer'))
h = int(input('Please enter the eighth integer'))
i = int(input('Please enter the ninth integer'))
j = int(input('Please enter the tenth integer'))
lst = [a, b, c, d, e, f, g, h, i, j]
lst.sort(reverse=True)
for x in range(11):
    if lst[x] % 2 == 0:
        print('The largest even number is', lst[x])
        break

else:
    print('Sorry there are not any even numbers in those integers')



'''
Finger Exercise 6
'''
a = int(input('Please enter an integer'))
root = 1
for pwr in range(2, 7):
    while 1 <= int(root) <= a:
        if root ** pwr == a:
            print('Here are the two integers, root:', root,'and pwr:', pwr,'.')
            break
        root += 1
else:
    print('Sorry, such pair of integers are not exist')


'''
Finger Exercise 7
'''



'''
Finger Exercise 8
'''
def multiplylist(list):
    output = 1
    for a in list:
        output *= a
    return output
list = [3,5,4]
print(multiplylist(list))

'''
Finger Exercise 9
'''

def f(x):
    if a <= x <= b:
        print('yes,',x,'is in range',a,'to',b)
    else:
        print('No,',x,'is not in range',a,'to',b)
    return x
a = float(input('Please input the start of the range'))
b = float(input('Please input the end of the range'))
x = float(input('Please enter a number'))
c = f(x)



'''
Finger Exercise 10
'''









