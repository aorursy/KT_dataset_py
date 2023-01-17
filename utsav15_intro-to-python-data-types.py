x = 3
x
type(x)
print(x + 1)   # Addition;
print(x - 1)   # Subtraction;
print(x * 2)   # Multiplication;
print(x ** 2)  # Exponentiation;
x += 1
print(x)  # Prints "4"
x *= 2
print(x)  # Prints "8"
y = 2.5
print(type(y)) # Prints "<type 'float'>"
print(y, y + 1) # Prints "2.5 3.5"
t, f = True, False
print(type(t)) # Prints "<type 'bool'>"
print(t and f) # Logical AND;
print(t or f)  # Logical OR;
print(not t)   # Logical NOT;
print(t != f)  # Logical XOR;
wishing = 'Welcome!'   # String literals can use single quotes
guest = "Mayur"   # or double quotes; it does not matter.
print(wishing, guest)
print('Welcome! Mayur')
print("Welcome! Sam")
s = "hello"
print(s.capitalize())  # Capitalize a string; prints "Hello"
print(s.upper())       # Convert a string to uppercase; prints "HELLO"
mylist = [1,2,3,'four'] # Create a list
mylist
mylist[2]
mylist[-1] # Negative indices count from the end of the list
mylist[3] = 'foo'
mylist
mylist.append('bar') # Add a new element to the end of the list
mylist
mylist.pop() # Remove the last element from the list
mylist
l = mylist.pop() # Remove and return the last element of the list
l
nums = [0,1,2,3,4]
print(nums)         # Prints "[0, 1, 2, 3, 4]"
print(nums[2:4])    # Get a slice from index 2 to 4 (exclusive); prints "[2, 3]"
print(nums[2:])     # Get a slice from index 2 to the end; prints "[2, 3, 4]"
print(nums[:2])     # Get a slice from the start to index 2 (exclusive); prints "[0, 1]"
print(nums[:])      # Get a slice of the whole list; prints ["0, 1, 2, 3, 4]"
print(nums[:-1])    # Slice indices can be negative; prints ["0, 1, 2, 3]"
nums[2:4] = [8, 9] # Assign a new sublist to a slice
print(nums)         # Prints "[0, 1, 8, 9, 4]"
d = {'cat': 'fluffy', 'dog': 'furry'}  # Create a new dictionary with some data
print(d['cat'])       # Get an entry from a dictionary; prints "cute"
print('cat' in d)     # Check if a dictionary has a given key; prints "True"
d['fish'] = 'wet'    # Set an entry in a dictionary
d      
t = (5, 6)
type(t)
t[0]
t[0] = 1 #Item assignment is not possible with tuple
age = int(input("What is your age?\n"))


if age<=10:
    print("You are too young for a driving licence.")
elif age>10 and age<18:
    print("Return back when you are 18 years of age.")
else:
    print("You can apply for a driving licence!")
print("Dividing two numbers")
a = int(input("Enter first number\n"))
b = int(input("Enter second number\n"))

try:
    c = a/b
    print("The quotient of two numbers is", c)
    
except:
    print("Can not divide by zero")