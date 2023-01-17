# ranges can have steps
for n in range(0,100,10):
    print(n)

# ranges can have negative steps
for n in range(10,0,-1):
    print(n)

#letters in strings can be accessed by index
text = "ABC"
index = len(text) - 1
print(text[index])

letter_list = ["C", "B", "A"]
result = "".join(letter_list)
print(result)

CBA

#use len get the length =f the text
# using either ranges and a for loop
# or a counter, and a while loop

#create a function that reverses a string (don't use the pre-made on built into python)
def reverse_text(text):
    output = []

    start = len(text) - 1
    
    #TODO - loop through letters backwards and append them to the output list.
    
    return "".join(output)

test_data = "ABC"
expected_result = "CBA"
result = reverse_text(test_data)
print(result)
# assert(result == expected_result)


for n in range(10,0,1):
    print(n)
for n in range(10,0,-1):
    print(n)
for n in range(10,0,1):
    print(n)
#letters in strings can be accessed by index
text = "ABC"
index = len(text) - 1
print(text[index])

letter_list = ["C", "B", "A"]
result = "".join(letter_list)
print(result)

CBA


CBA

#use len get the length =f the text
# using either ranges and a for loop
# or a counter, and a while loop

#create a function that reverses a string (don't use the pre-made on built into python)
def reverse_text(text):
    output = []

    start = len(text) - 1
    
    #TODO - loop through letters backwards and append them to the output list.
    
    return "".join(output)

test_data = "ABC"
expected_result = "CBA"
result = reverse_text(test_data)
print(result)
# assert(result == expected_result)
#use len get the length =f the text
text = "ABC"
f = len(text)

# using either ranges and a for loop
def reverse_text(text):
    output = []
 
    start = len(text) - 1
    
    for index in range(start, -1, -1):
#        print(index)
#        print(text[index])
        letter = text[index]
        output.append(letter)
        print(output)
        
    return "".join(output)

test_data = (text)
expected_result = "CBA"
result = reverse_text(test_data)
print(result)
list = "A,B,C,D,E,F"
f = len(list)
print(f)
start = len(list) - 1
print(start)
# using either ranges and a for loop
def reverse_list(list):
    output = []
 
    start = len(list) - 1
    print(start)
list = "A,B,C,D,E,F"
f = len(list)
print(f)
# using either ranges and a for loop
def reverse_list(list):
    output = []
 
    start = len(list) - 1

    for index in range(start, 0, -1):
#        print(index)
#        print(list[index])
        letter = list[index]
        output.append(letter)
        print(output)
        
    return "".join(output)

test_data = (list)
expected_result = "FEDCBA"
result = reverse_list(test_data)
print(result)