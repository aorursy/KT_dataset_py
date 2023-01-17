my_list = ['a', 'p', 'p', 'l', 'e']

my_string = 'apple'
print(my_list[0])

print(my_string[0])
# Print the string starting from the 2nd element until the end

# Note: remember the index starts from '0'

print(my_string[1:])



# Print the string starting from the 2nd element until the 4th element

print(my_string[1:4])



# Print the string from the start until the 4th element

print(my_string[:4])
# This is a function that return the second instance of provided keyword in a string

def find_second(string, keyword):

    return string.index(keyword, string.index(keyword)+1)



print(find_second('dance, dance, dance everyday', 'dance'))
# This is a function that return the string after the second instance of provided keyword

def find_second(string, keyword):

    first_occurence = string.index(keyword)

    second_occurence = string.index(keyword, first_occurence+1)

    

    return string[second_occurence:]



print(find_second('dance, dance, dance everyday', 'dance'))
# return function only return the value of the output without display it on the screen

def return_string(my_string):

    return my_string



# nothing will be shown

return_string('Hello world')





#######################################################################################



# print function directly display the value of the output on the screen

def print_string(my_string):

    print(my_string)

    

# you will see "Hellow world" on the screen

print_string('Hello world')
# So in order to display the output to the screen, you need to use print function here

print(return_string('Hello world'))
# So why we need return function if we can directly use print function to display the output?

# it is actually useful for assignment (store) of calculated value to a variable for later use



def return_sum_of_numbers(a, b):

    return a+b



def print_sum_of_numbers(a, b):

    print(a+b)

    

    

sum_1 = return_sum_of_numbers(3, 5)

sum_2 = print_sum_of_numbers(3, 5)
# let's check:



# 1. Add 10 to sum_1 and print, the answer should be 18

answer_1 = sum_1 + 10

print(answer_1)



# 1. Add 10 to sum_1 and print, the answer should be 18

answer_2 = sum_2 + 10

print(answer_2)
print(sum_2)