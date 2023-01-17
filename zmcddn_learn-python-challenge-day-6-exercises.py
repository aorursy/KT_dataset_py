# SETUP. You don't need to worry for now about what this code does or how it works. If you're ever curious about the 
# code behind these exercises, it's available under an open source license here: https://github.com/Kaggle/learntools/
import sys; sys.path.insert(0, '../input/learntools/pseudo_learntools')
from learntools.python import binder; binder.bind(globals())
from learntools.python.ex6 import *
print('Setup complete.')
a = ""
length = 0
q0.a.check()
b = "it's ok"
length = 7
q0.b.check()
c = 'it\'s ok'
length = 7
q0.c.check()
d = """hey"""
length = 3
q0.d.check()
e = '\n'
length = 1
q0.e.check()
def is_valid_zip(zip_code):
    """Returns whether the input string is a valid (5 digit) zip code
    """
    return len(zip_code) == 5 and zip_code.isdigit()

q1.check()
q1.hint()
q1.solution()
def word_search(doc_list, keyword):
    """
    Takes a list of documents (each document is a string) and a keyword. 
    Returns list of the index values into the original list for all documents 
    containing the keyword.

    Example:
    doc_list = ["The Learn Python Challenge Casino.", "They bought a car", "Casinoville"]
    >>> word_search(doc_list, 'casino')
    >>> [0]
    """
    index = []
    for item in doc_list:
        doc = item.split()
        word_list = [word.strip('.,').lower() for word in doc]
        if keyword.lower() in word_list:
            index.append(doc_list.index(item))
    return index

q2.check()
q2.hint()
q2.solution()
def multi_word_search(doc_list, keywords):
    """
    Takes list of documents (each document is a string) and a list of keywords.  
    Returns a dictionary where each key is a keyword, and the value is a list of indices
    (from doc_list) of the documents containing that keyword

    >>> doc_list = ["The Learn Python Challenge Casino.", "They bought a car and a casino", "Casinoville"]
    >>> keywords = ['casino', 'they']
    >>> multi_word_search(doc_list, keywords)
    {'casino': [0, 1], 'they': [1]}
    """
    word_dict = {}
    for key in keywords:
        word_dict[key] = word_search(doc_list, key)
    return word_dict

q3.check()
q3.solution()
def diamond(height):
    """Return a string resembling a diamond of specified height (measured in lines).
    height must be an even integer.
    """
    diamond_string = ['']*(height+1)
    for i in range(1, height+1):
        if i <= height//2:
#             print(' '*(height//2-i) + '/'*i + '\\'*i)
            diamond_string[i] = ' '*(height//2-i) + '/'*i + '\\'*i
        else:
#             print(' '*(i-height//2-1) + '\\'*(height+1-i) + '/'*(height+1-i))
            diamond_string[i] = ' '*(i-height//2-1) + '\\'*(height+1-i) + '/'*(height+1-i)
    diamond_string.pop(0)
    return '\n'.join(diamond_string)
# print(diamond(6))
q4.check()
d4 = """ /\\ 
//\\\\
\\\\//
 \\/ """
print(d4)
q4.hint()
q4.solution()
def conditional_roulette_probs(history):
    """

    Example: 
    conditional_roulette_probs([1, 3, 1, 5, 1])
    > {1: {3: 0.5, 5: 0.5}, 
       3: {1: 1.0},
       5: {1: 1.0}
      }
    """
    prob = {}
    digits = []
    for i in range(1, len(history)):
        pre_digit = str(history[i-1])
        next_digit = str(history[i])
        if pre_digit not in digits:
            digits.append(pre_digit)
            digits.append([next_digit, 1])
        else:
            place = digits.index(pre_digit)
            if next_digit not in digits[place+1]:
                digits[place+1].append(next_digit)
                digits[place+1].append(1)
            else:
                next_place = digits[place+1].index(next_digit)
                digits[place+1][next_place+1] += 1
    print(digits)
    for item_index in range(0, len(digits), 2):
#         print(digits[item_index])
        current_digit = int(digits[item_index])
        prob[current_digit] = {}
        for next_item_index in range(0, len(digits[item_index+1]), 2):
            next_digit = int(digits[item_index+1][next_item_index])
            total_times = 0
            for i in range(1, len(digits[item_index+1]), 2):
                total_times += digits[item_index+1][i]
            prob[current_digit][next_digit] = digits[item_index+1][next_item_index+1] / total_times
    return prob

q5.check()
q5.solution()