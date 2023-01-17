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
help(str.count)
s = '12345'
len(s)
def is_valid_zip(zip_code):
    """Returns whether the input string is a valid (5 digit) zip code
    """
    return (zip_code.isdigit() and len(zip_code) ==5)

q1.check()
q1.hint()
q1.solution()
help(str.strip)
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
    index_list = []
    for entry in doc_list:
        # unified
        new_entry = entry.lower()
        # get the single word
        # words is a list containing every single word of new_entry
        words = new_entry.split()
        for word in words:
            # remove the punctuation(,.)
            new_word = word.strip(',.')
            # compare
            if (new_word.find(keyword) >= 0 and len(keyword) == len(new_word)):
                index_list.append(doc_list.index(entry))
    return index_list

q2.check()
q2.hint()
q2.solution()
numbers = {'one':1, 'two':2, 'three':3}
d = {}
d['one'] = 1
d['two'] = 2
d
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
    d = {}
    for entry in keywords:
        d[entry] = word_search(doc_list, entry)
    return d
    
q3.check()
q3.solution()
diamond = "/"*4 + '\\'
print(diamond)
def diamond(height):
    """Return a string resembling a diamond of specified height (measured in lines).
    height must be an even integer.
    """
    diamond = ''
    for i in range(height//2):
        diamond1 = ' '*(height//2-1-i) + '/'*(i+1) + '\\'*(i+1) + ' '*(height//2-1-i) + '\n'
        diamond += diamond1
    for j in range(height//2):
        diamond2  = ' '*(j) + '\\'*(height//2-j) + '/'*(height//2-j) + ' '*(j) + '\n'
        diamond +=diamond2
    return diamond
# print(diamond(10))

q4.check()
d4 = """ /\\ 
//\\\\
\\\\//
 \\/ """
print(d4)
#q4.hint()
#q4.solution()
def conditional_roulette_probs(history):
    """

    Example: 
    conditional_roulette_probs([1, 3, 1, 5, 1])
    > {1: {3: 0.5, 5: 0.5}, 
       3: {1: 1.0},
       5: {1: 1.0}
      }
    """
    # layer by layer
    # although it is correct now
    # it will be wrong when it comes [1,3,1,5,1,5,1]
    # so the solution is the right way:
    # record the counts, then obtain the probabilities
    d = {}
    for entry in history:
        d[entry] = {}
    for i in range(1,len(history)):
        d[history[i-1]][history[i]] = 0
    for i in range(1,len(history)):
        d[history[i-1]][history[i]] = 1/len(d[history[i-1]])
    return d

# conditional_roulette_probs([1,2,1,5,1,5,1])
q5.check()
# according to the solution

def conditional_roulette_probs(history):
    counts = {}
    for i in range(1,len(history)):
        pre = history[i-1]
        if pre not in counts:
            counts[pre] = {}
        if history[i] not in counts[pre]:
            counts[pre][history[i]] = 0
        counts[pre][history[i]] += 1
    prob = {}
    for pre, roll in counts.items():
        prob[pre] = {}
        total = sum(roll.values())
        for entry in roll:
            sub_prob = roll[entry]/total
            prob[pre][entry] = sub_prob
    return prob

conditional_roulette_probs([1,2,1,5,1,5,1])
q5.solution()