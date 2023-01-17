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
    return (len(zip_code) == 5) & zip_code.isdecimal()

q1.check()
#q1.hint()
#q1.solution()
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
    keyword = keyword.strip().lower()
    found = []
    sep = [' ', '.', ',']
    for i,d in enumerate(doc_list):
        d = d.lower()
        idx = d.find(keyword)
        
        if idx < 0:
            # nothing found
            continue
        
        valid = True
        if idx > 0:
            # not at the beginning of the doc
            # preceded by a valid separation
            if d[idx-1] not in sep:
                valid = False
        
        if idx < len(d) - len(keyword):
            # not at the end of the doc
            # followed by a valid separation
            if d[idx+len(keyword)] not in sep:
                valid = False
        
        if valid:
            found.append(i)
            
    return found
    
q2.check()
#q2.hint()
#q2.solution()
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
    res = {}
    for keyword in keywords:
        res[keyword] = word_search(doc_list, keyword)
        
    return res

q3.check()
#q3.solution()
def diamond(height):
    """Return a string resembling a diamond of specified height (measured in lines).
    height must be an even integer.
    """
    height //= 2
    u = ''
    d = ''
    for i in range(height):
        u += ' '*(height-1-i) + '/'*(i+1) + '\\'*(i+1) + ' '*(height-1-i) +'\n'
        d = ' '*(height-1-i) + '\\'*(i+1) + '/'*(i+1) + ' '*(height-1-i) +'\n' + d
    
    return u+d
    
#print(diamond(2))

q4.check()
d4 = """ /\\ 
//\\\\
\\\\//
 \\/ """
print(d4)
#q4.hint()
#q4.solution()
history = [1, 3, 1, 5, 1]
numbers = [1,2,3]

# count occurrences
for i,j in zip(history[:-1], history[1:]):
    if i not in res.keys():
        # add primary key i if it doesn't exist
        res[i] = {}
    if j not in res[i].keys():
        # add secondary key i if it doesn't exist
        res[i][j] = 0
    res[i][j] += 1
        
# average
for k,v in res.items():
    tot = sum(v.values())
    for l,w in v.items():
        v[l] /= tot

res
for k,v in res.items():
    tot = sum(v.values())
    for l,w in v.items():
        v[l] /= tot
        
res
import numpy as np

def conditional_roulette_probs(history):
    """

    Example: 
    conditional_roulette_probs([1, 3, 1, 5, 1])
    > {1: {3: 0.5, 5: 0.5}, 
       3: {1: 1.0},
       5: {1: 1.0}
      }
    """
    res = {}
    # count occurrences
    for i,j in zip(history[:-1], history[1:]):
        if i not in res.keys():
            # add primary key i if it doesn't exist
            res[i] = {}
        if j not in res[i].keys():
            # add secondary key i if it doesn't exist
            res[i][j] = 0
        res[i][j] += 1

    # average
    for k,v in res.items():
        tot = sum(v.values())
        for l,w in v.items():
            v[l] /= tot

    return res

#conditional_roulette_probs([1, 1, 1, 1])

q5.check()
#q5.solution()