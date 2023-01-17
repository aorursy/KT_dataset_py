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
    return len(zip_code) == 5 and str.isdigit(zip_code)

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
    result = []
    for index, doc in enumerate(doc_list):
        tmp = doc.replace(',', '').replace('.', '').lower()
        tmp_list = tmp.split(' ')
        if keyword in tmp_list:
            result.append(index)
    return result


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
    result = {}
    for keyword in keywords:
        result[keyword] = []
    key_set = set(keywords)
    for index, doc in enumerate(doc_list):
        words = doc.replace(',', ' ').replace('.', ' ').lower().split()
        word_set = set(words)
        for word in key_set & word_set:
            result[word].append(index)

    return result

q3.check()
#q3.solution()
def diamond(height):
    """Return a string resembling a diamond of specified height (measured in lines).
    height must be an even integer.
    """
    result = ''
    half = int(height / 2)
    for row in range(half):
        result += (' ' * (half - row - 1))
        result += ('/' * (row + 1))
        result += ('\\' * (row + 1))
        result += '\n'
    for row in range(half, height):
        result += (' ' * (row - half))
        result += ('\\' * (height - row))
        result += ('/' * (height - row))
        result += '\n'
    return result

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
    result = {}
    for i in range(len(history)):
        if history[i] not in result:
            result[history[i]] = {}
        if i > 0:
            value = result[history[i - 1]]
            if history[i] in value:
                value[history[i]] += 1
            else:
                value[history[i]] = 1
    
    for k, v in result.items():
        total = sum(v.values())
        v.update((k_, v_ / total) for k_, v_ in v.items())
    
    return result
            


q5.check()
q5.solution()