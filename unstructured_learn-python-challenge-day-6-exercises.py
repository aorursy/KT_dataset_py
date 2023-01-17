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
    return zip_code.isnumeric() and (len(zip_code) == 5)

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
    # Replace '.' and ',' with ''. Then turn everything into lower case.
    def preprocess_document(doc):
        output_doc = doc.replace(',', '')
        output_doc = output_doc.replace('.', '')
        output_doc = output_doc.lower()
        return output_doc
    
    doc_replaced = [preprocess_document(doc) for doc in doc_list]
    
    # Split based on space and check for existance of keyword
    doc_split = [doc.split() for doc in doc_replaced]
    return [index for index, val in enumerate(doc_split) if keyword in val]


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
    return {keyword : word_search(doc_list, keyword) for keyword in keywords}

q3.check()
#q3.solution()
def diamond(height):
    """Return a string resembling a diamond of specified height (measured in lines).
    height must be an even integer.
    """
    if height%2 != 0:
        ValueError("Need height to be even")
        
    string_list = []
    for i in range(1, (height//2 + 1)):  # For height 10, iterate from 1 to 5
        leading_slash = "/" * i
        trailing_slash = "\\" * i
        justified_spaces = " " * ((height - 2*i)//2)  # Leading and trailing spaces so that total character count (space + text) == height
        base_string = "".join([justified_spaces, leading_slash, trailing_slash, justified_spaces])
        string_list.append(base_string)
        
    reversed_strings_list = [elem[::-1] for elem in string_list]  # Unreadable but quick code to reverse string
    return "\n".join(string_list + list(reversed(reversed_strings_list))) 


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
    # Create ordered pairs from data: (1, 3), (3, 1), (1, 5), (5, 1)
    ordered_pairs = []
    for i in range(len(history) - 1):
        ordered_pairs.append((history[i], history[i+1]))
        
    # Iterate over ordered_pairs and build up output dictionary
    output_dict = {}
    x_val = 0
    y_val = 1  # x and y coordinates of the elements of ordered_pair
    for elem in ordered_pairs:
        if elem[x_val] not in output_dict:
            output_dict[elem[x_val]] = { elem[y_val]: 1 }  # Add new key to output_dict and initialize with 2nd number in elem with count 1
        elif elem[y_val] not in output_dict[elem[x_val]]:
            output_dict[elem[x_val]][elem[y_val]] = 1  # Add new entry inside existing key of output_dict and initialize with count 1
        else:
            output_dict[elem[x_val]][elem[y_val]] += 1  # Increment count by 1 as another instance has been found
            
    def count_to_probability(input_dict):
        """Divides all dictionary values by the sum of these values (thus converting into a probability) and returns new dict
        """
        total_count = sum(input_dict.values())
        return { key: value / total_count for key, value in input_dict.items() }
    
    
    return { key: count_to_probability(value) for key, value in output_dict.items() }
            
    


q5.check()
q5.solution()