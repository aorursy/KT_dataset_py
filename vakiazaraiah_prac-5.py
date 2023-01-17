# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
"""

Algorithm find_second_to_last(L)
if L.head is None or L.head.next is None: {check if the list has 0 or 1 element}
return None {there is no second last node}
current = L.head
next = L.head.next
while next.next is not None: {if next has a next, then current is not second
last}
current = next
next = next.next
return current

"""
def count_nodes(L):
    if L.current is None:
        return 0
    count = 1
    original = L.current
    current = original
    while current.next != original:
        count += 1
    return count

"""
Algorithm same_circular_list(x, y):
current = x

while current.next is not y: {loop until y is found, indicating that it’s the same list}

if current.next is x: {or if x is found again, then they must be different lists}
return False
current = current.next

return True
"""
from positional_list import PositionalList

def list_to_positional_list(list_):
    """Construct a new PositionalList using the contents of list_"""
    pos_list = PositionalList()
    for element in list_:
        pos_list.add_last(element)
        return pos_list
"""A simple solution, taking advantage of PositionalList being iterable, is:"""
def max(self):
    """Return the maximal element."""
    return max(element for element in self)