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
push(5): 5
push(3): 5, 3
pop(): 5 return 3
push(2): 5, 2
push(8): 5, 2, 8
pop(): 5, 2 return 8
pop(): 5 return 2
push(9): 5, 9
push(1): 5, 9, 1
pop(): 5, 9 return 1
push(7): 5, 9, 7
push(6): 5, 9, 7, 6
pop(): 5, 9, 7 return 6
pop(): 5, 9 return 7
push(4): 5, 9, 4
pop(): 5, 9 return 4
pop(): 5 return 9
"""
"""
3 Empty errors from pops will not have removed an element. So 7 pops remove an element each.
25 - 7 = 18.
"""
def transfer(s, t):
    while not s.is_empty():
        t.push(s.pop())
"""
enqueue(5): 5
enqueue(3): 5, 3
dequeue(): 3 return 5
enqueue(2): 3, 2
enqueue(8): 3, 2, 8
dequeue(): 2, 8 return 3
dequeue(): 8 return 2
enqueue(9): 8, 9
enqueue(1): 8, 9, 1
dequeue(): 9, 1 return 8
enqueue(7): 9, 1, 7
enqueue(6): 9, 1, 7, 6
dequeue(): 1, 7, 6 return 9
dequeue(): 7, 6 return 1
enqueue(4): 7, 6, 4
dequeue(): 6, 4 return 7
dequeue(): 4 return 6
"""
"""
5 Empty errors from dequeues will not have removed an element. So 10 dequeues remove an
element each. 32 - 10 = 22.
"""
"""One solution is as follows
Operation Q D
[ ] [1, 2, 3, 4, 5, 6, 7, 8]
Q.enqueue(D.delete_first()) [1] [2, 3, 4, 5, 6, 7, 8]
Q.enqueue(D.delete_first()) [1, 2] [3, 4, 5, 6, 7, 8]
Q.enqueue(D.delete_first()) [1, 2, 3] [4, 5, 6, 7, 8]
D.add_last(D.delete_first()) [1, 2, 3] [5, 6, 7, 8, 4]
Q.enqueue(D.delete_first()) [1, 2, 3, 5] [6, 7, 8, 4]
Q.enqueue(D.delete_last()) [1, 2, 3, 5, 4] [6, 7, 8]
Q.enqueue(D.delete_first()) × 3 [1, 2, 3, 5, 4, 6, 7, 8] [ ]
D.add_first(D.dequeue()) × 8 [ ] [1, 2, 3, 5, 4, 6, 7, 8]
"""
"""
Operation S D
[ ] [1, 2, 3, 4, 5, 6, 7, 8]
S.push(D.delete_last()) [8] [1, 2, 3, 4, 5, 6, 7]
S.push(D.delete_last()) [8, 7] [1, 2, 3, 4, 5, 6]
S.push(D.delete_last()) [8, 7, 6] [1, 2, 3, 4, 5]
D.add_first(D.delete_last()) [8, 7, 6] [5, 1, 2, 3, 4]
S.push(D.delete_last()) [8, 7, 6, 4] [5, 1, 2, 3]
S.push(D.delete_first()) [8, 7, 6, 4, 5] [1, 2, 3]
S.push(D.delete_last()) × 3 [8, 7, 6, 4, 5, 3, 2, 1] [ ]
D.add_first(S.pop()) × 8 [ ] [1, 2, 3, 5, 4, 6, 7, 8]
"""