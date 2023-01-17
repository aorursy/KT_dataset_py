#!/bin/python3

import math
import os
import random
import re
import sys



class Multiset:
    def __init__(self):
        self.items=[]
    def add(self, val):
        # adds one occurrence of val from the multiset, if any
        self.items.append(val)
    def remove(self, val):
        # removes one occurrence of val from the multiset, if any
        if self.items.count(val)!=0:
            return self.items.remove(val)

    def __contains__(self, val):
        # returns True when val is in the multiset, else returns False
        return val in self.items
    
    def __len__(self):
        # returns the number of elements in the multiset
        return len(self.items)
if __name__ == '__main__':
    def performOperations(operations):
        m = Multiset()
        result = []
        for op_str in operations:
            elems = op_str.split()
            if elems[0] == 'size':
                result.append(len(m))
            else:
                op, val = elems[0], int(elems[1])
                if op == 'query':
                    result.append(val in m)
                elif op == 'add':
                    m.add(val)
                elif op == 'remove':
                    m.remove(val)
        return result

    q = int(input())
    operations = []
    for _ in range(q):
        operations.append(input())

    result = performOperations(operations)
    
    fptr = open(os.environ['OUTPUT_PATH'], 'w')
    fptr.write('\n'.join(map(str, result)))
    fptr.write('\n')
    fptr.close()

#
# Complete the 'reverse_words_order_and_swap_cases' function below.
#
# The function is expected to return a STRING.
# The function accepts STRING sentence as parameter.
#

def reverse_words_order_and_swap_cases(sentence):
    # Write your code here
    lst=[sentence]        
    lst1= lst[0].split()
    lst1.reverse()

    str2= ' '.join(map(str,lst1)) 
    str3=str2.swapcase()
    return str3
!/bin/python3

import math
import os
import random
import re
import sys


# Complete the 'filledOrders' function below.
#
# The function is expected to return an INTEGER.
# The function accepts following parameters:
#  1. INTEGER_ARRAY order
#  2. INTEGER k
def filledOrders(order, k):
    # Write your code here
    order.sort()
    total=0
    for i,j in enumerate(order):
        if total+j<=k:
            total+=j
        else:
            return i
    else:
        return len(order)

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    order_count = int(input().strip())
    order = []

    for _ in range(order_count):
        order_item = int(input().strip())
        order.append(order_item)

    k = int(input().strip())

    result = filledOrders(order, k)

    fptr.write(str(result) + '\n')

    fptr.close()
