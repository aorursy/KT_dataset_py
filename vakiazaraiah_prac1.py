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
#Write a short Python function, is_multiple(n, m), that takes two integer values and returns

#True if n is a multiple of m, that is, n = mi for some integer i, and False otherwise.

n=3

m=4



def is_multiple(n, m):

    return n % m == 0



is_multiple(n,m)

#Demonstrate how to use Python’s list comprehension syntax to produce the list [1, 2, 4, 8,

#16, 32, 64, 128, 256].

[2 ** i for i in range(0, 9)]
seq = [4,3,2,2]





def all_distant(seq):

    for i in range(0, len(seq) -1):

        for j in range(i + 1, len(seq)):

            if seq[i] == seq[j]:

                return True

    return False



all_distant(seq)
n = 20

def harmonic_gen(n):

    h = 0

    for i in range(1, n + 1):

        h+= 1/i

        yield(h)

        

    print(h)



    
