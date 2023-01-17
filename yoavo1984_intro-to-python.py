# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
print("Hello world")
year = 1984

print("I was born in :", year)
# Your code here.
a = 5

print(a, type(a))
s = "Yoav"

print(s, type(s))
f = 2.17

print(f, type(f))
print("This will be printed")

# print("This is a comment, will not be executed")

print("This will aslo be printed") # But this is a comment
a = 7

b = 2
a + b
a * b
a ** b