import os

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from IPython.display import display

import seaborn as sns



print(os.listdir("../input"))

q1=pd.read_csv("../input/top-5000-youtube-channels-data-from-socialblade/data.csv",na_values="-- ")

q1.head(10)
w1 = q1["Subscribers"]

pd.to_numeric(w1)

print("Mean: {}".format(w1.mean()))

# print("Mode: {}".format(w1.mode()))

print("Median: {}".format(w1.median()))
a = np.empty((3,3))

print (a)
a = np.array([[1,0],[1,1]])

b = np.array([[2,1],[3,4]])

c = a*b

print(c)