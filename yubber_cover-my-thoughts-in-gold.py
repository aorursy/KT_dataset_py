# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.datasets import load_iris # cover my thoughts in gold



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
iris = load_iris() # is np array

data = pd.DataFrame(iris.data) # contains column information unlike np arrays



data.columns = ["sepal length", "sepal width", "petal length", "petal width"] # .columns normally gives range index; here we reassign it to have the columns named. 

# run code block above if this doesn't work

data["class"] = iris.target



def label(index):

    '''transforms 0, 1, 2 in class to setosa, versiocolor, virginica'''

    return iris.target_names[index] # target names returns a np arr



data["class"] = data["class"].apply(label) # for every value in class column, value.label(value) and assign 



print(data.describe(),data.info(),data.groupby("class").describe(),sep="\n\n\n") # groupy gives stats per class

# good ways to sort are ones that are far apart and minimal data points are in gray area of prediction
import matplotlib.pyplot as plt # basic plotting tool

import seaborn as sns # pretty plotting tool



sns.boxplot(x="sepal length",y="petal length",data=data)

# sns.boxplot(x="class",y="sepal width",data=data)



# who needs expert opinions when you can just train your pet computer to vore data



dead = data.drop(data.sample(9).index)



print(dead)