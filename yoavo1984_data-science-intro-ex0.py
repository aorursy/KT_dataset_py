# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# Declare your variable in this cell.

print(type(i) == int, type(stu) == str, type(b1) == bool) # Should print True True True
import random

random_number = random.randint(0, 50) # Samples a random number between 0 to 50.
my_number = random.randint(0,20)

your_number = random.randint(0,20)
random_number_a = random.randint(0,50)

random_number_b = random.randint(0,50)
l = ["Boni", 1, True, "Clyde", "Star Wars", "IDC", False, 1.57]
random_number = random.randint(0, 50)

while() : # Statement should go between the brackets

    random_number = random.randint(0,50)

    

if random_number == 35:

    print("Well Done!")

else :

    print("Try Again")
l = ["this", "is", "a", "long", "list", "of", "words"]

# Reminder : l[0] will give you the first element



# Your code here
# Your code here
# Your code here
# Your function here
# Testing your function

x = 5

y = 2

if power(x,y) == 25 and power(3,3) == 27:

    print ("Sucesses")

else:

    print("Try Again.")
data_path = '../input/winemag-data-130k-v2.csv'

# use pd.read_csv to load the csv into a variable named df

df = pd.read_csv(data_path) # Loading the data to a pandas dataframe.

df = df.drop([df.columns[0]], axis=1)
# Your code goes here



# Your code goes here



# Here is an example of group by 'country' taking max, then the points columns and plotting a 'bar' plot

df.groupby('country').max()['points'].plot(kind='bar')

# ^1                 ^2     ^3        ^4
# Your code goes here



# Here is an example of listing the countries by the price of the wine.

df.groupby('country').mean().sort_values(by='price', ascending=False)
# Your code goes here


