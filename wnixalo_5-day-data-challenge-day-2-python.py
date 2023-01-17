# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # plotting



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
nutrition = pd.read_csv("../input/starbucks_drinkMenu_expanded.csv")
nutrition.describe()
nutrition.describe(include='all')
nutrition.columns
# (bins & edgecolor optional) # semicolon suppresses returned output

plt.figure(figsize=(12,12))

plt.hist(nutrition[' Sodium (mg)'], bins=9, edgecolor="black");

plt.title("Sodium in Starbucks Menu");

# labelling the axes:

plt.xlabel("Sodium (mg)")

plt.ylabel("Count")
# another way of plotting a histogram from the Pandas plotting API

nutrition.hist(column = " Sodium (mg)", figsize=(12,12))