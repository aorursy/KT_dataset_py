import pandas as pd
import os
import matplotlib.pyplot as plt
#print(os.listdir("../input/"))
cereal = pd.read_csv("../input/cereal.csv")
#cereal.describe()
print(cereal.columns)

cal = cereal['calories']
plt.hist(cal)

name = cereal['name']
# print(name)
plt.title("Calories from different food items")

plt.hist(cal,  edgecolor = "black")
plt.xlabel("Calories in Kcal")
plt.ylabel("Count")

cereal.hist(column="calories", figsize = (12,12))

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
