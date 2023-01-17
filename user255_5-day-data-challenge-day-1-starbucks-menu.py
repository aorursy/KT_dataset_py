# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
drinks =  pd.read_csv('../input/starbucks_drinkMenu_expanded.csv')

drinks.head()
drinks.describe()
drinks.columns
import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

ax, fig = plt.subplots(figsize =(10,10))

ax1 = plt.subplot(221)

sns.distplot(drinks.Calories, kde = False)

ax2 = plt.subplot(222)

sns.distplot(drinks[' Sugars (g)'], kde=False)

ax1.set_title('Calories for Starbucks Beverage')

ax2.set_title('Sugars (g) for Starbucks Beverage')