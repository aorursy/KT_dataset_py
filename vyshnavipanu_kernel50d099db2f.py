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
data=pd.read_csv("../input/new-york-city-airbnb-open-data/AB_NYC_2019.csv")
data.head()
data.columns
data.info()
data.describe()
import seaborn as sns

data.boxplot('reviews_per_month')

data.boxplot('minimum_nights')
data.boxplot('number_of_reviews')
data.isnull().sum()
import matplotlib.pyplot as plt

plt.scatter('number_of_reviews','price')