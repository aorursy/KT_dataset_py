# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



import plotly.graph_objs as go

from plotly import tools



import plotly.plotly as py

from plotly.plotly import iplot





# plotly

from plotly.offline import init_notebook_mode, iplot

init_notebook_mode(connected=True)





# word cloud library

from wordcloud import WordCloud



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/column_2C_weka.csv",sep=",") 
data.info()
data.head()
data["class"].value_counts(dropna =False)
f, ax = plt.subplots(figsize=(18,18))

sns.heatmap(data.corr(),annot=True, linewidths=.5, fmt='.2f', ax=ax)
x = data.pelvic_incidence.values.reshape(-1,1)

y = data.sacral_slope.values.reshape(-1,1)
# plot data

plt.scatter(x, y)

plt.xlabel("Pelvic incidence")

plt.ylabel("Sacral slope")

plt.show()

from sklearn.linear_model import LinearRegression



linear_reg = LinearRegression()
linear_reg.fit(x,y)
y_head = linear_reg.predict(x)
# plot data

plt.scatter(x, y)

plt.xlabel("Pelvic incidence")

plt.ylabel("Sacral slope")

plt.plot(x, y_head, color="red")

plt.show()

from sklearn.metrics import r2_score



print("r_score: ", r2_score(y,y_head))
# anaother method to find r_score :

print("r_score: ", linear_reg.score(x,y))