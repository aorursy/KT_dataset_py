# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data=pd.read_csv("../input/AppleStore.csv")
data.info()
data.columns
data.head()
#correlation map
f,ax=plt.subplots(figsize=(18,18))
sns.heatmap(data.corr(),annot=True,linewidths=.5,fmt=".1f",ax=ax)
plt.show()
#line plot
data.rating_count_tot.plot(kind="line",color="green",alpha=0.5,linestyle=":",grid=True,label="rating_count_tot")
data.rating_count_ver.plot(kind="line",color="blue",alpha=0.5,linestyle="-",grid=True,label="rating_count_ver")
plt.xlabel("rating_count_tot")
plt.ylabel("rating_count_ver")
plt.legend()
plt.title("mobile app lineplot")
plt.show()
#scatter plot
data.plot(kind="scatter",x="user_rating",y="user_rating_ver",color="red",)
plt.xlabel("user_rating")
plt.ylabel("user_rating_ver")
plt.legend()
plt.title("mobile app scatterplot")
plt.show()
#histogram
data.user_rating.plot(kind="hist",bins=50,figsize=(10,10))
plt.show()
#filtering

y=data["price"]>50
#print(y)
data[y]