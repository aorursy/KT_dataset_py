# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sb

from pandas.tseries.offsets import Hour, Minute



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

#for dirname, _, filenames in os.walk('/kaggle/input'):

#    for filename in filenames:

#        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.

data = pd.read_csv("/kaggle/input/pizza-restaurants-and-the-pizza-they-sell/8358_1.csv")

data.columns

data=data[['categories', 'city', 'country','menus.dateSeen','menus.name','name', 'postalCode']]



#i am beginner, please leave your feedbacks



#---Top 10 Restaurants in Los Angeles(as a order)

filter1=data["city"]=="Los Angeles"

data[filter1]["name"].value_counts().nlargest(10).plot(kind="barh")

#or 

data.query("city=='Los Angeles'")["name"].value_counts().nlargest(10).plot(kind="barh")

#filter 2 to find the best selling menu

filter2=data["menus.name"]==data["menus.name"].value_counts().nlargest(1).index[0]

data=data[filter2]



#it is to fix dates, hours and minute is not necessary us

data['menus.dateSeen']= data['menus.dateSeen'].str[0:10]



#to cover date format

data['menus.dateSeen']= pd.to_datetime(data['menus.dateSeen'])



#to show dates

years= pd.DatetimeIndex(data['menus.dateSeen']).year.value_counts().index

#to show number of order

data_annual=pd.DatetimeIndex(data['menus.dateSeen']).year.value_counts().values









fig, ax = plt.subplots(figsize=(6, 3), subplot_kw=dict(aspect="equal"))







data = data_annual

ingredients = years





def func(pct, allvals):

    absolute = int(pct/100.*np.sum(allvals))

    return "{:.1f}%\n({:d} g)".format(pct, absolute)



wedges, texts, autotexts = ax.pie(data, autopct=lambda pct: func(pct, data),

                                  textprops=dict(color="w"))



ax.legend(wedges, ingredients,

          title="Ingredients",

          loc="center left",

          bbox_to_anchor=(1, 0, 0.5, 1))



plt.setp(autotexts, size=8, weight="bold")



ax.set_title("Sales from 2015 to 2017")



plt.show()
