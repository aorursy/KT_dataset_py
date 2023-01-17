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
import pandas as pd

amazon = pd.read_csv("../input/forest-fires-in-brazil/amazon.csv", encoding = "ISO-8859-1", parse_dates=['date'])
amazon.head()

amazon['state'].unique()
import seaborn as sns

import matplotlib.pyplot as plt



total_state = amazon.groupby('state')

state_list = list(amazon['state'].unique())



mystate = []

myfire = []

for state in state_list:

    mystate.append(state)

    myfire.append(total_state.get_group(state).number.sum())





    

df_state_fire = pd.DataFrame(data={'States': state_list, 'total_fire': myfire})

top_5_df_state_fire = df_state_fire.sort_values(by='total_fire', ascending=False).iloc[:5]



plt.figure(figsize=(15,7))

ax = sns.barplot(data=top_5_df_state_fire, x='States', y='total_fire')
mg = amazon['state'] == 'Mato Grosso'

years = list(amazon['year'].unique())

amazon_mg = amazon[mg].groupby('year')

amazon_mg.head()



myyear = []

total_fire_year = []



for year in years:

    myyear.append(year)

    total_fire_year.append(amazon_mg.get_group(year).number.sum())

    

df_mato = pd.DataFrame(data={'Year': myyear, 'total_fire': total_fire_year})



plt.figure(figsize=(20,7))

ax = sns.barplot(data=df_mato, x='Year', y='total_fire')