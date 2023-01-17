import pandas as pd 

from IPython.display import display, HTML

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

import warnings

warnings.filterwarnings('ignore') 
df_horses = pd.read_csv("../input/horses.csv")
plt.figure()

plt.scatter(df_horses['age'], df_horses['prize_money'])

plt.xlabel('Age (years)')

plt.ylabel('Winnings')

plt.xlim([0, 14])

plt.ylim([0, 7000000])

plt.show()
age_cum_winnings = df_horses.groupby('age')['prize_money'].sum()

num_age = df_horses.groupby('age').count()

age_cum_winnings.plot()
age = df_horses.groupby('age')['prize_money'].mean()

age.plot()
num_age['prize_money'].plot()