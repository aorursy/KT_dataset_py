# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Data Visualisation

import matplotlib.pyplot as plt

import seaborn as sns

import plotly_express as px



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/GHG-Mexico-1990-2010.csv')
df.head()
plt.figure(figsize=(12,7))

sns.barplot(data=df, x='Sector', y='Amount', hue='GHG')
px.line(df[df['GHG'] == 'CO2'], x='Year', y='Amount', color='Subsector')
px.line(df[df['Subsector'] == 'Transport'], x='Year', y='Amount', color_discrete_sequence=['LimeGreen', 'Blue', 'Red'], color='GHG')
transport_CO2 = df[df['GHG'].apply(lambda x: x == 'CO2')][df['Subsector'] == 'Transport']

transport_CO2.head()