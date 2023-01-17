%matplotlib inline
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import sqlite3
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

conn = sqlite3.connect('../input/database.sqlite')
cursor = conn.cursor()

results = pd.DataFrame(cursor.execute("SELECT * FROM NationalNames").fetchall(),columns=['id','name','year','gender','count'])
print(results.head())

wendys = results[results.name=='Wendy']
print(wendys.head())
%matplotlib inline
plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
wendys[wendys.gender=='F'][['year','count']].groupby('year').sum().plot()
wendys[wendys.gender=='M'][['year','count']].groupby('year').sum().plot()
stateNames = pd.DataFrame(cursor.execute("SELECT * FROM StateNames").fetchall(),columns=['id','name','year','gender','state','count'])
print(stateNames.head())

stateNames[(stateNames.name=='Wendy') & (stateNames.gender=='F') & (stateNames.state=='CA')]
import Basemap
