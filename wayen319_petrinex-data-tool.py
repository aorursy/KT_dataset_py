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
months = ['01','02','03','04','05','06','07','08','09','10','11','12']

filenames = [] 



for month in months:

    filename = '../input/Vol_2018-'+month+'.CSV'

    filenames.append(filename)



dfs = []

for filename in filenames:

    df = pd.read_csv(filename, index_col='ProductionMonth')

    dfs.append(df)
print(dfs[0].head(4))

print(dfs[11].head(4))
cols = ['ReportingFacilitySubTypeDesc','ReportingFacilityLocation','ActivityID','ProductID','Volume','Hours']



nums = [0,1,2,3,4,5,6,7,8,9,10,11]

dfs_new = []



for num in nums:

    df_new = dfs[num][cols]

    dfs_new.append(df_new)



print(dfs_new[0].head(4)) 
dfs_con=[]



for num in nums:

    df_con = dfs_new[num][(dfs_new[num]['ProductID']=='WASTE') & (dfs_new[num]['ActivityID']=='INJ')]

    dfs_con.append(df_con)



print(dfs_con[0].head())

print(dfs_con[0].info())
import matplotlib.pyplot as plt



month = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']

rates = []

volumes = []



for num in nums:

    volume = dfs_con[num]['Volume'].astype(float).sum(axis = 0, skipna = True) 

    volumes.append(volume)

    

    # multiplied by 8760 to convert from m3/hr to m3/yr

    rate = volume/dfs_con[num]['Hours'].astype(float).sum(axis = 0, skipna = True) * 8760 

    rates.append(rate)



plt.figure(1)

plt.bar(month, volumes, color="blue")

plt.title('2018 Wastewater Disposal Volume by Month')

plt.xlabel('Month')

plt.ylabel('$m^3$')



plt.figure(2)

plt.bar(month, rates, color="green")

plt.title('2018 Wastewater Disposal Rate by Month')

plt.xlabel('Month')

plt.ylabel('$m^3$/year')



plt.show()