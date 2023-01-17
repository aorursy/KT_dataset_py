import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

import csv

%matplotlib inline
df = pd.read_csv("../input/2017-F500-diversity-data.csv")
df.head(12)
df.tail()
df.shape
df['data-avail'].unique()
df.describe()
df.sample(5)
from random import sample



randomIndex = np.array(sample(range(len(df)), 5))



dfSample = df.ix[randomIndex]



print(dfSample)
print(randomIndex)
dfChoice = df.ix[0]



print(dfChoice)
pd.isnull(df)
list(df)
#Totals for white, black, hispanic, and asian females

test = df.groupby(['WHF10', 'BLKF10', 'HISPF10', 'ASIANF10'])



test.size()
df1 = df[['f500-2017-rank','name','FT10', 'FT11']] #FT10=FEMALE-TOTAL, FT11=PREVIOUS-YEAR-FEMALE-TOTAL 



df1 = df1[df1.FT10 != 'n/a']



print(df1)
df1 = df[['FT10']]



df1 = df1[df1.FT10 != 'n/a']



print(df1)
df1['FT10'].median()
plt.plot(df1)



plt.legend(['2017 Female Total'])



plt.show()