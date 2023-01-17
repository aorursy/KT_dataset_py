# Let's take a look at the monthly trend of the training data. Simply take the average of the logerror for each month after removing the outliers (abs >0.4)

# We see repetitive trend for 2017 except for a lifted baseline, and we can extraplate the get an estimate for the last three month in 2017

# I manually imputed the last three points, but you could do a better job with smothing and extrapolate, which hopefully will be useful for the final prediction



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



#load training data



train_2017 = pd.read_csv("../input/train_2017.csv")

train_2016 = pd.read_csv("../input/train_2016_v2.csv")



#convert transactiondate to 1,2,....24

b=[]

for dt in train_2017['transactiondate']:

    b.append(int(dt[5:7])+12)

train_2017.transactiondate=b



b=[]

for dt in train_2016['transactiondate']:

    b.append(int(dt[5:7]))

train_2016.transactiondate=b



train_2017 = train_2017[train_2017.logerror > -0.4 ]

train_2017 = train_2017[train_2017.logerror < 0.4 ]

train_2016 = train_2016[train_2016.logerror > -0.4 ]

train_2016 = train_2016[train_2016.logerror < 0.4 ]



train_both = train_2016.append(train_2017, ignore_index=True)



ave = []

for i in range(22):

    aa = train_both.loc[train_both['transactiondate'] == i]

    ave.append(np.mean(aa.logerror))

ave_ex = [0.0159,0.017,0.0172]

plt.plot(ave,marker='o')

plt.plot([22,23,24], ave_ex, marker = 's')



plt.show()