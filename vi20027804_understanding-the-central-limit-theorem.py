from IPython.display import Image

Image("../input/imagedistribution/distribution.png")

Image("../input/imagedistribution/sampling.png")
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# We will check the what are the columns present and what datatypes are used and if there is any missing data.

df = pd.read_csv("../input/heart-disease-uci/heart.csv")

df.info()
df_chol = df['chol']

# Looking the statisc of cholestrol 

df_chol.describe()
# let's take out the mean of the chol data

df_chol.mean()
# plot all the observation in chol data

plt.hist(df_chol, bins=100)



plt.xlabel('cholestrol')

plt.ylabel('frequency')

plt.title('Histogram of Cholestrol frequency')

plt.axvline(x=df_chol.mean(),color='r')
#We will take sample size=10, samples=300

#Calculate the arithmetice mean and plot the mean of sample 300 times



array = []

n = 300

for i in range(1,n):

    array.append(df_chol.sample(n=30,replace= True).mean())



#print(array)

plt.hist(array, bins=100)



plt.xlabel('cholestrol')

plt.ylabel('frequency')

plt.title('Histogram of Cholestrol frequency')

plt.axvline(x=np.mean(array),color='r') # for giving mean line
#We will take sample size=20, 60 & 500 samples=300

#Calculate the arithmetice mean and plot the mean of sample 300 times



array1 = []

array2 = []

array3 = []

n = 300

for i in range(1,n):

    array1.append(df_chol.sample(n=20,replace= True).mean())

    array2.append(df_chol.sample(n=60,replace= True).mean())

    array3.append(df_chol.sample(n=500,replace= True).mean())



#print(array)

fig , (ax1,ax2,ax3) = plt.subplots(nrows=1, ncols=3,figsize=(18,5))

#plt.figure()



#plt.subplot(311)

ax1.hist(array1, bins=100,color='b')

ax1.set_xlabel('cholestrol')

ax1.set_ylabel('frequency')

ax1.set_title('Sample size = 20')

ax1.axvline(x=np.mean(array1),color='r') # for giving mean line



#ax2.subplot(312)

ax2.hist(array2, bins=100, color='g')

ax2.set_xlabel('cholestrol')

ax2.set_ylabel('frequency')

ax2.set_title('Sample size = 60')

ax2.axvline(x=np.mean(array2),color='r') # for giving mean line



#ax3.subplot(313)

ax3.hist(array3, bins=100)

ax3.set_xlabel('cholestrol')

ax3.set_ylabel('frequency')

ax3.set_title('Sample size = 500')

ax3.axvline(x=np.mean(array3),color='r') # for giving mean line
