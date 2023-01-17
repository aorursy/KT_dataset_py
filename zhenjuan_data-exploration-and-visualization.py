import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
%matplotlib inline
from scipy.stats import gaussian_kde
import time
import seaborn as sbn
# loading data
df_train = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')
# checking the trainning data 
df_train.head(5)
# checking the test data
df_test.head(5)
# information from the above tables:
# (1) x, y, accuracy and time are used to predict place_id
# (2) x, y stands for location, time is time, and accuracy is used to measure the location accuracy
# (1) check the accuracy data distribution
count1, bin1 = np.histogram(df_train['accuracy'], bins=50)
binc1 = bin1[:-1]+np.diff(bin1)/2
count2, bin2 =np.histogram(df_test['accuracy'],bins=50)
binc2=bin2[:-1]+np.diff(bin2)/2
                                   
plt.figure(0, figsize = (12,4))
plt.subplot(121)
plt.bar(binc1,count1/(count1.sum()*1.0),width=np.diff(bin1)[0])
plt.xlabel('Accuracy')
plt.ylabel('Fraction')
plt.title('Trainning Data')

plt.subplot(122)
plt.bar(binc2,count2/(count2.sum()*1.0),width=np.diff(bin2)[0])
plt.xlabel('Accuracy')
plt.ylabel('Fraction')
plt.title('Testing Data')

plt.show()
# data pattern in trainning set is very similar to that in testing data
# (2) checking the time distribution
count3, bin3 = np.histogram(df_train['time'], bins=50)
binc3 = bin3[:-1] +np.diff(bin3)/2

count4,bin4 = np.histogram(df_test['time'], bins=50)
binc4 = bin4[:-1] +np.diff(bin4)/2

plt.figure(1, figsize =(12,4))

plt.subplot(121)
plt.bar(binc3,count3/(count3.sum()*1.0),width = np.diff(bin3)[0], color = 'g')
plt.xlabel('Time')
plt.ylabel('Fraction')
plt.title('Trainning data')

plt.subplot(122)
plt.bar(binc4,count4/(count4.sum()*1.0),width = np.diff(bin4)[0], color = 'b')
plt.xlabel('Time')
plt.ylabel('Fraction')
plt.title('Testing data')

plt.show()
plt.figure(2,figsize=(12,4))
plt.bar(binc3,count3/(count3.sum()*1.0),width=np.diff(binc3)[0],color = 'g',label='Training')
plt.bar(binc4, count4/(count4.sum()*1.0),width=np.diff(binc4)[0],color='b',label='Testing')
plt.xlabel('Time')
plt.ylabel('Fraction')
plt.title('Test')
plt.legend()
plt.show()
# check how frequently different locations appear

df_pcounts = df_train['place_id'].value_counts()

count5, bin5 = np.histogram(df_pcounts.values, bins =50)
binc5 = bin5[:-1] + np.diff(bin5)/2

plt.figure(3, figsize=(12,4))
plt.bar(binc5,count5/(count5.sum()*1.0), width=np.diff(bin5)[0])
plt.xlabel('Number of place occurances')
plt.ylabel('Fraction')
plt.title('Tranning data')
plt.show()
# checking how accuracy of signal changes with time

plt.figure(4,figsize=(12,4))
plt.scatter(df_train['time'], df_train['accuracy'], s=1, c='g', alpha=0.1, label='Trainning')
plt.scatter(df_test['time'], df_test['accuracy'], s=1, c='b', alpha =0.1, label='Testing')
plt.xlabel('Time')
plt.ylabel('Accuracy')
plt.title('Accuracy-Time')
plt.legend()

plt.show()
# accuracy vs. location
df_train['xround'] = df_trai