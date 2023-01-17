# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
#from sklearn import linear_model
import math
import scipy.stats as st
import matplotlib.pyplot as plt

#setting random seed
np.random.seed(888)
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
os.chdir('../input')

data = pd.read_csv('bouts_out_new.csv')
# Any results you write to the current directory are saved as output.
data.info()
#setting thresh to 19 drops most of the nan values (most of them are in the reach_B columns)
data.drop(data.loc[:,'judge1_A':'judge3_B'], axis = 1, inplace = True)
data = data.dropna(how = 'any', thresh = 19)
#drops any duplicate in the dataset
data = data.drop_duplicates()
data.info()

plt.figure(figsize=(20,5))

plt.subplot(1,4,1)
data.boxplot(column = ['age_A' ,'age_B'])

plt.subplot(1,4,2)
data.boxplot(column = ['height_A', 'height_B'])

plt.subplot(1,4,3)
data.boxplot(column = ['reach_A', 'reach_B'])

plt.subplot(1,4,4)
data.boxplot(column = ['weight_A', 'weight_B'])
#   AGE

#deletes the row with suspcious age, we have boxer ages under 18 and over 60
data = data.drop(data[data['age_A'] < 18].index)
data = data.drop(data[data['age_A'] > 60].index)
data = data.drop(data[data['age_B'] < 18].index)
data = data.drop(data[data['age_B'] > 60].index)

#filling the remaining missing values with the median 
data['age_A'] = data['age_A'].fillna(data['age_A'].median())
data['age_B'] = data['age_B'].fillna(data['age_B'].median())

data['age_A'].describe()
data['age_B'].describe() 

#both graphs show that the ages between Boxer A and Boxer B are approximately the same

#   Height

#shortest boxer recorded was 147 cm 
# we have a boxer B that was 15 cm 

#deleting the rows with suspicious height
data = data.drop(data[data['height_A'] < 147].index)
data = data.drop(data[data['height_B'] < 147].index)

#filling the remaining missing values with the median 
data['height_A'] = data['height_A'].fillna(data['height_A'].median())
data['height_B'] = data['height_B'].fillna(data['height_B'].median())

data['height_A'].describe() 
data['height_B'].describe()

#Suspicious reach, 25 cm - 58cm, and 437 - 456 cm 

data['reach_A'].describe()
data['reach_B'].describe()

#deleting suspcious reach
data = data.drop(data[data['reach_A'] < 120].index)
data = data.drop(data[data['reach_A'] > 230].index)
data = data.drop(data[data['reach_B'] < 120].index)
data = data.drop(data[data['reach_B'] > 230].index)

#filling NA with median 
data['reach_A'] = data['reach_A'].fillna(data['reach_A'].median())
data['reach_B'] = data['reach_B'].fillna(data['reach_B'].median())
#
##   Stance
#data['stance_A'].describe()
#data['stance_B'].describe()

#   Weight
#restriction
data = data.drop(data[data['weight_A'] < 105].index)
data = data.drop(data[data['weight_A'] > 323].index)
data = data.drop(data[data['weight_B'] < 120].index)
data = data.drop(data[data['weight_B'] > 230].index)

#boxers should be equal or 3 lbs within eachother
#I am simply making the weight of boxer A equal to weight of boxer B if boxer A's weight is missing 
#and vice versa. Then fill the rest NA with its median

data['weight_A'] = data['weight_A'].fillna(data['weight_B'])
data['weight_A'] = data['weight_A'].fillna(data['weight_A'].median())
data['weight_B'] = data['weight_B'].fillna(data['weight_A'])

#wilson score = lower bound
def ci_lower_bound(wins, losses, a = 0.05):
    '''calculates the Wilson score of a boxer'''
    n = wins + losses
    if n == 0:
        return 0
    z = st.norm.ppf(1 - a / 2)
    phat = 1.0 * wins / n
    lower = (phat - z * z / (2 * n) - z * math.sqrt( (phat*(1 - phat) + z /(4*n))/ n ))/(1 + z*z/n)
    return lower

data['lower_bound_a'] = data.apply(lambda x: ci_lower_bound(x['won_A'], x['lost_A']), axis = 1)
data['lower_bound_b'] = data.apply(lambda x: ci_lower_bound(x['won_B'], x['lost_B']), axis = 1)

#   age difference

data['age_dif'] = data['age_A'] - data['age_B']

#   height dif

data['height_dif'] = data['height_A'] - data['height_B']

#   reach dif

data['reach_dif'] = data['reach_A'] - data['reach_B']

#   weight difference

data['weight_dif'] = data['weight_A'] - data['weight_B']

#   experience
#if A, then boxer_A has more experience and vice versa

def experience(bound_A, bound_B):
    if bound_A > bound_B:
        better_box = 'A'
    else:
        better_box = 'B'
    return better_box

data['experience'] = data.apply(lambda x: experience(x['lower_bound_a'],x['lower_bound_b']), axis = 1)

#   knock_out_perc

data['knock_out_perc_A'] = data['kos_A']/data['won_A']
data['knock_out_perc_B'] = data['kos_B']/data['won_B']
data[['knock_out_perc_A','knock_out_perc_B']] = data[['knock_out_perc_A','knock_out_perc_B']].fillna(0)

#removing columns we're not using
del data['decision'] #only care about winning or losing, so decision doesn't matter
del data['stance_A']
del data['stance_B']
del data['lower_bound_a']
del data['lower_bound_b']
data.drop(data.loc[:,'age_A':'weight_B'], axis = 1, inplace = True)
data.reset_index(inplace = True)
del data['index']

#adjusting the subplots
plt.subplots_adjust(left=.12, bottom=.05, right=2, top=2, wspace=.2, hspace=.2)

#age dif 

plt.subplot(2,2,1)
data['age_dif'].hist(bins = 20)
plt.xlabel('Age A - Age B')
plt.title('The Age Difference Between Boxer A and Boxer B')


#height dif

plt.subplot(2,2,2)
data['height_dif'].hist(bins = 20)
plt.xlabel('Height A - Height B')
plt.title('The Height Difference Between Boxer A and Boxer B')

#reach dif 
plt.subplot(2,2,3)
data['reach_dif'].hist(bins = 20)
plt.xlabel('Reach A - Reach B')
plt.title('The Reach Difference Between Boxer A and Boxer B')


#weight diff
plt.subplot(2,2,4)
data['weight_dif'].hist(bins = 20)
plt.xlabel('Weight A - Weight B')
plt.title('The Weight Difference Between Boxer A and Boxer B')

plt.show()


#print(data['age_dif'].describe())
#print(data['height_dif'].describe())
#print(data['reach_dif'].describe())
#print(data['weight_dif'].describe())

#experience
print(data['experience'].describe())

data.result =  data.result.map({'win_A' : 0 , 'win_B' : 1, 'draw' : 2})
data.experience =  data.experience.map({'A' : 0 , 'B' : 1})
#experience
data['experience'].describe()

data.reset_index(inplace = True)
del data['index']


corr = data.corr()
plt.figure(num=None, figsize=(8, 8), dpi=80, facecolor='w', edgecolor='k')
corrMat = plt.matshow(corr, fignum = 1)
plt.xticks(range(len(corr.columns)), corr.columns, rotation = 90)
plt.yticks(range(len(corr.columns)), corr.columns)
plt.colorbar(corrMat)
plt.show()

print(data['result'].value_counts())
