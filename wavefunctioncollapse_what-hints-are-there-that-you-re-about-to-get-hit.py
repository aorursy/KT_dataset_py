import matplotlib.pyplot as plt

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import sklearn.linear_model as sk

from sklearn import preprocessing



full_data_set = pd.read_csv('../input/nflplaybyplay2015.csv',low_memory=False)
## Pull out pass plays and sacks

Pass_Plays = full_data_set.loc[full_data_set.PlayType=='Pass']

Sack_Plays = full_data_set.loc[full_data_set.PlayType=='Sack']

## Form a single set

P_S_data = pd.concat([Pass_Plays,Sack_Plays])
# https://github.com/maksimhorowitz/nflscrapR/blob/master/R/PlayByPlayBoxScore.R

# description of columns which allows us to create

good_columns = ['Drive','qtr','down','TimeUnder','TimeSecs','PlayTimeDiff','yrdline100','ydstogo']

good_columns += ['ScoreDiff','PosTeamScore','DefTeamScore']

good_columns += ['Sack'] #this is our result field

uncleaned_data = P_S_data[good_columns]
#uncleaned_data.loc[uncleaned_data.down.isnull()==True].head()
uncleaned_data.qtr.unique() #checking what OT is assigned as
## Split quarter into a set of five binary variables

def quarter_binary(df,name,number):

    df[name] = np.where(df['qtr']==number,1,0)

    return df



for x in [['qt1',1],['qt2',2],['qt3',3],['qt4',4],['qt5',5]]:

    uncleaned_data = quarter_binary(uncleaned_data,x[0],x[1])



del uncleaned_data['qtr']

#uncleaned_data.head()
## We have some null values in the down columns which I can't explain, drop them and any other 

## nulls



cleaned_data = uncleaned_data.dropna()

explanatory_variables = cleaned_data.columns
def pandas_to_numpy(df):

    y = df['Sack'].values

    del df['Sack']

    X = df.values

    X = preprocessing.scale(X) # 0 mean and 1 std norming

    return X,y
X_all, y_all = pandas_to_numpy(cleaned_data)
logreg = sk.LogisticRegressionCV()

logreg.fit(X_all,y_all)

coef_array = np.abs(logreg.coef_) #careful now with signs when interpreting results
x = np.arange(1,coef_array.shape[1]+1,1)
plt.scatter(x,coef_array,marker='x',color='r')

plt.axhline(0, color='b')
explanatory_variables[1]  #note 0 vs 1 based indexing ~ this is the largest value
## Next phase is to add a boring constant variable to point out how often a pass play...

## ...will not end in a sack