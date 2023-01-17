# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

from sklearn.experimental import enable_iterative_imputer

from sklearn.impute import IterativeImputer

from sklearn.preprocessing import OneHotEncoder

from sklearn.ensemble import RandomForestClassifier

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
Train_data = pd.read_csv('../input/titanic/train.csv')



y_train = Train_data.Survived

#check for any missing y values 

print('are there any missing y Values?', y_train.isnull().any())

#there aren't any, what a treat

X_train= Train_data.drop(['Survived'], axis = 1)





def concat_df(train_data, test_data):

    # Returns a concatenated df of training and test set

    return pd.concat([train_data, test_data], sort=True).reset_index(drop=True)



def divide_df(all_data):

    # Returns divided dfs of training and test set

    return all_data.loc[:890], all_data.loc[891:]





X_test = pd.read_csv('../input/titanic/test.csv')

df_all = concat_df(X_train, X_test)



x_cols_missing = [col for col in X_train if X_train[col].isnull().any() ]

print(x_cols_missing)

print(df_all.isnull().sum())



print(df_all.info())

dfs = [X_train, X_test]





#yikes, so we have 177 missing age values

#find correllation coefficients 



corr = X_train.corr()



df_all_corr = df_all.corr().abs().unstack().sort_values(kind="quicksort", ascending=False).reset_index()



df_all_corr.rename(columns={"level_0": "Feature 1", "level_1": "Feature 2", 0: 'Correlation Coefficient'}, inplace=True)



print(df_all_corr[df_all_corr['Feature 1'] == 'Age'])





print(x_cols_missing)

#we have some missing column values, yikes.



#let's use a simple imputer for age



#X.head()

#next steps: impute values for age but more cleverly using 'miss, mr, etc.'

#decide what to do with Cabin, maybe make 1,0 for cabin or not, then break down based on cabin, look at layout of ship

#think about ways embarked could affect things, other than through other variables. Is it IV? 
total = df_all.isnull().sum().sort_values(ascending = False)

percent_1 = df_all.isnull().sum()/(df_all.count()+df_all.isnull().sum())*100

percent_2 = (round(percent_1,1)).sort_values(ascending = False)



table = pd.concat([total, percent_2],axis = 1, keys = ['Total', '%'])

table
#building a heatmap 

cmap = sns.diverging_palette(20,220, n = 200)



mask = np.triu(np.ones_like(corr, dtype=np.bool))



ax = sns.heatmap(corr, mask = mask,vmin = -1, vmax = 1, center = 0, cmap = cmap, square = True)



ax.set_xticklabels( ax.get_xticklabels(),rotation=45,horizontalalignment='right')
#fill in age values by looking at the median age along Pclass and sex. 



age_by_pclass_sex = df_all.groupby(['Sex', 'Pclass']).median()['Age']





for pclass in range(1, 4):

    for sex in ['female', 'male']:

        print('Median age of Pclass {} {}s: {}'.format(pclass, sex, age_by_pclass_sex[sex][pclass]))

print('Median age of all passengers: {}'.format(df_all['Age'].median()))



# Filling the missing values in Age with the medians of Sex and Pclass groups

df_all['Age'] = df_all.groupby(['Sex', 'Pclass'])['Age'].apply(lambda x: x.fillna(x.median()))

#dealing with missing Embarked values

df_all[df_all['Embarked'].isnull()]

#we call the index on the whole data to return the rows. How clever!
df_all['Embarked'] = df_all['Embarked'].fillna('S')
df_all[df_all['Fare'].isnull()]
med_fare = df_all.groupby(['Pclass', 'Parch', 'SibSp','Sex']).Fare.median()[3][0][0]['male']

df_all['Fare'] = df_all['Fare'].fillna(med_fare)
#drop Cabin column

df_all_no_C = df_all.drop('Cabin', axis = 1)

#mmmm that's better hun

df_all[['Name']]
titles = []

for name in df_all_no_C['Name']:

    if 'Mr.' in name:

        titles.append('Mr.')

    elif any(x in name for x in ['Mrs.', 'Ms.']):

        titles.append('Mrs.')

    elif 'Miss' in name: 

        titles.append('Miss')

    elif 'Master.' in name:

        titles.append('Master')

    elif any(x in name for x in ['Dr', 'Rev']):

        titles.append('Dr.')

    elif any(x in name for x in ['Col.', 'Major']):

        titles.append('Military')

    else:

        titles.append('Other')





#add new titles column this to that then.

df_all_no_C['Title'] = titles





#now we can drop the name! Not that handy lool



df_all_no_CN = df_all_no_C.drop('Name', axis = 1)

df_all_no_CN[['Title']]

s = (df_all_no_CN.dtypes == 'object')



object_cols = list(s[s].index)

object_cols
df_all
list_of_tickets = df_all.Ticket.value_counts()



dictionary = dict({})

for x in list_of_tickets.keys():

    count = list_of_tickets[x]

    if count not in dictionary.keys():

        dictionary[count] = 1

    else:

        dictionary[count] += 1

sorted_items = sorted(dictionary.items())        



x = pd.DataFrame(sorted_items)



x.columns = ['Number of people per ticket','no. of occurrences']



x
cat_features = ['Embarked','Sex','Title']

dfs = divide_df(df_all_no_CN)

X_train1 = dfs[0]

X_test1 = dfs[1]



OH_E = OneHotEncoder(handle_unknown = 'ignore', sparse = False)



OH_cols_train = pd.DataFrame(OH_E.fit_transform(X_train1[cat_features]))

OH_cols_test = pd.DataFrame(OH_E.transform(X_test1[cat_features]))



OH_cols_train.index = X_train1.index

OH_cols_test.index = X_test1.index



num_X_train1 = X_train1.drop(cat_features, axis = 1)

num_X_test1 = X_test1.drop(cat_features, axis = 1)



OH_X_train = pd.concat([num_X_train1,OH_cols_train], axis = 1)

OH_X_test = pd.concat([num_X_test1,OH_cols_test], axis = 1)



#change column names!!! 

OH_X_test.rename(columns = {0:'Emb_C', 1:'Emb_Q', 2:'Emb_S',3:'Male',4:'Female',5:'Dr.', 6:'Master',7:'Military',8:'Miss',9:'Mr.',10:'Mrs.',11:'Other'}, inplace = True)

OH_X_train.rename(columns = {0:'Emb_C', 1:'Emb_Q', 2:'Emb_S',3:'Male',4:'Female',5:'Dr.', 6:'Master',7:'Military',8:'Miss',9:'Mr.',10:'Mrs.',11:'Other'}, inplace = True)

OH_X_test.drop('Ticket', inplace = True, axis = 1)

OH_X_train.drop('Ticket', inplace = True, axis = 1)
OH_X_test

OH_X_train
my_regressor = RandomForestClassifier(criterion = 'gini')

my_regressor.fit(OH_X_train, y_train)

predictions = my_regressor.predict(OH_X_test)