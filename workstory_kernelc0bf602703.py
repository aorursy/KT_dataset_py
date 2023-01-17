# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('/kaggle/input/mai-ml-2019-linear-classification/train.csv');

train.head()
full_data = train.drop(columns = 'PassengerId')

test=pd.read_csv('/kaggle/input/mai-ml-2019-linear-classification/test.csv')

y_passengerid=test['PassengerId']

test=test.drop(columns='PassengerId')
full_data.isnull().sum()[full_data.isnull().sum() > 0]

tester_full_data=full_data

to_zero=['Fare']

to_na=['Age','Cabin']

to_zero = dict.fromkeys(to_zero, 0)

to_fill = dict.fromkeys(to_na, 'NA')

to_fill.update(to_zero)



full_data = full_data.fillna(value = to_fill)

test = test.fillna(value = to_fill)





# not_full_data=test.loc[test['Age'] !='NA']



# Age = {'NA' : not_full_data['Age'].mean() }

# test[['Age']] = test[['Age']].replace(Age)
full_data.columns
full_data=full_data.drop(columns='Cabin')

full_data=full_data.drop(columns='Ticket')



test=test.drop(columns='Cabin')

test=test.drop(columns='Ticket')

# print(tester_full_data['Name'].unique())



tester_full_data_Miss=tester_full_data

tester_full_data_Mrs=tester_full_data

tester_full_data_Mr=tester_full_data

tester_full_data_Master=tester_full_data

tester_full_data_Dr=tester_full_data

tester_full_data_Other=tester_full_data



Tester=tester_full_data['Name'].unique().tolist()

# print(Tester)

Miss=[]

for j in Tester:

    if j.rfind("Miss.")!=-1:

        Miss.append(j)



Mrs=[]

for j in Tester:

    if j.rfind("Mrs.")!=-1:

        Mrs.append(j)

        

Mr=[]

for j in Tester:

    if j.rfind("Mr.")!=-1:

        Mr.append(j)



Master=[]

for j in Tester:

    if ((j.rfind("Master.")!=-1)):

        Master.append(j)



Dr=[]

for j in Tester:

    if ((j.rfind("Dr.")!=-1)):

        Dr.append(j)  



        

Other=[]

for j in Tester:

    if ((j.rfind("Mrs.")==-1)&(j.rfind("Miss.")==-1)&(j.rfind("Mr.")==-1)&((j.rfind("Dr.")==-1))&(j.rfind("Master.")==-1)):

        Other.append(j)     



        



        

# print(mean_Dr,mean_Miss)

# test_for_Miss['Age'].unique()

for i in range(850):

    for j in Miss:

        if full_data['Name'][i]==j:

            full_data['Name'][i]='Miss'



for i in range(850):

    for j in Mrs:

        if full_data['Name'][i]==j:

            full_data['Name'][i]='Mrs'

            

for i in range(850):

    for j in Mr:

        if full_data['Name'][i]==j:

            full_data['Name'][i]='Mr'

            

for i in range(850):

    for j in Dr:

        if full_data['Name'][i]==j:

            full_data['Name'][i]='Dr'

            

for i in range(850):

    for j in Master:

        if full_data['Name'][i]==j:

            full_data['Name'][i]='Master'

            

for i in range(850):

    for j in Other:

        if full_data['Name'][i]==j:

            full_data['Name'][i]='Other'
full_data['Name'].unique()



full_data_Miss=full_data[(full_data['Name']=='Miss')&(full_data['Age']!='NA')]

full_data_Mr=full_data[(full_data['Name']=='Mr')&(full_data['Age']!='NA')]

full_data_Master=full_data[(full_data['Name']=='Master')&(full_data['Age']!='NA')]

full_data_Dr=full_data[(full_data['Name']=='Dr')&(full_data['Age']!='NA')]

full_data_Other=full_data[(full_data['Name']=='Other')&(full_data['Age']!='NA')]

full_data_Mrs=full_data[(full_data['Name']=='Mrs')&(full_data['Age']!='NA')]



mean_Miss=full_data_Miss['Age'].mean()

mean_Mr=full_data_Mr['Age'].mean()

mean_Master=full_data_Master['Age'].mean()

mean_Dr=full_data_Dr['Age'].mean()

mean_Other=full_data_Other['Age'].mean()

mean_Mrs=full_data_Mrs['Age'].mean()



# Embarked = {'S' : 3, 'C' : 2, 'Q':1  }

# full_data[['Embarked']] = full_data[['Embarked']].replace(Embarked)
Age = {'NA' : mean_Miss }

full_data[(full_data['Name']=='Miss')] = full_data[(full_data['Name']=='Miss')].replace(Age)



Age = {'NA' : mean_Mr }

full_data[(full_data['Name']=='Mr')] = full_data[(full_data['Name']=='Mr')].replace(Age)



Age = {'NA' : mean_Master }

full_data[(full_data['Name']=='Master')] = full_data[(full_data['Name']=='Master')].replace(Age)



Age = {'NA' : mean_Dr }

full_data[(full_data['Name']=='Dr')] = full_data[(full_data['Name']=='Dr')].replace(Age)



Age = {'NA' : mean_Other }

full_data[(full_data['Name']=='Other')] = full_data[(full_data['Name']=='Other')].replace(Age)



Age = {'NA' : mean_Mrs }

full_data[(full_data['Name']=='Mrs')] = full_data[(full_data['Name']=='Mrs')].replace(Age)
Tester=test['Name'].unique().tolist()

# print(Tester)

Miss=[]

for j in Tester:

    if j.rfind("Miss.")!=-1:

        Miss.append(j)



Mrs=[]

for j in Tester:

    if j.rfind("Mrs.")!=-1:

        Mrs.append(j)

        

Mr=[]

for j in Tester:

    if j.rfind("Mr.")!=-1:

        Mr.append(j)



Master=[]

for j in Tester:

    if ((j.rfind("Master.")!=-1)):

        Master.append(j)



Dr=[]

for j in Tester:

    if ((j.rfind("Dr.")!=-1)):

        Dr.append(j)  



        

Other=[]

for j in Tester:

    if ((j.rfind("Mrs.")==-1)&(j.rfind("Miss.")==-1)&(j.rfind("Mr.")==-1)&((j.rfind("Dr.")==-1))&(j.rfind("Master.")==-1)):

        Other.append(j)     

test.shape
for i in range(test.shape[0]):

    for j in Miss:

        if test['Name'][i]==j:

            test['Name'][i]='Miss'



for i in range(test.shape[0]):

    for j in Mrs:

        if test['Name'][i]==j:

            test['Name'][i]='Mrs'

            

for i in range(test.shape[0]):

    for j in Mr:

        if test['Name'][i]==j:

            test['Name'][i]='Mr'

            

for i in range(test.shape[0]):

    for j in Dr:

        if test['Name'][i]==j:

            test['Name'][i]='Dr'

            

for i in range(test.shape[0]):

    for j in Master:

        if test['Name'][i]==j:

            test['Name'][i]='Master'

            

for i in range(test.shape[0]):

    for j in Other:

        if test['Name'][i]==j:

            test['Name'][i]='Other'
full_data_Miss=test[(test['Name']=='Miss')&(test['Age']!='NA')]

full_data_Mr=test[(test['Name']=='Mr')&(test['Age']!='NA')]

full_data_Master=test[(test['Name']=='Master')&(test['Age']!='NA')]

full_data_Dr=test[(test['Name']=='Dr')&(test['Age']!='NA')]

full_data_Other=test[(test['Name']=='Other')&(test['Age']!='NA')]

full_data_Mrs=test[(test['Name']=='Mrs')&(test['Age']!='NA')]



mean_Miss=full_data_Miss['Age'].mean()

mean_Mr=full_data_Mr['Age'].mean()

mean_Master=full_data_Master['Age'].mean()

mean_Dr=full_data_Dr['Age'].mean()

mean_Other=full_data_Other['Age'].mean()

mean_Mrs=full_data_Mrs['Age'].mean()
Age = {'NA' : mean_Miss }

test[(test['Name']=='Miss')] = test[(test['Name']=='Miss')].replace(Age)



Age = {'NA' : mean_Mr }

test[(test['Name']=='Mr')] = test[(test['Name']=='Mr')].replace(Age)



Age = {'NA' : mean_Master }

test[(test['Name']=='Master')] = test[(test['Name']=='Master')].replace(Age)



Age = {'NA' : mean_Dr }

test[(test['Name']=='Dr')] = test[(test['Name']=='Dr')].replace(Age)



Age = {'NA' : mean_Other }

test[(test['Name']=='Other')] = test[(test['Name']=='Other')].replace(Age)



Age = {'NA' : mean_Mrs }

test[(test['Name']=='Mrs')] = test[(test['Name']=='Mrs')].replace(Age)
# full_data=full_data.drop(columns='Name')

# test=test.drop(columns='Name')
full_data.columns
s = (full_data.dtypes == 'object')

object_cols = list(s[s].index)



print("Categorical variables:")

print(object_cols)



object_cat=['Sex','Embarked']
# not_full_data_=full_data.loc[full_data['Age'] !='NA']



# Age = {'NA' : not_full_data_['Age'].mean() }

# full_data[['Age']] = full_data[['Age']].replace(Age)
full_data['Age'].unique()
s = (test.dtypes == 'object')

object_cols = list(s[s].index)



print("Categorical variables:")

print(object_cols)

test['Age'].unique()
# test=test[['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']]

# test['Embarked'].unique()
# Embarked = {'NA':0 }

# test[['Embarked']] = test[['Embarked']].replace(Embarked)
to_zero=['Fare']

to_na=['Age','Cabin','Embarked']

to_zero = dict.fromkeys(to_zero, 0)

to_fill = dict.fromkeys(to_na, 'NA')

to_fill.update(to_zero)

test = test.fillna(value = to_fill)

test['Embarked'].unique()



Embarked = {'S' : 3, 'C' : 2, 'Q':1 ,'NA':2 }

test[['Embarked']] = test[['Embarked']].replace(Embarked)





Embarked = {'S' : 3, 'C' : 2, 'Q':1  }

full_data[['Embarked']] = full_data[['Embarked']].replace(Embarked)
s = (test.dtypes == 'object')

object_cols = list(s[s].index)



print("Categorical variables:")

print(object_cols)
print(len(full_data[(full_data['Embarked']==1) & (full_data['Survived']==1)])/len(full_data[full_data['Embarked']==1]))

print(len(full_data[(full_data['Embarked']==2) & (full_data['Survived']==1)])/len(full_data[full_data['Embarked']==2]))

print(len(full_data[(full_data['Embarked']==3) & (full_data['Survived']==1)])/len(full_data[full_data['Embarked']==3]))
from sklearn.preprocessing import OneHotEncoder

OH_encoder=OneHotEncoder(sparse=False)



OH_cols_full_data=pd.DataFrame(OH_encoder.fit_transform(full_data[object_cols]))

OH_cols_test=pd.DataFrame(OH_encoder.fit_transform(test[object_cols]))



OH_cols_full_data.index=full_data.index

OH_cols_test.index=test.index



num_X_full_data=full_data.drop(object_cols,axis=1)

num_X_test=test.drop(object_cols,axis=1)



OH_X_full_data=pd.concat([num_X_full_data,OH_cols_full_data],axis=1)

OH_X_test=pd.concat([num_X_test,OH_cols_test],axis=1)
OH_X_full_data.head()
OH_X_test.head()
X_scaled=OH_X_full_data.drop(columns='Survived')

Y=OH_X_full_data['Survived']

import sklearn.metrics as sm

import numpy as np

import matplotlib.pyplot as plt



from matplotlib.colors import ListedColormap

from sklearn.datasets import load_breast_cancer, make_blobs

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import cross_val_score, cross_validate, train_test_split



%pylab inline

from tqdm import tqdm



C_values = np.arange(1, 200, 0.1)



best_score = 0

best_param = 0.1

best_model = None



for C in tqdm(C_values):

    model = LogisticRegression(C=C)

    score = np.mean(cross_val_score(model, X_scaled, Y, cv=5, scoring="accuracy"))

    

    if score > best_score:

        best_score = score

        best_model = model

        best_param = C

        

print(f"Hyperparameters selection finished, best score - {best_score:0.4f} at C={best_param}")

best_model

# OH_X_test.shape

OH_X_test


best_model.fit(X_scaled, Y)

result=best_model.predict(OH_X_test)

df = pd.DataFrame({

    'PassengerId': y_passengerid,

    'Survived': result 

})

df.to_csv("Maksim-Max.csv",index=False)