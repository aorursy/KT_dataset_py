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
# -*- coding: utf-8 -*-



import numpy as np

import pandas as pd 

import matplotlib as plt

import re



from sklearn.ensemble import GradientBoostingClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import ExtraTreesClassifier

from xgboost.sklearn import XGBClassifier



from sklearn.model_selection import GridSearchCV

from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import accuracy_score,roc_auc_score









def dummydata(df,nlist,keyword):

    for i,v in enumerate(nlist):

        df[keyword] = df[keyword].replace(v,i)

    return df





def familyname(df):

    pattern1 = r','

    df['Name'] = [re.split(pattern1,x)[0] for x in df['Name']]

    name_list = df['Name'].drop_duplicates().tolist()

    dummydata(df,name_list,'Name')

    return df

   



def tikets(df):

    df['Ticket'] = [re.split(' ',x)[-1] for x in df['Ticket']]

    dummydata(df,['LINE'],'Ticket')

    return df



def cabin(df):

    mid = pd.Series(df['Cabin'].values)

    mid = mid.str.replace(r'(\w+\s*)(.*)',lambda x:x.group(1)[0:1])

    df['Cabin'] = mid.to_frame()

    cabinMap = {elem:index for index,elem in enumerate(set(df["Cabin"]))}

    df['Cabin'] = df['Cabin'].map(cabinMap)    

    return df



def newcolumn(df):

    df['relative'] = df['SibSp']+df['Parch']+1

    df['isalone'] = 1

    df['isalone'].loc[df['relative']>1] = 0

    df['agebin'] = pd.cut(df['Age'].astype(int), 5)

    df['farebin'] = pd.qcut(df['Fare'], 4)

    label = LabelEncoder()

    df['agebin'] = label.fit_transform(df['agebin'])

    df['farebin'] = label.fit_transform(df['farebin'])

    return df







def loaddata(df):

    

    df['Age'] = df['Age'].fillna(-1)

    df['Cabin'] = df['Cabin'].fillna('N')

    df['Embarked'] = df['Embarked'].fillna('N')

    df['Fare'] = df['Fare'].fillna(df['Fare'].median())

    new_data = df[['Survived','Pclass','Name','Sex','Age','SibSp','Parch','Fare','Cabin','Embarked']]

    sex_list = ['male','female']

    embarked_list = ['C','Q','S','N']

    dummydata(new_data,sex_list,'Sex')

    dummydata(new_data,embarked_list,'Embarked')

    familyname(new_data)

    #tikets(new_data)

    cabin(new_data)

    newcolumn(new_data)

    #print(new_data.info())

    new_data = new_data.drop(columns=['agebin','farebin'])

       

    data_mid = np.array(new_data)

    data_mid = np.asarray(data_mid,'float32')

    length = data_mid.shape[0] 

    label = data_mid[0:length,0]

    data = data_mid[0:length,1:]

    

    return label,data

 



 

df1 = pd.read_csv('../input/titanic/train.csv')

df2 = pd.read_csv('../input/titanic/test.csv')

df = pd.concat([df1,df2])

x1 = df1.shape[0]

x2 = df2.shape[0]



[label_raw,data_raw] = loaddata(df)

label_train = label_raw[0:x1]

label_test = label_raw[x1:x1+x2]



data_train = data_raw[0:x1,:]

data_test = data_raw[x1:x1+x2,:]





###



###GBDT

'''

model = GradientBoostingClassifier(

    n_estimators=180, 

    learning_rate=0.05,

    max_depth=8, 

    min_samples_split = 5,

    min_samples_leaf = 40,

    )



model.fit(data_train,label_train)

label_test1 = model.predict(data_test)

score = accuracy_score(label_train,model.predict(data_train))

print('GradientBoosting:accuracy_score is %f' % score)

'''





###Random Foreast



model = RandomForestClassifier(n_estimators = 100,

                    bootstrap = True,

                    max_features = 'sqrt',

                    oob_score = True

                    )

            

model.fit(data_train,label_train)

label_test2 = model.predict(data_test)

score = accuracy_score(label_train,model.predict(data_train))

print('Random Forest:accuracy_score is %f' % score) #train准确率

print('oob_score is %f' %model.oob_score_) #用bootstrap之外的数据测试，模拟test





###XGBoost

'''

model = XGBClassifier()

model.fit(data_train,label_train)

label_test3 = model.predict(data_test)

score = accuracy_score(label_train,model.predict(data_train))

print('XGBoost:accuracy_score is %f' % score)

'''



###ExtraTree

'''

model  = ExtraTreesClassifier(n_estimators = 100,

                    bootstrap = True,

                    max_features = 'sqrt',

                    oob_score = True)



model.fit(data_train,label_train)

label_test4 = model.predict(data_test)

score = accuracy_score(label_train,model.predict(data_train))

print('ExtraTree:accuracy_score is %f' % score) #train准确率

print('oob_score is %f' %model.oob_score_)

'''







###

'''

params_test = {'n_estimators':range(10,200,5)}

reg1 = GridSearchCV(estimator = model, param_grid = params_test, scoring = 'accuracy',cv = 5)

reg1.fit(data_train,label_train)

print("%s,%s"%(reg1.best_params_, reg1.best_score_))

'''





###output



df = pd.read_csv('../input/titanic/test.csv')

label_test = np.asarray(label_test2,'int64')

new_df = df.loc[:,['PassengerId']]

new_df['Survived'] = pd.DataFrame(label_test)



export_csv = new_df.to_csv('export_RandomForest.csv',index = None, header = True)
