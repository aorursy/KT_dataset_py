# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')
train_df.head(5)
test_df.tail(3)
#Now to get the features of the data
print(train_df.columns.values)
train_df['type'],test_df['type'] = 'training', 'testing'
print(train_df.columns.values)
test_df.insert(1, 'Survived', 0)
test_df.head(5)
combine_df = pd.concat([train_df,test_df])
combine_df.tail(4)
combine_df['womenchild'] = combine_df['Sex'].apply(lambda x: 1 if x == 'female' else 0) + combine_df['Age'].apply(lambda x: 1 if x <=18 else 0)
combine_df.tail(5)
combine_df['familytot'] = combine_df['SibSp'] + combine_df['Parch'] + 1
#combine_df.head(5)
combine_df.describe(include='all')
combine_df['surname'] = combine_df['Name'].str.extract(r'([a-zA-Z]+)',expand=False)
#combine_df['surname'].head(10)
combine_df['titlename'] = combine_df['Name'].str.extract(r' ([a-zA-Z]+)\.',expand=False)
#combine_df['titlename'].head(100)


combine_df['Aloneperson'] = combine_df["familytot"].apply(lambda x:1 if x <=1 else 0)
combine_df.titlename.unique()


combine_df['titlename'] = combine_df['titlename'].str.replace('Ms','Miss')

combine_df['titlename'] = combine_df['titlename'].replace(['Don','Dr','Mme','Major','Lady','Sir','Mlle','Col','Countess','Jonkheer','Dona'],'HighClass')
combine_df['Age']= combine_df["Age"].fillna(combine_df.groupby(by = combine_df['titlename'])['Age'].transform('mean'))
combine_df['womenchild'] = combine_df['Sex'].apply(lambda x: 1 if x == 'female' else 0) + combine_df['Age'].apply(lambda x: 1 if x <=18 else 0)
combine_df.head(5)
combine_df['Fare'] = combine_df['Fare'].fillna(combine_df.groupby(by=combine_df['Pclass'])['Fare'].transform('mean'))
combine_df['Cabin'] = combine_df['Cabin'].fillna(0)
combine_df['CabinString'] = combine_df['Cabin'].str.extract(r'([A-Za-z]+)', expand=False)
combine_df['HasCabin'] = combine_df['CabinString'].apply(lambda x: 0 if pd.isnull(x) else 1)
combine_df['Embarked'] = combine_df['Embarked'].fillna(combine_df['Embarked'].value_counts().idxmax())
combine_df['Survived'] = combine_df['Survived'].fillna(0).astype('int')
combine_df.head(3)
combine_df.info()
col_objects = combine_df.select_dtypes(['object']).columns
combine_df[col_objects] = combine_df[col_objects].astype('category')
#combine_df.info()
col_objects = combine_df.select_dtypes(['category']).columns
combine_df[col_objects] = combine_df[col_objects].apply(lambda x:x.cat.codes)
combine_df.info()
correlationmap = combine_df[combine_df['type'] ==1].corr()
fig,ax = plt.subplots(figsize=(20,15))
heatmap = sns.heatmap(correlationmap,annot = True,cmap = plt.cm.RdBu,fmt='.1f',square= True)
combine_df.head(10)
x = combine_df[combine_df['type']==1].drop(['PassengerId','Survived','Name','Ticket','Cabin','type'],axis =1)
x.info()
y = combine_df[combine_df['type']==1]['Survived']
y.head(5)
from sklearn.ensemble import RandomForestClassifier
forestcl = RandomForestClassifier(max_depth=99, n_estimators=2000, random_state=0).fit(x, y)
feat = pd.DataFrame(data=forestcl.feature_importances_, index=x.columns, columns=['FeatureImportances']).sort_values(['FeatureImportances'], ascending=False)
feat[feat['FeatureImportances'] > 0.01].index
x = combine_df[combine_df['type'] == 1][feat[feat['FeatureImportances'] > 0.01].index]
y = combine_df[combine_df['type'] == 1]['Survived']
x.head(5)
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
x_train, x_validate, y_train, y_validate = train_test_split(x, y, random_state=0, test_size=0.25)
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression().fit(x_train, y_train)
logreg_ypredict = logreg.predict(x_validate)
logreg_cvsc = cross_val_score(logreg, x, y, cv=5, scoring='accuracy')
print(logreg_cvsc.mean())

forestclf = RandomForestClassifier(max_depth=20, min_samples_split=5, n_estimators=250, random_state=0).fit(x_train, y_train)
forestclf_ypredict = forestclf.predict(x_validate)
forestclf_cvsc = cross_val_score(forestclf, x, y, cv=5, scoring='accuracy')
print(forestclf_cvsc.mean())
x_test = combine_df[combine_df['type'] == 0][feat[feat['FeatureImportances'] > 0.01].index]
y_test = pd.DataFrame(forestclf.predict(x_test), columns=['survived'])
x_test.head(5)
y_test.head(5)
x_test.info()
out = pd.DataFrame({'PassengerId': combine_df[combine_df['type'] == 0]['PassengerId'], 'Survived': y_test['survived']})
out.to_csv('submission.csv', index=False)