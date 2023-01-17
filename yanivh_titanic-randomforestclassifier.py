# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
# Load the data

train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

train.head(5)
train.groupby('Sex').Survived.value_counts()
train.groupby(['Pclass','Sex']).Survived.value_counts()
id = pd.crosstab([train.Pclass,train.Sex],train.Survived.astype(float))

id.div(id.sum(1).astype(float),0)
#all the data to be in numerical format. As we can see below, our data set has 5 categorical variables which contain non-numerical values: Name, Sex, Ticket, Cabin and Embarked.

train.dtypes
#We then check the number of levels that each of the five categorical variables have.

for cat in ['Name', 'Sex', 'Ticket', 'Cabin', 'Embarked']:

    print("Number of levels in category'{0}' : {1:2.2f}".format(cat,train[cat].unique().size))
#explor Sex & Embarked

for cat in ['Sex', 'Embarked']:

    print ('category  :{} - Values : {}'.format(cat,train[cat].unique()))
#map Sex & Embarked

train['Sex'] = train['Sex'].map({'male':0,'female':1})

train['Embarked']=train['Embarked'].map({'S':0,'C':1,'Q':2})

train.head()
#For nan i.e. the missing values, we simply replace them with a placeholder value (-999). In fact, we perform this replacement for the entire data set

train=train.fillna(-999)

pd.isnull(train).any()
train.head()
#For Cabin, we encode the levels as digits using Scikit-learn's MultiLabelBinarizer and treat them as new features.

from sklearn.preprocessing import MultiLabelBinarizer

mlb= MultiLabelBinarizer()

CabinTrans = mlb.fit_transform([{str(val)} for val in train['Cabin'].values ])

CabinTrans
#Name and Ticket have so many levels , drop 

#Survives drop target 

#Cabin drop - set new feature

train_new = train.drop(['Name','Ticket','Cabin','Survived'],axis=1)
#check correct encoding done

assert (len(train['Cabin'].unique()) == len(mlb.classes_)),"not equal"
train_new= np.hstack((train_new.values,CabinTrans))
print (np.isnan(train_new).any())

print (train_new[0].size)
#store the Survived labels, which we need to predict, in a separate variable.

train_class=train['Survived'].values
from sklearn.model_selection import train_test_split

training_features, testing_features, training_classes, testing_classes = train_test_split(train_new, train_class, train_size=0.75, test_size=0.25)
from sklearn.ensemble import RandomForestClassifier

export_pipline = RandomForestClassifier(bootstrap=False,max_features=0.4,min_samples_leaf=1,min_samples_split=9)

export_pipline.fit(training_features,training_classes)

result = export_pipline.predict(testing_features)

result
s_prob = pd.DataFrame(export_pipline.predict_proba(testing_features),columns=['P({})'.format(x) for x in export_pipline.classes_])

s_prob.head()

print(export_pipline.score(train_new,train_class))
export_pipline.classes_
s_prob= pd.DataFrame(export_pipline.predict_proba(training_features),columns=['P({})'.format(x) for x in export_pipline.classes_] )

s_prob
#prediction on test data

test.describe()
test.dtypes
for cat in ['Cabin','Name','Sex','Embarked','Ticket']:

    print ("unique values in {} : {}".format(cat,test[cat].unique().size))
for var in ['Cabin','Name','Ticket']:

    new = list(set(test[var])- set(train[var]))

    test.ix[test[var].isin(new) ,var] =-999
test['Sex'] =test['Sex'].map({'male':0,'female':1})

test['Embarked'] = test['Embarked'].map({'S':0,'C':1,'Q':2})
test=test.fillna(-999)

pd.isnull(test).any()
from sklearn.preprocessing import MultiLabelBinarizer

mlb= MultiLabelBinarizer()

subCabinTrans=mlb.fit([{str(val)} for val in train['Cabin'].values]).transform([{str(val)} for val in test['Cabin'].values ])
test = test.drop(['Name','Ticket','Cabin'], axis=1)


# Form the new submission data set

test_new = np.hstack((test.values,subCabinTrans))
# Ensure equal number of features in both the final training and submission dataset

assert (train_new.shape[1] == train_new.shape[1]), "Not Equal"
submission  = export_pipline.predict(test_new)

submission
# Create the submission file

result = pd.DataFrame({'PassengerId':test['PassengerId'], 'Survived': submission})

result
result.to_csv('submission.csv',index=False)