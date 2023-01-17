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
import pandas as pd

import numpy as np



train = pd.read_csv("../input/titanic/train.csv")

gender_model = pd.read_csv("../input/titanic/gender_submission.csv")

test = pd.read_csv("../input/titanic/test.csv")

#gender_class_model = pd.read_csv("../input/titanic/genderclassmodel.csv")
train.head()
train = train.fillna(value=0, method=None)

train = train.set_index("PassengerId")

train.head()
#Survived is out target variable

y_train = train['Survived']

X_train = train.iloc[:, train.columns != 'Survived']
print ("lenght of X_train before split :" , len(X_train))

print ("lenght of y_train before split :" , len(y_train))
#spliting the data for further analysis

from sklearn.model_selection import train_test_split

X_train1,X_test1,y_train1,y_test1 = train_test_split(X_train,y_train,random_state=0)



print ('length of X_train1 after split :' , len(X_train1))

print ('length of X_test1 after split :' , len(X_test1))
X_train1['Sex']


from sklearn import preprocessing

from sklearn.preprocessing import MinMaxScaler



le = preprocessing.LabelEncoder()

a1 = le.fit(X_train1['Sex'])



X_train1['Sex1'] = a1.transform(X_train1.loc[:,'Sex'])

a2 = le.fit(X_train1.loc[:,'Pclass'])

X_train1['Pclass1'] = a2.transform(X_train1.loc[:,'Pclass'])



c1 = le.fit(X_test1.loc[:,'Sex'])

X_test1['Sex1'] = c1.transform(X_test1.loc[:,'Sex'])

c2 = le.fit(X_test1.loc[:,'Pclass'])

X_test1['Pclass1'] = c2.transform(X_test1.loc[:,'Pclass'])



X_train1 = X_train1.loc[:,('Age', 'SibSp','Parch','Fare','Sex1', 'Pclass1')]

X_test1 = X_test1.loc[:,('Age', 'SibSp','Parch','Fare','Sex1', 'Pclass1')]



#preprocessing for test data 

b1 = le.fit(test.loc[:,'Sex'])

test['Sex1'] = b1.transform(test.loc[:,'Sex'])

b2 = le.fit(test.loc[:,'Pclass'])

test['Pclass1'] = b2.transform(test.loc[:,'Pclass'])





scaler = MinMaxScaler()

X_train1 = scaler.fit_transform(X_train1)

# we must apply the scaling to the test set that we computed for the training set

X_test1 = scaler.transform(X_test1)

#test_new2 = scaler.transform(test_new1)
# #Preprocessing the data before model fit



# from sklearn import preprocessing

# from sklearn.preprocessing import MinMaxScaler



# le = preprocessing.LabelEncoder()

# a1 = le.fit(X_train1.loc[:,'Sex']

# X_train1['Sex1'] = a1.transform(X_train1.loc[:,'Sex'])

# a2 = le.fit(X_train1.loc[:,'Pclass'])

# X_train1['Pclass1'] = a2.transform(X_train1.loc[:,'Pclass'])



# c1 = le.fit(X_test1.loc[:,'Sex'])

# X_test1['Sex1'] = c1.transform(X_test1.loc[:,'Sex'])

# c2 = le.fit(X_test1.loc[:,'Pclass'])

# X_test1['Pclass1'] = c2.transform(X_test1.loc[:,'Pclass'])



# X_train1 = X_train1.loc[:,('Age', 'SibSp','Parch','Fare','Sex1', 'Pclass1')]

# X_test1 = X_test1.loc[:,('Age', 'SibSp','Parch','Fare','Sex1', 'Pclass1')]



# #preprocessing for test data 

# b1 = le.fit(test.loc[:,'Sex'])

# test['Sex1'] = b1.transform(test.loc[:,'Sex'])

# b2 = le.fit(test.loc[:,'Pclass'])

# test['Pclass1'] = b2.transform(test.loc[:,'Pclass'])





# scaler = MinMaxScaler()

# X_train1 = scaler.fit_transform(X_train1)

# # we must apply the scaling to the test set that we computed for the training set

# X_test1 = scaler.transform(X_test1)

# #test_new2 = scaler.transform(test_new1)



#Applying logistic regression to get output if person survived or not

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import roc_curve, auc , roc_auc_score



clf = LogisticRegression().fit(X_train1, y_train1)





y_scor1 = clf.decision_function(X_test1)

fpr_grad, tpr_grad, _ = roc_curve(y_test1, y_scor1)





print("Model accuracy Score : " , clf.score(X_test1, y_test1))

print("Model AUC Score : " , auc(fpr_grad, tpr_grad))
#further enhancing the model



clf = LogisticRegression(penalty='l1',C=10,class_weight='balanced').fit(X_train1, y_train1)





y_scor1 = clf.decision_function(X_test1)

fpr_grad, tpr_grad, _ = roc_curve(y_test1, y_scor1)





print("Model accuracy Score : " , clf.score(X_test1, y_test1))

print("Model AUC Score : " , auc(fpr_grad, tpr_grad))

#predcting if the Passenger survived or not

test = test[['Age', 'SibSp','Parch','Fare','Sex1', 'Pclass1']]

test = test.fillna(value=0, method=None)

i=test.index

j=clf.predict_proba(test)

prob = j[:,0]

series1 = pd.Series(prob,index=i)

df = pd.DataFrame({"Survival probability": prob})

df.index.name='PassengerID'

print("probability that Passenger survived :\n ",df)