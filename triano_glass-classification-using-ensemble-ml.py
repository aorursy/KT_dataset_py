#import python packages 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
import sklearn
from sklearn.model_selection import train_test_split  
from sklearn.metrics import accuracy_score 
%matplotlib inline
#import dataset from draft environment
data = pd.read_csv('../input/glass.csv')
data.head()
data.info()
#correlation of each the datasets 
corr = data.corr()
plt.figure(figsize=(12,12))
sns.heatmap(corr, cbar = True,  square = True, annot=True, fmt= '.2f',annot_kws={'size': 15}
            , alpha = 0.7, cmap= 'coolwarm')
plt.show()
# make boxplot to correction is there outlier or no
# you can repeat this code for all feature
fig, axes = plt.subplots(nrows=2,ncols=2)
fig.set_size_inches(10,10)
sns.boxplot(x=data['RI'],color = 'blue', ax=axes[0][0])
sns.boxplot(x=data['Na'],color = 'Red', ax=axes[0][1])
sns.boxplot(x=data['Mg'],color = 'Green', ax=axes[1][0])
sns.boxplot(x=data['Al'],color = 'Orange', ax=axes[1][1])
dt = data['Type'].value_counts()
print ('The number of each Type class = \n')
print (dt)
sns.countplot(data['Type'])
plt.show()
#import packages for imbalance-learn for balancing class
from imblearn.over_sampling import SMOTE

from sklearn.model_selection import train_test_split

x = data.drop('Type', axis=1)
y = data['Type']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

print("Number  X_train dataset: ", x_train.shape)
print("Number y_train dataset: ", y_train.shape)
print("Number X_test dataset: ", x_test.shape)
print("Number y_test dataset: ", y_test.shape)
print("Before OverSampling, counts of label '1': {}".format(sum(y_train==1)))
print("Before OverSampling, counts of label '2': {}".format(sum(y_train==2)))
print("Before OverSampling, counts of label '3': {}".format(sum(y_train==3)))
print("Before OverSampling, counts of label '5': {}".format(sum(y_train==5)))
print("Before OverSampling, counts of label '6': {}".format(sum(y_train==6)))
print("Before OverSampling, counts of label '7': {} \n".format(sum(y_train==7)))

sm = SMOTE(random_state=2)
x_train_res, y_train_res = sm.fit_sample(x_train, y_train.ravel())

print('After OverSampling, the shape of train_X: {}'.format(x_train_res.shape))
print('After OverSampling, the shape of train_y: {} \n'.format(y_train_res.shape))

print("After OverSampling, counts of label '1': {}".format(sum(y_train_res==1)))
print("After OverSampling, counts of label '2': {}".format(sum(y_train_res==2)))
print("After OverSampling, counts of label '3': {}".format(sum(y_train_res==3)))
print("After OverSampling, counts of label '5': {}".format(sum(y_train_res==5)))
print("After OverSampling, counts of label '6': {}".format(sum(y_train_res==6)))
print("After OverSampling, counts of label '7': {}".format(sum(y_train_res==7)))
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

RFC = RandomForestClassifier(n_estimators = 500, criterion = 'entropy', random_state = 42, max_depth = 10 )
RFC.fit(x_train_res, y_train_res.ravel())

#predict 
pred_train = RFC.predict(x_train_res)
#Confusion Matrik of train dataset  
print(confusion_matrix(y_train_res,pred_train))
print ('\n')
print(classification_report(y_train_res,pred_train))
# Confusion Matriks of Test Dataset  
Pred_RFC =RFC.predict(x_test)

print('Confusion Matrix : ','\n',confusion_matrix(y_test,Pred_RFC))
print ('\n')
print(classification_report(y_test,Pred_RFC))
print('\n')
print ('Accuracy_R.Forest_Classifier : ', 
                     accuracy_score(y_test,Pred_RFC)*100,'%')
from sklearn.ensemble import BaggingClassifier
BS = BaggingClassifier(RandomForestClassifier(), n_estimators = 300 )
BS.fit(x_train_res, y_train_res.ravel())

#predict 
pred_train_BS = BS.predict(x_train_res)
pred_test_BS = BS.predict(x_test)
# Confusion Matriks of Test Dataset  
print('Confusion Matrix : ','\n',confusion_matrix(y_test,pred_test_BS))
print ('\n')
print(classification_report(y_test,pred_test_BS))
print('\n')
print ('Accuracy_Bagging Classifier : ', 
                     accuracy_score(y_test,pred_test_BS)*100,'%')
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
AB = AdaBoostClassifier(DecisionTreeClassifier(), n_estimators = 300 )
AB.fit(x_train_res, y_train_res.ravel())

#predict 
pred_train_AB = AB.predict(x_train_res)
pred_test_AB = AB.predict(x_test)
# Confusion Matriks of Test Dataset  
print('Confusion Matrix : ','\n',confusion_matrix(y_test,pred_test_AB))
print ('\n')
print(classification_report(y_test,pred_test_AB))
print('\n')
print ('Accuracy_ AdaBoost Classifier : ', 
                     accuracy_score(y_test,pred_test_AB)*100,'%')
# split dataset to be train and test
train = pd.concat([x_train,y_train], axis = 1)
print(train.head())
test = pd.concat([x_test,y_test], axis = 1)
print(test.head())
dt=train['Type'].groupby(train['Type']).count()

print ('The number of each Type class = \n')
print (dt)
# we will calculate for each the number of class
C3 = train[train['Type']==3]
C3 = pd.concat([C3]*5)

C5 = train[train['Type']==5]
C5 = pd.concat([C5]*5)

C6 = train[train['Type']==6]
C6 =pd.concat([C6]*8)

C7 = train[train['Type']==7]
C7 = pd.concat([C7]*2)

C1 = train[train['Type']==1]

C2 = train[train['Type']==2]

#Combain of every dataframe above with new variable name 
data_balanced=pd.concat([C1,C2,C3,C5,C6,C7])
data_balanced.head()

data_balanced.shape
type=data_balanced['Type'].groupby(data_balanced['Type']).count()
type
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

x_train_A = data_balanced.drop('Type', axis=1)
y_train_A = data_balanced['Type']

RFC = RandomForestClassifier(n_estimators = 300, criterion = 'entropy', random_state = 42, max_depth = 10 )
RFC.fit(x_train_A, y_train_A)
pred_train = RFC.predict(x_train_A)
print(confusion_matrix(y_train_A,pred_train))
print ('\n')
print(classification_report(y_train_A,pred_train))
#predict test dataset 
x_test_A = test.drop('Type', axis=1)
y_test_A = test['Type']

pred_test = RFC.predict(x_test_A)
print('Confusion Matrix : ','\n',confusion_matrix(y_test_A,pred_test))
print ('\n')
print(classification_report(y_test_A,pred_test))

print ('Accuracy_R.Forest_Classifier_B : ', 
                     accuracy_score(y_test_A,pred_test)*100,'%')
from sklearn.ensemble import BaggingClassifier
bs = BaggingClassifier(RandomForestClassifier(), n_estimators = 300 )
bs.fit(x_train_res, y_train_res.ravel())

#predict 
pred_train_bs = bs.predict(x_train_res)
pred_test_bs = bs.predict(x_test)
# Confusion Matriks of Test Dataset  
print('Confusion Matrix : ','\n',confusion_matrix(y_test,pred_test_bs))
print ('\n')
print(classification_report(y_test,pred_test_bs))
print('\n')
print ('Accuracy_Bagging Classifier : ', 
                     accuracy_score(y_test,pred_test_bs)*100,'%')
