# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input/testing'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
from sklearn import svm 

import pandas as pd

from sklearn.metrics import accuracy_score

import warnings

warnings.filterwarnings('ignore')

from matplotlib import pyplot as plt

import numpy as np

train=pd.read_csv('/kaggle/input/testing/c_4.csv')

by_class_0 = train.groupby({'A64':['0']})

by_class_1 = train.groupby({'A64':['1']})

by_class_2 = train.groupby({'A64':['2']})

by_class_3 = train.groupby({'A64':['3']})

by_class_4 = train.groupby({'A64':['4']})

by_class_5 = train.groupby({'A64':['5']})

by_class_6 = train.groupby({'A64':['6']})



df_count = by_class_0





target_count = train.A64.value_counts()

print('class 3',target_count[3])

print('class 5',target_count[4])





target_count.plot(kind='bar', title='Count (A64)');
#correlations_data =train.corr()['A64']

#print(correlations_data)



#for x in correlations_data:

 #   print(x)
#X=train.drop(['A11','A12','A18','A19','A20','A21','A26','A27','A28','A29','A34','A35','A36','A37','A42','A42','A43','A44','A45','A51','A52'],axis=1)

#print(X)
from sklearn.model_selection import train_test_split

y=train['A64']

from sklearn.model_selection import cross_val_score

train1=train.drop(['A64'],axis=1)

X_train, X_test, y_train, y_test = train_test_split(train1, y, test_size=0.85, random_state=1,shuffle=True)



target_count.plot(kind='bar', title='Count (A64)');

target_count = train.A64.value_counts()

print('class 3',target_count[3])

print('class 4',target_count[4])
from sklearn.svm import SVC

from sklearn import metrics

from sklearn.metrics import confusion_matrix 

from sklearn.metrics import f1_score

classifier_linear = SVC(kernel='linear',gamma='auto', random_state = 1)

scores = cross_val_score(classifier_linear, train1, y, cv=5)

print(scores)

classifier_rbf = SVC(kernel='rbf',gamma='auto',random_state=1)

scores1 = cross_val_score(classifier_rbf, train1, y, cv=5)

print(scores1)

classifier_rbf.fit(train1,y)

classifier_linear.fit(train1,y)

Y_pred_rbf=classifier_rbf.predict(X_test)

Y_linear = classifier_linear.predict(X_test)

results_=confusion_matrix(Y_linear,y_test)

print(results_)

results_1=confusion_matrix(Y_pred_rbf,y_test)

print(results_1)

print("Accuracy_linear:",metrics.accuracy_score(y_test, Y_linear))



print(classifier_linear.n_support_)

print(classifier_rbf.n_support_)

print (classifier_linear.support_vectors_)

f1_score(y_test,Y_linear, average='macro') 

print('class 3',target_count[3])

print('class 4',target_count[4])
from sklearn.metrics import classification_report

target_names=['class 3','class 4']



print(classification_report(y_test,Y_linear, target_names=target_names))

print (classifier_linear.support_)
df=classifier_linear.support_

train_2=train.drop(train.index[[ 27,48,108,114,149,174,176,177,178,179,211,309,320,337,391,406,410,431,474,524,529,531,552,608,611,632,633,647 , 685 , 719  ,729 , 738 , 748  ,751 , 758 , 850 , 853,

  929 , 962 , 999 ,1055 ,1088 ,1090 ,1105, 1131 ,1133 ,1135 ,1160 ,1190 ,1222 ,1227,1297 ,1385 ,1404 ,1409 ,1410 ,1458 ,1474 ,1581 ,1590 ,1592, 1618 ,1639 ,1646 ,1685 ,1711 ,1714 ,1716, 1737 ,1751 ,1752 ,1837 ,1911 ,1960 ,2004]])

print(train_2)



target_count = train_2.A64.value_counts()

print(target_count[3])

print(target_count[4])
train3=train_2.drop(['A64'],axis=1)

y_1=train_2['A64']

X_train, X_test, y_train, y_test = train_test_split(train3, y_1, test_size=0.85, random_state=1,shuffle=True)

classifier_linear1 = SVC(kernel='linear',gamma='auto', random_state = 1)

scores = cross_val_score(classifier_linear, train3, y_1, cv=10)

print(scores)





classifier_linear.fit(train3,y_1)

Y_linear_1 = classifier_linear.predict(X_test)

results_=confusion_matrix(Y_linear_1,y_test)

print(results_)

from sklearn.metrics import classification_report

target_names=['class 3','class 4']



print(classification_report(y_test,Y_linear_1, target_names=target_names))
from sklearn.metrics import classification_report

target_names=['class 3','class 4']

print(classification_report(y_test,Y_linear_1, target_names=target_names))            
from imblearn.over_sampling import SMOTE 

sm = SMOTE(kind='regular',k_neighbors=2,random_state=78) 

X_train_res, y_train_res = sm.fit_sample(X_train, y_train.ravel()) 

print('After OverSampling, the shape of train_X: {}'.format(X_train_res.shape)) 

print('After OverSampling, the shape of train_y: {} \n'.format(y_train_res.shape)) 

  

print("After OverSampling, counts of label '1': {}".format(sum(y_train_res == 3))) 

print("After OverSampling, counts of label '0': {}".format(sum(y_train_res == 4))) 
from sklearn.linear_model import LogisticRegression 

lr1 = LogisticRegression() 

lr1.fit(X_train_res, y_train_res.ravel()) 

predictions = lr1.predict(X_test) 

# print classification report 

print(classification_report(y_test, predictions)) 
from imblearn.under_sampling import NearMiss 

nr = NearMiss() 

  

X_train_miss, y_train_miss = nr.fit_sample(X_train, y_train.ravel()) 

print('After Undersampling, the shape of train_X: {}'.format(X_train_miss.shape)) 

print('After Undersampling, the shape of train_y: {} \n'.format(y_train_miss.shape)) 

  

print("After Undersampling, counts of label '1': {}".format(sum(y_train_miss == 3))) 

print("After Undersampling, counts of label '0': {}".format(sum(y_train_miss == 4))) 
from sklearn.metrics import confusion_matrix

from matplotlib import pyplot as plt







labels = ['3','4']

fig = plt.figure()

ax = fig.add_subplot(111)

cax = ax.matshow(conf_mat, cmap=plt.cm.Blues)

fig.colorbar(cax)

ax.set_xticklabels([''] + labels)

ax.set_yticklabels([''] + labels)

plt.xlabel('Predicted')

plt.ylabel('Expected')

plt.show()
count_class_3,count_class_4 = train.A64.value_counts()



# Divide by class

class_3 = train[train['A64'] == 3]

class_4 = train[train['A64'] == 4]



print(count_class_3)

print(count_class_4)
def plot_2d_space(X, y, label='Classes'):   

    colors = ['#1F77B4', '#FF7F0E']

    markers = ['o', 's']

    for l, c, m in zip(np.unique(y), colors, markers):

        plt.scatter(

            X[y==l, 0],

            X[y==l, 1],

            c=c, label=l, marker=m

        )

    plt.title(label)

    plt.legend(loc='upper right')

    plt.show()
class_0_under = class_0.sample(count_class_1,replace='true')

df_test_under = pd.concat([class_0_under, class_1], axis=0)

print(df_test_under.A64.value_counts())



df_test_under.A64.value_counts().plot(kind='bar', title='Count (A64)')
from sklearn.feature_selection import RFECV

from sklearn.svm import SVR

model = SVC(kernel='linear',gamma='auto')

rfe = RFECV(model,step=1,cv=5)

RFE_X_Train = rfe.fit_transform(train,y)

RFE_X_Test = rfe.transform(X_test)

rfe.fit(RFE_X_Train,y)

RFE_X_Test=rfe.predict(RFE_X_Test)



print("Overall Accuracy using RFE: ", metrics.accuracy_score(RFE_X_Test,y_test))