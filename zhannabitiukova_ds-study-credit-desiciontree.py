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
train_file_path = '../input/GiveMeSomeCredit/cs-training.csv'

test_file_path = '../input/GiveMeSomeCredit/cs-test.csv'

train_data = pd.read_csv(train_file_path)

test_data = pd.read_csv(test_file_path)
#print columns headers of the dataset

train_data.columns
#drop lines with missing data (NA)

train_data.dropna(axis=1)

cols_with_missing = [col for col in train_data.columns

                         if train_data[col].isnull().any()]

reduced_train_data = train_data.drop(cols_with_missing, axis=1)

reduced_test_data = test_data.drop(cols_with_missing,axis=1)
reduced_train_data.columns
#specify the target variable

train_y = reduced_train_data.SeriousDlqin2yrs

test_y = reduced_test_data.SeriousDlqin2yrs
#create list of features

feature_names = ['RevolvingUtilizationOfUnsecuredLines', 'age',

       'NumberOfTime30-59DaysPastDueNotWorse', 'DebtRatio',

       'NumberOfOpenCreditLinesAndLoans', 'NumberOfTimes90DaysLate',

       'NumberRealEstateLoansOrLines', 'NumberOfTime60-89DaysPastDueNotWorse']

#create data corresponding to the features

train_X = reduced_train_data[feature_names]

test_X = reduced_test_data[feature_names]
#review data

print(train_X.describe)

print(train_X.head)
#split train data into train and test set. I am not sure if this is necessary, 

#because we have a train test separately, but I am not sure why my attempts to use it fail.

#So I am just trying to work by example form other notebooks



from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(train_X, train_y, test_size = 0.3, random_state = 0)
from sklearn.tree import DecisionTreeRegressor

#specify the model, set any numeric valye as parameter to ensure reproducibility 

credit_model = DecisionTreeRegressor(random_state=1)



#fit the model

credit_model.fit(x_train,y_train)
#make predictions

predictions_train = credit_model.predict(x_train)

y_pred = credit_model.predict(x_test)
#this section investigates resulting data, I had to do this because confusion matrix was throwing errors

print(predictions_train)

print(y_pred)

print(y_pred.shape)

print(y_pred.dtype)

print(y_test.shape)

print(y_test.dtype)
#conver float to int

y_predi = y_pred.astype(int)
#create confuson matrics in text view

from sklearn.metrics import confusion_matrix

tn, fp, fn, tp = confusion_matrix( y_test,y_predi).ravel()

(tn, fp, fn, tp)
#create consusion matrix and plot

import scikitplot as skplt

skplt.metrics.plot_confusion_matrix(y_test,y_predi,figsize=(6,6))
from sklearn import metrics

#calculate ROC

fpr, tpr, thresholds = metrics.roc_curve(y_test, y_predi)

print(fpr)

print(fpr.shape)

print(tpr)

print(tpr.shape)

print(thresholds)
from sklearn import metrics

#calculate AUC

roc_auc = metrics.auc(fpr, tpr)
# method I: plt

import matplotlib.pyplot as plt

plt.title('Receiver Operating Characteristic')

plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)

plt.legend(loc = 'lower right')

plt.plot([0, 1], [0, 1],'r--')

plt.xlim([0, 1])

plt.ylim([0, 1])

plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate')

plt.show()
#calculate F1 score

from sklearn.metrics import f1_score

f1_score(y_test, y_predi) #do we need to use average=None as the third param?
#accuracy score

from sklearn.metrics import accuracy_score

accuracy_score(y_test, y_predi) #do we need to use normalize=False as the third param?
#precision

from sklearn.metrics import precision_score

precision_score(y_test, y_predi)
#recall

from sklearn.metrics import recall_score

recall_score(y_test, y_predi)
#cost-sensitive accuracy

fp_cost = 1

fn_cost = 0

cost_sensitive_accuracy = (tp + tn) / (tp + tn + fp*fp_cost + fn*fn_cost)

print(cost_sensitive_accuracy)