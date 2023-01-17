# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from pandas import DataFrame

import re

import matplotlib.pyplot as plt

try:

    import seaborn as sns

except:

    !pip install seaborn

    import seaborn as sns

%matplotlib inline
sns.set_style('whitegrid')
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
train_data.head()
train_data.columns = train_data.columns.str.replace('-', '_').str.replace(': ', '')
train_data.columns
def format_vertical_headers(df):

    """Display a dataframe with vertical column headers"""

    styles = [dict(selector="th", props=[('width', '40px')]),

              dict(selector="th.col_heading",

                   props=[("writing-mode", "vertical-rl"),

                          ('transform', 'rotateZ(180deg)'), 

                          ('height', '290px'),

                          ('vertical-align', 'top')])]

    return (df.fillna('').style.set_table_styles(styles))
format_vertical_headers(train_data.head())
test_data.columns
test_data.columns = test_data.columns.str.replace('-', '_').str.replace(': ', '')

test_data.columns
format_vertical_headers(test_data.head())
print(train_data.info())

print("\n=======================================\n")

print(test_data.info())
format_vertical_headers(train_data.describe())
format_vertical_headers(test_data.describe())
#define function to draw boxplots of train and test data

def draw_boxplots_tran_test(df_train, df_test, column):

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3))

    sns.boxplot(x=train_data[column], ax=ax1)

    ax1.set_title("Train", fontsize=18)

    sns.boxplot(x=test_data[column], color="green", ax=ax2)

    ax2.set_title("Test", fontsize=18)

    plt.show()

    return 
#let's visualize data for each data column
draw_boxplots_tran_test(train_data,test_data,'RevolvingUtilizationOfUnsecuredLines')
draw_boxplots_tran_test(train_data,test_data,'NumberOfTime30_59DaysPastDueNotWorse')
draw_boxplots_tran_test(train_data,test_data,'DebtRatio')
draw_boxplots_tran_test(train_data,test_data,'MonthlyIncome')
fig = plt.figure(figsize=(8, 5))

ax = fig.add_subplot(111)



train_data[(train_data.MonthlyIncome.notnull())].MonthlyIncome.hist(bins=100, ax=ax)

plt.xlabel('MonthlyIncome')

plt.ylabel('Numper of people')

plt.title('Histogram of MonthlyIncome')
draw_boxplots_tran_test(train_data,test_data,'NumberOfOpenCreditLinesAndLoans')
draw_boxplots_tran_test(train_data,test_data,'NumberOfTimes90DaysLate')
draw_boxplots_tran_test(train_data,test_data,'NumberRealEstateLoansOrLines')
draw_boxplots_tran_test(train_data,test_data,'NumberOfTime60_89DaysPastDueNotWorse')
draw_boxplots_tran_test(train_data,test_data,'NumberOfDependents')
#the number of people who returned the credit

sns.countplot(x='SeriousDlqin2yrs',data=train_data)
#Most peoplt did not have serious due diligence in 2 years
def draw_heatmaps(df_train,df_test):

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18,8))

    sns.heatmap(df_train.isnull(), cmap='coolwarm', yticklabels=False, cbar=False, ax=ax1)

    ax1.set_title("Train", fontsize=18)

    sns.heatmap(df_test.isnull(), cmap='coolwarm', yticklabels=False, cbar=False, ax=ax2)

    ax1.set_title("Test", fontsize=18)

    return
#lets look at the number of missing data using a heatmap

draw_heatmaps(train_data,test_data)
train_data = train_data[train_data.RevolvingUtilizationOfUnsecuredLines < 15000]

test_data = test_data[test_data.RevolvingUtilizationOfUnsecuredLines < 15000]

draw_boxplots_tran_test(train_data,test_data,'RevolvingUtilizationOfUnsecuredLines')
train_data = train_data[(train_data.age >= 40) & (train_data.age < 65)]

test_data = test_data[(test_data.age >= 40) & (test_data.age < 65)]

draw_boxplots_tran_test(train_data,test_data,'age')
train_data = train_data[train_data.NumberOfTime30_59DaysPastDueNotWorse < 18]

test_data = test_data[test_data.NumberOfTime30_59DaysPastDueNotWorse < 18]

draw_boxplots_tran_test(train_data,test_data,'NumberOfTime30_59DaysPastDueNotWorse')
train_data = train_data[train_data.DebtRatio < 7500]

test_data = test_data[test_data.DebtRatio < 7500]

draw_boxplots_tran_test(train_data,test_data,'DebtRatio')
train_data = train_data[train_data.DebtRatio < 7500]

test_data = test_data[test_data.DebtRatio < 7500]

draw_boxplots_tran_test(train_data,test_data,'DebtRatio')
train_data = train_data[train_data.MonthlyIncome < 10000]

test_data = test_data[test_data.MonthlyIncome < 10000]

draw_boxplots_tran_test(train_data,test_data,'MonthlyIncome')
train_data = train_data[train_data.NumberOfOpenCreditLinesAndLoans < 20]

test_data = test_data[test_data.NumberOfOpenCreditLinesAndLoans < 20]

draw_boxplots_tran_test(train_data,test_data,'NumberOfOpenCreditLinesAndLoans')
train_data = train_data[train_data.NumberOfTimes90DaysLate < 20]

test_data = test_data[test_data.NumberOfTimes90DaysLate < 20]

draw_boxplots_tran_test(train_data,test_data,'NumberOfTimes90DaysLate')
train_data = train_data[train_data.NumberRealEstateLoansOrLines < 5]

test_data = test_data[test_data.NumberRealEstateLoansOrLines < 5]

draw_boxplots_tran_test(train_data,test_data,'NumberRealEstateLoansOrLines')
train_data = train_data[train_data.NumberOfTime60_89DaysPastDueNotWorse < 20]

test_data = test_data[test_data.NumberOfTime60_89DaysPastDueNotWorse < 20]

draw_boxplots_tran_test(train_data,test_data,'NumberOfTime60_89DaysPastDueNotWorse')
train_data = train_data[train_data.NumberOfDependents < 10]

test_data = test_data[test_data.NumberOfDependents < 10]

draw_boxplots_tran_test(train_data,test_data,'NumberOfDependents')
#let's see how data look after removing outliners

format_vertical_headers(train_data.describe())
format_vertical_headers(test_data.describe())
draw_heatmaps(train_data,test_data)
#specify the target variable

train_y = train_data.SeriousDlqin2yrs

test_y = test_data.SeriousDlqin2yrs
#create list of features

feature_names = ['RevolvingUtilizationOfUnsecuredLines', 'age',

       'NumberOfTime30_59DaysPastDueNotWorse', 'DebtRatio',

       'NumberOfOpenCreditLinesAndLoans', 'NumberOfTimes90DaysLate',

       'NumberRealEstateLoansOrLines', 'NumberOfTime60_89DaysPastDueNotWorse']

#create data corresponding to the features

train_X = train_data[feature_names]

test_X = test_data[feature_names]
#split train data into train and test set. I am not sure if this is necessary, 

#because we have a train test separately, but I am not sure why my attempts to use it fail.

#So I am just trying to work by example form other notebooks



from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(train_X, train_y, test_size = 0.3, random_state = 0)
from sklearn.linear_model import LogisticRegression

#specify the model, set any numeric valye as parameter to ensure reproducibility 

credit_model = LogisticRegression(random_state=1)



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

print(roc_auc)
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