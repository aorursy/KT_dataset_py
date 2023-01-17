# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# read files
path = '../input'
test = pd.read_csv(path + '/test.csv')
test_shape = test.shape
print('test_shape',test_shape)

train = pd.read_csv(path + '/train.csv')
train_shape = train.shape
print('train_shape',train_shape)
train.head(10)
# segment data by sex and calculate the mean. compare the relationship between Survived and Sex
import matplotlib.pyplot as plt

sex_pivot = train.pivot_table(index='Sex',values = 'Survived')
sex_pivot # from the result shows, females survived in higher proportions than males did.
# segment data by Pclass.

pclass_pivot = train.pivot_table(index='Pclass', values = 'Survived')
#pclass_pivot
pclass_pivot.plot.bar()
plt.show() # the survived factor also relates to classes.
# Sex and PClass are categorical features. Now let's check the age column. Age column is a continuous numerical column. 
#One way to look at distribution of values in a continuous numerical set is to use histograms. 
#We can create two histograms to compare visually the those that survived vs those who died across different age ranges:
train['Age'].describe()
# Age is fractional if passengers are less than one. Some ages info are missing. 
train[train['Survived'] == 1]
survived = train[train['Survived'] == 1]
died = train[train['Survived'] == 0]

survived['Age'].plot.hist(alpha=0.5, color='red', bins=50)
#plt.legend()
#plt.show()
died['Age'].plot.hist(alpha=0.5, color='blue', bins=50)
plt.legend(['Survived','Died'])
plt.show()
#plt.show()
# separate continuous feature into a categorical feature by dividing it into ranges
def process_age(df, cut_points, label_names):
    df['Age'] = df['Age'].fillna(-0.5)  # filling the missing data, Replace all NaN elements.
    df['Age_categories'] = pd.cut(df['Age'], cut_points, labels=label_names)
    return df

cut_points = [-1, 0, 5, 12, 18, 35, 60, 100]
label_names = ['Missing', 'Infant', 'Child', 'Teenager', 'Young Adult', 'Adult', 'Senior']

train = process_age(train, cut_points, label_names)
test = process_age(test, cut_points, label_names)

#train.head(10)
age_cat_pivot = train.pivot_table(index='Age_categories', values='Survived')
#train.pivot_table(index='Age_categories', values='Survived')
age_cat_pivot.plot.bar()
plt.show()

# prepare the colums for machine learning. convert our values into numbers
#train['Pclass'].value_counts()

#column_name = 'Pclass'
#df = train
#dummies = pd.get_dummies(df[column_name], prefix=column_name)
#dummies.head()

def create_dummies(df,column_name):
    dummies = pd.get_dummies(df[column_name], prefix=column_name)
    df = pd.concat([df, dummies], axis=1)
    return df

train = create_dummies(train, 'Pclass')
test = create_dummies(test,'Pclass')

train = create_dummies(train, 'Sex')
test = create_dummies(test,'Sex')

train = create_dummies(train, 'Age_categories')
test = create_dummies(test,'Age_categories')

train.head()
#train.shape
# Now, the data has been repared. 
# Logistics Regression
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
# using fit() method to train model.
columns = ['Pclass_1', 'Pclass_2', 'Pclass_3', 'Sex_female', 'Sex_male',
       'Age_categories_Missing','Age_categories_Infant',
       'Age_categories_Child', 'Age_categories_Teenager',
       'Age_categories_Young Adult', 'Age_categories_Adult',
       'Age_categories_Senior']
lr.fit(train[columns], train['Survived'])

#Congratulations, you've trained your first machine learning model! 
#Our next step is to find out how accurate our model is, and to do that, we'll have to make some predictions.
#train.shape
# find Out accurate.
# split train dataframe into two 
from sklearn.model_selection import train_test_split

holdout = test # for discretion the test, we rename it as holdout 

columns = ['Pclass_1', 'Pclass_2', 'Pclass_3', 'Sex_female', 'Sex_male',
       'Age_categories_Missing','Age_categories_Infant',
       'Age_categories_Child', 'Age_categories_Teenager',
       'Age_categories_Young Adult', 'Age_categories_Adult',
       'Age_categories_Senior']

all_X = train[columns]
all_y = train['Survived']

# test_size control the proportions of data are split into,
train_X, test_X, train_y, test_y = train_test_split(all_X, all_y, test_size = 0.2, random_state = 0)

train_X.shape

from sklearn.metrics import accuracy_score

lr = LogisticRegression()
lr.fit(train_X, train_y)
predictions = lr.predict(test_X)

accuracy = accuracy_score(test_y, predictions)
accuracy
from sklearn.metrics import confusion_matrix

conf_matrix = confusion_matrix(test_y, predictions)
pd.DataFrame(conf_matrix, columns=['Survived','Died'], index=[['Survived','Died']])
# get more accurate error measurement by cross validation
# this test data is quite small so it probabily overfitting. We can use cross validation to train and test our model
from sklearn.model_selection import cross_val_score
import numpy as np

scores = cross_val_score(lr, all_X, all_y, cv=10)
np.mean(scores)

#holdout.head()
lr.fit(all_X, all_y)
holdout_predictions = lr.predict(holdout[columns])
holdout_predictions
# submission
holdout_ids = holdout['PassengerId']
submission_df = {'PassengerId': holdout_ids, 
                 'Survived': holdout_predictions}
submission = pd.DataFrame(submission_df)
submission.to_csv('titanic_submission.csv', index=False)