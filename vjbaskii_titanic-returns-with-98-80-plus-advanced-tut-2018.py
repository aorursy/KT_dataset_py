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

# we go two type of data 1.train data 2. test data
# step
#     acquire and load the data
#     overview the data(summary head tail info) and understading the features/columns
#     check for missing/improper data's in the dataframes and correct them (isnull dropna fillna)
#     check for correlation between the column/features to output column (correlation ploting)
#     alter the column data if needed(feature scaling-increase performances, merge data's 
#     train the data (make the computer understand the data)
#     predict with the new data (best method is to try many algorithms)
    
# load the data which are to be processed.
df_train = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')
df_gender_submission = pd.read_csv('../input/gender_submission.csv')

# Yeah do check your dimension of the dataframes loaded. understand the data present. it help us to choose algorithm( based on type of data)
print(df_train.shape)
df_train.head(2)
print(df_test.shape)
df_test.head(2)
# Survived column gone missing in test data
print(df_gender_submission.shape)  
print(df_gender_submission.head(2)) 

# it seem we got test output(survived) as a seperate dataframe(df_gender_submission)
# so me gonna merge df_gender_submission with test data
df_test = pd.merge(df_test,df_gender_submission , how='left', on=['PassengerId'])  
df_test.tail() 

# Just overview the data we got. 	Name	Sex	Age	SibSp	Parch	Ticket	Fare	Cabin	Embarked	Survived
# just checking column datatype.int= integer(1 or 5878787) note: float is number with decimal values ie 1.2 or 0.0038
# object datatypes are string ie something like "your name". 
df_test.info()
# we checking for missing values in each column because some data manupulation/modifying_functions does not take null, NA, nill, - etc eg SVM algorithm.
# But filling missing values with mean/median/mode is also a prediction which may not be 100% accurate, 
# instead you can use models like Decision Trees and Random Forest which handle missing values very well.

# find the no. of missing values in train data.
df_train.isnull().sum()
# find the no. of missing values in test data.
df_test.isnull().sum()
# exploting data, we can see that survived mean is 0.383838
# which represent's 38 percent survived logically.
df_train.describe()
# we use corr() function to see which feature(Sex Pclass SibSpBool ParchBool FamilyBool Fare) contribute to the output(survived).
# Looks like Pclass has got highest negative correlation with "Survived" followed by Fare, Parch and Age
df_train.corr()["Survived"].sort_values()
# here data manupulation starts we are trying to make compilation faster and simpler.
# what ever you do to train data replicate it for test data. otherwise it may lead to wrong calculation.
# changing the male female string to 0 and 1 ie boolean datatype. not necessary for small amounts of data.
df_train["Sex"].loc[df_train["Sex"] == 'female'] = 0
df_train["Sex"].loc[df_train["Sex"] == 'male'] = 1
# df_train[df_train["Survived"] == 1].describe()
# converting the age 0-15yr as 0 and 15-80 as 1,ie 1 is older and 0 is young.
df_train['AgeRange'] = pd.cut(df_train['Age'], [0, 15, 80], labels=[0, 1])
# merging two columns with similar data for predictions. family count/size equals sibiling + parch
df_train['FamilyCount'] = (df_train['SibSp'] + df_train['Parch'])
# 1 if parch > 0 or sibiling > 0 ie if passenger is alone it will be 0
df_train['FamilyBool'] = (df_train['SibSp'] > 0) | (df_train['Parch'] > 0)
# 1 if parch > 0 ie if passenger does not have sibiling it will be 0
df_train['ParchBool'] = (df_train['Parch'] > 0)
# 1 sibiling > 0 ie if passenger does not have parch it will be 0
df_train['SibSpBool'] = (df_train['SibSp'] > 0) 
# verifying all went well.
df_train.tail()
# repeating data manupulation for test data.
df_test["Sex"].loc[df_test["Sex"] == 'female'] = 0
df_test["Sex"].loc[df_test["Sex"] == 'male'] = 1
# df_train[df_train["Survived"] == 1].describe()
df_test['AgeRange'] = pd.cut(df_test['Age'], [0, 15, 80], labels=[0, 1])
df_test['FamilyCount'] = (df_test['SibSp'] + df_test['Parch'])
df_test['FamilyBool'] = (df_test['SibSp'] > 0) | (df_test['Parch'] > 0)
df_test['ParchBool'] = (df_test['Parch'] > 0)
df_test['SibSpBool'] = (df_test['SibSp'] > 0) 

df_test.tail()
# we are going to analyse data by plotting. 
# dropping missing value rows by using dropna()
df_train_clean_age = df_train.dropna(subset=['Age'])
def scatter_plot_class(pclass):
    g = sns.FacetGrid(df_train_clean_age[df_train_clean_age['Pclass'] == pclass], 
                      col='Sex',
                      col_order=[0, 1],
                      hue='Survived', 
                      hue_kws=dict(marker=['v', '^']), 
                      size=6)
    g = (g.map(plt.scatter, 'Age', 'Fare', edgecolor='w', alpha=0.7, s=80).add_legend())
    plt.subplots_adjust(top=0.9)
    g.fig.suptitle('CLASS {}'.format(pclass))

# plotted separately because the fare scale for the first class makes it difficult to visualize second and third class charts
scatter_plot_class(1)
scatter_plot_class(2)
scatter_plot_class(3)

# we can find that younger female category survival is higher orange color upper arrows.
survived_by_class = df_train_clean_age.groupby('Pclass')['Survived'].mean()
survived_by_class
# the outcome says  better the class better survival rate. may sure you get 1st class ticket in your journey.
survived_by_sex = df_train_clean_age.groupby('Sex')['Survived'].mean()
survived_by_sex
# 0's are female, they survival is 75%. It's god's plan nothing we can do.
survived_by_age = df_train_clean_age.groupby('AgeRange')['Survived'].mean()
survived_by_age
# younger survival rate is better. Yup.
# he have choosen features based on which feature contribute most towards output.
# cols is array of column name we have choosen neglecting under correlated features
cols = ['Pclass','Sex','Fare','FamilyBool','ParchBool','SibSpBool']
# cols + 'survived'
tcols = np.append(['Survived'],cols)

df_train = df_train.loc[:,tcols].dropna().reindex()
X = df_train.loc[:,cols]
y = np.ravel(df_train.loc[:,['Survived']])
print(X.shape)
print(y.shape)
# flatten() always returns a copy.
# ravel() returns a view of the original array whenever possible. This isn't visible in the printed output, but if you modify the array returned by ravel, it may modify the entries in the original array. If you modify the entries in an array returned from flatten this will never happen. ravel will often be faster since no memory is copied, but you have to be more careful about modifying the array it returns.
# reshape((-1,)) gets a view whenever the strides of the array allow it even if that means you don't always get a contiguous array.
df_test = df_test.loc[:,tcols].dropna().reindex()
X_test = df_test.loc[:,cols]
y_test = np.ravel(df_test.loc[:,['Survived']])
print(X_test.shape)
print(y_test.shape)
# Now step inn for prediction
# Loading sklearn libs
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn import svm
 
clf_log = LogisticRegression()
# training the model with train data .fit(cols,'survived')
clf_log = clf_log.fit(X,y)
# predicting the model with test data
y_pred = clf_log.predict(X_test)
print('prediction complete')
acc_tree = accuracy_score(y_test, y_pred) * 100
print('Accuracy for our model: {}'.format(acc_tree))

acc_tree = accuracy_score(y_test, y_pred,normalize=False)
print('We predicted {} out of {} successfully.'.format(acc_tree,y_test.shape))
print(classification_report(y_test, y_pred))

print('star me and spread support')