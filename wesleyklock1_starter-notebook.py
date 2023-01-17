# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



%matplotlib inline

import seaborn as sns

color = sns.color_palette()

sns.set_style('darkgrid')





import warnings

def ignore_warn(*args, **kwargs):

    pass

warnings.warn = ignore_warn #ignore annoying warning (from sklearn and seaborn)



# Any results you write to the current directory are saved as output.



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#Now let's import and put the train and test datasets in  pandas dataframe



train = pd.read_csv('../input/qmi-comp-spring-2020/train.csv')

test = pd.read_csv('../input/qmi-comp-spring-2020/test.csv')



print(train.shape)

print(test.shape)
train.head(5)
import imblearn

from imblearn.over_sampling import SMOTE





temp = test['f1']



for index,value in enumerate(temp):

    print(value)
# check which features are categorical and numerical

# by using value counts



cols = train.columns



for col in cols:

    # print the number of unique values for each column

    print(col,len(train[col].value_counts()))

    

# anything with over 10 categories is probably not categorical
# for each of these features look at the distribution using seaborn



import seaborn as sns

import matplotlib.pyplot as plt



sns.distplot(train['f10'])
# make a list of categorical or numerical features 

# NOTE: what is presented here may or may not be the best grouping



categorical = ['f2','f5','f6','f9','f11','f18','f20','f21','f22','f24']



numerical = ['f1','f3','f4','f7','f8','f10','f11','f12',

             'f13','f14','f15','f16','f17','f19','f23']
# then divide your numerical features into either normal or skew features

# do this by using sns.distplot like before

# or by doing a test for normality



normal = ['f1','f4','f14']



skew = ['f3','f7','f8','f10','f11','f12',

             'f13','f15','f16','f17','f19','f23']
#Now drop the  'Id' colum since it's unnecessary for  the prediction process.

train.drop("Id", axis = 1, inplace = True)

test.drop("Id", axis = 1, inplace = True)
# the value majority of samples are 1's 

print(train['Y'].value_counts())

print("Percent 1s:",6546/(6546+395))



# what does this mean for modeling?



# what should we do when training to account a significant amount of

# class bias
# what should we do about missing data?

total = train.isnull().sum().sort_values(ascending=False)

print(total.sum())



# hey looks like we don't have to worry about it 😊
# we should log transform the very skewed numeric features

def normalize(df):

    skewed_feats = train[skew].apply(lambda x: skew(x))

    # here i used > 0.75 but you can use a different bar for skewness

    skewed_feats = skewed_feats[skewed_feats > 0.75]

    skewed_feats = skewed_feats.index

    df[skewed_feats] = np.log1p(df[skewed_feats])

    df = df.replace(-np.Inf,-2147483648)

    return df

# you probably want to do something about handling outliers

# I leave this up to you 😊
# modeling section



from sklearn.model_selection import train_test_split

# train_test split



x_train, y_train = train.drop(columns=['Y']), train['Y']



full_training, full_holdout = train_test_split(train,test_size=0.30)



x_training, y_training = full_training.drop(columns=['Y']), full_training['Y']



x_holdout, y_holdout = full_holdout.drop(columns=['Y']), full_holdout['Y']


# ranking function (how are we going to rank our models)

from sklearn.metrics import roc_auc_score

def roc(model):

    model.fit(x_training,y_training)

    y_pred = model.predict(x_holdout)

    return roc_auc_score(y_holdout, y_pred)





# function use to return a dataframe for predictions of test set

# remember to transform the test set first in the same way you 

# transformed the training set during the EDA/feature engineering stage

def make_pred(model):

    model.fit(x_train,y_train)

    return pd.DataFrame({'Id': test.index, 'Y': model.predict(test)})
# IMPORTANT REMEMBER TO TRANSFORM THE TEST SET IN THE SAME WAY YOU 

# TRANSFORMED THE TRAINING SET
# our first model 



from sklearn.tree import DecisionTreeClassifier



dtc_model = DecisionTreeClassifier()

print(roc(dtc_model))



submission = make_pred(dtc_model)
submission
submission.to_csv('submission.csv', index=False)
# ok so i finished this part what do i do now?



# try different models? There are like a billion different models

# https://towardsdatascience.com/machine-learning-classifiers-a5cc4e1b0623



# change the way you preprocessed the data

#    change the skew threshold

#    over-sample the 0 class

#    remove some features you think are useless

#    analyze correlations



# try stacking,ensembling models

# https://towardsdatascience.com/ensemble-methods-bagging-boosting-and-stacking-c9214a10a205?gi=b8bb1ee97c46



# try optimizing hyperparameters 

# https://scikit-learn.org/stable/modules/grid_search.html