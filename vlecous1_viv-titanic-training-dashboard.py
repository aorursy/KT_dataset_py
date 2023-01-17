%matplotlib inline



# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt



# Import statements required for Plotly 

import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.tools as tls





from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score, log_loss, roc_curve

from imblearn.over_sampling import SMOTE

import xgboost
train_data = pd.read_csv('../input/train.csv')

print("There are %d observations and %d variables in the train dataset" %(train_data.shape))
# Let's print some observations to see the data

train_data.head(10)
# print list of variables

print(list(train_data))



# print statistics

print(train_data.describe())





# Let's do some plotting

from pandas.plotting import scatter_matrix



scatter_matrix(train_data, alpha=0.2, figsize=(20, 20), diagonal='kde')
train_data.Age.hist(bins = 40)
# Need to code the gender of the passenger as numeric

train_data.Sex.dtypes



def gender(x):

    if x == "Male":

        return 1

    else:

        return 0





train_data["Sex"] = train_data["Sex"].apply(gender)

train_data.Sex.dtypes



# Need to code the port of embarkation as C = 0, Q = 1, S = 2

def embarkation(x):

    switcher = {

        "C": 0,

        "Q": 1,

        "S": 2,

    }

    return switcher.get(x)



train_data["Embarked"] = train_data["Embarked"].apply(embarkation)

train_data.Embarked.dtypes

train_data.Embarked.head(10)



train_data.dtypes

# Train model

    # Select the features used. Starts with almost all of them 

    # (excluding tip amount of course)

features = ['Pclass', 

            'Sex', 

            'Age', 

            'SibSp', 

            'Parch', 

            'Fare', 

            'Embarked',

            'Survived']



target = 'Survived'



# Drop any of the row that contains a NaN

df = train_data[features].dropna(axis=0, how='any')



df.dtypes

print(df[features].describe())


    # Create first a non-optimized random forest regressor using the aforementioned features

    # This will serve as a baseline for benchmarking 

    # We will get a feel on the important features and decide which ones to keep

features = ['Pclass', 

            'Sex', 

            'Age', 

            'SibSp', 

            'Parch', 

            'Fare', 

            'Embarked']    

rfc = RandomForestClassifier(n_estimators=50,

                             oob_score=True,

                             max_features=None,

                             n_jobs=-1)

    

    # Train the random forest

rfc.fit(df[features], df[target])



print("\nBenchmarking RFC:")

    

# Print the oob score, which is the R2 based on oob_predictions

print("The out of bag score is: %.4f." %(rfc.oob_score_))



y_pred = rfc.predict(df[features])



rfc.score(df[features], df[target])

sum(y_pred == df[target]).sum()/len(y_pred)
fpr_rt_lm, tpr_rt_lm, _ = roc_curve(df[target], y_pred)



plt.figure(1)

plt.plot([0, 1], [0, 1], 'k--')

plt.plot(fpr_rt_lm, tpr_rt_lm, label='RT')

plt.xlabel('False positive rate')

plt.ylabel('True positive rate')

plt.title('ROC curve')

plt.legend(loc='best')

plt.show()
