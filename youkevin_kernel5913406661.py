# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import string

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.



import warnings

warnings.filterwarnings("ignore")
#setup

from sklearn.model_selection import train_test_split

# Read the data

X = pd.read_csv('/kaggle/input/titanic/train.csv', index_col='PassengerId')

X_test_full = pd.read_csv('/kaggle/input/titanic/test.csv', index_col='PassengerId')

pd.set_option('display.max_columns', None)  

pd.set_option('display.expand_frame_repr', False)

pd.set_option('max_colwidth', -1)













correlations = X.corr(method='pearson')

print(correlations)

X.dropna(axis=0, subset=['Survived'], inplace=True)

y = X.Survived             



X.drop(['Survived'], axis=1, inplace=True)





print(X.head())
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import MinMaxScaler



# "Cardinality" means the number of unique values in a column

# Select categorical columns with relatively low cardinality (convenient but arbitrary)

categorical_cols = [cname for cname in X.columns if X[cname].dtype == "object"]





# Select numeric columns

numeric_cols = [cname for cname in X.columns if X[cname].dtype in ['int64', 'float64']]

scaler = MinMaxScaler()

X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

X_test_full[numeric_cols] = scaler.fit_transform(X_test_full[numeric_cols])


from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import OneHotEncoder

from sklearn.compose import ColumnTransformer

# Preprocessing for numerical data

numerical_transformer = SimpleImputer(strategy='median')



# Preprocessing for categorical data

categorical_transformer = Pipeline(steps=[

    ('imputer', SimpleImputer(strategy='most_frequent')),

    ('onehot', OneHotEncoder(handle_unknown='ignore'))

])



# Bundle preprocessing for numerical and categorical data

preprocessor = ColumnTransformer(

    transformers=[

        ('num', numerical_transformer, numeric_cols),

        ('cat', categorical_transformer, categorical_cols)

    ])
from xgboost import XGBClassifier



from sklearn.metrics import mean_absolute_error

from sklearn.metrics import accuracy_score



import statistics



from sklearn.model_selection import cross_val_score

def pipelinefunction(model):

    my_pipeline = Pipeline(steps=[('preprocessor', preprocessor),

                              ('model', model)

                             ])

    my_pipeline.fit(X, y)

    preds = my_pipeline.predict(X_test_full)

    scores = -1 * cross_val_score(my_pipeline, X, y,

                                  cv=5,

                                  scoring='neg_mean_absolute_error')

    print(scores)

    print("acc:",1-statistics.mean(scores))

    output = pd.DataFrame({'PassengerId': X_test_full.index,'Survived': preds})

    output.Survived = output.Survived.astype(int)

    output_final = output.set_index('PassengerId')

    return output



model = XGBClassifier(max_depth=3,n_estimators=40,learning_rate=0.005,booster='gbtree',random_state=1)

output = pipelinefunction(model)











from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import VotingClassifier

from sklearn.tree import ExtraTreeClassifier



print("DecisionTreeClassifier")

model1 = DecisionTreeClassifier(criterion='gini',splitter='best',max_depth=3)

pipelinefunction(model1)

print()

print("KNeightborsClassifier")

model2 = KNeighborsClassifier(algorithm='ball_tree')

pipelinefunction(model2)

print()

print("LogisticRegression")

model3= LogisticRegression(solver='newton-cg',class_weight='balanced')

pipelinefunction(model3)

print()



print("ExtraTreeClassifier")

model5 = ExtraTreeClassifier(criterion='gini',splitter='best',max_depth=4)

output = pipelinefunction(model5)

print()

from sklearn.calibration import CalibratedClassifierCV

print("CalibratedClassifierCV")

model6 = CalibratedClassifierCV(method='sigmoid',cv=5)

print(model6.get_params())

output = pipelinefunction(model6)

print()

output.to_csv('submission.csv', index=False)
from sklearn.ensemble import GradientBoostingClassifier

print("GradientBoostingClassifier")

model7= GradientBoostingClassifier(loss='exponential',learning_rate=0.12,n_estimators=60,criterion='friedman_mse',max_features=None)

pipelinefunction(model7)

print()
print("VotingClassifier")

model4= VotingClassifier(estimators=[('lr', model), ('rf', model7), ('gnb', model6)],voting='hard')

output = pipelinefunction(model4)



print()

print("XGBClassifier")

from xgboost import XGBClassifier

model0 = XGBClassifier(max_depth=3,n_estimators=50,learning_rate=0.02)

output = pipelinefunction(model0)
