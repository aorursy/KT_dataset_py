# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns



from sklearn.model_selection import train_test_split

from sklearn.preprocessing import OneHotEncoder

from sklearn.impute import SimpleImputer

from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression, LogisticRegression

from sklearn.metrics import mean_squared_error

from sklearn.model_selection import cross_val_score

from sklearn.decomposition import PCA


# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

csv = []

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        path = os.path.join(dirname, filename)

        csv.append(path)



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# Importing the train and test datasets



train = pd.read_csv('/kaggle/input/titanic/train.csv')

test = pd.read_csv('/kaggle/input/titanic/test.csv')
train_set, test_set = train_test_split(train, test_size=0.2, random_state=109)
train_set.head()
# Checking data quality. 

# Missing a lot of cabin data: drop

# SOme age missing, investigate the spread of the age and then fill with: mean or median 

train_set.info()
test_set = test_set.drop(['Name', 'Ticket', 'PassengerId', 'Cabin'], axis=1)

train_set = train_set.drop(['Name', 'Ticket', 'PassengerId', 'Cabin'], axis=1)
# Inspecting age distribution. Since the age is skewed to the left, fillna with median

age_group = train_set[["Age", "Survived"]]

age_group.hist(column="Age")
# Creating a function to group categories and plot 

def groupby_plot(df, category):

    group_df = df[["Survived",category]]

    group_df = group_df.groupby(category).sum() / group_df.groupby(category).count()

    print("Plotting for {}".format(category))

    return group_df.plot.bar()
# [1] First inspection of correlation between gender and survival through bar plot

groupby_plot(train_set, "Sex")
# [2] Initial inspection for correlation between fare and survival

group_fare = train_set[["Survived", "Fare"]]

group_fare = group_fare.groupby("Survived").mean()

group_fare.plot.bar()
# [3] Inspection for correlation between Embarked and survival

groupby_plot(train_set, "Embarked")
def split_agegroup(df):

    bins = [0, 3, 8, 13, 16, 35, 65, 120]

    labels = ['infant','Toddler', 'Child','Teenager', 'Young Adult','Adult', 'Old']

    df['age_group'] = pd.cut(df["Age"], bins=bins, labels = labels, right=False)

    df = df.drop('Age',axis=1)

    return df
# [4] Inspectino for correlation between Age and Survival

# Splitting the ages into age groups

test_set = split_agegroup(test_set)

train_set = split_agegroup(train_set)

groupby_plot(train_set, "age_group")
train_set.head()
# Now I have completed a basic EDA, now I will create a data pipeline to prepare for modelling



num_pipeline = Pipeline([

    ('imputer', SimpleImputer(strategy='median')),

    ('std_scaler', StandardScaler())

])



cat_pipeline = Pipeline([

    ('imputer', SimpleImputer(strategy='most_frequent')),

    ('1hotenc', OneHotEncoder(handle_unknown='ignore'))

])

pca_pipeline = Pipeline([

    ('pca', PCA(n_components=0.95))

])
num_attribs = ["Pclass", "SibSp", "Parch", "Fare"]

cat_attribs = ['Sex', 'Embarked', 'age_group']

columns = ["Pclass", "Sex", "SibSp","Parch", "Fare","Embarked", "age_group"]



# Separating the features and label

train_label = train_set['Survived']

training = train_set.drop('Survived',axis=1)



# Running the training data through the data pipeline

full_pipeline = ColumnTransformer([

    ('num', num_pipeline, num_attribs),

    ('impute', cat_pipeline, cat_attribs)

])

train_prepared = pd.DataFrame(full_pipeline.fit_transform(training))

train_prepared
# Training a logistic regression model

log_reg = LogisticRegression()

log_reg.fit(train_prepared, train_label)
some_data = training.iloc[:5]

some_labels = train_label.iloc[:5]

some_data_prepared = full_pipeline.transform(some_data)



print('Log Predictions: ', log_reg.predict(some_data_prepared))

print('Log labels: ', list(some_labels))
print(log_reg.coef_)
scores = cross_val_score(log_reg, train_prepared, train_label,

                        scoring='neg_mean_squared_error', cv=10)

log_rmse_scores = np.sqrt(-scores)
def display_scores(scores):

    print('Scores:', scores)

    print('Mean:', scores.mean())

    print('Standard Deviation:', scores.std())
test_label = test_set['Survived']

test_features = test_set.drop('Survived',axis=1)
test_prepared = pd.DataFrame(full_pipeline.transform(test_features))

test_prepared
prediction = log_reg.predict(test_prepared)

np.sum(prediction ==test_label)/len(prediction)
# num_attribs = ["Pclass", "SibSp", "Parch", "Fare"]

# cat_attribs = ['Sex', 'Embarked', 'age_group']

final_prediction = pd.DataFrame(test['PassengerId'])

test = test.drop(['Name', 'Ticket', 'PassengerId', 'Cabin'], axis=1)
split_agegroup(test)
test
final_pre = full_pipeline.transform(test)

pd.DataFrame(final_pre)
prediction = pd.DataFrame(log_reg.predict(final_pre), columns=['Survived'])

final_prediction['Survived'] = prediction

final_prediction
final_prediction.to_csv('prediction.csv', index=False)