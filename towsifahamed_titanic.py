# Import Dependencies

%matplotlib inline



# Start Python Imports

import math, time, random, datetime



# Data Manipulation

import numpy as np

import pandas as pd

import csv as csv



# Visualization 

import matplotlib.pyplot as plt

import missingno

import seaborn as sns

plt.style.use('seaborn-whitegrid')



# Preprocessing

from sklearn.preprocessing import OneHotEncoder, LabelEncoder, label_binarize



# Machine learning

import catboost

from sklearn.model_selection import train_test_split

from sklearn import model_selection, tree, preprocessing, metrics, linear_model

from sklearn.svm import LinearSVC

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import LinearRegression, LogisticRegression, SGDClassifier

from sklearn.tree import DecisionTreeClassifier

from catboost import CatBoostClassifier, Pool, cv



#Shuffle the datasets

from sklearn.utils import shuffle



#Learning curve

from sklearn.model_selection import learning_curve

from sklearn.model_selection import ShuffleSplit



# Let's be rebels and ignore warnings for now

import warnings

warnings.filterwarnings('ignore')



#import seaborn as sns

#Output plots in notebook

#%matplotlib inline 



addpoly = True

plot_lc = 0   # 1--display learning curve/ 0 -- don't display
#loading the data sets from the csv files

train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

gender_submission = pd.read_csv('../input/gender_submission.csv')



print('train dataset: %s, test dataset %s' %(str(train.shape), str(test.shape)) )

train.head()
test.head()
gender_submission.head()
# Plot graphic of missing values

missingno.matrix(train, figsize = (30,10))

# Let's write a little function to show us how many missing values

# there are

def find_missing_values(df, columns):

    """

    Finds number of rows where certain columns are missing values.

    ::param_df:: = target dataframe

    ::param_columns:: = list of columns

    """

    missing_vals = {}

    print("Number of missing or NaN values for each column:")

    df_length = len(df)

    for column in columns:

        total_column_values = df[column].value_counts().sum()

        missing_vals[column] = df_length-total_column_values

        #missing_vals.append(str(column)+ " column has {} missing or NaN values.".format())

    return missing_vals



missing_values = find_missing_values(train, columns=train.columns)

missing_values
df_bin = pd.DataFrame() # for discretised continuous variables

df_con = pd.DataFrame() # for continuous variables
train.dtypes
# How many people survived?

fig = plt.figure(figsize=(10,1))

sns.countplot(y='Survived', data=train);

print(train.Survived.value_counts())
# Let's add this to our subset dataframes

df_bin['Survived'] = train['Survived']

df_con['Survived'] = train['Survived']
sns.distplot(train.Pclass)
df_bin['Pclass'] = train['Pclass']

df_con['Pclass'] = train['Pclass']
# Let's view the distribution of Sex

plt.figure(figsize=(10, 1))

sns.countplot(y="Sex", data=train);
# add Sex to the subset dataframes

df_bin['female'] = train['Sex']

df_bin['female'] = np.where(df_bin['female'] == 'female', 1, 0) # change sex to 0 for male and 1 for female



df_con['Sex'] = train['Sex']
# How does the Sex variable look compared to Survival?

# We can see this because they're both binarys.

fig = plt.figure(figsize=(10,5))

sns.distplot(df_bin.loc[df_bin['Survived'] == 1]['female'], kde_kws={'label': 'Survived'});

sns.distplot(df_bin.loc[df_bin['Survived'] == 0]['female'], kde_kws={'label': 'Did not survive'});
df_bin.head()
def plot_count_dist(data, bin_df, label_column, target_column, figsize=(20, 5), use_bin_df=False):

    """

    Function to plot counts and distributions of a label variable and 

    target variable side by side.

    ::param_data:: = target dataframe

    ::param_bin_df:: = binned dataframe for countplot

    ::param_label_column:: = binary labelled column

    ::param_target_column:: = column you want to view counts and distributions

    ::param_figsize:: = size of figure (width, height)

    ::param_use_bin_df:: = whether or not to use the bin_df, default False

    """

    if use_bin_df: 

        fig = plt.figure(figsize=figsize)

        plt.subplot(1, 2, 1)

        sns.countplot(y=target_column, data=bin_df);

        plt.subplot(1, 2, 2)

        sns.distplot(data.loc[data[label_column] == 1][target_column], 

                     kde_kws={"label": "Survived"});

        sns.distplot(data.loc[data[label_column] == 0][target_column], 

                     kde_kws={"label": "Did not survive"});

    else:

        fig = plt.figure(figsize=figsize)

        plt.subplot(1, 2, 1)

        sns.countplot(y=target_column, data=data);

        plt.subplot(1, 2, 2)

        sns.distplot(data.loc[data[label_column] == 1][target_column], 

                     kde_kws={"label": "Survived"});

        sns.distplot(data.loc[data[label_column] == 0][target_column], 

                     kde_kws={"label": "Did not survive"});
df_bin.head()
# What values are there?

train.SibSp.value_counts()
# Add SibSp to subset dataframes

df_bin['SibSp'] = train['SibSp']

df_con['SibSp'] = train['SibSp']

# Visualise the counts of SibSp and the distribution of the values

# against Survived

plot_count_dist(train, 

                bin_df=df_bin, 

                label_column='Survived', 

                target_column='SibSp', 

                figsize=(25,7))
# What values are there?

train.Parch.value_counts()

# Add Parch to subset dataframes

df_bin['Parch'] = train['Parch']

df_con['Parch'] = train['Parch']
# Visualise the counts of Parch and the distribution of the values

# against Survived

plot_count_dist(train, 

                bin_df=df_bin,

                label_column='Survived', 

                target_column='Parch', 

                figsize=(25,7))
df_bin.head()
df_con.head()
# How many kinds of ticket are there?

train.Ticket.value_counts()
# How many kinds of fare are there?

train.Fare.value_counts()
# Add Fare to sub dataframes

df_con['Fare'] = train['Fare'] 

df_bin['Fare'] = pd.cut(train['Fare'], bins=5) # discretised 

# What do our Fare bins look like?

df_bin.Fare.value_counts()
# Visualise the Fare bin counts as well as the Fare distribution versus Survived.

plot_count_dist(data=train,

                bin_df=df_bin,

                label_column='Survived', 

                target_column='Fare', 

                figsize=(25,7), 

                use_bin_df=True)
# Add Embarked to sub dataframes

df_bin['Embarked'] = train['Embarked']

df_con['Embarked'] = train['Embarked']

# Remove Embarked rows which are missing values

print(len(df_con))

df_con = df_con.dropna(subset=['Embarked'])

df_bin = df_bin.dropna(subset=['Embarked'])

print(len(df_con))
# One-hot encode binned variables

one_hot_cols = df_bin.columns.tolist()

one_hot_cols.remove('Survived')

df_bin_enc = pd.get_dummies(df_bin, columns=one_hot_cols)



df_bin_enc.head()
# Label Encode all continuous values using LabelEncoder()

df_con_enc = df_con.apply(LabelEncoder().fit_transform)



df_con_enc.head(20)
# Seclect the dataframe we want to use first for predictions

selected_df = df_con_enc

# Split the dataframe into data and labels

X_train = selected_df.drop('Survived', axis=1) # data

y_train = selected_df.Survived # labels
# View the data for the CatBoost model

X_train.head()
# View the labels for the CatBoost model

y_train.head()
# Define the categorical features for the CatBoost model

cat_features = np.where(X_train.dtypes != np.float)[0]

cat_features
# Use the CatBoost Pool() function to pool together the training data and categorical feature labels

train_pool = Pool(X_train, 

                  y_train,

                  cat_features)
# CatBoost model definition

catboost_model = CatBoostClassifier(iterations=10000,

                                    custom_loss=['Accuracy'],

                                    loss_function='Logloss')



# Fit CatBoost model

catboost_model.fit(train_pool,

                   plot=True)



# CatBoost accuracy

acc_catboost = round(catboost_model.score(X_train, y_train) * 100, 2)
# Perform CatBoost cross-validation

start_time = time.time()



# Set params for cross-validation as same as initial model

cv_params = catboost_model.get_params()



# Run the cross-validation for 10-folds (same as the other models)

cv_data = cv(train_pool,

             cv_params,

             fold_count=10,

             plot=True)



# How long did it take?

catboost_time = (time.time() - start_time)



# CatBoost CV results save into a dataframe (cv_data), let's withdraw the maximum accuracy score

acc_cv_catboost = round(np.max(cv_data['test-Accuracy-mean']) * 100, 2)
# Print out the CatBoost model metrics

print("---CatBoost Metrics---")

print("Accuracy: {}".format(acc_catboost))

print("Accuracy CV 10-Fold: {}".format(acc_cv_catboost))

print("Running Time: {}".format(datetime.timedelta(seconds=catboost_time)))
# Create a list of columns to be used for the predictions

wanted_test_columns = X_train.columns

wanted_test_columns
# Make a prediction using the CatBoost model on the wanted columns

predictions = catboost_model.predict(test[wanted_test_columns]

                                     .apply(LabelEncoder().fit_transform))
# Our predictions array is comprised of 0's and 1's (Survived or Did Not Survive)

predictions[:20]
# Create a submisison dataframe and append the relevant columns

submission = pd.DataFrame()

submission['PassengerId'] = test['PassengerId']

submission['Survived'] = predictions # our model predictions on the test dataset

submission.head()
# Let's convert our submission dataframe 'Survived' column to ints

submission['Survived'] = submission['Survived'].astype(int)

print('Converted Survived column to integers.')
# Are our test and submission dataframes the same length?

if len(submission) == len(test):

    print("Submission dataframe is the same length as test ({} rows).".format(len(submission)))

else:

    print("Dataframes mismatched, won't be able to submit to Kaggle.")
# Convert submisison dataframe to csv for submission to csv 

# for Kaggle submisison

submission.to_csv('catboost_submission.csv', index=False)

print('Submission CSV is ready!')