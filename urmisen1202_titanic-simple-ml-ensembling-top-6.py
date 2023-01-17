# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd

import numpy as np



# Data visualisation & images

import matplotlib.pyplot as plt

import seaborn as sns



# Pipeline and machine learning algorithms

from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import StandardScaler

from sklearn.neighbors import KNeighborsClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import SVC

from xgboost import XGBClassifier

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.ensemble import ExtraTreesClassifier

from sklearn.ensemble import AdaBoostClassifier

from sklearn.gaussian_process import GaussianProcessClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.ensemble import BaggingClassifier

from sklearn.ensemble import VotingClassifier



# Model fine-tuning and evaluation

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import cross_val_predict

from sklearn import model_selection



%matplotlib inline
#Load the train and test data from the dataset

df_train=pd.read_csv('../input/titanic/train.csv')

df_test=pd.read_csv('../input/titanic/test.csv')
# Join all data into one file

ntrain = df_train.shape[0]

ntest = df_test.shape[0]



# Creating y_train variable; we'll need this when modelling, but not before

y_train = df_train['Survived'].values



# Saving the passenger ID's ready for our submission file at the very end

passId = df_test['PassengerId']



# Create a new all-encompassing dataset

data = pd.concat((df_train, df_test))



# Printing overall data shape

print("data size is: {}".format(data.shape))
df_train.info()
df_train.head()
# Returning descriptive statistics of the train dataset

df_train.describe(include = 'all')
df_test.info()
df_test.head()
# Initiate correlation matrix

corr = df_train.corr()  # Pandas dataframe.corr() is used to find the pairwise correlation of all columns in the dataframe. 

# Set-up mask

mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True

# Set-up figure

plt.figure(figsize=(14, 8))

# Title

plt.title('Overall Correlation of Titanic Features', fontsize=18)

# Correlation matrix

sns.heatmap(corr, mask=mask, annot=False,cmap='RdYlGn', linewidths=0.2, annot_kws={'size':20})

plt.show()
# Feature: Survived



# Plot for survived

fig = plt.figure(figsize = (10,5))

sns.countplot(x='Survived', data = df_train)

print(df_train['Survived'].value_counts())
# Feature: Pclass

# Bar chart of each Pclass type

fig = plt.figure(figsize = (10,10))

ax1 = plt.subplot(2,1,1)

ax1 = sns.countplot(x = 'Pclass', hue = 'Survived', data = df_train)

ax1.set_title('Ticket Class Survival Rate')

ax1.set_xticklabels(['1 Upper','2 Middle','3 Lower'])

ax1.set_ylim(0,400)

ax1.set_xlabel('Ticket Class')

ax1.set_ylabel('Count')

ax1.legend(['No','Yes'])



# Pointplot Pclass type

ax2 = plt.subplot(2,1,2)

sns.pointplot(x='Pclass', y='Survived', data=df_train)

ax2.set_xlabel('Ticket Class')

ax2.set_ylabel('Percent Survived')

ax2.set_title('Percentage Survived by Ticket Class')
# Feature: Age

# Bar chart of age mapped against sex. For now, missing values have been dropped and will be dealt with later

survived = 'survived'

not_survived = 'not survived'

fig, axes = plt.subplots(nrows=1, ncols=2,figsize=(10, 4))

women = df_train[df_train['Sex']=='female']

men = df_train[df_train['Sex']=='male']

ax = sns.distplot(women[women['Survived']==1].Age.dropna(), bins=20, label = survived, ax = axes[0], kde =False)

ax = sns.distplot(women[women['Survived']==0].Age.dropna(), bins=20, label = not_survived, ax = axes[0], kde =False)

ax.legend()

ax.set_title('Female')

ax = sns.distplot(men[men['Survived']==1].Age.dropna(), bins=20, label = survived, ax = axes[1], kde = False)

ax = sns.distplot(men[men['Survived']==0].Age.dropna(), bins=20, label = not_survived, ax = axes[1], kde = False)

ax.legend()

_ = ax.set_title('Male')
# Feature: SibSp & ParCh

# Plotting survival rate vs Siblings or Spouse on board

fig = plt.figure(figsize = (10,12))

ax1 = plt.subplot(2,1,1)

ax1 = sns.countplot(x = 'SibSp', hue = 'Survived', data = df_train)

ax1.set_title('Survival Rate with Total of Siblings and Spouse on Board')

ax1.set_ylim(0,500)

ax1.set_xlabel('# of Sibling and Spouse')

ax1.set_ylabel('Count')

ax1.legend(['No','Yes'],loc = 1)



# Plotting survival rate vs Parents or Children on board

ax2 = plt.subplot(2,1,2)

ax2 = sns.countplot(x = 'Parch', hue = 'Survived', data = df_train)

ax2.set_title('Survival Rate with Total Parents and Children on Board')

ax2.set_ylim(0,500)

ax2.set_xlabel('# of Parents and Children')

ax2.set_ylabel('Count')

ax2.legend(['No','Yes'],loc = 1)
# Feature: Fare

# Bar chart of each Fare type

fig = plt.figure(figsize = (10,10))

ax1 = sns.countplot(x = 'Pclass', hue = 'Survived', data = df_train)

ax1.set_title('Ticket Class Survival Rate with respect to fare')

ax1.set_xticklabels(['1 Upper','2 Middle','3 Lower'])

ax1.set_xlabel('Ticket Class')

ax1.set_ylabel('Fare')

ax1.legend(['No','Yes'])
# Graph to display fare paid per the three ticket types

fig = plt.figure(figsize = (10,5))

sns.swarmplot(x="Pclass", y="Fare", data=df_train, hue='Survived')
print("TRAIN DATA:")

df_train.isnull().sum()
sns.heatmap(df_train.isnull(),yticklabels=False,cbar=False,cmap='viridis')

print("TEST DATA:")

df_test.isnull().sum()
sns.heatmap(df_test.isnull(),yticklabels=False,cbar=False,cmap='viridis')

# Extract last name

data['Last_Name'] = data['Name'].apply(lambda x: str.split(x, ",")[0])



# Fill in missing Fare value by overall Fare mean

data['Fare'].fillna(data['Fare'].mean(), inplace=True)



# Setting coin flip (e.g. random chance of surviving)

default_survival_chance = 0.5

data['Family_Survival'] = default_survival_chance



# Grouping data by last name and fare - looking for families

for grp, grp_df in data[['Survived','Name', 'Last_Name', 'Fare', 'Ticket', 'PassengerId',

                           'SibSp', 'Parch', 'Age', 'Cabin']].groupby(['Last_Name', 'Fare']):

    

    # If not equal to 1, a family is found 

    # Then work out survival chance depending on whether or not that family member survived

    if (len(grp_df) != 1):

        for ind, row in grp_df.iterrows():

            smax = grp_df.drop(ind)['Survived'].max()

            smin = grp_df.drop(ind)['Survived'].min()

            passID = row['PassengerId']

            if (smax == 1.0):

                data.loc[data['PassengerId'] == passID, 'Family_Survival'] = 1

            elif (smin == 0.0):

                data.loc[data['PassengerId'] == passID, 'Family_Survival'] = 0



# Print the headline

print("Number of passengers with family survival information:", 

      data.loc[data['Family_Survival']!=0.5].shape[0])
# If not equal to 1, a group member is found

# Then work out survival chance depending on whether or not that group member survived

for _, grp_df in data.groupby('Ticket'):

    if (len(grp_df) != 1):

        for ind, row in grp_df.iterrows():

            if (row['Family_Survival'] == 0) | (row['Family_Survival']== 0.5):

                smax = grp_df.drop(ind)['Survived'].max()

                smin = grp_df.drop(ind)['Survived'].min()

                passID = row['PassengerId']

                if (smax == 1.0):

                    data.loc[data['PassengerId'] == passID, 'Family_Survival'] = 1

                elif (smin==0.0):

                    data.loc[data['PassengerId'] == passID, 'Family_Survival'] = 0



# Print the headline

print("Number of passenger with family/group survival information: " 

      +str(data[data['Family_Survival']!=0.5].shape[0]))
# Reset index for remaining feature engineering steps

data = data.reset_index(drop=True)

data = data.drop('Survived', axis=1)

data.tail()
# Visualising fare data

plt.hist(data['Fare'], bins=40)

plt.xlabel('Fare')

plt.ylabel('Count')

plt.title('Distribution of fares')

plt.show()
# Turning fare into 4 bins due to heavy skew in data

data['Fare'] = pd.qcut(data['Fare'], 4)



# I will now use Label Encoder to convert the bin ranges into numbers

lbl = LabelEncoder()

data['Fare'] = lbl.fit_transform(data['Fare'])
# Visualise new look fare variable

sns.countplot(data['Fare'])

plt.xlabel('Fare Bin')

plt.ylabel('Count')

plt.title('Fare Bins')
# Inspecting the first five rows of Name

df_train['Name'].head()
# New function to return name title only

def get_title(name):

    if '.' in name:

        return name.split(',')[1].split('.')[0].strip()

    else:

        return 'Unknown'
# Creating two lists of titles, one for each dataset

titles_data = sorted(set([x for x in data['Name'].map(lambda x: get_title(x))]))



# Printing list length and items in each list

print(len(titles_data), ':', titles_data)
# New function to classify each title into 1 of 4 overarching titles

def set_title(x):

    title = x['Title']

    if title in ['Capt', 'Col', 'Don', 'Jonkheer', 'Major', 'Rev', 'Sir']:

        return 'Mr'

    elif title in ['the Countess', 'Mme', 'Lady','Dona']:

        return 'Mrs'

    elif title in ['Mlle', 'Ms']:

        return 'Miss'

    elif title =='Dr':

        if x['Sex']=='male':

            return 'Mr'

        else:

            return 'Mrs'

    else:

        return title
# Applying the get_title function to create the new 'Title' feature

data['Title'] = data['Name'].map(lambda x: get_title(x))

data['Title'] = data.apply(set_title, axis=1)
# Printing values of the title column (checking function worked!)

print(data['Title'].value_counts())
# Returning NaN within Age across Train & Test set

print('Total missing age data: ', pd.isnull(data['Age']).sum())

# Check which statistic to use in imputation

print(data['Age'].describe(exclude='NaN'))
#Imputing Age within the train & test set with the Median, grouped by Pclass and title

data['Age'] = data.groupby('Title')['Age'].apply(lambda x: x.fillna(x.median()))
# Visualise new look age variable

plt.hist(data['Age'], bins=40)

plt.xlabel('Age')

plt.ylabel('Count')

plt.title('Distribution of ages')

plt.show()
# Turning data into 4 bins due to heavy skew in data

data['Age'] = pd.qcut(data['Age'], 4)



# Transforming bins to numbers

lbl = LabelEncoder()

data['Age'] = lbl.fit_transform(data['Age'])
# Visualise new look fare variable

plt.xticks(rotation='90')

sns.countplot(data['Age'])

plt.xlabel('Age Bin')

plt.ylabel('Count')

plt.title('Age Bins')
#transferring the titles over to numbers ready for Machine Learning

data['Title'] = data['Title'].replace(['Mr', 'Miss', 'Mrs', 'Master'], [0, 1, 2, 3])
# Recoding sex to numeric values with use of a dictionary for machine learning model compatibility

data['Sex'] = data['Sex'].replace(['male', 'female'], [0, 1])
# Inspecting the first five rows of Embarked

data['Embarked'].head()
data['Embarked'].describe()
# Filling in missing embarked values with the mode (S)

data['Embarked'] = data['Embarked'].fillna(data['Embarked'].mode()[0])



# Converting to numeric values

data['Embarked'] = data['Embarked'].replace(['S', 'C', 'Q'], [0, 1, 2])
# Inspecting head of Cabin column

data['Cabin'].head()
# Labelling all NaN values as 'Unknown'

data['Cabin'].fillna('Unknown',inplace=True)
# Extracting the first value in the each row of Cabin

data['Cabin'] = data['Cabin'].map(lambda x: x[0])
# Return the counts of each unique value in the Cabin column

data['Cabin'].value_counts()
# New function to classify known cabins as 'Known', otherwise 'Unknown'

def unknown_cabin(cabin):

    if cabin != 'U':

        return 1

    else:

        return 0

    

# Applying new function to Cabin feature

data['Cabin'] = data['Cabin'].apply(lambda x:unknown_cabin(x))
# Creating two features of relatives and not alone

data['FamilySize'] = data['SibSp'] + data['Parch'] + 1

data['IsAlone'] = 1 #initialize to yes/1 is alone

data['IsAlone'].loc[data['FamilySize'] > 1] = 0 # now update to no/0 if family size is greater than 1

# Final look at the data

data.head()
# Dropping what we know need for Machine Learning

data = data.drop(['Name', 'Parch', 'SibSp', 'Ticket', 'Last_Name', 'PassengerId'], axis = 1)
# Return to train/test sets

train = data[:ntrain]

test = data[ntrain:]
# Set up feature and target variables in train set, and remove Passenger ID from test set

X_test = test

X_train = train



# Scaling data to support modelling

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)
# Initiate 11 classifier models

ran = RandomForestClassifier(random_state=1)

knn = KNeighborsClassifier()

log = LogisticRegression()

xgb = XGBClassifier()

gbc = GradientBoostingClassifier()

svc = SVC(probability=True)

ext = ExtraTreesClassifier()

ada = AdaBoostClassifier()

gnb = GaussianNB()

gpc = GaussianProcessClassifier()

bag = BaggingClassifier()



# Prepare lists

models = [ran, knn, log, xgb, gbc, svc, ext, ada, gnb, gpc, bag]         

scores = []



# Sequentially fit and cross validate all models

for mod in models:

    mod.fit(X_train, y_train)

    acc = cross_val_score(mod, X_train, y_train, scoring = "accuracy", cv = 10)

    scores.append(acc.mean())
# Creating a table of results, ranked highest to lowest

results = pd.DataFrame({

    'Model': ['Random Forest', 'K Nearest Neighbour', 'Logistic Regression', 'XGBoost', 'Gradient Boosting', 'SVC', 'Extra Trees', 'AdaBoost', 'Gaussian Naive Bayes', 'Gaussian Process', 'Bagging Classifier'],

    'Score': scores})



result_df = results.sort_values(by='Score', ascending=False).reset_index(drop=True)

result_df.head(11)
# Plot results

sns.barplot(x='Score', y = 'Model', data = result_df, color = 'c')

plt.title('Machine Learning Algorithm Accuracy Score \n')

plt.xlabel('Accuracy Score (%)')

plt.ylabel('Algorithm')

plt.xlim(0.80, 0.86)
# Function for new graph

def importance_plotting(data, x, y, palette, title):

    sns.set(style="whitegrid")

    ft = sns.PairGrid(data, y_vars=y, x_vars=x, size=5, aspect=1.5)

    ft.map(sns.stripplot, orient='h', palette=palette, edgecolor="black", size=15)

    

    for ax, title in zip(ft.axes.flat, titles):

    # Set a different title for each axes

        ax.set(title=title)

    # Make the grid horizontal instead of vertical

        ax.xaxis.grid(False)

        ax.yaxis.grid(True)

    plt.show()
# Building feature importance into a DataFrame

fi = {'Features':train.columns.tolist(), 'Importance':xgb.feature_importances_}

importance = pd.DataFrame(fi, index=None).sort_values('Importance', ascending=False)
# Creating graph title

titles = ['The most important features in predicting survival on the Titanic: XGBoost']



# Plotting graph

importance_plotting(importance, 'Importance', 'Features', 'Reds_r', titles)
# Building feature importance into a DataFrame

fi = {'Features':train.columns.tolist(), 'Importance':np.transpose(log.coef_[0])}

importance = pd.DataFrame(fi, index=None).sort_values('Importance', ascending=False)
# Creating graph title

titles = ['The most important features in predicting survival on the Titanic: Logistic Regression']



# Plotting graph

importance_plotting(importance, 'Importance', 'Features', 'Reds_r', titles)
# Getting feature importances for the 5 models where we can

gbc_imp = pd.DataFrame({'Feature':train.columns, 'gbc importance':gbc.feature_importances_})

xgb_imp = pd.DataFrame({'Feature':train.columns, 'xgb importance':xgb.feature_importances_})

ran_imp = pd.DataFrame({'Feature':train.columns, 'ran importance':ran.feature_importances_})

ext_imp = pd.DataFrame({'Feature':train.columns, 'ext importance':ext.feature_importances_})

ada_imp = pd.DataFrame({'Feature':train.columns, 'ada importance':ada.feature_importances_})



# Merging results into a single dataframe

importances = gbc_imp.merge(xgb_imp, on='Feature').merge(ran_imp, on='Feature').merge(ext_imp, on='Feature').merge(ada_imp, on='Feature')



# Calculating average importance per feature

importances['Average'] = importances.mean(axis=1)



# Ranking top to bottom

importances = importances.sort_values(by='Average', ascending=False).reset_index(drop=True)



# Display

importances
# Building feature importance into a DataFrame

fi = {'Features':importances['Feature'], 'Importance':importances['Average']}

importance = pd.DataFrame(fi, index=None).sort_values('Importance', ascending=False)
# Creating graph title

titles = ['The most important features in predicting survival on the Titanic: 5 model average']



# Plotting graph

importance_plotting(importance, 'Importance', 'Features', 'Reds_r', titles)
# Drop redundant features

train = train.drop(['Embarked','IsAlone'], axis=1)

test = test.drop(['Embarked', 'IsAlone'], axis=1)



# Re-build model variables

X_train = train

X_test = test



# Transform

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)
# Initiate models

ran = RandomForestClassifier(random_state=1)

knn = KNeighborsClassifier()

log = LogisticRegression()

xgb = XGBClassifier(random_state=1)

gbc = GradientBoostingClassifier(random_state=1)

svc = SVC(probability=True)

ext = ExtraTreesClassifier(random_state=1)

ada = AdaBoostClassifier(random_state=1)

gnb = GaussianNB()

gpc = GaussianProcessClassifier()

bag = BaggingClassifier(random_state=1)



# Lists

models = [ran, knn, log, xgb, gbc, svc, ext, ada, gnb, gpc, bag]         

scores_v2 = []



# Fit & cross validate

for mod in models:

    mod.fit(X_train, y_train)

    acc = cross_val_score(mod, X_train, y_train, scoring = "accuracy", cv = 10)

    scores_v2.append(acc.mean())
# Creating a table of results, ranked highest to lowest

results = pd.DataFrame({

    'Model': ['Random Forest', 'K Nearest Neighbour', 'Logistic Regression', 'XGBoost', 'Gradient Boosting', 'SVC', 'Extra Trees', 'AdaBoost', 'Gaussian Naive Bayes', 'Gaussian Process', 'Bagging Classifier'],

    'Original Score': scores,

    'Score with feature selection': scores_v2})



result_df = results.sort_values(by='Score with feature selection', ascending=False).reset_index(drop=True)

result_df.head(11)
# Plot results

sns.barplot(x='Score with feature selection', y = 'Model', data = result_df, color = 'c')

plt.title('Machine Learning Algorithm Accuracy Score \n')

plt.xlabel('Accuracy Score (%)')

plt.ylabel('Algorithm')

plt.xlim(0.80, 0.86)
# Parameter's to search

n_estimators = [10, 25, 50, 75, 100]

max_depth = [3, None]

max_features = [1, 3, 5, 7]

min_samples_split = [2, 4, 6, 8, 10]

min_samples_leaf = [2, 4, 6, 8, 10]



# Setting up parameter grid

hyperparams = {'n_estimators': n_estimators, 'max_depth': max_depth, 'max_features': max_features,

               'min_samples_split': min_samples_split, 'min_samples_leaf': min_samples_leaf}



# Run GridSearch CV

gd=GridSearchCV(estimator = RandomForestClassifier(), param_grid = hyperparams, 

                verbose=True, cv=5, scoring = "accuracy")



# Fitting model and return results

gd.fit(X_train, y_train)

print(gd.best_score_)

print(gd.best_estimator_)


# Parameter's to search

n_neighbors = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20]

algorithm = ['auto']

weights = ['uniform', 'distance']

leaf_size = [1, 2, 3, 4, 5, 10, 15, 20, 25, 30]



# Setting up parameter grid

hyperparams = {'algorithm': algorithm, 'weights': weights, 'leaf_size': leaf_size, 

               'n_neighbors': n_neighbors}



# Run GridSearch CV

gd=GridSearchCV(estimator = KNeighborsClassifier(), param_grid = hyperparams, 

                verbose=True, cv=5, scoring = "accuracy")



# Fitting model and return results

gd.fit(X_train, y_train)

print(gd.best_score_)

print(gd.best_estimator_)


# Parameter's to search

penalty = ['l1', 'l2']

C = np.logspace(0, 4, 10)



# Setting up parameter grid

hyperparams = {'penalty': penalty, 'C': C}



# Run GridSearch CV

gd=GridSearchCV(estimator = LogisticRegression(), param_grid = hyperparams, 

                verbose=True, cv=5, scoring = "accuracy")



# Fitting model and return results

gd.fit(X_train, y_train)

print(gd.best_score_)

print(gd.best_estimator_)
# Parameter's to search

learning_rate = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.2]

n_estimators = [10, 25, 50, 75, 100, 250, 500, 750, 1000]



# Setting up parameter grid

hyperparams = {'learning_rate': learning_rate, 'n_estimators': n_estimators}



# Run GridSearch CV

gd=GridSearchCV(estimator = XGBClassifier(), param_grid = hyperparams, 

                verbose=True, cv=5, scoring = "accuracy")



# Fitting model and return results

gd.fit(X_train, y_train)

print(gd.best_score_)

print(gd.best_estimator_)
max_depth = [3, 4, 5, 6, 7, 8, 9, 10]

min_child_weight = [1, 2, 3, 4, 5, 6]



hyperparams = {'max_depth': max_depth, 'min_child_weight': min_child_weight}



gd=GridSearchCV(estimator = XGBClassifier(learning_rate=0.0001, n_estimators=10), param_grid = hyperparams, 

                verbose=True, cv=5, scoring = "accuracy")



gd.fit(X_train, y_train)

print(gd.best_score_)

print(gd.best_estimator_)
gamma = [i*0.1 for i in range(0,5)]



hyperparams = {'gamma': gamma}



gd=GridSearchCV(estimator = XGBClassifier(learning_rate=0.0001, n_estimators=10, max_depth=3, 

                                          min_child_weight=1), param_grid = hyperparams, 

                verbose=True, cv=5, scoring = "accuracy")



gd.fit(X_train, y_train)

print(gd.best_score_)

print(gd.best_estimator_)
subsample = [0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1]

colsample_bytree = [0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1]

    

hyperparams = {'subsample': subsample, 'colsample_bytree': colsample_bytree}



gd=GridSearchCV(estimator = XGBClassifier(learning_rate=0.0001, n_estimators=10, max_depth=3, 

                                          min_child_weight=1, gamma=0), param_grid = hyperparams, 

                verbose=True, cv=5, scoring = "accuracy")



gd.fit(X_train, y_train)

print(gd.best_score_)

print(gd.best_estimator_)
reg_alpha = [1e-5, 1e-2, 0.1, 1, 100]

    

hyperparams = {'reg_alpha': reg_alpha}



gd=GridSearchCV(estimator = XGBClassifier(learning_rate=0.0001, n_estimators=10, max_depth=3, 

                                          min_child_weight=1, gamma=0, subsample=0.6, colsample_bytree=0.9),

                                         param_grid = hyperparams, verbose=True, cv=5, scoring = "accuracy")



gd.fit(X_train, y_train)

print(gd.best_score_)

print(gd.best_estimator_)


# Parameter's to search

learning_rate = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.2]

n_estimators = [100, 250, 500, 750, 1000, 1250, 1500]



# Setting up parameter grid

hyperparams = {'learning_rate': learning_rate, 'n_estimators': n_estimators}



# Run GridSearch CV

gd=GridSearchCV(estimator = GradientBoostingClassifier(), param_grid = hyperparams, 

                verbose=True, cv=5, scoring = "accuracy")



# Fitting model and return results

gd.fit(X_train, y_train)

print(gd.best_score_)

print(gd.best_estimator_)
# Parameter's to search

Cs = [0.001, 0.01, 0.1, 1, 5, 10, 15, 20, 50, 100]

gammas = [0.001, 0.01, 0.1, 1]



# Setting up parameter grid

hyperparams = {'C': Cs, 'gamma' : gammas}



# Run GridSearch CV

gd=GridSearchCV(estimator = SVC(probability=True), param_grid = hyperparams, 

                verbose=True, cv=5, scoring = "accuracy")



# Fitting model and return results

gd.fit(X_train, y_train)

print(gd.best_score_)

print(gd.best_estimator_)


# Parameter's to search

n_estimators = [10, 25, 50, 75, 100]

max_depth = [3, None]

max_features = [1, 3, 5, 7]

min_samples_split = [2, 4, 6, 8, 10]

min_samples_leaf = [2, 4, 6, 8, 10]



# Setting up parameter grid

hyperparams = {'n_estimators': n_estimators, 'max_depth': max_depth, 'max_features': max_features,

               'min_samples_split': min_samples_split, 'min_samples_leaf': min_samples_leaf}



# Run GridSearch CV

gd=GridSearchCV(estimator = ExtraTreesClassifier(), param_grid = hyperparams, 

                verbose=True, cv=5, scoring = "accuracy")



# Fitting model and return results

gd.fit(X_train, y_train)

print(gd.best_score_)

print(gd.best_estimator_)


# Parameter's to search

n_estimators = [10, 25, 50, 75, 100, 125, 150, 200]

learning_rate = [0.001, 0.01, 0.1, 0.5, 1, 1.5, 2]



# Setting up parameter grid

hyperparams = {'n_estimators': n_estimators, 'learning_rate': learning_rate}



# Run GridSearch CV

gd=GridSearchCV(estimator = AdaBoostClassifier(), param_grid = hyperparams, 

                verbose=True, cv=5, scoring = "accuracy")



# Fitting model and return results

gd.fit(X_train, y_train)

print(gd.best_score_)

print(gd.best_estimator_)
# Parameter's to search

n_restarts_optimizer = [0, 1, 2, 3]

max_iter_predict = [1, 2, 5, 10, 20, 35, 50, 100]

warm_start = [True, False]



# Setting up parameter grid

hyperparams = {'n_restarts_optimizer': n_restarts_optimizer, 'max_iter_predict': max_iter_predict, 'warm_start': warm_start}



# Run GridSearch CV

gd=GridSearchCV(estimator = GaussianProcessClassifier(), param_grid = hyperparams, 

                verbose=True, cv=5, scoring = "accuracy")



# Fitting model and return results

gd.fit(X_train, y_train)

print(gd.best_score_)

print(gd.best_estimator_)
# Parameter's to search

n_estimators = [10, 15, 20, 25, 50, 75, 100, 150]

max_samples = [1, 2, 3, 5, 7, 10, 15, 20, 25, 30, 50]

max_features = [1, 3, 5, 7]



# Setting up parameter grid

hyperparams = {'n_estimators': n_estimators, 'max_samples': max_samples, 'max_features': max_features}



# Run GridSearch CV

gd=GridSearchCV(estimator = BaggingClassifier(), param_grid = hyperparams, 

                verbose=True, cv=5, scoring = "accuracy")



# Fitting model and return results

gd.fit(X_train, y_train)

print(gd.best_score_)

print(gd.best_estimator_)
# Initiate tuned models



ran = RandomForestClassifier(n_estimators=50,

                             max_depth=3, 

                             max_features=7,

                             min_samples_leaf=8, 

                             min_samples_split=6,  

                             random_state=1)



knn = KNeighborsClassifier(algorithm='auto', 

                           leaf_size=3, 

                           n_neighbors=10, 

                           weights='uniform')



log = LogisticRegression(C=21.544346900318832,

                         penalty='l2')



xgb = XGBClassifier(learning_rate=0.0001, 

                    n_estimators=10,

                    random_state=1)



gbc = GradientBoostingClassifier(learning_rate=0.0005,

                                 n_estimators=1250,

                                 random_state=1)



svc = SVC(C=50, gamma=0.01, probability=True)





ext = ExtraTreesClassifier(max_depth=3, 

                           max_features=7,

                           min_samples_leaf=8, 

                           min_samples_split=4,

                           n_estimators=25,

                           random_state=1)



ada = AdaBoostClassifier(learning_rate=0.5, 

                         n_estimators=25,

                         random_state=1)



gpc = GaussianProcessClassifier(max_iter_predict=1)



bag = BaggingClassifier(max_features=7, max_samples=50, n_estimators=20,random_state=1)



# Lists

models = [ran, knn, log, xgb, gbc, svc, ext, ada, gnb, gpc, bag]         

scores_v3 = []



# Fit & cross-validate

for mod in models:

    mod.fit(X_train, y_train)

    acc = cross_val_score(mod, X_train, y_train, scoring = "accuracy", cv = 10)

    scores_v3.append(acc.mean())
# Creating a table of results, ranked highest to lowest

results = pd.DataFrame({

    'Model': ['Random Forest', 'K Nearest Neighbour', 'Logistic Regression', 'XGBoost', 'Gradient Boosting', 'SVC', 'Extra Trees', 'AdaBoost', 'Gaussian Naive Bayes', 'Gaussian Process', 'Bagging Classifier'],

    'Original Score': scores,

    'Score with feature selection': scores_v2,

    'Score with tuned parameters': scores_v3})



result_df = results.sort_values(by='Score with tuned parameters', ascending=False).reset_index(drop=True)

result_df.head(11)
# Plot results

sns.barplot(x='Score with tuned parameters', y = 'Model', data = result_df, color = 'c')

plt.title('Machine Learning Algorithm Accuracy Score \n')

plt.xlabel('Accuracy Score (%)')

plt.ylabel('Algorithm')

plt.xlim(0.82, 0.86)
#Hard Vote or majority rules w/Tuned Hyperparameters

grid_hard = VotingClassifier(estimators = [('Random Forest', ran), 

                                           ('Logistic Regression', log),

                                           ('XGBoost', xgb),

                                           ('Gradient Boosting', gbc),

                                           ('Extra Trees', ext),

                                           ('AdaBoost', ada),

                                           ('Gaussian Process', gpc),

                                           ('SVC', svc),

                                           ('K Nearest Neighbour', knn),

                                           ('Bagging Classifier', bag)], voting = 'hard')



grid_hard_cv = model_selection.cross_validate(grid_hard, X_train, y_train, cv = 10, return_train_score=True)

grid_hard.fit(X_train, y_train)

 



print("Hard voting on train set score mean: {:.2f}". format(grid_hard_cv['train_score'].mean()*100))

print("Hard voting on test set score mean: {:.2f}". format(grid_hard_cv['test_score'].mean()*100))

grid_soft = VotingClassifier(estimators = [('Random Forest', ran), 

                                           ('Logistic Regression', log),

                                           ('XGBoost', xgb),

                                           ('Gradient Boosting', gbc),

                                           ('Extra Trees', ext),

                                           ('AdaBoost', ada),

                                           ('Gaussian Process', gpc),

                                           ('SVC', svc),

                                           ('K Nearest Neighbour', knn),

                                           ('Bagging Classifier', bag)], voting = 'soft')



grid_soft_cv = model_selection.cross_validate(grid_soft, X_train, y_train, cv = 10, return_train_score=True)

grid_soft.fit(X_train, y_train)



print("Soft voting on train set score mean: {:.2f}". format(grid_soft_cv['train_score'].mean()*100)) 

print("Soft voting on test set score mean: {:.2f}". format(grid_soft_cv['test_score'].mean()*100))

# Final predictions

predictions = grid_hard.predict(X_test)



submission = pd.concat([pd.DataFrame(passId), pd.DataFrame(predictions)], axis = 'columns')



submission.columns = ["PassengerId", "Survived"]

submission.to_csv('titanic_submission.csv', header = True, index = False)