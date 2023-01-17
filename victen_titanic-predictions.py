# to handle datasets

import pandas as pd

import numpy as np



# for text / string processing

import re



# for plotting

import matplotlib.pyplot as plt

%matplotlib inline



# to divide train and test set

from sklearn.model_selection import train_test_split



# feature scaling

from sklearn.preprocessing import MinMaxScaler



# for tree binarisation

from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import cross_val_score





# to build the models

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import AdaBoostClassifier

from sklearn.ensemble import RandomForestClassifier

import xgboost as xgb



# to evaluate the models

from sklearn.metrics import roc_auc_score

from sklearn import metrics



pd.pandas.set_option('display.max_columns', None)

pd.pandas.set_option('display.max_rows', None)



import warnings

warnings.filterwarnings('ignore')
# load dataset

data = pd.read_csv('../input/titanic/train.csv')

data.head()
# Load the dataset for submission (the one on which our model will be evaluated by Kaggle)

# it contains exactly the same variables, but not the target



submission = pd.read_csv('../input/titanic/test.csv')

submission.head()
# let's inspect the type of variables in pandas

data.dtypes
print('Number of PassengerId labels: ', len(data.PassengerId.unique()))

print('Number  of passengers on the Titanic: ', len(data))
# find categorical variables

categorical = [var for var in data.columns if data[var].dtype=='O']

print('There are {} categorical variables'.format(len(categorical)))
# find numerical variables

numerical = [var for var in data.columns if data[var].dtype!='O']

print('There are {} numerical variables'.format(len(numerical)))
# view of categorical variables

data[categorical].head()
# view of numerical variables

data[numerical].head()
# let's visualise the values of the discrete variables

for var in ['Pclass',  'SibSp', 'Parch']:

    print(var, ' values: ', data[var].unique())
# let's visualise the percentage of missing values

data.isnull().mean()
numerical = [var for var in numerical if var not in['Survived', 'PassengerId']]

numerical
# let's make boxplots to visualise outliers in the continuous variables 

# Age and Fare



plt.figure(figsize=(15,6))

plt.subplot(1, 2, 1)

fig = data.boxplot(column='Age')

fig.set_title('')

fig.set_ylabel('Age')



plt.subplot(1, 2, 2)

fig = data.boxplot(column='Fare')

fig.set_title('')

fig.set_ylabel('Fare')
# first we plot the distributions to find out if they are Gaussian or skewed.

# Depending on the distribution, we will use the normal assumption or the interquantile

# range to find outliers



plt.figure(figsize=(15,6))

plt.subplot(1, 2, 1)

fig = data.Age.hist(bins=20)

fig.set_ylabel('Number of passengers')

fig.set_xlabel('Age')



plt.subplot(1, 2, 2)

fig = data.Fare.hist(bins=20)

fig.set_ylabel('Number of passengers')

fig.set_xlabel('Fare')
# find outliers



# Age

Upper_boundary = data.Age.mean() + 3* data.Age.std()

Lower_boundary = data.Age.mean() - 3* data.Age.std()

print('Age outliers are values < {lowerboundary} or > {upperboundary}'.format(lowerboundary=Lower_boundary, upperboundary=Upper_boundary))



# Fare

IQR = data.Fare.quantile(0.75) - data.Fare.quantile(0.25)

Lower_fence = data.Fare.quantile(0.25) - (IQR * 3)

Upper_fence = data.Fare.quantile(0.75) + (IQR * 3)

print('Fare outliers are values < {lowerboundary} or > {upperboundary}'.format(lowerboundary=Lower_fence, upperboundary=Upper_fence))

# outlies in discrete variables

for var in ['Pclass',  'SibSp', 'Parch']:

    print(data[var].value_counts() / np.float(len(data)))

    print()
for var in categorical:

    print(var, ' contains ', len(data[var].unique()), ' labels')
# Cabin

data['Cabin_numerical'] = data.Cabin.str.extract('(\d+)') # extracts number from string

data['Cabin_numerical'] = data['Cabin_numerical'].astype('float') # parses the above variable to float type



data['Cabin_categorical'] = data['Cabin'].str[0] # captures first letter of string (the letter of the cabin)



# same for submission data

submission['Cabin_numerical'] = submission.Cabin.str.extract('(\d+)')

submission['Cabin_numerical'] = submission['Cabin_numerical'].astype('float')



submission['Cabin_categorical'] = submission['Cabin'].str[0]



data[['Cabin', 'Cabin_numerical', 'Cabin_categorical']].head()
# drop the original variable

data.drop(labels='Cabin', inplace=True, axis=1)

submission.drop(labels='Cabin', inplace=True, axis=1)
#  Ticket

# extract the last bit of ticket as number

data['Ticket_numerical'] = data.Ticket.apply(lambda s: s.split()[-1])

data['Ticket_numerical'] = np.where(data.Ticket_numerical.str.isdigit(), data.Ticket_numerical, np.nan)

data['Ticket_numerical'] = data['Ticket_numerical'].astype('float')



# extract the first part of ticket as category

data['Ticket_categorical'] = data.Ticket.apply(lambda s: s.split()[0])

data['Ticket_categorical'] = np.where(data.Ticket_categorical.str.isdigit(), np.nan, data.Ticket_categorical)



# submission

submission['Ticket_numerical'] = submission.Ticket.apply(lambda s: s.split()[-1])

submission['Ticket_numerical'] = np.where(submission.Ticket_numerical.str.isdigit(), submission.Ticket_numerical, np.nan)

submission['Ticket_numerical'] = submission['Ticket_numerical'].astype('float')



# extract the first part of ticket as category

submission['Ticket_categorical'] = submission.Ticket.apply(lambda s: s.split()[0])

submission['Ticket_categorical'] = np.where(submission.Ticket_categorical.str.isdigit(), np.nan, submission.Ticket_categorical)



data[['Ticket', 'Ticket_numerical', 'Ticket_categorical']].head()
# let's explore the ticket categorical part a bit further

data.Ticket_categorical.unique()
# it contains several labels, some of them seem very similar apart from the punctuation

# I will try to reduce this number of labels a bit further



# remove non letter characters from string

text = data.Ticket_categorical.apply(lambda x: re.sub("[^a-zA-Z]", '', str(x)))



# to visualise the output and compare with input

pd.concat([text, data.Ticket_categorical], axis=1).head()
# set to upper case: we reduce the number of labels quite a bit

text = text.str.upper()

text.unique()
# process the variable in submission as well

data['Ticket_categorical'] = text



submission['Ticket_categorical'] = submission.Ticket_categorical.apply(lambda x: re.sub("[^a-zA-Z]", '', str(x)))

submission['Ticket_categorical'] = submission['Ticket_categorical'].str.upper()
# drop the original variable

data.drop(labels='Ticket', inplace=True, axis=1)

submission.drop(labels='Ticket', inplace=True, axis=1)
def get_title(passenger):

    # extracts the title from the name variable

    line = passenger

    if re.search('Mrs', line):

        return 'Mrs'

    elif re.search('Mr', line):

        return 'Mr'

    elif re.search('Miss', line):

        return 'Miss'

    elif re.search('Master', line):

        return 'Master'

    else:

        return 'Other'

    

data['Title'] = data['Name'].apply(get_title)

submission['Title'] = submission['Name'].apply(get_title)



data[['Name', 'Title']].head()
# drop the original variable

data.drop(labels='Name', inplace=True, axis=1)

submission.drop(labels='Name', inplace=True, axis=1)
# create a variable indicating family size (including the passenger)

# sums siblings and parents



data['Family_size'] = data['SibSp']+data['Parch']+1

submission['Family_size'] = submission['SibSp']+submission['Parch']+1



print(data.Family_size.value_counts()/ np.float(len(data)))



(data.Family_size.value_counts() / np.float(len(data))).plot.bar()
# variable indicating if passenger was a mother

data['is_mother'] = np.where((data.Sex =='female')&(data.Parch>=1)&(data.Age>18),1,0)

submission['is_mother'] = np.where((submission.Sex =='female')&(submission.Parch>=1)&(submission.Age>18),1,0)



data[['Sex', 'Parch', 'Age', 'is_mother']].head()
data.loc[data.is_mother==1, ['Sex', 'Parch', 'Age', 'is_mother']].head()
print('there were {} mothers in the Titanic'.format(data.is_mother.sum()))
data[['Cabin_numerical', 'Ticket_numerical', 'is_mother', 'Family_size']].isnull().mean()
# first we plot the distributions to find out if they are Gaussian or skewed.

# Depending on the distribution, we will use the normal assumption or the interquantile

# range to find outliers



plt.figure(figsize=(15,6))

plt.subplot(1, 2, 1)

fig = data.Cabin_numerical.hist(bins=50)

fig.set_ylabel('Number of passengers')

fig.set_xlabel('Cabin number')



plt.subplot(1, 2, 2)

fig = data.Ticket_numerical.hist(bins=50)

fig.set_ylabel('Number of passengers')

fig.set_xlabel('Ticket number')
# let's visualise outliers with the boxplot and whiskers

plt.figure(figsize=(15,6))

plt.subplot(1, 2, 1)

fig = data.boxplot(column='Cabin_numerical')

fig.set_title('')

fig.set_ylabel('Cabin number')



plt.subplot(1, 2, 2)

fig = data.boxplot(column='Ticket_numerical')

fig.set_title('')

fig.set_ylabel('Ticket number')
# Ticket numerical

IQR = data.Ticket_numerical.quantile(0.75) - data.Ticket_numerical.quantile(

    0.25)

Lower_fence = data.Ticket_numerical.quantile(0.25) - (IQR * 3)

Upper_fence = data.Ticket_numerical.quantile(0.75) + (IQR * 3)

print(

    'Ticket number outliers are values < {lowerboundary} or > {upperboundary}'.

    format(lowerboundary=Lower_fence, upperboundary=Upper_fence))

passengers = len(data[data.Ticket_numerical > Upper_fence]) / np.float(

    len(data))

print('Number of passengers with ticket values higher than {upperboundary}: {passengers}'.format(upperboundary=Upper_fence, passengers=passengers))
data[['Cabin_categorical', 'Ticket_categorical', 'Title']].isnull().mean()
for var in ['Cabin_categorical', 'Ticket_categorical', 'Title']:

    print(var, ' contains ', len(data[var].unique()), ' labels')
# rare / infrequent labels (less than 1% of passengers)

for var in ['Cabin_categorical', 'Ticket_categorical', 'Title']:

    print(data[var].value_counts() / np.float(len(data)))

    print()
# Let's separate into train and test set



X_train, X_test, y_train, y_test = train_test_split(data, data.Survived, test_size=0.2,

                                                    random_state=0)

X_train.shape, X_test.shape
# let's group again the variables into categorical or numerical

# now considering the newly created variables



def find_categorical_and_numerical_variables(dataframe):

    cat_vars = [col for col in data.columns if data[col].dtypes == 'O']

    num_vars  = [col for col in data.columns if data[col].dtypes != 'O']

    return cat_vars, num_vars

                 

categorical, numerical = find_categorical_and_numerical_variables(data)       
categorical
numerical = [var for var in numerical if var not in ['Survived','PassengerId']]

numerical
# print variables with missing data

for col in numerical:

    if X_train[col].isnull().mean()>0:

        print(col, X_train[col].isnull().mean())
def impute_na(X_train, df, variable):

    # make temporary df copy

    temp = df.copy()

    

    # extract random from train set to fill the na

    random_sample = X_train[variable].dropna().sample(temp[variable].isnull().sum(), random_state=0)

    

    # pandas needs to have the same index in order to merge datasets

    random_sample.index = temp[temp[variable].isnull()].index

    temp.loc[temp[variable].isnull(), variable] = random_sample

    return temp[variable]
# Age and ticket

# add variable indicating missingness

for df in [X_train, X_test, submission]:

    for var in ['Age', 'Ticket_numerical']:

        df[var+'_NA'] = np.where(df[var].isnull(), 1, 0)

    

# replace by random sampling

for df in [X_train, X_test, submission]:

    for var in ['Age', 'Ticket_numerical']:

        df[var] = impute_na(X_train, df, var)

    



# Cabin numerical

extreme = X_train.Cabin_numerical.mean() + X_train.Cabin_numerical.std()*3

for df in [X_train, X_test, submission]:

    df.Cabin_numerical.fillna(extreme, inplace=True)
# print variables with missing data

for col in categorical:

    if X_train[col].isnull().mean()>0:

        print(col, X_train[col].isnull().mean())
# add label indicating 'Missing' to Cabin categorical

# or replace by most frequent label in Embarked



for df in [X_train, X_test, submission]:

    df['Embarked'].fillna(X_train['Embarked'].mode()[0], inplace=True)

    df['Cabin_categorical'].fillna('Missing', inplace=True)
# check absence of null values

X_train.isnull().sum()
X_test.isnull().sum()
submission.isnull().sum()
# Fare in the submission dataset contains one null value, I will replace it by the median 

submission.Fare.fillna(X_train.Fare.median(), inplace=True)
def top_code(df, variable, top):

    return np.where(df[variable]>top, top, df[variable])



for df in [X_train, X_test, submission]:

    df['Age'] = top_code(df, 'Age', 73)

    df['SibSp'] = top_code(df, 'SibSp', 4)

    df['Parch'] = top_code(df, 'Parch', 2)

    df['Family_size'] = top_code(df, 'Family_size', 7)
# let's check that it worked

for var in ['Age',  'SibSp', 'Parch', 'Family_size']:

    print(var, ' max value: ', X_train[var].max())
# let's check that it worked

for var in ['Age',  'SibSp', 'Parch', 'Family_size']:

    print(var, ' max value: ', submission[var].max())
# find quantiles and discretise train set

X_train['Fare'], bins = pd.qcut(x=X_train['Fare'], q=8, retbins=True, precision=3, duplicates='raise')

X_test['Fare'] = pd.cut(x = X_test['Fare'], bins=bins, include_lowest=True)

submission['Fare'] = pd.cut(x = submission['Fare'], bins=bins, include_lowest=True)
submission.Fare.isnull().sum()
t1 = X_train.groupby(['Fare'])['Fare'].count() / np.float(len(X_train))

t2 = X_test.groupby(['Fare'])['Fare'].count() / np.float(len(X_test))

t3 = submission.groupby(['Fare'])['Fare'].count() / np.float(len(submission))



temp = pd.concat([t1,t2,t3], axis=1)

temp.columns = ['train', 'test', 'submission']

temp.plot.bar(figsize=(12,6))
# find quantiles and discretise train set

X_train['Ticket_numerical'], bins = pd.qcut(x=X_train['Ticket_numerical'], q=8, retbins=True, precision=3, duplicates='raise')

X_test['Ticket_numerical'] = pd.cut(x = X_test['Ticket_numerical'], bins=bins, include_lowest=True)

submission['Ticket_numerical_temp'] = pd.cut(x = submission['Ticket_numerical'], bins=bins, include_lowest=True)
X_test.Ticket_numerical.isnull().sum()
submission.Ticket_numerical_temp.isnull().sum()
submission[submission.Ticket_numerical_temp.isnull()][['Ticket_numerical', 'Ticket_numerical_temp']]
# inspect the ticket bins in training set

X_train.Ticket_numerical.unique()
X_train.Ticket_numerical.unique()[0]
submission.loc[submission.Ticket_numerical_temp.isnull(), 'Ticket_numerical_temp'] = X_train.Ticket_numerical.unique()[0]

submission.Ticket_numerical_temp.isnull().sum()
submission['Ticket_numerical'] = submission['Ticket_numerical_temp']

submission.drop(labels=['Ticket_numerical_temp'], inplace=True, axis=1)

submission.head()
# find unfrequent labels in categorical variables

for var in categorical:

    print(var, X_train[var].value_counts()/np.float(len(X_train)))

    print()
def rare_imputation(variable, which='rare'):    

    # find frequent labels

    temp = X_train.groupby([variable])[variable].count()/np.float(len(X_train))

    frequent_cat = [x for x in temp.loc[temp>0.01].index.values]

    

    # create new variables, with Rare labels imputed

    if which=='frequent':

        # find the most frequent category

        mode_label = X_train.groupby(variable)[variable].count().sort_values().tail(1).index.values[0]

        X_train[variable] = np.where(X_train[variable].isin(frequent_cat), X_train[variable], mode_label)

        X_test[variable] = np.where(X_test[variable].isin(frequent_cat), X_test[variable], mode_label)

        submission[variable] = np.where(submission[variable].isin(frequent_cat), submission[variable], mode_label)

    

    else:

        X_train[variable] = np.where(X_train[variable].isin(frequent_cat), X_train[variable], 'Rare')

        X_test[variable] = np.where(X_test[variable].isin(frequent_cat), X_test[variable], 'Rare')

        submission[variable] = np.where(submission[variable].isin(frequent_cat), submission[variable], 'Rare')
rare_imputation('Cabin_categorical', 'frequent')

rare_imputation('Ticket_categorical', 'rare')
# let's check that it worked

for var in categorical:

    print(var, X_train[var].value_counts()/np.float(len(X_train)))

    print()
# let's check that it worked

for var in categorical:

    print(var, submission[var].value_counts()/np.float(len(submission)))

    print()
X_train.head()
categorical
for df in [X_train, X_test, submission]:

    df['Sex']  = pd.get_dummies(df.Sex, drop_first=True)
X_train.Sex.unique()
X_test.Sex.unique()
submission.Sex.unique()
def encode_categorical_variables(var, target):

        # make label to risk dictionary

        ordered_labels = X_train.groupby([var])[target].mean().to_dict()

        

        # encode variables

        X_train[var] = X_train[var].map(ordered_labels)

        X_test[var] = X_test[var].map(ordered_labels)

        submission[var] = submission[var].map(ordered_labels)



# enccode labels in categorical vars

for var in categorical:

    encode_categorical_variables(var, 'Survived')
# parse discretised variables to object before encoding

for df in [X_train, X_test, submission]:

    df.Fare = df.Fare.astype('O')

    df.Ticket_numerical = df.Ticket_numerical.astype('O')
# encode labels

for var in ['Fare', 'Ticket_numerical']:

    print(var)

    encode_categorical_variables(var, 'Survived')
#let's inspect the dataset

X_train.head()
X_train.describe()
variables_that_need_scaling = ['Pclass', 'Age', 'Sibsp', 'Parch', 'Cabin_numerical', 'Family_size']
training_vars = [var for var in X_train.columns if var not in ['PassengerId', 'Survived']]

training_vars
# fit scaler

scaler = MinMaxScaler() # create an instance

scaler.fit(X_train[training_vars]) #  fit  the scaler to the train set and then transform it
xgb_model = xgb.XGBClassifier()



eval_set = [(X_test[training_vars], y_test)]

xgb_model.fit(X_train[training_vars], y_train, eval_metric="auc", eval_set=eval_set, verbose=False)



pred = xgb_model.predict_proba(X_train[training_vars])

print('xgb train roc-auc: {}'.format(roc_auc_score(y_train, pred[:,1])))

pred = xgb_model.predict_proba(X_test[training_vars])

print('xgb test roc-auc: {}'.format(roc_auc_score(y_test, pred[:,1])))
rf_model = RandomForestClassifier()

rf_model.fit(X_train[training_vars], y_train)



pred = rf_model.predict_proba(X_train[training_vars])

print('RF train roc-auc: {}'.format(roc_auc_score(y_train, pred[:,1])))

pred = rf_model.predict_proba(X_test[training_vars])

print('RF test roc-auc: {}'.format(roc_auc_score(y_test, pred[:,1])))
ada_model = AdaBoostClassifier()

ada_model.fit(X_train[training_vars], y_train)



pred = ada_model.predict_proba(X_train[training_vars])

print('Adaboost train roc-auc: {}'.format(roc_auc_score(y_train, pred[:,1])))

pred = ada_model.predict_proba(X_test[training_vars])

print('Adaboost test roc-auc: {}'.format(roc_auc_score(y_test, pred[:,1])))
logit_model = LogisticRegression()

logit_model.fit(scaler.transform(X_train[training_vars]), y_train)



pred = logit_model.predict_proba(scaler.transform(X_train[training_vars]))

print('Logit train roc-auc: {}'.format(roc_auc_score(y_train, pred[:,1])))

pred = ada_model.predict_proba(scaler.transform(X_test[training_vars]))

print('Logit test roc-auc: {}'.format(roc_auc_score(y_test, pred[:,1])))
pred_ls = []

for model in [xgb_model, rf_model, ada_model, logit_model]:

    pred_ls.append(pd.Series(model.predict_proba(X_test[training_vars])[:,1]))



final_pred = pd.concat(pred_ls, axis=1).mean(axis=1)

print('Ensemble test roc-auc: {}'.format(roc_auc_score(y_test,final_pred)))
tpr, tpr, thresholds = metrics.roc_curve(y_test, final_pred)

thresholds
accuracy_ls = []

for thres in thresholds:

    y_pred = np.where(final_pred>thres,1,0)

    accuracy_ls.append(metrics.accuracy_score(y_test, y_pred, normalize=True))

    

accuracy_ls = pd.concat([pd.Series(thresholds), pd.Series(accuracy_ls)],

                        axis=1)

accuracy_ls.columns = ['thresholds', 'accuracy']

accuracy_ls.sort_values(by='accuracy', ascending=False, inplace=True)

accuracy_ls.head()
pred_ls = []

for model in [xgb_model, rf_model, ada_model, logit_model]:

    pred_ls.append(pd.Series(model.predict_proba(submission[training_vars])[:,1]))



final_pred = pd.concat(pred_ls, axis=1).mean(axis=1)
final_pred = pd.Series(np.where(final_pred>0.40,1,0))
temp = pd.concat([submission.PassengerId, final_pred], axis=1)

temp.columns = ['PassengerId', 'Survived']

temp.head()
temp.to_csv('submit_titanic.csv', index=False)
importance = pd.Series(rf_model.feature_importances_)

importance.index = training_vars

importance.sort_values(inplace=True, ascending=False)

importance.plot.bar(figsize=(12,6))
importance = pd.Series(xgb_model.feature_importances_)

importance.index = training_vars

importance.sort_values(inplace=True, ascending=False)

importance.plot.bar(figsize=(12,6))
importance = pd.Series(np.abs(logit_model.coef_.ravel()))

importance.index = training_vars

importance.sort_values(inplace=True, ascending=False)

importance.plot.bar(figsize=(12,6))