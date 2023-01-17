import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
train_df = pd.read_csv('../input/train.csv', parse_dates=['DateTime'])
train_df.head()
train_df.info()
train_df.describe()
test_df = pd.read_csv('../input/test.csv', parse_dates=['DateTime'])
test_df.info()
test_df.describe(include = 'all')
train_df.rename(columns = {'OutcomeType': 'Outcome1', 'OutcomeSubtype': 'Outcome2', 'AnimalType': 'Animal', 'SexuponOutcome': 'Sex', 'AgeuponOutcome': 'Age'}, inplace=True)
test_df.rename(columns = {'AnimalType': 'Animal', 'SexuponOutcome': 'Sex', 'AgeuponOutcome': 'Age'}, inplace=True)
train_df.drop('AnimalID', axis = 1, inplace = True)
test_ID = test_df['ID']
test_df.drop('ID', axis = 1, inplace = True)
train_df.isnull().sum()/train_df.shape[0]
test_df.isnull().sum()/test_df.shape[0]
sns.countplot(x = 'Outcome1', data = train_df)
train_df['Age'].unique()
def convert_age(col):
    try:
        num = col.split()[0]
        unit = col.split()[1]

        if unit == 'year' or unit == 'years':
            if num == '0':
                return 365/2
            return int(num) * 365
        if unit == 'month' or unit == 'months':
            return int(num) * 30
        if unit == 'week' or unit == 'weeks':
            return int(num) * 7
        if unit == 'day' or unit == 'days':
            return int(num)
    except AttributeError: 
        pass
train_df['Age'] = train_df['Age'].apply(convert_age)
test_df['Age'] = test_df['Age'].apply(convert_age)
train_df['Age'].fillna(train_df['Age'].mean(), inplace = True)
test_df['Age'].fillna(test_df['Age'].mean(), inplace = True)
date = train_df['DateTime']
date.dt.dayofweek.head()
def convert_date(df, col):
    fld = df[col]
    targ_pre = 'Date'
    for n in ('day', 'dayofweek', 'dayofyear', 'hour', 'is_month_start', 'is_month_end', 'is_quarter_start', 'is_quarter_end', 
             'is_year_end', 'is_year_start', 'month', 'quarter', 'week', 'weekofyear', 'year'):
        df[targ_pre + n] = getattr(fld.dt, n.lower())
    df.drop(col, axis = 1, inplace = True)
convert_date(train_df, 'DateTime')
convert_date(test_df, 'DateTime')
test_df.columns
train_df['Name'] = train_df['Name'].fillna('NaN')
train_df['Sex'] = train_df['Sex'].fillna('NaN')
test_df['Name'] = test_df['Name'].fillna('NaN')
y = train_df['Outcome1']
X = train_df.drop(['Outcome1', 'Outcome2'], axis = 1)
from sklearn import preprocessing
labelEnc = preprocessing.LabelEncoder()
for col in test_df.columns.values:
    if test_df[col].dtypes == 'object':
        tmp = X[col].append(test_df[col])
        labelEnc.fit(tmp.values)
        X[col]=labelEnc.transform(X[col])
        test_df[col]=labelEnc.transform(test_df[col])
X.head()
import os
os.makedirs('tmp', exist_ok = True)
X.to_feather('tmp/X')
test_df.to_feather('tmp/test_df')
X = pd.read_feather('tmp/X')
test_df = pd.read_feather('tmp/test_df')
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=100)
rf.fit(X, y)
pred = rf.predict(test_df)
pred.shape
sub_df = pd.read_csv('../input/sample_submission.csv')
sub_df.head()
pred_prob = rf.predict_proba(test_df)
rf.classes_
submit_df = pd.DataFrame(pred_prob, columns = ['Adoption', 'Died', 'Euthanasia', 'Return_to_owner', 'Transfer'])
submit_df.insert(0, 'ID', test_ID)
submit_df.head()
submit_df.to_csv('submission.csv', index = False)
