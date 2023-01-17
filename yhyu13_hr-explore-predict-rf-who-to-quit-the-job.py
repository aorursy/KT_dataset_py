import numpy as np

import pandas as pd



import matplotlib.pyplot as plt

%matplotlib inline



import seaborn as sns
df = pd.read_csv('../input/HR_comma_sep.csv')
df.info()
df.head()
df['sales'].unique()
df['promotion_last_5years'].unique()
df['salary'].unique()
df.mean()
df.mean()['average_montly_hours']/30
print('# of people left = {}'.format(df[df['left']==1].size))

print('# of people stayed = {}'.format(df[df['left']==0].size))

print('protion of people who left in 5 years = {}%'.format(int(df[df['left']==1].size/df.size*100)))
corrmat = df.corr()

f, ax = plt.subplots(figsize=(4, 4))

# Draw the heatmap using seaborn

sns.heatmap(corrmat, vmax=.8, square=True)

plt.show()
corrmat_low = df[df['salary'] == 'low'].corr()

corrmat_medium = df[df['salary'] == 'medium'].corr()

corrmat_high = df[df['salary'] == 'high'].corr()



sns.heatmap(corrmat_low, vmax=.8, square=True,annot=True,fmt='.2f')
sns.heatmap(corrmat_medium, vmax=.8, square=True,annot=True,fmt='.2f')
sns.heatmap(corrmat_high, vmax=.8, square=True,annot=True,fmt='.2f')
df_low = df[df['salary'] == 'low']

df_medium = df[df['salary'] == 'medium']

df_high = df[df['salary'] == 'high']



print('# of low salary employees= ',df_low.shape[0])

print('# of medium salary employees= ',df_medium.shape[0])

print('# of high salary employees= ',df_high.shape[0])
fmt = '{:<22}{:<25}{}'



print(fmt.format('', 'mean', 'std'))

for i, (mean, std) in enumerate(zip(df_low.mean(), df_low.std())):

    print(fmt.format(df_low.columns[i], mean, std))

print('\n')

for i, (mean, std) in enumerate(zip(df_medium.mean(), df_medium.std())):

    print(fmt.format(df_low.columns[i], mean, std))

print('\n')

for i, (mean, std) in enumerate(zip(df_high.mean(), df_high.std())):

    print(fmt.format(df_low.columns[i], mean, std))
sns.factorplot("sales", col="salary", col_wrap=4, data=df, kind="count", size=10, aspect=.8)
df.groupby('sales').mean()['satisfaction_level'].plot(kind='bar',color='r')
from sklearn.linear_model import SGDClassifier

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.calibration import CalibratedClassifierCV

import xgboost as xgb

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import StratifiedKFold

from sklearn.feature_selection import SelectFromModel

from sklearn.linear_model import LogisticRegression

from sklearn import svm
df_copy = pd.get_dummies(df)

df_copy.head()
df1 = df_copy

y = df1['left'].values

df1 = df1.drop(['left'],axis=1)

X = df1.values
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.50)
log_reg = LogisticRegression()

log_reg.fit(Xtrain, ytrain)

y_val_l = log_reg.predict_proba(Xtest)

print("Validation accuracy: ", sum(pd.DataFrame(y_val_l).idxmax(axis=1).values

                                   == ytest)/len(ytest))
sdg = SGDClassifier()

sdg.fit(Xtrain, ytrain)

y_val_l = sdg.predict(Xtest)

print("Validation accuracy: ", sum(y_val_l

                                   == ytest)/len(ytest))
radm = RandomForestClassifier()

radm.fit(Xtrain, ytrain)

y_val_l = radm.predict_proba(Xtest)

print("Validation accuracy: ", sum(pd.DataFrame(y_val_l).idxmax(axis=1).values

                                   == ytest)/len(ytest))
clf = radm
indices = np.argsort(radm.feature_importances_)[::-1]



# Print the feature ranking

print('Feature ranking:')



for f in range(df1.shape[1]):

    print('%d. feature %d %s (%f)' % (f+1 , indices[f], df1.columns[indices[f]],

                                      radm.feature_importances_[indices[f]]))
df_copy = df

y = LabelEncoder().fit(df['salary']).transform(df['salary'])

df2 = df_copy.drop(['salary'],axis=1)

X = pd.get_dummies(df2).values
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2)
radm = RandomForestClassifier()

radm.fit(Xtrain, ytrain)

y_val_l = radm.predict_proba(Xtest)

print("Validation accuracy: ", sum(pd.DataFrame(y_val_l).idxmax(axis=1).values

                                   == ytest)/len(ytest))
log_reg = LogisticRegression()

log_reg.fit(Xtrain, ytrain)

y_val_l = log_reg.predict_proba(Xtest)

print("Validation accuracy: ", sum(pd.DataFrame(y_val_l).idxmax(axis=1).values

                                   == ytest)/len(ytest))
df_copy = df

y = LabelEncoder().fit(df['sales']).transform(df['sales'])

df2 = df_copy.drop(['sales'],axis=1)

X = pd.get_dummies(df2).values
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2)
radm = RandomForestClassifier()

radm.fit(Xtrain, ytrain)

y_val_l = radm.predict_proba(Xtest)

print("Validation accuracy: ", sum(pd.DataFrame(y_val_l).idxmax(axis=1).values

                                   == ytest)/len(ytest))
log_reg = LogisticRegression()

log_reg.fit(Xtrain, ytrain)

y_val_l = log_reg.predict_proba(Xtest)

print("Validation accuracy: ", sum(pd.DataFrame(y_val_l).idxmax(axis=1).values

                                   == ytest)/len(ytest))
stay = df[df['left'] == 0]

stay_copy = pd.get_dummies(stay)
df1 = stay_copy

y = df1['left'].values

df1 = df1.drop(['left'],axis=1)

X = df1.values
pred = clf.predict_proba(X)
sum(pred[:,1]==1)
stay['will leave the job'] = pred[:,1]
stay[stay['will leave the job']>=0.5]