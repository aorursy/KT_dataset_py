import numpy as np

import pandas as pd



df = pd.read_csv('../input/HR_comma_sep.csv')

df.info()
# rename some columns

df.rename(columns={'average_montly_hours':'average_monthly_hours', 'sales':'department'}, 

          inplace=True)

df.describe()
print ('Departments:')

print (df['department'].value_counts())

print ('\nSalary:')

print (df['salary'].value_counts())
df.corr()
import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

plt.style.use('ggplot')
# Attrition by department

plot = sns.factorplot(x='department', y='left', kind='bar', data=df)

plot.set_xticklabels(rotation=45, horizontalalignment='right');
# Attrition by salary level

plot = sns.factorplot(x='salary', y='left', kind='bar', data=df);
df[df['department']=='management']['salary'].value_counts().plot(kind='pie', title='Management salary level distribution');
df[df['department']=='RandD']['salary'].value_counts().plot(kind='pie', title='R&D dept salary level distribution');
# Satisfaction level vs. Attrition

bins = np.linspace(0.0001, 1.0001, 21)

plt.hist(df[df['left']==1]['satisfaction_level'], bins=bins, alpha=0.7, label='Employees Left')

plt.hist(df[df['left']==0]['satisfaction_level'], bins=bins, alpha=0.5, label='Employees Stayed')

plt.xlabel('satisfaction_level')

plt.xlim((0,1.05))

plt.legend(loc='best');
# Last evaluation vs. Attrition

bins = np.linspace(0.3501, 1.0001, 14)

plt.hist(df[df['left']==1]['last_evaluation'], bins=bins, alpha=1, label='Employees Left')

plt.hist(df[df['left']==0]['last_evaluation'], bins=bins, alpha=0.4, label='Employees Stayed')

plt.xlabel('last_evaluation')

plt.legend(loc='best');
# Number of projects vs. Attrition

bins = np.linspace(1.5, 7.5, 7)

plt.hist(df[df['left']==1]['number_project'], bins=bins, alpha=1, label='Employees Left')

plt.hist(df[df['left']==0]['number_project'], bins=bins, alpha=0.4, label='Employees Stayed')

plt.xlabel('number_project')

plt.grid(axis='x')

plt.legend(loc='best');
# Average monthly hours vs. Attrition

bins = np.linspace(75, 325, 11)

plt.hist(df[df['left']==1]['average_monthly_hours'], bins=bins, alpha=1, label='Employees Left')

plt.hist(df[df['left']==0]['average_monthly_hours'], bins=bins, alpha=0.4, label='Employees Stayed')

plt.xlabel('average_monthly_hours')

plt.legend(loc='best');
# Years at company vs. Attrition

bins = np.linspace(1.5, 10.5, 10)

plt.hist(df[df['left']==1]['time_spend_company'], bins=bins, alpha=1, label='Employees Left')

plt.hist(df[df['left']==0]['time_spend_company'], bins=bins, alpha=0.4, label='Employees Stayed')

plt.xlabel('time_spend_company')

plt.xlim((1,11))

plt.grid(axis='x')

plt.xticks(np.arange(2,11))

plt.legend(loc='best');
# Attrition by whether employee had work accident

plot = sns.factorplot(x='Work_accident', y='left', kind='bar', data=df);
# Attrition by whether employee had promotion in last 5 years

plot = sns.factorplot(x='promotion_last_5years', y='left', kind='bar', data=df);
# Percentage of employees who had promotion in last 5 years

df['promotion_last_5years'].mean()
X = df.drop('left', axis=1)

y = df['left']

X.drop(['department','salary'], axis=1, inplace=True)
# One-hot encoding

salary_dummy = pd.get_dummies(df['salary'])

department_dummy = pd.get_dummies(df['department'])



# from EDA, only management and R&D had attrition different from the rest.

X[['managment','RandD']] = department_dummy[['management', 'RandD']]

X[['salary_high', 'salary_medium']] = salary_dummy[['high', 'medium']]
# Split Training Set from Testing Set (70/30)

from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
# Scaling features

from sklearn.preprocessing import StandardScaler



stdsc = StandardScaler()

# transform our training features

X_train_std = stdsc.fit_transform(X_train)

# transform the testing features in the same way

X_test_std = stdsc.transform(X_test)
# Cross validation

from sklearn.model_selection import ShuffleSplit



cv = ShuffleSplit(n_splits=20, test_size=0.3)
# Model #1: kNN

from sklearn.neighbors import KNeighborsClassifier



knn = KNeighborsClassifier()
# Search for best parameters

from sklearn.model_selection import GridSearchCV



parameters = {'n_neighbors': range(1,11), 'weights': ['uniform', 'distance']}

clf = GridSearchCV(knn, parameters, cv=cv)

clf.fit(X_train_std, y_train)

print('Parameters with best score:')

print(clf.best_params_)

print('Cross validation score:', clf.best_score_)
best_knn = clf.best_estimator_

print('Test score:', best_knn.score(X_test_std, y_test))
# Model #2: Random Forest

from sklearn.ensemble import RandomForestClassifier



rf_model = RandomForestClassifier()
rf_param = {'n_estimators': range(1,11)}

rf_grid = GridSearchCV(rf_model, rf_param, cv=cv)

rf_grid.fit(X_train, y_train)

print('Parameter with best score:')

print(rf_grid.best_params_)

print('Cross validation score:', rf_grid.best_score_)
best_rf = rf_grid.best_estimator_

print('Test score:', best_rf.score(X_test, y_test))
# feature importance scores

features = X.columns

feature_importances = best_rf.feature_importances_



features_df = pd.DataFrame({'Features': features, 'Importance Score': feature_importances})

features_df.sort_values('Importance Score', inplace=True, ascending=False)



features_df
features_df['Importance Score'][:5].sum()
# Model #3: Logistic Regression

from sklearn.linear_model import LogisticRegression



logreg = LogisticRegression()
log_param = {'C': np.linspace(1e-5, 1e5, 21), 'class_weight': [None, 'balanced']}

log_grid = GridSearchCV(logreg, log_param, cv=cv)

log_grid.fit(X_train_std, y_train)

print('Parameter with best score:')

print(log_grid.best_params_)

print('Cross validation score:', log_grid.best_score_)
best_logreg = log_grid.best_estimator_

print('Test score:', best_logreg.score(X_test_std,y_test))
reduced_features = ['satisfaction_level', 'time_spend_company', 

                    'number_project', 'average_monthly_hours', 'last_evaluation']

X2_train = X_train[reduced_features]

X2_test = X_test[reduced_features]

stdsc2 = StandardScaler()

X2_train_std = stdsc2.fit_transform(X2_train)

X2_test_std = stdsc2.transform(X2_test)
logreg2 = LogisticRegression()

log_grid2 = GridSearchCV(logreg2, log_param, cv=cv)

log_grid2.fit(X2_train_std, y_train)

print('Parameter with best score:')

print(log_grid2.best_params_)

print('Cross validation score:', log_grid2.best_score_)
best_logreg2 = log_grid2.best_estimator_

print('Test score:', best_logreg2.score(X2_test_std,y_test))
# Model #4: K-mean clustering

from sklearn.cluster import KMeans



# Fit entire dataset. Reduced features (top 5 from RF importance scores); scaled.

X2 = X[reduced_features]

X2_std = stdsc2.fit_transform(X2)
# Inertia vs. # of clusters

x1 = []

y1 = []

for n in range(2,11):

    km = KMeans(n_clusters=n, random_state=7)

    km.fit(X2_std)

    x1.append(n)

    y1.append(km.inertia_)

plt.scatter(x1, y1)

plt.plot(x1, y1);
km = KMeans(n_clusters=7, n_init=20, random_state=7)

km.fit(X2_std)

columns = {str(x): stdsc2.inverse_transform(km.cluster_centers_[x]) for x in range(0,len(km.cluster_centers_))}

pd.DataFrame(columns, index=X2.columns)
# Percentage of employees left for each cluster. Helps identify which cluster to direct our focus.

kmpredict = pd.DataFrame(data=df['left'])

kmpredict['cluster'] = km.labels_

kmpredict.groupby('cluster').mean()