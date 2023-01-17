# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from scipy.stats import chi2_contingency
from sklearn import svm
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold, cross_val_score
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train_data = pd.read_csv("/kaggle/input/titanic/train.csv")
test_data = pd.read_csv("/kaggle/input/titanic/test.csv")
# alternate - df.isnull().values.any()
df_all = pd.concat([train_data, test_data], sort=True).reset_index(drop=True)
age_by_pclass_sex = df_all.groupby(['Sex', 'Pclass']).median()['Age']
for pclass in range(1, 4):
    for sex in ['female', 'male']:
        print('Median age of Pclass {} {}s: {}'.format(pclass, sex, age_by_pclass_sex[sex][pclass]))
print('Median age of all passengers: {}'.format(df_all['Age'].median()))
df_all['Age'] = df_all.groupby(['Sex', 'Pclass'])['Age'].apply(lambda x: x.fillna(x.median()))
df_all['Embarked'] = df_all['Embarked'].fillna('S')
df_all['Deck'] = df_all['Cabin'].apply(lambda s: s[0] if pd.notnull(s) else 'M')
# train_data['Deck'].head()
# Passenger in the T deck is changed to A
idx = df_all[df_all['Deck'] == 'T'].index
df_all.loc[idx, 'Deck'] = 'A'
df_all['Deck'] = df_all['Deck'].replace(['A', 'B', 'C'], 'ABC')
df_all['Deck'] = df_all['Deck'].replace(['D', 'E'], 'DE')
df_all['Deck'] = df_all['Deck'].replace(['F', 'G'], 'FG')


df_all = df_all.drop(['Cabin'],axis=1)

ct_deck = pd.crosstab(df_all['Deck'],df_all['Survived'])
ct_deck.plot.bar()
med_fare = df_all.groupby(['Pclass', 'Parch', 'SibSp']).Fare.median()[3][0][0]
# Filling the missing value in Fare with the median Fare of 3rd class alone passenger
df_all['Fare'] = df_all['Fare'].fillna(med_fare)
df_all.isnull().sum()
train_data = df_all[df_all["Survived"]>=0]
train_data.head()
# train_data.isnull().sum()
test_data = df_all[df_all["Survived"].isna()]
test_data = test_data.drop(['Survived'],axis=1).reset_index(drop=True)
test_data.head()
features = ["Pclass", "Sex", "SibSp", "Parch","Deck","Fare","Age"]
X = pd.get_dummies(train_data[features])

y = train_data["Survived"]

X_test = pd.get_dummies(test_data[features])
corr_df=X[["Pclass", "SibSp", "Parch","Fare","Age"]]  #New dataframe to calculate correlation between numeric features
cor= corr_df.corr(method='pearson')
print(cor)
csq=chi2_contingency(pd.crosstab(train_data['Survived'], train_data['Sex']))
print("P-value: ",csq[1])
csq2=chi2_contingency(pd.crosstab(train_data['Survived'], train_data['Embarked']))
print("P-value: ",csq2[1])
csq3=chi2_contingency(pd.crosstab(train_data['Survived'], train_data['Pclass']))
print("P-value: ",csq3[1])
ct = pd.crosstab(train_data.Sex,train_data.Survived)
ct.plot.bar()
ct_e = pd.crosstab(train_data.Pclass,train_data.Survived)
ct_e.plot.bar()
ct_em = pd.crosstab(train_data.Embarked,train_data.Survived)
ct_em.plot.bar()
sns.catplot(x="Survived",y="Age",data = train_data)
sns.catplot(x="Survived",y="Fare",data = train_data)
#A pipeline is made - first scaling and then model fitting
# x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.3) 
# model_svc = make_pipeline(StandardScaler(),svm.SVC(kernel='linear'))
model_svc = svm.SVC()
# model_svc.fit(x_train,y_train)
# svc_predicted = model_svc.predict(x_test)
# accuracy_score(y_test,svc_predicted)
k_fold = KFold(n_splits=5)
model_svc.fit(X,y)
scores = cross_val_score(model_svc,X,y,cv=k_fold)
scores.mean()
def plot_learning_curve(estimator, title, X, y, axes=None, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate 3 plots: the test and training learning curve, the training
    samples vs fit times curve, the fit times vs score curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    axes : array of 3 axes, optional (default=None)
        Axes to use for plotting the curves.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

          - None, to use the default 5-fold cross-validation,
          - integer, to specify the number of folds.
          - :term:`CV splitter`,
          - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : int or None, optional (default=None)
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    train_sizes : array-like, shape (n_ticks,), dtype float or int
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the dtype is float, it is regarded as a
        fraction of the maximum size of the training set (that is determined
        by the selected validation method), i.e. it has to be within (0, 1].
        Otherwise it is interpreted as absolute sizes of the training sets.
        Note that for classification the number of samples usually have to
        be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))
    """
    if axes is None:
        _, axes = plt.subplots(1, 3, figsize=(20, 5))

    axes[0].set_title(title)
    if ylim is not None:
        axes[0].set_ylim(*ylim)
    axes[0].set_xlabel("Training examples")
    axes[0].set_ylabel("Score")

    train_sizes, train_scores, test_scores, fit_times, _ = \
        learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs,
                       train_sizes=train_sizes,
                       return_times=True)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)

    # Plot learning curve
    axes[0].grid()
    axes[0].fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
    axes[0].fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1,
                         color="g")
    axes[0].plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Training score")
    axes[0].plot(train_sizes, test_scores_mean, 'o-', color="g",
                 label="Cross-validation score")
    axes[0].legend(loc="best")

    # Plot n_samples vs fit_times
    axes[1].grid()
    axes[1].plot(train_sizes, fit_times_mean, 'o-')
    axes[1].fill_between(train_sizes, fit_times_mean - fit_times_std,
                         fit_times_mean + fit_times_std, alpha=0.1)
    axes[1].set_xlabel("Training examples")
    axes[1].set_ylabel("fit_times")
    axes[1].set_title("Scalability of the model")

    # Plot fit_time vs score
    axes[2].grid()
    axes[2].plot(fit_times_mean, test_scores_mean, 'o-')
    axes[2].fill_between(fit_times_mean, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1)
    axes[2].set_xlabel("fit_times")
    axes[2].set_ylabel("Score")
    axes[2].set_title("Performance of the model")

    return plt
#  ig, axes = plt.subplots(3, 1, figsize=(7, 10))

# # train_sizes, train_scores, valid_scores = learning_curve(SVC(kernel='linear'), X, y, train_sizes=[50, 80, 110], cv=5)
# title = "learning curve"
# # SVC is more expensive so we do a lower number of CV iterations:
# cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
# estimator = svm.SVC(gamma=0.001)
# plot_learning_curve(estimator, title, X, y, axes=axes[:,], ylim=(0.58, 0.66),
#                     cv=cv, n_jobs=4)
# plt.show()
model = RandomForestClassifier(n_estimators=100,max_depth=5,random_state=1)
model.fit(X,y)
predictions  = model.predict(X_test)
# output = pd.DataFrame({'PassengerId':test_data.PassengerId,'Survived':predictions})
# output.to_csv('my_submission.csv', index=False)
# print("Your submission was successfully saved!")
# from sklearn.model_selection import KFold, cross_val_score
k_fold = KFold(n_splits=5)
scores = cross_val_score(model,X,y,cv=k_fold)
scores.mean()
# from cabin create hasCabin
train_data['hasCabin']=train_data['Cabin'].apply(lambda x: 0 if x==0 else 1)
test_data['hasCabin']=test_data['Cabin'].apply(lambda x: 0 if x==0 else 1)
# combine SibSp and Parch features to create new one FamilyMem
train_data['FamilyMem']=train_data.apply(lambda x: x['SibSp']+x['Parch'], axis=1)
test_data['FamilyMem']=test_data.apply(lambda x: x['SibSp']+x['Parch'], axis=1)

features.append("hasCabin")
features.append("FamilyMem")

#updating final train data and test data
X = pd.get_dummies(train_data[features])
X_test = pd.get_dummies(test_data[features])
X.head()

rfc = RandomForestClassifier(n_jobs=-1,max_features= 'sqrt' ,n_estimators=50, oob_score = True) 

param_grid = { 
    'n_estimators': [200, 700],
    'max_features': ['auto', 'sqrt', 'log2']
}

CV_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv= 5)
CV_rfc.fit(X, y)
CV_rfc.best_params_
#update Model parameters according to Grid Search 
rfc = RandomForestClassifier(max_features= 'log2' ,n_estimators=700, oob_score = True) 
#fit the new model :D
rfc.fit(X,y)
fig, axes = plt.subplots(3, 1, figsize=(7, 10))

# train_sizes, train_scores, valid_scores = learning_curve(SVC(kernel='linear'), X, y, train_sizes=[50, 80, 110], cv=5)
title = "learning curve"
# SVC is more expensive so we do a lower number of CV iterations:
cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
estimator = RandomForestClassifier(max_features= 'auto' ,n_estimators=200, oob_score = True)
plot_learning_curve(estimator, title, X, y, axes=axes[:,], ylim=(0.5, 1),
                    cv=cv, n_jobs=4)
plt.show()
predictions  = rfc.predict(X_test)
output = pd.DataFrame({'PassengerId':test_data.PassengerId,'Survived':predictions})
output.to_csv('submission.csv', index=False)
print("Your submission was successfully saved!")