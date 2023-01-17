import smtplib

from matplotlib import style

import seaborn as sns

sns.set(style='ticks', palette='RdBu')

#sns.set(style='ticks', palette='Set2')

import pandas as pd

import numpy as np

import time

import datetime 

%matplotlib inline

import matplotlib.pyplot as plt

from subprocess import check_output

pd.options.display.max_colwidth = 1000

from time import gmtime, strftime

Time_now = strftime("%Y-%m-%d %H:%M:%S", gmtime())

import timeit

start = timeit.default_timer()

pd.options.display.max_rows = 100



from sklearn.decomposition import PCA

from sklearn.model_selection import cross_val_score

from sklearn.feature_selection import RFECV, SelectKBest

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier

from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.naive_bayes import GaussianNB, BernoulliNB

from sklearn.neighbors import KNeighborsClassifier

from sklearn.linear_model import SGDClassifier

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.calibration import CalibratedClassifierCV

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import StratifiedKFold

from sklearn.feature_selection import SelectFromModel

from sklearn import svm

from scipy.stats import skew

from scipy.stats.stats import pearsonr

from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, LassoCV, LassoLarsCV

from sklearn.model_selection import cross_val_score

import matplotlib

import matplotlib.pyplot as plt
data = pd.read_csv('../input/analysis4/studentanalysis2.csv')

df = data
data.columns
data.head()
data.describe()
categorical_features = (data.select_dtypes(include=['object']).columns.values)

categorical_features
numerical_features = data.select_dtypes(include = ['float64', 'int64']).columns.values

numerical_features
df.isnull().sum()
df = df[pd.notnull(df['practicum_grade'])]
d = pd.DataFrame(data=df)

corr = d.corr()



# Generate a mask for the upper triangle

mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True



# Set up the matplotlib figure

f, ax = plt.subplots(figsize=(11, 9))



# Generate a custom diverging colormap

cmap = sns.diverging_palette(220, 10, as_cmap=True)



# Draw the heatmap with the mask and correct aspect ratio

sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,

            square=True, linewidths=.5, cbar_kws={"shrink": .5})
mod_df = df 



gender = {'M':1, 

              'F':2}



race = {  'group A': 1,

                     'group B': 2,

                     'group C': 3,

                     'group D': 4,

                     'group E': 5

                     }

parental_education =  {"bachelor's degree": 1,

                     'some college': 2,

                     "master's degree": 3,

                     "associate's degree": 4,

                     'high school': 5,

                     'some high school': 6

                     }

programmecat = {'DEG':1,

               'DIP':2,

               'PGDE':3}

programmegroup = {'BA (Ed) (Pr)':1,

'BA (Ed) (Sec)':2,

                 'BA (Ed) (Sec)':3,

'BSc (Ed) (Pr)':4,

'BSc (Ed) (Sec)':5,

'Dip AR Ed (Pr)':6,

'Dip AR Ed (Sec)':7,

'Dip Ed (CL)(Pr)':8,

'Dip Ed (ML)(Pr)':9,

'Dip Ed (TL)(Pr)':10,

'Dip HE Ed':11,

'Dip MU Ed (Pr)':12,

 'Dip PE (Pr)':13,

 'DISE':14,

'PGDE (Art) (Pr)':15,

 'PGDE (CL) (Pr)':16,

 'PGDE (CL) (Sec)':17,

'PGDE (JC)':18,

'PGDE (ML) (Sec)':19,

 'PGDE (MU) (Pr)':20,

 'PGDE (PE) (Pr)':21,

'PGDE (PE)(Sec)':22,

 'PGDE (Pr)':23,

'PGDE (Sec)':24,

 'PGDE (TL) (Pr)':25,

 'PGDE (TL) (Sec)':26}

practicum_grade = {'A*':3,

                  'B*':2,

                  'P':1}



studentabsencedays = {'Under-7':0,

                          'Above-7':1}





mod_df.gender  = mod_df.gender.map(gender)

mod_df.race     = mod_df.race.map(race)

mod_df.parental_education     = mod_df.parental_education.map(parental_education)

mod_df.practicum_grade     = mod_df.practicum_grade.map(practicum_grade)

mod_df.programmegroup     = mod_df.programmegroup.map(programmegroup)

mod_df.programmecat     = mod_df.programmecat.map(programmecat)

mod_df.studentabsencedays     = mod_df.studentabsencedays.map(studentabsencedays)
from sklearn.model_selection import train_test_split

from catboost import Pool, CatBoostRegressor, cv



X=mod_df.drop(columns=['cgpa'])





print(X.columns)

categorical_features_indices =[0,1, 3,4,5,6,7]

y=mod_df['cgpa']



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2) 

                                                    



X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2)
X = mod_df[['gender','race','programmecat','programmegroup','studentabsencedays','parental_education','mathscore','readingscore','writingscore']]

y = mod_df['cgpa']



from sklearn import model_selection

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

from sklearn.linear_model import LinearRegression

kfold = model_selection.KFold(n_splits=10)

lr = LinearRegression()

scoring = 'r2'

results = model_selection.cross_val_score(lr, X, y, cv=kfold, scoring=scoring)

lr.fit(X_train,y_train)

lr_predictions = lr.predict(X_test)

print('Coefficients: \n', lr.coef_)

from sklearn.metrics import r2_score

print("R_square score: ", r2_score(y_test,lr_predictions))
from sklearn.tree import DecisionTreeRegressor

dtr = DecisionTreeRegressor(random_state = 42)

dtr.fit(X_train,y_train)

dtr_predictions = dtr.predict(X_test) 



# R^2 Score

print("R_square score: ", r2_score(y_test,dtr_predictions))
from sklearn.ensemble import RandomForestRegressor

rfr = RandomForestRegressor(n_estimators = 100)

rfr.fit(X_train,y_train)

rfr_predicitions = rfr.predict(X_test) 



# R^2 Score

print("R_square score: ", r2_score(y_test,rfr_predicitions))
from sklearn import ensemble

clf = ensemble.GradientBoostingRegressor(n_estimators = 400, max_depth = 5, min_samples_split = 2,

          learning_rate = 0.1, loss = 'ls')

clf.fit(X_train, y_train)

clf_predicitions = clf.predict(X_test) 

print("R_square score: ", r2_score(y_test,clf_predicitions))
from sklearn.model_selection import train_test_split

from catboost import Pool, CatBoostRegressor, cv



X=mod_df.drop(columns=['cgpa'])





print(X.columns)

categorical_features_indices =[0,1, 3,4,5,6,7]

y=mod_df['cgpa']



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2) 

                                                    



X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2)
def perform_model(X_train, y_train,X_valid, y_valid,X_test, y_test):

    model = CatBoostRegressor(

        random_seed = 400,

        loss_function = 'RMSE',

        iterations=400,

    )

    

    model.fit(

        X_train, y_train,

        cat_features = categorical_features_indices,

        eval_set=(X_valid, y_valid),

        verbose=False

    )

    

    print("RMSE on training data: "+ model.score(X_train, y_train).astype(str))

    print("RMSE on test data: "+ model.score(X_test, y_test).astype(str))

    

    return model
model=perform_model(X_train, y_train,X_valid, y_valid,X_test, y_test)
feature_score = pd.DataFrame(list(zip(X.dtypes.index, model.get_feature_importance(Pool(X, label=y, cat_features=categorical_features_indices)))),

                columns=['Feature','Score'])



feature_score = feature_score.sort_values(by='Score', ascending=False, inplace=False, kind='quicksort', na_position='last')
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



df_copy = pd.get_dummies(mod_df)



df1 = df_copy

y = np.asarray(df1['cgpa'], dtype="|S6")

df1 = df1.drop(['cgpa'],axis=1)

X = df1.values

Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.50)



radm = RandomForestClassifier()

radm.fit(Xtrain, ytrain)

clf = radm

indices = np.argsort(radm.feature_importances_)[::-1]



# Print the feature ranking

print('Feature ranking:')



for f in range(df1.shape[1]):

    print('%d. feature %d %s (%f)' % (f+1 , 

                                      indices[f], 

                                      df1.columns[indices[f]], 

                                      radm.feature_importances_[indices[f]]))