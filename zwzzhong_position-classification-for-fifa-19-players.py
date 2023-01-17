import pandas as pd

import numpy as np

import random as rnd

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

import warnings

warnings.filterwarnings('ignore')

from sklearn import preprocessing

from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_validate

from sklearn.metrics import recall_score
# Read in the dataset

fifa = pd.read_csv("../input/data.csv")

fifa.head()
fifa.info()
fifa.describe()
df2 = fifa.loc[:, 'Crossing':'Release Clause']

df1 = fifa[['ID', 'Name', 'Age', 'Nationality', 'Overall', 'Club', 'Value', 'Wage', 'Preferred Foot', 'Skill Moves', 'Position', 'Height', 'Weight']]

df3 = pd.concat([df1, df2], axis=1)

df = df3.drop(['GKDiving', 'GKHandling', 'GKKicking', 'GKPositioning', 'GKReflexes'], axis=1)

df = df.loc[df['Position'] != 'GK', :]
df = df.dropna()
def value_to_int(df_value):

    try:

        value = float(df_value[1:-1])

        suffix = df_value[-1:]



        if suffix == 'M':

            value = value * 1000000

        elif suffix == 'K':

            value = value * 1000

    except ValueError:

        value = 0

    return value

  

df['Value_float'] = df['Value'].apply(value_to_int)

df['Wage_float'] = df['Wage'].apply(value_to_int)

df['Release_Clause_float'] = df['Release Clause'].apply(lambda m: value_to_int(m))
def weight_to_int(df_weight):

    value = df_weight[:-3]

    return value

  

df['Weight_int'] = df['Weight'].apply(weight_to_int)

df['Weight_int'] = df['Weight_int'].apply(lambda x: int(x))
def height_to_int(df_height):

    try:

        feet = int(df_height[0])

        dlm = df_height[-2]



        if dlm == "'":

            height = round((feet * 12 + int(df_height[-1])) * 2.54, 0)

        elif dlm != "'":

            height = round((feet * 12 + int(df_height[-2:])) * 2.54, 0)

    except ValueError:

        height = 0

    return height



df['Height_int'] = df['Height'].apply(height_to_int)
df.loc[df['Preferred Foot'] == 'Left', 'Preferred_Foot'] = 1

df.loc[df['Preferred Foot'] == 'Right', 'Preferred_Foot'] = 0
for i in ['ST', 'CF', 'LF', 'LS', 'LW', 'RF', 'RS', 'RW']:

  df.loc[df.Position == i , 'Pos'] = 'Strikers' 



for i in ['CAM', 'CDM', 'LCM', 'CM', 'LAM', 'LDM', 'LM', 'RAM', 'RCM', 'RDM', 'RM']:

  df.loc[df.Position == i , 'Pos'] = 'Midfielder' 



for i in ['CB', 'LB', 'LCB', 'LWB', 'RB', 'RCB', 'RWB']:

  df.loc[df.Position == i , 'Pos'] = 'Defender' 



#df.loc[df['Pos'] == 'Strikers', 'Pos_int'] = 1

#df.loc[df['Pos'] == 'Midfielder', 'Pos_int'] = 2

#df.loc[df['Pos'] == 'Defender', 'Pos_int'] = 3
df = df.drop(['Value', 'Wage', 'Release Clause', 'Weight', 'Height'], axis=1)
df = df[df['Overall'] >=70]

df.head()
plt.figure(figsize=(12, 8))

sns.countplot(x = 'Pos', data =df)
plt.figure(figsize=(12, 8))



# Set up the matplotlib figure

f, axes = plt.subplots(2, 2, figsize=(15, 15), sharex=False)

sns.despine(left=True)



sns.boxplot('Pos', 'Overall', data = df, ax=axes[0, 0])

sns.boxplot('Pos', 'Age', data = df, ax=axes[0, 1])



sns.boxplot('Pos', 'Height_int', data = df, ax=axes[1, 1])

sns.boxplot('Pos', 'Weight_int', data = df, ax=axes[1, 0])

f, axes = plt.subplots(ncols= 3, figsize=(30, 10), sharex=False)

sns.despine(left=True)



sns.boxplot('Pos', 'Value_float', data = df, showfliers=False, ax=axes[0])

sns.boxplot('Pos', 'Wage_float', data = df, showfliers=False, ax=axes[1])

sns.boxplot('Pos', 'Release_Clause_float', data = df, showfliers=False, ax=axes[2])
plt.figure(figsize=(15,10))



a = df[df['Pos'] == 'Strikers']

b = df[df['Pos'] == 'Defender']

c = df[df['Pos'] == 'Midfielder']



sns.distplot(a['Skill Moves'], color='blue', label = 'Strikers', kde=False)

sns.distplot(b['Skill Moves'], color='red', label = 'Defender',  kde=False)

sns.distplot(c['Skill Moves'], color='green', label = 'Midfielder',  kde=False)



plt.legend(fontsize = 'xx-large')

plt.figure(figsize=(15,10))

sns.countplot(x='Preferred Foot', data=df, hue='Pos')
cols = ['Age', 'Skill Moves', 'Crossing', 'Finishing', 'HeadingAccuracy', 'ShortPassing', 'Volleys',

       'Dribbling', 'Curve', 'FKAccuracy', 'LongPassing', 'BallControl',

       'Acceleration', 'SprintSpeed', 'Agility', 'Reactions', 'Balance',

       'ShotPower', 'Jumping', 'Stamina', 'Strength', 'LongShots',

       'Aggression', 'Interceptions', 'Positioning', 'Vision', 'Penalties',

       'Composure', 'Marking', 'StandingTackle', 'SlidingTackle',

       'Value_float', 'Wage_float', 'Release_Clause_float',

       'Weight_int', 'Height_int', 'Preferred_Foot']





y = ['Pos']

x = cols

x_train, x_test, y_train, y_test = train_test_split(df[x],df[y], test_size=0.2)
#Common Model Algorithms

from sklearn import svm, tree, linear_model, neighbors, naive_bayes, ensemble, discriminant_analysis, gaussian_process

from xgboost import XGBClassifier

#Common Model Evaluations

from sklearn import feature_selection

from sklearn import model_selection

from sklearn import metrics



from sklearn.model_selection import train_test_split

import warnings

warnings.filterwarnings('ignore')
MLA = [

    #Ensemble Methods

    ensemble.AdaBoostClassifier(),

    ensemble.BaggingClassifier(),

    ensemble.ExtraTreesClassifier(),

    ensemble.GradientBoostingClassifier(),

    ensemble.RandomForestClassifier(),



    #GLM

    linear_model.LogisticRegressionCV(),

    linear_model.SGDClassifier(),

    

    #Nearest Neighbor

    neighbors.KNeighborsClassifier(),

    

    #SVM

    svm.SVC(probability=True),

    svm.NuSVC(probability=True),

    svm.LinearSVC(),

    

    #Trees    

    tree.DecisionTreeClassifier(),

    tree.ExtraTreeClassifier(),

    

    #Discriminant Analysis

    discriminant_analysis.LinearDiscriminantAnalysis(),

    discriminant_analysis.QuadraticDiscriminantAnalysis(),



    #xgboost: 

    XGBClassifier()    

    ]



#split dataset in cross-validation with this splitter class

cv_split = model_selection.ShuffleSplit(n_splits = 10, test_size = .3, train_size = .6, random_state = 0 ) # run model 10x with 60/30 split intentionally leaving out 10%



#create table to compare MLA metrics

MLA_columns = ['MLA Name','MLA Train Accuracy Mean', 'MLA Test Accuracy Mean', 'MLA Test Accuracy 3*STD' ,'MLA Time']

MLA_compare = pd.DataFrame(columns = MLA_columns)



#create table to compare MLA predictions

MLA_predict = y_train[y]



#index through MLA and save performance to table

row_index = 1

for alg in MLA:



    #set name and parameters

    MLA_name = alg.__class__.__name__

    MLA_compare.loc[row_index, 'MLA Name'] = MLA_name

    

    #score model with cross validation

    cv_results = model_selection.cross_validate(alg,x_train[x],  y_train[y], cv  = cv_split)



    MLA_compare.loc[row_index, 'MLA Time'] = cv_results['fit_time'].mean()

    MLA_compare.loc[row_index, 'MLA Train Accuracy Mean'] = cv_results['train_score'].mean()

    MLA_compare.loc[row_index, 'MLA Test Accuracy Mean'] = cv_results['test_score'].mean()   

    #if this is a non-bias random sample, then +/-3 standard deviations (std) from the mean, should statistically capture 99.7% of the subsets

    MLA_compare.loc[row_index, 'MLA Test Accuracy 3*STD'] = cv_results['test_score'].std()*3   #let's know the worst that can happen!

    



    #save MLA predictions 

    alg.fit(x_train[x],  y_train[y])

    MLA_predict[MLA_name] = alg.predict(x_train[x])

    

    row_index+=1



#print and sort table

MLA_compare.sort_values(by = ['MLA Test Accuracy Mean'], ascending = False, inplace = True)

MLA_compare
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

from sklearn.metrics import roc_auc_score

from sklearn.model_selection import StratifiedKFold

from xgboost import XGBClassifier

from datetime import datetime
def timer(start_time=None):

    if not start_time:

        start_time = datetime.now()

        return start_time

    elif start_time:

        thour, temp_sec = divmod((datetime.now() - start_time).total_seconds(), 3600)

        tmin, tsec = divmod(temp_sec, 60)

        print('\n Time taken: %i hours %i minutes and %s seconds.' % (thour, tmin, round(tsec, 2)))
# A parameter grid for XGBoost

params = {

        'min_child_weight': [1, 5, 10],

        'gamma': [0.5, 1, 1.5, 2, 5],

        'subsample': [0.6, 0.8, 1.0],

        'colsample_bytree': [0.6, 0.8, 1.0],

        'max_depth': [3, 4, 5]

        }
xgb = XGBClassifier(learning_rate=0.02, n_estimators=600, objective='binary:logistic',

                    silent=True, nthread=1)
folds = 3

param_comb = 5



skf = StratifiedKFold(n_splits=folds, shuffle = True, random_state = 1001)



random_search = RandomizedSearchCV(xgb, param_distributions=params, n_iter=param_comb, n_jobs=4, cv=skf.split(x_train,y_train), verbose=3, random_state=1001 )



start_time = timer(None) # timing starts from this point for "start_time" variable

random_search.fit(x_train,y_train)

timer(start_time) # timing ends here for "start_time" variable
print('\n All results:')

print(random_search.cv_results_)

print('\n Best estimator:')

print(random_search.best_estimator_)

print('\n Best normalized gini score for %d-fold search with %d parameter combinations:' % (folds, param_comb))

print(random_search.best_score_ * 2 - 1)

print('\n Best hyperparameters:')

print(random_search.best_params_)

results = pd.DataFrame(random_search.cv_results_)
xgb = XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,

       colsample_bytree=0.8, gamma=1.5, learning_rate=0.02,

       max_delta_step=0, max_depth=5, min_child_weight=1, missing=None,

       n_estimators=600, n_jobs=1, nthread=1, objective='multi:softprob',

       random_state=0, reg_alpha=0, reg_lambda=1, scale_pos_weight=1,

       seed=None, silent=True, subsample=0.6)



xgb.fit(df[x],df[y])



y_pred = xgb.predict(x_test)



print('Accuracy of XGBClassifier on test set: {:.2f}'.format(xgb.score(x_test, y_test)))
from yellowbrick.classifier import ROCAUC



classes=[0,1,2]



# Instantiate the visualizer with the classification model

visualizer = ROCAUC(xgb, classes=classes)



visualizer.fit(x_train, y_train)  # Fit the training data to the visualizer

visualizer.score(x_test, y_test)  # Evaluate the model on the test data

g = visualizer.poof()             # Draw/show/poof the data