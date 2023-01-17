# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split

from sklearn.ensemble import ExtraTreesClassifier

from sklearn import metrics

import matplotlib.pyplot as plt

from xgboost import XGBClassifier

from sklearn.feature_selection import SelectFromModel

from sklearn.metrics import roc_auc_score

from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import AdaBoostClassifier

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.svm import SVC







# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# http://www.davidsbatista.net/blog/2018/02/23/model_optimization/

class EstimatorSelectionHelper:



    def __init__(self, models, params):

        if not set(models.keys()).issubset(set(params.keys())):

            missing_params = list(set(models.keys()) - set(params.keys()))

            raise ValueError("Some estimators are missing parameters: %s" % missing_params)

        self.models = models

        self.params = params

        self.keys = models.keys()

        self.grid_searches = {}



    def fit(self, X, y, cv=3, n_jobs=3, verbose=1, scoring=None, refit=False):

        for key in self.keys:

            print("Running GridSearchCV for %s." % key)

            model = self.models[key]

            params = self.params[key]

            gs = GridSearchCV(model, params, cv=cv, n_jobs=n_jobs,

                              verbose=verbose, scoring=scoring, refit=refit,

                              return_train_score=True)

            gs.fit(X,y)

            self.grid_searches[key] = gs    



    def score_summary(self, sort_by='mean_score'):

        def row(key, scores, params):

            d = {

                 'estimator': key,

                 'min_score': min(scores),

                 'max_score': max(scores),

                 'mean_score': np.mean(scores),

                 'std_score': np.std(scores),

            }

            return pd.Series({**params,**d})



        rows = []

        for k in self.grid_searches:

            print(k)

            params = self.grid_searches[k].cv_results_['params']

            scores = []

            for i in range(self.grid_searches[k].cv):

                key = "split{}_test_score".format(i)

                r = self.grid_searches[k].cv_results_[key]        

                scores.append(r.reshape(len(params),1))



            all_scores = np.hstack(scores)

            for p, s in zip(params,all_scores):

                rows.append((row(k, s, p)))



        df = pd.concat(rows, axis=1).T.sort_values([sort_by], ascending=False)



        columns = ['estimator', 'min_score', 'mean_score', 'max_score', 'std_score']

        columns = columns + [c for c in df.columns if c not in columns]



        return df[columns]
# Preparing data

train_data = pd.read_csv("../input/train.csv")

test_data = pd.read_csv("../input/test.csv")



# cleaning data

remove = []

c = train_data.columns

for i in range(len(c)-1):

    v = train_data[c[i]].values

    for j in range(i+1,len(c)):

        if np.array_equal(v,train_data[c[j]].values):

            remove.append(c[j])



train_data.drop(remove, axis=1, inplace=True)



# Split into validation and training data

train_X, test_X, train_y, test_y = train_test_split(train_data.drop(["TARGET","ID"],axis=1),

                                                  train_data.TARGET, 

                                                  random_state=0,

                                                  test_size=0.25)
# Feature selection

# need to select the most important features

# will be using ExtraTreesClasifier for this task



feature_selection_model = ExtraTreesClassifier()

feature_selection_model.fit(train_X, train_y)

sel = SelectFromModel(feature_selection_model, prefit = True)



important_features = train_X.columns[(sel.get_support())]

# print(important_features)



feat_imp = pd.Series(feature_selection_model.feature_importances_, index = train_X.columns.values).sort_values(ascending=False)

feat_imp[:len(important_features)].plot(kind='bar', title='Most important features based on ExtraTreesClassifier', figsize=(16, 8))
#Finding best model with best hyperparameters

models1 = {

    'ExtraTreesClassifier': ExtraTreesClassifier(),

    'RandomForestClassifier': RandomForestClassifier(),

    'AdaBoostClassifier': AdaBoostClassifier(),

    'GradientBoostingClassifier': GradientBoostingClassifier(),

    'XGBClassifier': XGBClassifier()

}



params1 = {

    'ExtraTreesClassifier': { 'n_estimators': [64, 128, 256, 512, 1024] },

    'RandomForestClassifier': { 'n_estimators': [64, 128, 256, 512, 1024], 'min_samples_split': [2, 4, 8, 16] },

    'AdaBoostClassifier':  { 'n_estimators': [64, 128, 256, 512, 1024] },

    'GradientBoostingClassifier': { 'n_estimators': [64, 128, 256, 512, 1024], 'learning_rate': [0.8, 1.0] },

    'XGBClassifier': { 'n_estimators': [64, 128, 256, 512, 1024], 'max_depth': [1, 2, 3, 4, 5, 6]}

}



helper1 = EstimatorSelectionHelper(models1, params1)

helper1.fit(train_X[important_features], train_y, scoring='roc_auc', n_jobs=2, refit= True)



helper1.score_summary(sort_by='mean_score')
# Selecting the best estimator

clf = helper1.grid_searches.get('XGBClassifier').best_estimator_

print(helper1.grid_searches.get('XGBClassifier').best_params_)
#building final model and submitting to competition



#refit model with all train data

# clf = XGBClassifier(n_estimators=256, max_depth=1)

clf.fit(train_data.drop(["TARGET","ID"],axis=1)[important_features], train_data.TARGET)



submission_preds = clf.predict_proba(test_data[important_features])[:,1]



output = pd.DataFrame({'ID': test_data.ID,

                      'TARGET': submission_preds})

output.to_csv('submission.csv', index=False)