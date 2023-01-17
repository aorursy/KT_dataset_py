import os

import numpy as np

import pandas as pd



from sklearn.model_selection import train_test_split, ShuffleSplit, KFold, cross_val_score

from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression,Ridge, RidgeCV

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import roc_auc_score



from lightgbm import LGBMClassifier

import xgboost as xgb

from catboost import CatBoostClassifier



import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

sns.set()



import time

import datetime



#import shap

# load JS visualization code to notebook

#shap.initjs()



import warnings

warnings.filterwarnings("ignore")
%%time

PATH_TO_DATA = '../input/'



sample_submission = pd.read_csv(os.path.join(PATH_TO_DATA, 'sample_submission.csv'), 

                                    index_col='match_id_hash')

df_train_features = pd.read_csv(os.path.join(PATH_TO_DATA, 'train_features.csv'), 

                                    index_col='match_id_hash')

df_train_targets = pd.read_csv(os.path.join(PATH_TO_DATA, 'train_targets.csv'), 

                                   index_col='match_id_hash')

df_test_features = pd.read_csv(os.path.join(PATH_TO_DATA, 'test_features.csv'), 

                                   index_col='match_id_hash')
df_train_features.head(3)
df_test_features.head(3)
df_train_targets.head(3)
print('Shape of Training set: {0}\nShape of Test set: {1}'.format(df_train_features.shape,df_test_features.shape))
target = pd.Series(df_train_targets['radiant_win'].map({True: 1, False: 0}))
plt.hist(target);

plt.title('Target distribution');
general_features = ['game_time', 'game_mode', 'lobby_type', 'objectives_len', 'chat_len']

gen_feat_df = df_train_features[general_features].copy()

gen_feat_df['target'] = target

plt.figure(figsize=(8, 5));

ax = sns.heatmap(gen_feat_df.corr(),annot=True,)
plt.figure(figsize=(8, 5));

mask = np.zeros_like(gen_feat_df.corr())

mask[np.triu_indices_from(mask)] = True

with sns.axes_style("white"):

    ax = sns.heatmap(gen_feat_df.corr(), mask=mask,annot=True)
print('Top 10 features correlated with target (abs values):')

print(abs(df_train_features.corrwith(target)).sort_values(ascending=False)[0:10])
r_y_coord = ['r{0}_y'.format(i) for i in range(1,6)]

r_x_coord = ['r{0}_x'.format(i) for i in range(1,6)]

r_coord = r_y_coord+r_x_coord



d_y_coord = ['d{0}_y'.format(i) for i in range(1,6)]

d_x_coord = ['d{0}_x'.format(i) for i in range(1,6)]

d_coord = d_y_coord+d_x_coord
coord_feat_df = df_train_features[r_coord+d_coord].copy()

coord_feat_df['target'] = target

plt.figure(figsize=(16, 10));

ax = sns.heatmap(coord_feat_df.corr(),annot=True,)
print('Min y coordinate for Radiant: {0}'.format(coord_feat_df[r_y_coord].min(axis=0).sort_values(ascending=True)[0:1].values))

print('Max y coordinate for Radiant: {0}'.format(coord_feat_df[r_y_coord].max(axis=0).sort_values(ascending=False)[0:1].values)) 

print('Min x coordinate for Radiant: {0}'.format(coord_feat_df[r_x_coord].min(axis=0).sort_values(ascending=True)[0:1].values))

print('Max x coordinate for Radiant: {0}'.format(coord_feat_df[r_x_coord].max(axis=0).sort_values(ascending=False)[0:1].values)) 
print('Min y coordinate for Dire: {0}'.format(coord_feat_df[d_y_coord].min(axis=0).sort_values(ascending=True)[0:1].values))

print('Max y coordinate for Dire: {0}'.format(coord_feat_df[d_y_coord].max(axis=0).sort_values(ascending=False)[0:1].values)) 

print('Min x coordinate for Dire: {0}'.format(coord_feat_df[d_x_coord].min(axis=0).sort_values(ascending=True)[0:1].values))

print('Max x coordinate for Dire: {0}'.format(coord_feat_df[d_x_coord].max(axis=0).sort_values(ascending=False)[0:1].values)) 
#https://stackoverflow.com/questions/38783027/jupyter-notebook-display-two-pandas-tables-side-by-side

from IPython.display import display_html 

def display_side_by_side(*args):

    html_str=''

    for df in args:

        html_str+=df.to_html()

    display_html(html_str.replace('table','table style="display:inline"'),raw=True)
display_side_by_side(coord_feat_df[coord_feat_df['target']==1].describe().T,coord_feat_df[coord_feat_df['target']==0].describe().T)
coord_feat_df_mean = coord_feat_df.copy()

coord_feat_df_mean['target'] = target



coord_feat_df_mean['r_y_mean'] = coord_feat_df_mean[r_y_coord].mean(axis=1)

coord_feat_df_mean['r_x_mean'] = coord_feat_df_mean[r_x_coord].mean(axis=1)

coord_feat_df_mean['d_y_mean'] = coord_feat_df_mean[d_y_coord].mean(axis=1)

coord_feat_df_mean['d_x_mean'] = coord_feat_df_mean[d_x_coord].mean(axis=1)

mean_cols = ['r_y_mean', 'r_x_mean', 'd_y_mean', 'd_x_mean']
coord_feat_df_mean.head(3)
plt.figure(figsize=(8, 5));

ax = sns.heatmap(coord_feat_df_mean[mean_cols+['target']].corr(),annot=True,)
sns_plot = sns.pairplot(coord_feat_df_mean[mean_cols+['target']])

sns_plot.savefig('pairplot.png')
from sklearn.manifold import TSNE
%%time

tsne = TSNE(random_state=17)

tsne_representation = tsne.fit_transform(coord_feat_df_mean[mean_cols]) #https://habr.com/ru/company/ods/blog/323210/
plt.scatter(tsne_representation[:, 0], tsne_representation[:, 1], 

            c=coord_feat_df_mean['target'].map({0: 'blue', 1: 'orange'}));
r_kills = ['r{0}_kills'.format(i) for i in range(1,6)]

r_deaths = ['r{0}_deaths'.format(i) for i in range(1,6)]

r_assists = ['r{0}_assists'.format(i) for i in range(1,6)]

r_kda = r_kills+r_deaths+r_assists



d_kills = ['d{0}_kills'.format(i) for i in range(1,6)]

d_deaths = ['d{0}_deaths'.format(i) for i in range(1,6)]

d_assists = ['d{0}_assists'.format(i) for i in range(1,6)]

d_kda = d_kills+d_deaths+d_assists



kda_feat_df = df_train_features[r_kda+d_kda].copy()

kda_feat_df['target'] = target



kda_feat_df['r_tot_kills'] = kda_feat_df[r_kills].sum(axis=1)

kda_feat_df['r_tot_deaths'] = kda_feat_df[r_deaths].sum(axis=1)

kda_feat_df['r_tot_assists'] = kda_feat_df[r_assists].sum(axis=1)



kda_feat_df['d_tot_kills'] = kda_feat_df[d_kills].sum(axis=1)

kda_feat_df['d_tot_deaths'] = kda_feat_df[d_deaths].sum(axis=1)

kda_feat_df['d_tot_assists'] = kda_feat_df[d_assists].sum(axis=1)



tot_cols = ['r_tot_kills', 'r_tot_deaths', 'r_tot_assists', 'd_tot_kills', 'd_tot_deaths', 'd_tot_assists']



display(kda_feat_df.head(3))
plt.figure(figsize=(8, 5));

ax = sns.heatmap(kda_feat_df[tot_cols+['target']].corr(),annot=True,)
kda_feat_df['r_kda'] = (kda_feat_df['r_tot_kills']+kda_feat_df['r_tot_assists'])/kda_feat_df['r_tot_deaths']

kda_feat_df['d_kda'] = (kda_feat_df['d_tot_kills']+kda_feat_df['d_tot_assists'])/kda_feat_df['d_tot_deaths']
plt.figure(figsize=(4.8, 3));

ax = sns.heatmap(kda_feat_df[['r_kda','d_kda','target']].corr(),annot=True,)
X = df_train_features

y = df_train_targets['radiant_win']

X_test = df_test_features

y_cat = pd.Series(df_train_targets['radiant_win'].map({True: 1, False: 0})) #catboost doesn't understand True,False 

#X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.3, random_state=17) #for holdout, don't use in kernel

n_fold = 5

folds = KFold(n_splits=n_fold, shuffle=True, random_state=11)

cv = ShuffleSplit(n_splits=n_fold, test_size=0.3, random_state=17) #same as in https://www.kaggle.com/c/mlcourse-dota2-win-prediction/kernels starter kernel 
%%time



model_rf = RandomForestClassifier(n_estimators=100, n_jobs=4,

                                   max_depth=None, random_state=17)



# calcuate ROC-AUC for each split

cv_scores_rf = cross_val_score(model_rf, X, y, cv=cv, scoring='roc_auc')
%%time



model_lgb = LGBMClassifier(random_state=17)

cv_scores_lgb = cross_val_score(model_lgb, X, y, cv=cv, 

                                scoring='roc_auc', n_jobs=4)
%%time



model_xgb = xgb.XGBClassifier(random_state=17)

cv_scores_xgb = cross_val_score(model_xgb, X, y, cv=cv,

                                scoring='roc_auc', n_jobs=4)
%%time 

model_cat = CatBoostClassifier(random_state=17,silent=True)

cv_scores_cat = cross_val_score(model_cat, X, y_cat, cv=cv,

                                scoring='roc_auc', n_jobs=1) #pay attention n_jobs=1 here, just freezes with any other value
cv_results = pd.DataFrame(data={'RF': cv_scores_rf, 'LGB':cv_scores_lgb, 'XGB':cv_scores_xgb, 'CAT':cv_scores_cat})

display_side_by_side(cv_results, cv_results.describe())
#just visit https://lightgbm.readthedocs.io/en/latest/Parameters-Tuning.html

#https://medium.com/@pushkarmandot/https-medium-com-pushkarmandot-what-is-lightgbm-how-to-implement-it-how-to-fine-tune-the-parameters-60347819b7fc

params = {#'num_leaves': 31, # number of leaves in full tree (31 by default) 

         'learning_rate': 0.01, #this determines the impact of each tree on the final outcome. 



         'min_data_in_leaf': 50,

         'min_sum_hessian_in_leaf': 12.0,

         'objective': 'binary', 

         'max_depth': -1,

         'boosting': 'gbdt', #'dart' 

         'bagging_freq': 5,

         'bagging_fraction': 0.81,

         'boost_from_average':'false',

         'bagging_seed': 17,

         'metric': 'auc',

         'verbosity': -1,

         }
%%time

# this part is based on great kernel https://www.kaggle.com/artgor/seismic-data-eda-and-baseline by @artgor

oof = np.zeros(len(X))

prediction = np.zeros(len(X_test))

scores = []

feature_importance = pd.DataFrame()

for fold_n, (train_index, valid_index) in enumerate(folds.split(X, y)):

    print('Fold', fold_n, 'started at', time.ctime())

    X_train, X_valid = X.iloc[train_index], X.iloc[valid_index]

    y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]

    

    model = LGBMClassifier(**params, n_estimators = 2000, nthread = 5, n_jobs = -1)

    model.fit(X_train, y_train, 

              eval_set=[(X_train, y_train), (X_valid, y_valid)], eval_metric='auc',

              verbose=200, early_stopping_rounds=200)

            

    y_pred_valid = model.predict_proba(X_valid)[:, 1]

    y_pred = model.predict_proba(X_test, num_iteration=model.best_iteration_)[:, 1]

        

    oof[valid_index] = y_pred_valid.reshape(-1,)

    scores.append(roc_auc_score(y_valid, y_pred_valid))

    prediction += y_pred    

    

    # feature importance

    fold_importance = pd.DataFrame()

    fold_importance["feature"] = X.columns

    fold_importance["importance"] = model.feature_importances_

    fold_importance["fold"] = fold_n + 1

    feature_importance = pd.concat([feature_importance, fold_importance], axis=0)



prediction /= n_fold
print('CV mean score: {0:.4f}, std: {1:.4f}.'.format(np.mean(scores), np.std(scores)))

    

feature_importance["importance"] /= n_fold

    

cols = feature_importance[["feature", "importance"]].groupby("feature").mean().sort_values(

            by="importance", ascending=False)[:50].index



best_features = feature_importance.loc[feature_importance.feature.isin(cols)]



plt.figure(figsize=(14, 16));

sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False));

plt.title('LGB Features (avg over folds)');
lgb = LGBMClassifier(random_state=17)

lgb.fit(X, y)



X_test = df_test_features.values

y_test_pred = lgb.predict_proba(X_test)[:, 1]

df_submission = pd.DataFrame({'radiant_win_prob': y_test_pred}, 

                                 index=df_test_features.index)

submission_filename = 'lgb_{}.csv'.format(

    datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))

df_submission.to_csv(submission_filename)

print('Submission saved to {}'.format(submission_filename))
df_submission.head() #just to check that everything allright 