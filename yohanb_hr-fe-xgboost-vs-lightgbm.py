import numpy as np     # lin. algebra

import pandas as pd    # data tables

import seaborn as sns            # visualization

import matplotlib.pyplot as plt  # visualization



# Evaluation metrics:

from sklearn.metrics import roc_auc_score, r2_score, accuracy_score, classification_report 

from sklearn.model_selection import train_test_split



# Frameworks

import xgboost as xgb   # extreme gradient boosting framework

import lightgbm as lgb  # gradient boosting framework



# Fancy dataframe print

from IPython.display import display, HTML

def show_dataframe(X, rows = 2):

    display(HTML(X.to_html(max_rows=rows)))

%matplotlib inline
# read data

frame = pd.read_csv('../input/HR_comma_sep.csv')



# basic info about the data

show_dataframe(frame, 2)

frame.info()

for c in frame.columns.values: print(c + ':', len(frame[c].unique()))
cat_cols = ['sales', 'salary'] # categorical columns

plt.figure(figsize=(18,12))

# for every categorical column

for i in range(len(cat_cols)):

    c = cat_cols[i] # get colunm name

    means = frame.groupby(c).left.mean()          # mean of the target on each category

    stds = frame.groupby(c).left.std().fillna(0)  # std of the target on each category

    # put all statistics into a dataframe

    ddd = pd.concat([means, stds], axis=1); ddd.columns = ['means', 'stds']

    ddd.sort_values('means', inplace=True)  # sorting by means

    # Countplot:

    plt.subplot(2,2,2*i+1)

    ax = sns.countplot(frame[c], order=ddd.index.values)

    for p in ax.patches:

        x=p.get_bbox().get_points()[:,0]

        y=p.get_bbox().get_points()[1,1]

        ax.annotate('{:.0f}'.format(y), (x.mean(), y), ha='center', va='bottom')

    # Means and stds:

    plt.subplot(2,2,2*i+2)

    plt.fill_between(range(len(frame[c].unique())), 

                     ddd.means.values - ddd.stds.values,

                     ddd.means.values + ddd.stds.values,

                     alpha=0.1

                    )

    plt.xticks(range(len(frame[c].unique())), ddd.index.values, rotation=45)

    plt.plot(ddd.means.values, color='b', marker='.', linestyle='dashed', linewidth=0.5)

    plt.xlabel(c + ': Means(+-)STDs')

    plt.ylim(0, 1)
def build_funcs(col_name):

    assert col_name in cat_cols

    new_col = frame.groupby(col_name).left.mean().sort_values().reset_index()

    new_col = new_col.reset_index().set_index(col_name).drop('left', axis=1)

    new_col.columns = ['nums']

    new_col = new_col.nums.to_dict()

    print('### ' + col_name + ':')

    for k in new_col.keys():

        print(k, '=', new_col[k])

    def add_new_col(x):

        if x not in new_col.keys(): return int(len(new_col.keys())/2)

        return new_col[x]

    return add_new_col



for c in cat_cols:

    f_new_col  = build_funcs(c)

    frame[c + '_new'] = frame[c].apply(f_new_col)
for c in ['number_project', 'average_montly_hours', 'time_spend_company', 

          'Work_accident', 'left', 'promotion_last_5years', 'sales_new', 'salary_new']:

    frame[c] = frame[c].astype(np.int16)

X = frame.drop(cat_cols, axis=1)

show_dataframe(X, 1)

X.info(max_cols=0)
y = X.pop('left')  # get target

X_train, X_test, y_train, y_test = train_test_split(X, y, 

                                                    test_size=0.30,  # 30% on the test

                                                    stratify=y,      # balancing the split

                                                    random_state=42) # fixed random state
### XGBoost

# prepare dict of params for xgboost to run with

xgb_params = {

    'n_trees': 100, 

    'eta': 0.1,

    'max_depth': 7,

    'subsample': 0.75,

    'colsample_bytree': 0.75,

    'objective': 'binary:logistic',

    'scale_pos_weight': float(len(y_train)-sum(y_train)) / sum(y_train),

    'eval_metric': 'auc',

    'silent': 1

}

# form DMatrices for Xgboost training

dtrain_xgb = xgb.DMatrix(X_train, y_train)

dtest_xgb = xgb.DMatrix(X_test, y_test)

# xgboost, cross-validation

cv_result_xgb = xgb.cv(xgb_params, 

                   dtrain_xgb, 

                   num_boost_round=5000,

                   nfold = 5,

                   stratified=True,

                   early_stopping_rounds=50,

                   verbose_eval=100, 

                   show_stdv=True

                  )

num_boost_rounds_xgb = len(cv_result_xgb)

print('num_boost_rounds=' + str(num_boost_rounds_xgb))

# train model

model_xgb = xgb.train(dict(xgb_params, silent=0), 

                      dtrain_xgb, 

                      num_boost_round=num_boost_rounds_xgb)



### Visualizations about the training process:

plt.figure(figsize=(10,5))

# Features importance

plt.subplot(1,2,1)

features_score_xgb = pd.Series(model_xgb.get_fscore()).sort_values(ascending=False)

sns.barplot(x=features_score_xgb.values, 

            y=features_score_xgb.index.values, 

            orient='h', color='b')

# CV scores

plt.subplot(1,2,2)

train_scores = cv_result_xgb['train-auc-mean']

train_stds = cv_result_xgb['train-auc-std']

plt.plot(train_scores, color='blue')

plt.fill_between(range(len(cv_result_xgb)), 

                 train_scores - train_stds, 

                 train_scores + train_stds, 

                 alpha=0.1, color='blue')

test_scores = cv_result_xgb['test-auc-mean']

test_stds = cv_result_xgb['test-auc-std']

plt.plot(test_scores, color='red')

plt.fill_between(range(len(cv_result_xgb)), 

                 test_scores - test_stds, 

                 test_scores + test_stds, 

                 alpha=0.1, color='red')

plt.title('Train and test cv scores (AUC)')

plt.ylim(0.96,1)

plt.show()



### Evaluation

threshold = 0.5

y_pred_xgb = model_xgb.predict(dtest_xgb)

y_cl_xgb = [1 if x > threshold else 0 for x in y_pred_xgb]

print('Threshold:', threshold)

print('Accuracy:  {:.2f} %'.format(accuracy_score(y_test, y_cl_xgb)*100))

print('R2:        {:.4f}'.format(r2_score(y_test, y_cl_xgb)))

print('AUC:       {:.4f}'.format(roc_auc_score(y_test, y_cl_xgb)))

mis = sum(np.abs(y_test - np.array(y_cl_xgb)))

print('Misclass.: {} (~{:.2f} %) out of {}'.format(mis, 

                                                   float(mis)/len(y_test)*100, 

                                                   len(y_test)))

print(classification_report(y_test, 

                            y_cl_xgb, 

                            labels=[0,1], 

                            target_names=['stay', 'left'], 

                            digits=4))
# LightGBM 

lgb_params = {

    'learning_rate': 0.1,

    'max_depth': 7,

    'num_leaves': 40, 

    'objective': 'binary',

    'feature_fraction': 0.75,

    'bagging_fraction': 0.75,

    'metric': 'auc',

    'max_bin': 100}

# form LightGBM datasets

dtrain_lgb = lgb.Dataset(X_train, label=y_train)

dtest_lgb = lgb.Dataset(X_test, label=y_test)

# LightGBM, cross-validation

cv_result_lgb = lgb.cv(lgb_params, 

                       dtrain_lgb, 

                       num_boost_round=5000, 

                       nfold=5, 

                       stratified=True, 

                       early_stopping_rounds=50, 

                       verbose_eval=100, 

                       show_stdv=True)

num_boost_rounds_lgb = len(cv_result_lgb['auc-mean'])

print('num_boost_rounds_lgb=' + str(num_boost_rounds_lgb))

# train model

model_lgb = lgb.train(lgb_params, 

                      dtrain_lgb, 

                      num_boost_round=num_boost_rounds_lgb)



### Visualizations about the training process:

plt.figure(figsize=(10,5))

# Features importance

plt.subplot(1,2,1)

feature_imp = pd.Series(dict(zip(X_train.columns, 

                                 model_lgb.feature_importance())

                            )

                       ).sort_values(ascending=False)

sns.barplot(x=feature_imp.values, y=feature_imp.index.values, orient='h', color='g')

# CV scores

plt.subplot(1,2,2)

train_scores = np.array(cv_result_lgb['auc-mean'])

train_stds = np.array(cv_result_lgb['auc-stdv'])

plt.plot(train_scores, color='green')

plt.fill_between(range(len(cv_result_lgb['auc-mean'])), 

                 train_scores - train_stds, 

                 train_scores + train_stds, 

                 alpha=0.1, color='green')

plt.title('LightGMB CV-results')

plt.ylim(0.96,1)

plt.show()



### Evaluation

threshold = 0.5

y_pred_lgb = model_lgb.predict(X_test)

y_cl_lgb = [1 if x > threshold else 0 for x in y_pred_lgb]

print('Threshold:', threshold) 

print('Accuracy:  {:.2f} %'.format(accuracy_score(y_test, y_cl_lgb)*100))

print('R2:        {:.4f}'.format(r2_score(y_test, y_cl_lgb)))

print('AUC:       {:.4f}'.format(roc_auc_score(y_test, y_cl_lgb)))

mis = sum(np.abs(y_test - np.array(y_cl_lgb)))

print('Misclass.: {} (~{:.2f} %) out of {}'.format(mis, 

                                                   float(mis)/len(y_test)*100, 

                                                   len(y_test)))

print(classification_report(y_test, 

                            y_cl_lgb, 

                            labels=[0,1], 

                            target_names=['stay', 'left'], 

                            digits=4))