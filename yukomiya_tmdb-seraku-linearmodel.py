import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os
for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

from matplotlib import pyplot as plt

import pickle
with open('/kaggle/input/test-jhk-preocessed/train_JHK_processed0.pickle', 'rb') as f:

    train_JHK = pickle.load(f)
with open('/kaggle/input/preprocessed-data/train_KN_use.pickle', 'rb') as f:

    train_KNuse = pickle.load(f)

with open('/kaggle/input/preprocessed-data/test_KN_use.pickle', 'rb') as f:

    test_KNuse = pickle.load(f)

with open('/kaggle/input/preprocessed-data/train_KN_Tfid_tagline.pickle', 'rb') as f:

    train_Tfid_tagline = pickle.load(f)

with open('/kaggle/input/preprocessed-data/test_KN_Tfid_tagline.pickle', 'rb') as f:

    test_Tfid_tagline = pickle.load(f)
with open('/kaggle/input/preprocessed-data/traintest_YK_processed.pickle', 'rb') as f:

    traintest_YK = pickle.load(f)
# index を 1~3000 にそろえる

train_JHK.index = train_KNuse.index

train_Tfid_tagline.index = train_KNuse.index

train_YK = traintest_YK.loc[:3000]
drop_cols = ['id', 'belongs_to_collection', 'budget', 'genres', 'homepage',

       'original_language', 'popularity', 'production_companies',

       'production_countries', 'release_date', 'runtime', 'spoken_languages',

       'tagline', 'title', 'Keywords', 'crew', 'all_cast', 'all_crew']

train_JHK = train_JHK.drop(drop_cols, axis=1)
train_YK[("original_language", "af")].value_counts()
train_all = pd.concat([train_YK, train_KNuse, train_Tfid_tagline, train_JHK ], axis=1).drop("zero", axis=1)

train_all["log_budget"] = np.log10(train_all["budget"]+1)

train_all = train_all.drop("budget", axis=1)
train_all.head()
# 数値ではない列

no_numeric = train_all.apply(lambda s:pd.to_numeric(s, errors='coerce')).isnull().all()

no_numeric[no_numeric]
[x for x in train_all.columns if "revenue" in str(x)]
from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.metrics import mean_squared_error 

from sklearn.preprocessing import StandardScaler
X_train_all = train_all.drop("revenue", axis=1)

y_train_all = np.log(train_all["revenue"]+1)
# 標準化

X_train_all_mean = X_train_all.mean()

X_train_all_std  = X_train_all.std()

X_train_all = (train_all-X_train_all_mean)/X_train_all_std
X_train_all = X_train_all.dropna(axis=1)
train_X, val_X, train_y, val_y = train_test_split(X_train_all, 

                                                  y_train_all, 

                                                  test_size=0.25)
X_train_all.isnull().sum().sort_values()
from sklearn.linear_model import Lasso, Ridge
train_X = train_X.drop("collection_av_logrevenue", axis=1)
val_X = val_X.drop("collection_av_logrevenue", axis=1)
clf = Lasso(alpha=0.1, max_iter=3000, random_state=1)  # default alpha=1, max_iter=1000

clf.fit(train_X, train_y)
coef = pd.Series(clf.coef_, index=train_X.columns)

coef[coef!=0]
df_coef = pd.DataFrame(coef[coef!=0], columns=["coef"])

df_coef[abs(df_coef["coef"])>0.1].sort_values("coef", ascending=False)
val_pred = clf.predict(val_X)

np.sqrt(mean_squared_error(val_pred, val_y))
params = {

      'alpha':[0.01, 0.03, 0.1, 0.3, 1]}

#params = {

#      'alpha':[10, 100, 1000]}
gscv = GridSearchCV(Lasso(), params, cv=4, verbose=2, scoring="neg_mean_squared_error")
gscv.fit(train_X, train_y)

df_gsresult = pd.DataFrame.from_dict(gscv.cv_results_)
df_gsresult
df_gsresult["mean_RMSE"] = np.sqrt(-df_gsresult.loc[:,"split0_test_score":"split3_test_score"]).mean(axis=1)
df_gsresult[["param_alpha", "mean_RMSE"]]
from sklearn.ensemble import RandomForestRegressor
clf2 = RandomForestRegressor(random_state=1, n_jobs=3)  # default alpha=1, max_iter=1000

clf2.fit(train_X, train_y)
val_pred2 = clf2.predict(val_X)

np.sqrt(mean_squared_error(val_pred2, val_y))
df_importance = pd.DataFrame([clf2.feature_importances_], columns=train_X.columns, index=["importance"]).T

df_importance.sort_values("importance", ascending=False).head(20)
params = {

      'max_depth':[4,7,10,13], 

        "max_features":["auto", "sqrt", "log2"]}
gscv2 = GridSearchCV(clf2, params, cv=4, verbose=2, scoring="neg_mean_squared_error")

gscv2.fit(train_X, train_y)

df_gsresult2 = pd.DataFrame.from_dict(gscv2.cv_results_)

df_gsresult2
params = {

      'max_depth':[12, 15], 

        "max_features":["auto"], 

        'min_samples_split' : [3, 5, 10, 20, 30]}
gscv2 = GridSearchCV(clf2, params, cv=4, verbose=2, scoring="neg_mean_squared_error")

gscv2.fit(train_X, train_y)

df_gsresult2 = pd.DataFrame.from_dict(gscv2.cv_results_)

df_gsresult2