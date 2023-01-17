import pandas as pd

import numpy as np



# métrica y datasets

from sklearn.metrics import roc_auc_score

from sklearn.model_selection import train_test_split



# modelos

from xgboost import XGBClassifier

from sklearn.linear_model import LogisticRegression



# feature engineering

from sklearn.feature_selection import f_classif

from sklearn.preprocessing import StandardScaler



# visualización

from seaborn import heatmap



import os

print(os.listdir("../input"))
data = pd.concat([

       pd.read_csv("../input/banco-galicia-dataton-2019/pageviews/pageviews.csv", parse_dates=["FEC_EVENT"]),

       pd.read_csv("../input/banco-galicia-dataton-2019/pageviews_complemento/pageviews_complemento.csv", parse_dates=["FEC_EVENT"])

])



# Test

X_test = []

for c in data.drop(["USER_ID", "FEC_EVENT"], axis=1).columns:

    print("haciendo", c)

    temp = pd.crosstab(data.USER_ID, data[c])

    temp.columns = [c + "_" + str(v) for v in temp.columns]

    X_test.append(temp.apply(lambda x: x / x.sum(), axis=1))

X_test = pd.concat(X_test, axis=1)



# Train

data = data[data.FEC_EVENT.dt.month < 10]

X_train = []

for c in data.drop(["USER_ID", "FEC_EVENT"], axis=1).columns:

    print("haciendo", c)

    temp = pd.crosstab(data.USER_ID, data[c])

    temp.columns = [c + "_" + str(v) for v in temp.columns]

    X_train.append(temp.apply(lambda x: x / x.sum(), axis=1))

X_train = pd.concat(X_train, axis=1)



features = list(set(X_train.columns).intersection(set(X_test.columns)))

X_train = X_train[features]

X_test = X_test[features]



y_prev = pd.read_csv("../input/banco-galicia-dataton-2019/conversiones/conversiones.csv")

y_train = pd.Series(0, index=X_train.index)

idx = set(y_prev[y_prev.mes >= 10].USER_ID.unique()).intersection(

        set(X_train.index))

y_train.loc[list(idx)] = 1
def model_auc_score(model, X_t, y_t, X_v, y_v):

    model.fit(X_t, y_t)

    y_pred_t = model.predict_proba(X_t)[:,1]

    y_pred_v = model.predict_proba(X_v)[:,1]

    auc_t = roc_auc_score(y_t, y_pred_t)

    auc_v = roc_auc_score(y_v, y_pred_v)

    return auc_t, auc_v



def report_model(label, auc_t, auc_v):

    print('------- ' + label + '  --------')

    print('Training auc:   ' + str(auc_t))

    print('Validation auc: ' + str(auc_v))

    

def normalize(x_train, x_test):

    scaler = StandardScaler()

    x_train = scaler.fit_transform(x_train)

    x_test = scaler.transform(x_test)

    return x_train, x_test
SEED = 18 # mi número de la suerte :)

x_train = X_train.values

y_tr = y_train.values



# dividiendo el dataset

X_t, X_v, y_t, y_v = train_test_split(x_train, y_tr, test_size=0.20, random_state=SEED)

# normalizando

X_t, X_v = normalize(X_t, X_v)



# logistic regression

lr = LogisticRegression(random_state=SEED)

auc_t, auc_v = model_auc_score(lr, X_t, y_t, X_v, y_v)

report_model('Logistic Regression', auc_t, auc_v)





# xgb

xgb = XGBClassifier(random_state=SEED, booster='gbtree')

auc_t, auc_v = model_auc_score(xgb, X_t, y_t, X_v, y_v)

report_model('XGB', auc_t, auc_v)
features = X_train.columns

f_scores = f_classif(X_train, y_train)[0] # el [1] son los p-values.



df_fscores = pd.DataFrame({'features':features,'score':f_scores})

df_fscores = df_fscores.sort_values('score', ascending = False)

df_fscores.head(10)
# eliminando las features que no son relevantes

LIM_IRR = 10 # ajustable

df_fscores = df_fscores[df_fscores['score'] > LIM_IRR]

X_sel = X_train[df_fscores['features']]

X_sel.shape
xcorr = X_sel.corr().abs()

heatmap(xcorr)
# seleccionando las features redundantes

LIM_COR = 0.9 # ajustable

xcorr = xcorr[xcorr > LIM_COR].fillna(0)

index = []

column = []

for idx in list(xcorr.index):

    for col in list(xcorr.columns):

        # la matriz es diagonal

        if idx == col:

            break

        if (xcorr.loc[idx,col] != 0):

            index = index + [idx]

            column = column + [col]

df_fcorr = pd.DataFrame({'index':index, 'col':column})

df_fcorr.head(10)
# dropeamos las features correlacionadas que estan menos correlacionadas con el target

for idx in df_fcorr.index:

    f_idx = df_fcorr.loc[idx,'index']

    f_col = df_fcorr.loc[idx,'col']

    score_idx = df_fscores.loc[df_fscores['features'] == f_idx, 'score'].ravel()

    score_col = df_fscores.loc[df_fscores['features'] == f_col, 'score'].ravel()

    if score_idx > score_col:

        df_fcorr.loc[idx, 'drop'] = f_col

    else:

        df_fcorr.loc[idx, 'drop'] = f_idx 

drop_features = list(df_fcorr['drop'].unique())



X_sel.drop(columns=drop_features, inplace=True)

X_sel.shape
x_sel = X_sel.values

X_t, X_v, y_t, y_v = train_test_split(x_sel, y_tr, test_size=0.20, random_state=SEED)

X_t, X_v = normalize(X_t, X_v)
lr = LogisticRegression(random_state=SEED)

[auc_t, auc_v] = model_auc_score(lr, X_t, y_t, X_v, y_v)

report_model('Logistic Regression', auc_t, auc_v)



xgb = XGBClassifier(random_state=SEED)

[auc_t, auc_v] = model_auc_score(xgb, X_t, y_t, X_v, y_v)

report_model('XGB', auc_t, auc_v)