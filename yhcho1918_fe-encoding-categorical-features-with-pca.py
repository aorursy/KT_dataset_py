import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

pd.set_option('max_columns', 10, 'max_rows', 20)
df_train = pd.read_csv('../input/X_train.csv', encoding='cp949')
df_test = pd.read_csv('../input/X_test.csv', encoding='cp949')
df = pd.concat([df_train, df_test])
df
df_train.columns
from sklearn.decomposition import PCA

def dummy_to_pca(tr, column_name:str, features) :
    max_seq = 300
    max_d = 15
    col_count = tr.groupby(column_name)[column_name].count()
    if len(col_count) > max_seq:
        tops = col_count.sort_values(ascending=False)[0:max_seq].index
        f =tr.loc[tr[column_name].isin(tops)][['cust_id', column_name]]
    else:
        tops = col_count.index
        f =tr[['cust_id', column_name]]
    f = pd.get_dummies(f, columns=[column_name])  # This method performs One-hot-encoding
    f = f.groupby('cust_id').mean()
    if len(tops) < max_d:
        max_d = len(tops)
    pca = PCA(n_components=max_d)
    pca.fit(f)
    cumsum = np.cumsum(pca.explained_variance_ratio_) #분산의 설명량을 누적합
    num_d = np.argmax(cumsum >= 0.99) + 1 # 분산의 설명량이 99%이상 되는 차원의 수
    if num_d == 1:
        num_d = max_d
    pca = PCA(n_components=num_d)    
    result = pca.fit_transform(f)
    result = pd.DataFrame(result)
    result.columns = [column_name + '_' + str(column) for column in result.columns]
    result.index = f.index
    return pd.concat([features, result], axis=1, join_axes=[features.index])
# Extract Numeric features
f = df.groupby('cust_id').agg({
    'amount': [('총구매액', 'sum'),('구매건수', 'size'),('평균구매가격', 'mean'),('최대구매액', 'max')],
    'gds_grp_nm': [('구매상품다양성', lambda x: x.nunique()), 
               ('구매상품다양성비', lambda x: x.nunique()/x.count())],
    'tran_date': [
        ('내점일수',lambda x: x.str[:10].nunique()),
        ('내점비율',lambda x: x.str[:10].nunique()/x.count()),
        ('주말방문비율', lambda x: np.mean(pd.to_datetime(x).dt.dayofweek>4)),
        ('봄-구매비율', lambda x: np.mean( pd.to_datetime(x).dt.month.isin([3,4,5]))),
        ('여름-구매비율', lambda x: np.mean( pd.to_datetime(x).dt.month.isin([6,7,8]))),
        ('가을-구매비율', lambda x: np.mean( pd.to_datetime(x).dt.month.isin([9,10,11]))),
        ('겨울-구매비율', lambda x: np.mean( pd.to_datetime(x).dt.month.isin([1,2,12])))
    ],
    }).reset_index()

# Encode Categorical features
f.columns = f.columns.get_level_values(1)
f.rename(columns={'': 'cust_id'}, inplace=True)
f = dummy_to_pca(df, 'goods_id', f)
f = dummy_to_pca(df, 'gds_grp_nm', f)
f = dummy_to_pca(df, 'gds_grp_mclas_nm', f)
f = dummy_to_pca(df, 'store_nm', f)
df['month'] = pd.to_datetime(df['tran_date']).dt.month.astype(str)
f = dummy_to_pca(df, 'month', f)
df['week'] = pd.to_datetime(df['tran_date']).dt.dayofweek.astype(str)
f = dummy_to_pca(df, 'week', f)
f
# Features extracted from derived features
f['평균내점구매액'] = f['총구매액']/f['내점일수']
f['주중방문비율'] = 1 - f['주말방문비율']
f['주말방문수'] = (f['주말방문비율'] * f['내점일수']).astype('int64')
f['주중방문수'] = (f['주중방문비율'] * f['내점일수']).astype('int64')
f['내점당평균구매건수'] = f['구매건수']/f['내점일수']
f['주중구매액'] = (f['총구매액'] * f['주중방문비율']).astype('int64')
f['주말구매액'] = (f['총구매액'] * f['주말방문비율']).astype('int64')
# Split Data
X_train = pd.DataFrame({'cust_id': df_train.cust_id.unique()})
X_train = pd.merge(X_train, f, how='left')
display(X_train)

X_test = pd.DataFrame({'cust_id': df_test.cust_id.unique()})
X_test = pd.merge(X_test, f, how='left')
display(X_test)
IDtest = X_test.cust_id;
X_train.drop(['cust_id'], axis=1, inplace=True)
X_test.drop(['cust_id'], axis=1, inplace=True)
y_train = pd.read_csv('../input/y_train.csv').gender
# Learn XGB
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
import sys
import warnings
if not sys.warnoptions: warnings.simplefilter("ignore")

# clf = XGBClassifier()
# parameters = {
#     'max_depth': [4],
#     'subsample': [0.9],
#     'colsample_bytree': [1.0],
#     'learning_rate' : [0.05],
#     'min_child_weight': [5],
#     'silent': [True],
#     'n_estimators': [200]
# }
# clf = GridSearchCV(clf, parameters, n_jobs=1, cv=2)
# clf.fit(X_train, y_train)
# best_est = clf.best_estimator_
# print(best_est)

parameters = {'max_depth': 4, 'subsample': 0.9, 'colsample_bytree': 1.0, 'learning_rate': 0.05, 
              'min_child_weight': 5, 'silent': True, 'n_estimators': 200}
model = XGBClassifier(**parameters, random_state=0, n_jobs=-1)
score = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc')

print('{}\nmean = {:.5f}\nstd = {:.5f}'.format(score, score.mean(), score.std()))
pred = model.fit(X_train, y_train).predict_proba(X_test)[:,1]
fname = 'submission.csv'
submissions = pd.concat([IDtest, pd.Series(pred, name="gender")] ,axis=1)
submissions.to_csv(fname, index=False)
print("'{}' is ready to submit." .format(fname))