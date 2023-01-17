# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import lightgbm as lgb



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

test = pd.read_csv("../input/data-science-bowl-2019/test.csv")

specs = pd.read_csv("../input/data-science-bowl-2019/specs.csv")

train = pd.read_csv("../input/data-science-bowl-2019/train.csv")

train_labels = pd.read_csv("../input/data-science-bowl-2019/train_labels.csv")

sample_submission = pd.read_csv("../input/data-science-bowl-2019/sample_submission.csv")
#typeがAssesmentのレコード、かつ重複していない行を抽出

keep_id = train[train.type == "Assessment"][['installation_id']].drop_duplicates()

train = pd.merge(train, keep_id, on="installation_id", how="inner")
#train_labelsにgame_sessionが存在するレコードのみにフィルタ

train = train[train.game_session.isin(train_labels.game_session.unique())]

train.shape
train = train.merge(train_labels[['installation_id','game_session','accuracy_group']],

                 how='left', on=['installation_id','game_session'])
train = train.dropna()
# train_labels_gr0 = train_labels.loc[train_labels['accuracy_group'] == 0]

# train_labels_gr1 = train_labels.loc[train_labels['accuracy_group'] == 1]

# train_labels_gr2 = train_labels.loc[train_labels['accuracy_group'] == 2]

# train_labels_gr3 = train_labels.loc[train_labels['accuracy_group'] == 3]

# train_labels_gr0.describe(),train_labels_gr1.describe(), train_labels_gr2.describe(), train_labels_gr3.describe()
train_x = train.drop(['accuracy_group'], axis=1)

train_y = train.drop(['event_id','game_session','timestamp','event_data','installation_id','event_count','event_code','game_time','title','type','world'], axis=1)

del train
# 両方のセットへ「is_train」のカラムを追加

# 1 = trainのデータ、0 = testデータ

train_x['is_train'] = 1

test['is_train'] = 0

# trainのデータをtestと連結

train_test_combine = pd.concat([train_x,test],axis=0)
# 「event_id」をキーにして、トレーニングデータ、テストデータとspecsを左外部結合

train_test_combine_specs = pd.merge(train_test_combine, specs, on='event_id', how='left')

del train_test_combine

del specs
train_test_combine_specs['timestamp'] = pd.to_datetime(train_test_combine_specs['timestamp'])
Weekday = pd.DataFrame(train_test_combine_specs.timestamp.dt.dayofweek)

time = pd.DataFrame(train_test_combine_specs.timestamp.dt.hour)

train_test_combine_specs['Weekday'] = Weekday

train_test_combine_specs['time'] = time

train_test_combine_specs.head()
from sklearn.preprocessing import OneHotEncoder

cat_cols = ['title', 'world']

# OneHotEncoderでのencoding

ohe = OneHotEncoder(sparse=False, categories='auto')

ohe.fit(pd.DataFrame(train_test_combine_specs[cat_cols]))



# ダミー変数の列名の作成

columns = []

for i, c in enumerate(cat_cols):

    columns += [f'{c}_{v}' for v in ohe.categories_[i]]



# 生成されたダミー変数をデータフレームに変換

dummy_vals_train = pd.DataFrame(ohe.transform(train_test_combine_specs[cat_cols]), columns=columns)



# 残りの変数と結合

combine_x = pd.concat([train_test_combine_specs.drop(cat_cols, axis=1), dummy_vals_train], axis=1)

del train_test_combine_specs

del dummy_vals_train

combine_x.head()
import category_encoders as ce



# Eoncodeしたい列をリストで指定。もちろん複数指定可能。

list_cols = ['event_id', 'game_session',

       'event_code', 'type', 'info', 'args']



# 序数をカテゴリに付与して変換

ce_oe = ce.OrdinalEncoder(cols=list_cols,handle_unknown='impute')

df_session_ce_ordinal = ce_oe.fit_transform(combine_x)

del combine_x

df_session_ce_ordinal.head()
# 「is_train」のフラグでcombineからtestとtrainへ切り分ける

df_test = df_session_ce_ordinal.loc[df_session_ce_ordinal['is_train'] == 0]

df_train = df_session_ce_ordinal.loc[df_session_ce_ordinal['is_train'] == 1]



del df_session_ce_ordinal



# 「is_train」をtrainとtestのデータフレームから落とす

test_x = df_test.drop(['installation_id', 'is_train','timestamp', 'event_data'], axis=1)

train_x = df_train.drop(['installation_id', 'is_train','timestamp', 'event_data'], axis=1)



# サイズの確認をしておきましょう

train_x.shape, test_x.shape
# train_a = df_train.merge(train_labels[['installation_id','accuracy_group']],

#                  how='left')

# train_a = train_a.dropna()



# del df_train

# del train_labels

# from sklearn.preprocessing import Imputer



# imr = Imputer(missing_values='NaN', strategy='median', axis=0)

# imr = imr.fit(train_a.values)

# imputed_data = pd.DataFrame(imr.transform(train_a.values))

# imputed_data.isnull().sum()

# imputed_data.columns = ['event_id','game_session','timestamp','event_data','event_count','event_code','game_time','title','type','world','info','args','accuracy_group']

# imputed_data.head()



# train_x = df_train.drop(['installation_id'], axis=1)

# train_x
# from xgboost import XGBClassifier



# model = XGBClassifier(n_estimators=20, random_state=71, max_depth=10)



# model.fit(train_x, train_y)
from sklearn.model_selection import KFold

from sklearn.metrics import accuracy_score, log_loss, make_scorer





kf = KFold(n_splits=4, shuffle=True, random_state=71)

tr_idx, va_idx = list(kf.split(train_x))[0]

tr_x, va_x = train_x.iloc[tr_idx], train_x.iloc[va_idx]

tr_y, va_y = train_y.iloc[tr_idx], train_y.iloc[va_idx]



# del train_x

# del train_y

# del va_idx



# -----------------------------------

# lightgbmの実装

# -----------------------------------

import lightgbm as lgb

from sklearn.metrics import log_loss



# 特徴量と目的変数をlightgbmのデータ構造に変換する

lgb_train = lgb.Dataset(tr_x, tr_y)

lgb_eval = lgb.Dataset(va_x, va_y)



# ハイパーパラメータの設定

params = {'objective': 'multiclass', 'seed': 71, 'verbose': 0, 'metrics': 'multi_logloss', 'num_class': 4, 'lambda_l1': 0.1,

        'lambda_l2': 1}

#params = {'objective': 'regression', 'seed': 71, 'verbose': 0, 'metrics': 'l2'}



num_round = 1000



# 学習の実行

# カテゴリ変数をパラメータで指定している

# バリデーションデータもモデルに渡し、学習の進行とともにスコアがどう変わるかモニタリングする

#categorical_features = ['product', 'medical_info_b2', 'medical_info_b3']



model = lgb.train(params, lgb_train, num_boost_round=num_round,early_stopping_rounds=50,

                  #categorical_feature=categorical_features,

                  valid_names=['train', 'valid'], valid_sets=[lgb_train, lgb_eval] 

                  )



# バリデーションデータでのスコアの確認

va_pred = model.predict(va_x)

score = log_loss(va_y, va_pred)

print(f'logloss: {score:.4f}')
# train_labels_gr0 = train_labels.loc[train_labels['accuracy_group'] == 0]

# train_labels_gr1 = train_labels.loc[train_labels['accuracy_group'] == 1]

# train_labels_gr2 = train_labels.loc[train_labels['accuracy_group'] == 2]

# train_labels_gr3 = train_labels.loc[train_labels['accuracy_group'] == 3]

# train_labels_gr0.describe(),train_labels_gr1.describe(), train_labels_gr2.describe(), train_labels_gr3.describe()
# test_a = df_test.drop(['installation_id'], axis=1)

# preds = model.predict(test_a)
# preds = np.where((preds >= 0) & (preds <= 0.3), 0, preds)

# preds = np.where((preds >= 0.31) & (preds <= 0.45), 1, preds)

# preds = np.where((preds >= 0.46) & (preds <= 0.9), 2, preds)

# preds = np.where((preds >= 0.91) & (preds <= 1), 3, preds)
#test_a = df_test.drop(['installation_id'], axis=1)

preds = model.predict(test_x)



y_pred = []



for x in preds:

    y_pred.append(np.argmax(x))



#del test_a



y_preds = pd.Series(y_pred)



# テストデータのIDと予測値を連結

#test = test.installation_id.nunique()

submit = pd.concat([test.installation_id, y_preds], axis=1)

submit.columns = ['installation_id', 'accuracy_group']



group_gsgb_pred = pd.DataFrame(submit.groupby(['installation_id'])['accuracy_group'].agg(lambda x:x.value_counts().index[0])) 

group_gsgb_pred = group_gsgb_pred.round().astype(int)

group_gsgb_pred.head(10)





# カラム名をメルカリの提出指定の名前をつける

#submit['accuracy_group'] = test.groupby('installation_id').last()

#submit = submit.groupby('installation_id').agg('accuracy_group')

#submit = submit.groupby('installation_id')['accuracy_group'].agg(lambda x:x.value_counts().index[0])



# 提出ファイルとしてCSVへ書き出し

#submit.to_csv('submission.csv', index=False)
finalsubmission_gsgb = pd.DataFrame({'installation_id': group_gsgb_pred.index,'accuracy_group': group_gsgb_pred['accuracy_group']})

finalsubmission_gsgb.index = sample_submission.index

finalsubmission_gsgb.to_csv('submission.csv', index=False)