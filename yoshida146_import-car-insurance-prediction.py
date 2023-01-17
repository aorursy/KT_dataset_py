!pip install optuna==2.0.0rc0
import pandas as pd # データフレームやデータ処理を行うライブラリ

import numpy as np # 数値計算を行うライブラリ

import os # PythonからOSの機能を使用するライブラリ

import joblib # データの保存や並列処理

import matplotlib.pyplot as plt # 可視化

import seaborn as sns # pltをラッパーした可視化

import optuna # パラメータチューニング

import itertools

%matplotlib inline

sns.set() # snsでpltの設定をラッパー



import warnings

warnings.filterwarnings('ignore')
SEED = 42
from sklearn.metrics import mean_squared_error, make_scorer

def rmse(y_true, y_pred):

    return np.sqrt(mean_squared_error(y_true, y_pred))

rmse_scorer = make_scorer(rmse)
_input_path = os.path.join("..", "input", '1056lab-import-car-insurance-prediction')

os.listdir(_input_path)
target_col = "symboling"

df_train = pd.read_csv(os.path.join(_input_path, "train.csv"), index_col=0)

df_test = pd.read_csv(os.path.join(_input_path, "test.csv"), index_col=0)
df_train.head()
df_test.head()
df_train.info()
df_test.info()
sns.countplot(df_train[target_col])

plt.plot()
makers = df_train['make'].unique()

dict_makers = {}

for maker in makers:

    df_maker_tmp = df_train[df_train['make'] == maker].copy()

    

    dict_makers[maker] = {

        'train_length': len(df_maker_tmp),

        'test_length': len(df_test[df_test['make'] == maker]),

        'max': df_maker_tmp[target_col].max(), 

        'min': df_maker_tmp[target_col].min()

    }
dict_makers
df_train.sort_values(by=[target_col, 'make'])
def check_unique(X, X_test, col):

    return pd.concat([X[col], X_test[col]]).unique()
rep_misssing_val = {"?": np.nan}

df_train.replace(rep_misssing_val, inplace=True)

df_test.replace(rep_misssing_val, inplace=True)
# trainとtestで値にどんなものがあるか確認

pd.concat([df_train["num-of-cylinders"], df_test["num-of-cylinders"]]).unique()
# 英語から数字へと置き換え

rep_num_cylinders = {"four": 4, "five": 5, "six": 6, "three": 3, "twelve": 12, "two": 2, "eight": 8}

df_train["num-of-cylinders"].replace(rep_num_cylinders, inplace=True)

df_test["num-of-cylinders"].replace(rep_num_cylinders, inplace=True)
pd.concat([df_train["num-of-doors"], df_test["num-of-doors"]]).unique()
rep_num_doors = {"two": 2, "four": 4}

df_train["num-of-doors"].replace(rep_num_doors, inplace=True)

df_test["num-of-doors"].replace(rep_num_doors, inplace=True)
def extract_obj_cols(X):

    obj_cols = []

    for col, typ in zip(X.columns, X.dtypes):

        if typ != "float" and typ != "int":

            try:

                X[col].astype(float)

            except:

                obj_cols.append(col)

                

    return obj_cols





object_cols = []

for col, types in zip(df_test.columns, df_test.dtypes):

    if types != "float" and types != "int":

        try:

            df_train[col] = df_train[col].astype(float)

            df_test[col] = df_test[col].astype(float)

        except:

            object_cols.append(col)
df_train.info()
df_train.describe()
from sklearn.base import BaseEstimator, TransformerMixin

import category_encoders as ce

from sklearn.preprocessing import LabelEncoder



class CountEncoder(BaseEstimator, TransformerMixin):

    def __init__(self, return_df=False, return_full=False, ignore_cols=[], trans_cols=[]):

        self.return_df = return_df

        self.return_full = return_full

        self.ignore_cols = ignore_cols

        self.trans_cols = trans_cols

    

    def _check_in_cols(self, X):

        if isinstance(X, pd.core.series.Series):

            return not X.name in self.trans_cols

        if isinstance(X, pd.core.frame.DataFrame):

            pass

    

    def fit(self, X, y=None):

        assert isinstance(X, pd.core.frame.DataFrame) or isinstance(X, pd.core.series.Series), "XはDataFrameかSeriesにしてください"

        

        if len(self.trans_cols) <= 0 or self.trans_cols is None:

            if isinstance(X, pd.core.series.Series):

                cols = [X.name]

            elif isinstance(X, pd.core.frame.DataFrame):

#                 cols = X.columns

                cols = extract_obj_cols(X)

                cols = list(set(cols) - set(self.ignore_cols))

            self.trans_cols = cols



        count = {}

        for col in self.trans_cols:

            if isinstance(X, pd.core.series.Series):

                count[col] = X.value_counts().to_dict()

            else:

                count[col] = X[col].value_counts().to_dict()

        self.count = count



        return self

    

    def transform(self, X, y=None):

#         print(self.trans_cols)

#         print(self.count)

        assert self.trans_cols is not None, "変換する列名が設定されていません"

#         assert not False in [i in X.columnsself.trans_cols]

        

        df_transed = X.copy()

        

        for col in self.trans_cols:

            if isinstance(X, pd.core.series.Series):

                diff = list([set(self.count[col].keys())][0] - set(X.unique()))

#                 print(diff)

                for i in diff: self.count[i] = 0

                df_transed = df_transed.map(self.count[col])

            else:

                diff = list(set([self.count[col].keys()][0]) - set(X[col].unique()))

                for i in diff: self.count[i] = 0

                df_transed["CE_"+col] = df_transed[col].map(self.count[col])

        

        if self.return_full or isinstance(X, pd.core.series.Series):

            return df_transed

        else:

            return df_transed[["CE_"+col for col in self.trans_cols]]
class FrequencyEncoder(BaseEstimator, TransformerMixin):

    def __init__(self):

        pass

    

    def fit(self, X: pd.Series, y=None):

        self._dict = X.value_counts().to_dict()

        return self

    

    def transform(self, X: pd.Series, y=None):

        transed = X.map(self._dict) / X.count()

        return transed
class CombinCountEncoder(BaseEstimator, TransformerMixin):

    def __init__(self, trans_cols=[], prefix='', return_full=True):

        self.trans_cols = trans_cols

        self.prefix = prefix

        self.return_full = return_full

        

    def fit(self, X: pd.DataFrame, y=None):

        if len(self.trans_cols) <= 0:

            self.trans_cols = extract_obj_cols(X.columns)

        return self

    

    def transform(self, X: pd.DataFrame, y=None):

        X_ = X.copy()

        transed_cols = []

        for cols in list(itertools.combinations(self.trans_cols, 2)):

            col_name = '{}{}_{}'.format(self.prefix, cols[0], cols[1])

            _tmp = X_[cols[0]].astype(str) + '_' + X_[cols[1]].astype(str)

            cnt_map = _tmp.value_counts().to_dict()

            X_['CCE_'+col_name] = _tmp.map(cnt_map)

            transed_cols.append('CCE_'+col_name)

        

        return X_ if self.return_full else X_[transed_cols]
class AutoCalcEncoder(BaseEstimator, TransformerMixin):

    def __init__(self, num_cols=[], return_full=True, target_col='symboling'):

        self.num_cols = num_cols

        self.return_full = return_full

        self.target_col = target_col

        

    def fit(self, X: pd.DataFrame, y=None):

        if len(self.num_cols) <= 0:

            self.num_cols = list(set(X.columns) - set(extract_obj_cols(X)))

        if target_col in self.num_cols:

            self.num_cols.remove(target_col)

        return self

    

    def transform(self, X: pd.DataFrame, y=None):

        X_ = X.copy()

        transed_cols = []

        for cols in list(itertools.combinations(self.num_cols, 2)):

            left, right = X_[cols[0]].astype(float), X_[cols[1]].astype(float)

            X_['{}_plus_{}'.format(cols[0], cols[1])] = left + right

            transed_cols.append('{}_plus_{}'.format(cols[0], cols[1]))

            X_['{}_mul_{}'.format(cols[0], cols[1])] = left * right

            transed_cols.append('{}_mul_{}'.format(cols[0], cols[1]))

            try:

                X_['{}_div_{}'.format(cols[0], cols[1])] = left / right

                transed_cols.append('{}_div_{}'.format(cols[0], cols[1]))



#                 X_['{}_div_{}'.format(cols[0], cols[1])] = right / left

#                 transed_cols.append('{}_div_{}'.format(cols[1], cols[0]))

                    

            except:

                print('{}_div_{}'.format(cols[0], cols[1]))

            

        return X_ if self.return_full else X_[transed_cols]
class NullCounter(BaseEstimator, TransformerMixin):

    def __init__(self, count_cols=[], encoded_feateure_name='null_count'):

        self.count_cols = count_cols

        self.encoded_feateure_name = encoded_feateure_name

        

    def fit(self, X: pd.DataFrame, y=None):

        return self

    

    def transform(self, X: pd.DataFrame, y=None):

        X[self.encoded_feateure_name] = X.isnull().sum(axis=1)

        return X
def preprocess(X, X_test=None, target_col='symboling'):

    X_, X_test_ = X.copy(), X_test.copy()

#     X_['length_cylinder'] = X_['num-of-cylinders'] * X_['stroke']

#     X_test_['length_cylinder'] = X_test_['num-of-cylinders'] * X_test_['stroke']

    

#     X_['engine_quality'] = X_['bore'] * X_['stroke']

#     X_test_['engine_quality'] = X_test_['bore'] * X_test_['stroke']

    

#     X_['area'] = X_['length'] * X_['width']

#     X_test_['area'] = X_test_['length'] * X_test_['width']



    X_['volume'] = X_['length'] * X_['width'] * X_['height'] / 10000

    X_test_['volume'] = X_test_['length'] * X_test_['width'] * X_test_['height'] / 10000



#     X_['weight_per_vol'] = X_['curb-weight'] / X_['volume']

#     X_test_['weight_per_vol'] = X_test_['curb-weight'] / X_test_['volume']

    

    null_counter = NullCounter()

    X_ = null_counter.fit_transform(X_)

    X_test_ = null_counter.transform(X_test_)

    

    object_cols = extract_obj_cols(X_)

    

    cce = CombinCountEncoder(trans_cols=object_cols, return_full=True)

    _tmp_y_test = X_test_[target_col].values if target_col in X_test_.columns else None

#     print(_tmp_y_test)

    _tmp_y = X_[target_col].values

    _X_full = cce.fit_transform(

        pd.concat([X_.drop(target_col, axis=1), X_test_], sort=False).reset_index(drop=True))

    X_ = _X_full[: len(X_)]

    X_test_ = _X_full[len(X_): ].reset_index(drop=True)

    X_[target_col] = _tmp_y

    if _tmp_y_test is not None:

#         print(_tmp_y_test)

        X_test_[target_col] = _tmp_y_test

    del _tmp_y, _X_full

    

    num_cols = list(set(X_test.columns) - set(object_cols) - set(target_col))

    ace = AutoCalcEncoder(num_cols=num_cols, return_full=True)

    X_ = ace.fit_transform(X_)

    X_test_ = ace.transform(X_test_)

#     print(X_test_[target_col])

    for col in object_cols:

#         le = LabelEncoder()

        le = ce.OrdinalEncoder()

        te = ce.TargetEncoder()

        cel = CountEncoder()

        fe = FrequencyEncoder()

        X_['LE_' + col] = le.fit_transform(X_[col])

        X_test_['LE_' + col] = le.transform(X_test_[col])

        X_['TE_' + col] = te.fit_transform(X_[col], X_[target_col])

        X_test_['TE_' + col] = te.transform(X_test_[col])

        X_["CE_"+col] = cel.fit_transform(X_[col])

        X_test_["CE_"+col] = cel.transform(X_test_[col])

        X_['FE_'+col] = fe.fit_transform(X_[col])

        X_test_['FE_'+col] = fe.transform(X_test_[col])

    X_.drop(object_cols, axis=1, inplace=True)

    X_test_.drop(object_cols, axis=1, inplace=True)

        

    return X_, X_test_
df_train_, df_test_ = preprocess(df_train, df_test)

plt.figure(figsize=(20, 20))

sns.heatmap(df_train_.corr().abs(), vmin=-1, vmax=1, square=True)

plt.show()
import lightgbm as lgb



# X = df_train.drop(target_col, axis=1).values

# y = df_train[target_col].values

# X_test = df_test.values



X = df_train.drop(np.append(target_col, object_cols), axis=1)#.values

y = df_train[target_col].values

X_test = df_test.drop(object_cols, axis=1)#.values
import optuna.integration.lightgbm as op_lgb

from sklearn.model_selection import train_test_split



optuna.logging.set_verbosity(optuna.logging.ERROR)



# X_train, X_valid, y_train, y_valid = train_test_split(X, y)

X_train, X_valid = train_test_split(df_train)

X_train, X_valid = preprocess(X_train, X_valid)

y_train, y_valid = X_train[target_col], X_valid[target_col]

X_train.drop(target_col, axis=1, inplace=True)

X_valid.drop(target_col, axis=1, inplace=True)



lgb_params = {

    'objective': 'mean_squared_error',

    'metric': 'rmse',

    'seed': SEED,

}

dtrain = op_lgb.Dataset(X_train, label=y_train)

dvalid = op_lgb.Dataset(X_valid, label=y_valid)



best_params, history = {}, []

model = op_lgb.train(lgb_params, train_set=dtrain, valid_sets=dvalid,

                    num_boost_round=10000, early_stopping_rounds=1000,

                    verbose_eval=False)
best_params = model.params

best_params
X = df_train.copy()

X_test = df_test.copy()

X, X_test = preprocess(X, X_test)

y = X[target_col]

X.drop(target_col, axis=1, inplace=True)

dtrain_full = lgb.Dataset(X, label=y)

model = lgb.train(best_params, dtrain_full)
# df_feat = pd.DataFrame(model.feature_importance(), index=df_test.drop(object_cols, axis=1).columns, columns=['feateure'])

df_feat = pd.DataFrame(model.feature_importance(), index=X_test.columns, columns=['feateure'])

df_feat.sort_values(by='feateure', ascending=False, inplace=True)

df_feat
from sklearn.linear_model import LinearRegression

from sklearn.model_selection import cross_validate

from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import StandardScaler



lr = Pipeline([

    ("imputer", SimpleImputer(strategy="mean")),

    ("scaler", StandardScaler()),

    ("model", LinearRegression())])

cross_validate(lgb.LGBMModel(**best_params), X, y, scoring=rmse_scorer)["test_score"].mean()
import shap

shap.initjs()

explainer = shap.TreeExplainer(model)

shap_val = explainer.shap_values(X)

shap.summary_plot(shap_val,X, max_display=10)
shap.summary_plot(shap_val, X, plot_type='bar')
best_params
from sklearn.model_selection import StratifiedKFold



def runner(X, X_test, n_fold=10, target_col='symboling', best_params={}, base_params={}):

    splitter = StratifiedKFold(n_splits=n_fold, shuffle=True)

    cv = []

    preds = np.zeros(len(X_test))

    oof = np.zeros(len(X))

    for i, ids in enumerate(splitter.split(X.drop(target_col, axis=1), X[target_col])):

        print('{} Fold'.format(i+1))



        X_train, X_valid = preprocess(X.iloc[ids[0]], X.iloc[ids[1]])

        y_train, y_valid = X_train[target_col], X_valid[target_col]

        X_train.drop(target_col, axis=1, inplace=True)

        X_valid.drop(target_col, axis=1, inplace=True)



        lgb_params = {

            'objective': 'mean_squared_error',

            'metric': 'rmse',

            'seed': 42,

        }

        dtrain = op_lgb.Dataset(X_train, label=y_train)

        dvalid = op_lgb.Dataset(X_valid, label=y_valid)

        

        if len(best_params) <= 0:

            model = op_lgb.train(base_params, train_set=dtrain, valid_sets=dvalid,

                                num_boost_round=10000, early_stopping_rounds=1000,

                                verbose_eval=False)

        else:

            model = lgb.LGBMRegressor()

            model.fit(X_train, y_train)

        

        oof[ids[1]] += model.predict(X_valid)

        cv.append(rmse(X.iloc[ids[1]][target_col], oof[ids[1]]))

        print('Crossvalidation RMSE : {}'.format(cv[i]))

        

        X_test_tmp = X_test.copy()

        X_train, X_test_tmp = preprocess(X.iloc[ids[0]], X_test_tmp)

        preds += model.predict(X_test_tmp)

    print('Finish CV : {}'.format(rmse(X[target_col], oof)))

    preds /= n_fold

    return preds
best_params
predict = runner(df_train, df_test, base_params=lgb_params)
predict
submit = pd.read_csv(os.path.join(_input_path, "sampleSubmission.csv"))

submit[target_col] = predict

submit.to_csv("submit.csv", index=False)
import catboost as cat

cat_model = cat.CatBoostRegressor(

#     cat_features=cat_features,

    verbose=100, random_seed=SEED,

    eval_metric='RMSE',

    num_boost_round=1000,

)
X = df_train.copy()

X_test = df_test.copy()

X, X_test = preprocess(X, X_test)

y = X[target_col]

X.drop(target_col, axis=1, inplace=True)



# X_train, X_valid, y_train, y_valid = train_test_split(X, y)

X_train, X_valid = train_test_split(df_train)

X_train, X_valid = preprocess(X_train, X_valid)

y_train, y_valid = X_train[target_col], X_valid[target_col]

X_train.drop(target_col, axis=1, inplace=True)

X_valid.drop(target_col, axis=1, inplace=True)



# cat_features = ['LE_'+feat for feat in cat_features]

cat_features = [col for col in X_test.columns if col.startswith('LE_')]

for col in cat_features:

    X_train[col] = X_train[col].astype(int)

    X_valid[col] = X_valid[col].astype(int)

cat_model.fit(X_train, y_train, cat_features=cat_features, eval_set=(X_valid, y_valid),

             early_stopping_rounds=100)

# cat_model.fit(X, y, cat_features=cat_features)
cat_predict = cat_model.predict(X_test)

cat_predict = np.clip(cat_predict, -2, 3)



submit = pd.read_csv(os.path.join(_input_path, "sampleSubmission.csv"))

submit[target_col] = cat_predict

submit.to_csv("cat_submit.csv", index=False)