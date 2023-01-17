!ls ../input

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt # Visualization
import seaborn as sns # For easier statistical plotting
sns.set_style("whitegrid")
import xgboost
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV

class Experiment:
    # transformers and estimator are to make_pipeline
    transformers = []
    def __init__(self, estimator, remove_outliers=False):
        self.options = {'outliers': remove_outliers}

        self.estimator = estimator
        
        df = pd.read_csv('../input/train.csv')
        df_test = pd.read_csv('../input/test.csv')

        # Drop houses (rows) where the target is missing
        df.dropna(axis=0, subset=['SalePrice'], inplace=True)
        if remove_outliers:
            df = self.remove_outliers(df)
        target = df.SalePrice

        # Drop Id and SalePrice columns from train and test data
        df.drop(['Id', 'SalePrice'], axis=1, inplace=True)
        
        self.train_df = df
        self.train_target = target
        self.test_df = df_test.drop(['Id'], axis=1)
        self.test_ids = df_test.Id
        return
    
    def impute(self):
        self.options['impute'] = True
        self.transformers += [SimpleImputer()]
        return self

    
    def normalize(self):
        self.options['normalize'] = True
        self.transformers += [StandardScaler()]
        return self

    
    def log_scale(self, use_log_scale=False):
        self.options['log_scale'] = use_log_scale
        return self
        

    def num_columns(self):
        return self.train_df.select_dtypes(exclude=np.object).columns

    def not_num_columns(self):
        return self.train_df.select_dtypes(include=np.object).columns

    def categorical_columns(self):
        return self.train_df.select_dtypes(include=pd.Categorical).columns

    # Numeric norminal columns should be take into OHE
    numeric_norminal_columns = ['MSSubClass']

    ordinal_columns = {
            'LotShape': ['IR3', 'IR2', 'IR1', 'Reg'],
            'Utilities': ['ELO', 'NoSeWa', 'NoSewr', 'AllPub'],
            'LandSlope': ['Sev', 'Mod', 'Gtl'],
            'ExterQual': ['Po', 'Fa', 'TA', 'Gd', 'Ex'],
            'ExterCond': ['Po', 'Fa', 'TA', 'Gd', 'Ex'],
            'BsmtQual': ['Po', 'Fa', 'TA', 'Gd', 'Ex'],
            'BsmtCond': ['Po', 'Fa', 'TA', 'Gd', 'Ex'],
            'BsmtExposure': ['No', 'Mn', 'Av', 'Gd'],
            'BsmtFinType1': ['Unf', 'LwQ', 'Rec', 'BLQ', 'ALQ', 'GLQ'],
            'BsmtFinType2': ['Unf', 'LwQ', 'Rec', 'BLQ', 'ALQ', 'GLQ'],
            'HeatingQC': ['Po', 'Fa', 'TA', 'Gd', 'Ex'],
            'Electrical': ['Mix', 'FuseP', 'FuseF', 'FuseA', 'SBrkr'],
            'KitchenQual': ['Po', 'Fa', 'TA', 'Gd', 'Ex'],
            'Functional': ['Sql', 'Sev', 'Maj2', 'Maj1', 'Mod', 'Min2', 'Min1', 'Typ'],
            'FireplaceQu': ['Po', 'Fa', 'TA', 'Gd', 'Ex'],
            'GarageFinish': ['Unf', 'RFn', 'Fin'],
            'GarageQual': ['Po', 'Fa', 'TA', 'Gd', 'Ex'],
            'GarageCond': ['Po', 'Fa', 'TA', 'Gd', 'Ex'],
            'PavedDrive': ['N', 'P', 'Y'],
            'PoolQC': ['Fa', 'TA', 'Gd', 'Ex'],
            'Fence': ['MnWw', 'GdWo', 'MnPrv', 'GdPrv']
    }

    @staticmethod
    def convert_ordinal_to_code(df, col, categories):
        df[col] = pd.Categorical(df[col], ordered=True, categories=categories).codes
        
    def convert_ordinals_to_code(self):
        self.options['ordinal_code'] = True
        for col, categories in self.ordinal_columns.items():
            self.convert_ordinal_to_code(self.train_df, col, categories)
            self.convert_ordinal_to_code(self.test_df, col, categories)
        return self

    def nominal_columns(self):
        return self.not_num_columns().difference(self.ordinal_columns)

    def convert_numeric_norminals_to_string(self):
        self.options['num_normial_str'] = True
        cols = self.numeric_norminal_columns
        self.train_df[cols] = self.train_df[cols].astype("str")
        self.test_df[cols] = self.test_df[cols].astype("str")
        return self
    
    # Numeric columns having NaN
    def num_cols_having_nan(self):
        df = self.train_df.select_dtypes(exclude=np.object)
        return df.columns[df.isna().any()]
    
    
    # Mark numeric columns those had NaN
    num_cols_mark_missing = []
    def mark_num_cols_having_nan(self):
        self.options['mark_nan'] = True
        num_cols_mark_missing = []
        for col in self.num_cols_having_nan():
            new_col = col + '_was_missing'
            num_cols_mark_missing += [new_col]
            self.train_df[new_col] = self.train_df[col].isnull()
            self.test_df[new_col] = self.test_df[col].isnull()
        self.num_cols_mark_missing = num_cols_mark_missing
        return self
    

    # all the columns with object or category dtype will be onehot encoded
    # , dummy_na=True, drop_first=True
    def convert_to_onehot(self):
        self.options['OHE'] = True
        train_df = pd.get_dummies(self.train_df)
        test_df  = pd.get_dummies(self.test_df)
        # Align one hot columns in train and test dataframe
        self.train_df, self.test_df = train_df.align(test_df, join='inner', axis=1)
        return self
    

    def make_pipeline(self):
        return make_pipeline(*(self.transformers + [self.estimator]))
    
    
    cv_score = None
    def cross_val_score(self):
        # multiple by -1 to make positive MAE score instead of neg value returned as sklearn convention
        pipeline = self.make_pipeline()
        X = self.train_df[self.num_columns()]
        y = np.log(self.train_target) if self.options.get('log_scale') else self.train_target
        neg_mse = cross_val_score(pipeline, X, y,
                                  scoring = 'neg_mean_squared_error', 
                                  cv = 5, error_score='raise'
                                 ).mean()
        self.cv_score = np.sqrt(-neg_mse)
#         print("RMSE = {0}".format(self.cv_score))
        return self
    
    
    df_experiment_log = pd.DataFrame()
    def log_experiment(self, lb_score=None):
        cv_score = self.cv_score
        Experiment.df_experiment_log = pd.concat([Experiment.df_experiment_log,
            pd.DataFrame({ k:[v] for k,v in {
                **{'lb_score': lb_score, 'cv_score': cv_score},
                **self.options, 
                **{ k: self.estimator.get_params()[k] for k in
       ['colsample_bytree', 'learning_rate', 'max_depth', 
        'min_child_weight', 'n_estimators', 'subsample']}
            }.items() })],
            ignore_index=True, sort=False
        )
        Experiment.df_experiment_log.to_csv('experiment_log.csv')
        return Experiment.df_experiment_log

    # fit on all data from the competition
    def fit(self):
        pipeline = self.make_pipeline()
        X = self.train_df[self.num_columns()]
        y = np.log(self.train_target) if self.options.get('log_scale') else self.train_target
        pipeline.fit(X, y)
        predictions = pipeline.predict(self.test_df)
        self.predictions = np.exp(predictions) if self.options.get('log_scale') else predictions
        return self


    def submit_predictions(self, filepath):
        self.submission = pd.DataFrame({'Id': self.test_ids,
                      'SalePrice': self.predictions}
                    )
        self.submission.to_csv(filepath, index=False)
        return self
    
    def __str__(self):
        str = ('options = {0}\n\n'.format(self.options) + 
               'estimators = {0}\n\n'.format(self.estimators) +
               'not_num_columns = {0}\n\n'.format(self.not_num_columns().tolist()) +
#                'num_columns = {0}\n\n'.format(self.num_columns().tolist()) +
               'ordinal_columns = {0}\n\n'.format(self.ordinal_columns.keys()) +
               'num_columns - ordinal_columns = {0}\n\n'.format(self.num_columns().difference(
                   self.ordinal_columns.keys()).tolist()) +
               'num_cols_mark_missing = {0}\n\n'.format(self.num_cols_mark_missing)
              )
        
        return str
    
    @staticmethod
    def outliers(df):
        return pd.DataFrame(
            [
                df.LotFrontage > 300,
#                 df.Utilities == 1,
                df.BsmtFinSF1 > 4000,
                df.TotalBsmtSF > 6000,
                df['1stFlrSF'] > 4000,
                pd.DataFrame([
                    df.GrLivArea > 4000, 
                    np.log(df.SalePrice) < 13]).all(),
                pd.DataFrame([
                    df.GarageArea > 1200, 
                    np.log(df.SalePrice) < 12.5]).all(),
                pd.DataFrame([
                    df.OpenPorchSF > 500, 
                    np.log(df.SalePrice) < 11]).all(),
            ]
        ).any()
    
    def remove_outliers(self, df):
        self.options['outliers'] = True
        self.removed_outliers = df[self.outliers(df)]
        return df[~self.outliers(df)]
        

exp = (Experiment(XGBRegressor(), remove_outliers=True)
.impute()
.normalize()
.log_scale(True)
      )
# print("### Initial:\n{0}".format(exp))
(exp
 .convert_ordinals_to_code()
 .mark_num_cols_having_nan()
)
# print("### After preprocess:\n{0}".format(exp))
(exp
 .convert_numeric_norminals_to_string()
 .convert_to_onehot()
)
# print("### After OHE:\n{0}".format(exp))
exp.cross_val_score()
exp.fit().submit_predictions('sample_submit.csv')
# print(exp.submission.describe())

print(exp.train_df.columns.tolist())
exp.log_experiment()
exp.removed_outliers
!cat experiment_log.csv
(Experiment(XGBRegressor())
#  .impute()
 .normalize()
 .log_scale(True)
#  .convert_ordinals_to_code()
#  .mark_num_cols_having_nan()
#  .convert_numeric_norminals_to_string()
#  .convert_to_onehot()
 .cross_val_score().log_experiment())
(Experiment(XGBRegressor(), remove_outliers=True)
#  .impute()
 .normalize()
 .log_scale(True)
#  .convert_ordinals_to_code()
#  .mark_num_cols_having_nan()
#  .convert_numeric_norminals_to_string()
#  .convert_to_onehot()
 .cross_val_score().log_experiment())
exp = (Experiment(XGBRegressor())
       .impute()
#        .normalize()
 .log_scale(True)
#  .convert_ordinals_to_code()
#  .mark_num_cols_having_nan()
#  .convert_numeric_norminals_to_string()
#  .convert_to_onehot()
 .cross_val_score())
exp.log_experiment()
exp = (Experiment(XGBRegressor())
       .impute()
#        .normalize()
 .log_scale(True)
 .convert_ordinals_to_code()
#  .mark_num_cols_having_nan()
#  .convert_numeric_norminals_to_string()
#  .convert_to_onehot()
 .cross_val_score())
exp.log_experiment()
(Experiment(XGBRegressor())
       .impute()
       .normalize()
 .log_scale(True)
 .convert_ordinals_to_code()
#  .mark_num_cols_having_nan()
#  .convert_numeric_norminals_to_string()
#  .convert_to_onehot()
 .cross_val_score()
 .log_experiment())
(Experiment(XGBRegressor())
       .impute()
       .normalize()
 .log_scale(True)
 .convert_ordinals_to_code()
 .mark_num_cols_having_nan()
#  .convert_numeric_norminals_to_string()
#  .convert_to_onehot()
 .cross_val_score()
 .log_experiment())
(Experiment(XGBRegressor())
       .impute()
       .normalize()
 .log_scale(True)
 .convert_ordinals_to_code()
#  .mark_num_cols_having_nan()
 .convert_numeric_norminals_to_string()
 .convert_to_onehot()
 .cross_val_score()
 .log_experiment())
xgb = (Experiment(XGBRegressor())
       .impute()
       .normalize()
#  .log_scale(True)
 .convert_ordinals_to_code()
 .mark_num_cols_having_nan()
 .convert_numeric_norminals_to_string()
 .convert_to_onehot()
 .fit())

xgb_log = (Experiment(XGBRegressor())
       .impute()
       .normalize()
 .log_scale(True)
 .convert_ordinals_to_code()
 .mark_num_cols_having_nan()
 .convert_numeric_norminals_to_string()
 .convert_to_onehot()
 .fit())

print(pd.Series(xgb.predictions).describe())
print(pd.Series(xgb_log.predictions).describe())

diff = pd.Series(xgb.predictions - xgb_log.predictions)
print(diff.describe())
diff_sigma = diff.std()/(1+abs(diff.mean()))
print(diff_sigma)
assert(diff_sigma < 5)

xgb_log.submit_predictions('xgb_default.csv').submission.describe()
params = {'colsample_bytree': 0.7, 'learning_rate': 0.07, 
          'max_depth': 4, 'min_child_weight': 2, 
          'n_estimators': 190, 'nthread': 4, 
          'objective': 'reg:linear', 'subsample': 0.8}
exp = (Experiment(XGBRegressor(params=params)).impute().normalize()
 .log_scale(True)
 .convert_ordinals_to_code()
 .mark_num_cols_having_nan()
 .convert_numeric_norminals_to_string()
 .convert_to_onehot()
 .cross_val_score()
 .fit()
 .submit_predictions('xgb_13097.log.csv'))

print(exp.submission.describe())
exp.log_experiment(lb_score= .13097)
params = {'colsample_bytree': 0.7, 'learning_rate': 0.07, 
          'max_depth': 4, 'min_child_weight': 2, 
          'n_estimators': 190, 'nthread': 4, 
          'objective': 'reg:linear', 'subsample': 0.8}
exp = (Experiment(XGBRegressor(params=params), remove_outliers=True).impute().normalize()
 .log_scale(True)
 .convert_ordinals_to_code()
 .mark_num_cols_having_nan()
 .convert_numeric_norminals_to_string()
 .convert_to_onehot()
 .cross_val_score()
 .fit()
 .submit_predictions('xgb_13097.outliers.csv'))

print(exp.submission.describe())
exp.log_experiment()
xgb = XGBRegressor(
        colsample_bytree=0.8, learning_rate=0.05, max_depth=3, min_child_weight=2, 
        n_estimators=1000, nthread=4, objective='reg:linear', subsample=0.8)

exp = (Experiment(xgb).impute().normalize()
 .log_scale(True)
 .convert_ordinals_to_code()
 .mark_num_cols_having_nan()
 .convert_numeric_norminals_to_string()
 .convert_to_onehot()
 .cross_val_score()
 .fit()
 .submit_predictions('xgb_13045.log.csv'))
print(exp.submission.describe())
exp.log_experiment(lb_score= .13045)
xgb = XGBRegressor(
        colsample_bytree=0.8, learning_rate=0.05, max_depth=3, min_child_weight=2, 
        n_estimators=1000, nthread=4, objective='reg:linear', subsample=0.8)

exp = (Experiment(xgb, remove_outliers=True).impute().normalize()
 .log_scale(True)
 .convert_ordinals_to_code()
 .mark_num_cols_having_nan()
 .convert_numeric_norminals_to_string()
 .convert_to_onehot()
 .cross_val_score()
 .fit()
 .submit_predictions('xgb_13045.outliers.csv'))
print(exp.submission.describe())
exp.log_experiment()
xgb = XGBRegressor(
        colsample_bytree=0.8, learning_rate=0.05, max_depth=3, min_child_weight=2, 
        n_estimators=1000, nthread=4, objective='reg:linear', subsample=0.8
    )
exp = (Experiment(xgb).impute().normalize()
 .log_scale(True)
 .convert_ordinals_to_code()
 .mark_num_cols_having_nan()
 .convert_numeric_norminals_to_string()
 .convert_to_onehot()
 .cross_val_score()
 .fit()
 .submit_predictions('xgb.13020.log.csv'))
print(exp.submission.describe())
exp.log_experiment(lb_score= .13020)
xgb = XGBRegressor(
        colsample_bytree=0.8, learning_rate=0.05, max_depth=3, min_child_weight=2, 
        n_estimators=1000, nthread=4, objective='reg:linear', subsample=0.8
    )
exp = (Experiment(xgb, remove_outliers=True).impute().normalize()
 .log_scale(True)
 .convert_ordinals_to_code()
 .mark_num_cols_having_nan()
 .convert_numeric_norminals_to_string()
 .convert_to_onehot()
 .cross_val_score()
 .fit()
 .submit_predictions('xgb.13020.outliers.csv'))
print(exp.submission.describe())
exp.log_experiment()
params = {'colsample_bytree': 0.7, 'learning_rate': 0.07, 'max_depth': 4, 'min_child_weight': 2, 
          'n_estimators': 500, 'nthread': 4, 'objective': 'reg:linear', 'subsample': 0.8}
xgb = XGBRegressor()
xgb.set_params(**params)
exp = (Experiment(xgb).impute().normalize()
 .log_scale(True)
 .convert_ordinals_to_code()
 .mark_num_cols_having_nan()
 .convert_numeric_norminals_to_string()
 .convert_to_onehot()
 .cross_val_score()
 .fit()
 .submit_predictions('xgb.13020.v2.log.csv'))
print(exp.submission.describe())
exp.log_experiment(lb_score= .13020)
params = {'colsample_bytree': 0.7, 'learning_rate': 0.07, 'max_depth': 4, 'min_child_weight': 2, 
          'n_estimators': 500, 'nthread': 4, 'objective': 'reg:linear', 'subsample': 0.8}
xgb = XGBRegressor()
xgb.set_params(**params)
exp = (Experiment(xgb, remove_outliers=True).impute().normalize()
 .log_scale(True)
 .convert_ordinals_to_code()
 .mark_num_cols_having_nan()
 .convert_numeric_norminals_to_string()
 .convert_to_onehot()
 .cross_val_score()
 .fit()
 .submit_predictions('xgb.13020.v2.outliers.csv'))
print(exp.submission.describe())
exp.log_experiment()
params = {'colsample_bytree': 0.8, 'learning_rate': 0.07, 'max_depth': 2, 'min_child_weight': 2, 
          'n_estimators': 1000, 'nthread': 4, 'objective': 'reg:linear', 'subsample': 0.8}
xgb = XGBRegressor()
xgb.set_params(**params)
exp = (Experiment(xgb, remove_outliers=True).impute().normalize()
 .log_scale(True)
 .convert_ordinals_to_code()
 .mark_num_cols_having_nan()
 .convert_numeric_norminals_to_string()
 .convert_to_onehot()
 .cross_val_score()
 .fit()
 .submit_predictions('xgb.12714.outliers.csv'))
print(exp.submission.describe())
exp.log_experiment(lb_score=0.12714)
from sklearn.model_selection import GridSearchCV
# Can run 2.5 fits in 1 sec, or 153 fits in 1 min.
# XGBoost early_stopping with GridSearchCV https://github.com/dmlc/xgboost/issues/412

param_grid_light = {'nthread':[4],
#                       'objective':['reg:linear'],
                      'learning_rate': [.05,.07], #so called `eta` value
                      'max_depth': [2,4],
                      'min_child_weight': [1,2],
                      'subsample': [.8],
                      'colsample_bytree': [.7,.8],
                      'n_estimators': [100,200,500,600,900,1000],
                   }

param_grid_rigid = {'nthread':[4],
              'objective':['reg:linear'],
              'learning_rate': np.array(range(5,8)) * .01, #so called `eta` value
              'max_depth': range(2,5),
              'min_child_weight': [1,2],
              'subsample': np.array(range(8,11)) * .1,
              'colsample_bytree': np.array(range(7,10)) * .1,
              'n_estimators': range(190,240,5)}
FAST_TEST = True
gs_estimator = GridSearchCV(XGBRegressor(),
                            param_grid_light if FAST_TEST else param_grid_rigid,
                            cv = 5, scoring='neg_mean_squared_error',
                            refit=True,
                            verbose=1)
gs_experiment = (
    Experiment(gs_estimator).impute().normalize()
    .log_scale(True)
    .convert_ordinals_to_code()
    .mark_num_cols_having_nan()
    .convert_numeric_norminals_to_string()
    .convert_to_onehot()
    .fit()
    .submit_predictions('xgb_gs_log.csv'))
print(gs_experiment.submission.describe())
print(np.sqrt(-gs_estimator.best_score_), gs_estimator.best_params_, gs_estimator.best_estimator_, gs_estimator)
xgb = XGBRegressor()
xgb.set_params(**gs_estimator.best_params_)

(Experiment(xgb).impute().normalize()
 .log_scale(True)
 .convert_ordinals_to_code()
 .mark_num_cols_having_nan()
 .convert_numeric_norminals_to_string()
 .convert_to_onehot()
 .cross_val_score().log_experiment())
FAST_TEST = True
gs_estimator = GridSearchCV(XGBRegressor(),
                            param_grid_light if FAST_TEST else param_grid_rigid,
                            cv = 5, scoring='neg_mean_squared_error',
                            refit=True,
                            verbose=1)
gs_experiment = (
    Experiment(gs_estimator, remove_outliers=True).impute().normalize()
    .log_scale(True)
    .convert_ordinals_to_code()
    .mark_num_cols_having_nan()
    .convert_numeric_norminals_to_string()
    .convert_to_onehot()
    .fit()
    .submit_predictions('xgb_gs_outliers.csv'))
print(gs_experiment.submission.describe())
print(np.sqrt(-gs_estimator.best_score_), gs_estimator.best_params_, gs_estimator.best_estimator_, gs_estimator)
xgb = XGBRegressor()
xgb.set_params(**gs_estimator.best_params_)

(Experiment(xgb, remove_outliers=True).impute().normalize()
 .log_scale(True)
 .convert_ordinals_to_code()
 .mark_num_cols_having_nan()
 .convert_numeric_norminals_to_string()
 .convert_to_onehot()
 .cross_val_score().log_experiment())
Experiment.df_experiment_log.to_csv('experiments.csv', index=False)
Experiment.df_experiment_log