import pandas as pd
import missingno as msno
# Some constants
# Otherwise, pandas will try to interpret this column as an integer 
# (which is wrong according to the competition's guidelines).
VISITOR_ID_COL = "fullVisitorId"
DTYPES = {VISITOR_ID_COL: 'str'}
TARGET_COL = "transactionRevenue"
TRAIN_DATA_PATH = "../input/train.csv"
train_df = pd.read_csv(TRAIN_DATA_PATH, dtype=DTYPES)
train_df.sample(2).T
msno.matrix(train_df)
RAW_TARGET_COL = "totals"
raw_target_s = train_df[RAW_TARGET_COL]
for index, raw_target_row in raw_target_s.sample(30).iteritems():
    print(eval(raw_target_row))
records = []
for index, raw_target_row in raw_target_s.iteritems():
    parsed_target_row = eval(raw_target_row)
    records.append(parsed_target_row)
parsed_target_df = pd.DataFrame(records)
# Don't forget the visitor id!
parsed_target_df[VISITOR_ID_COL] = train_df[VISITOR_ID_COL]
parsed_target_df.sample(3).T
msno.matrix(parsed_target_df)
def percentage_of_missing(df, col):
    return 100 * df[col].isnull().sum() / df.shape[0]

missing_target_percent = percentage_of_missing(parsed_target_df, 
                                              TARGET_COL)
"The target column contains {}% missing data!".format(missing_target_percent.round(2))
target_df = (parsed_target_df.loc[:, [TARGET_COL, VISITOR_ID_COL]]
                            .assign(**{TARGET_COL: lambda df: df[TARGET_COL].fillna(0.0)
                                                                            .astype(int)}))
target_df.sample(5)
import seaborn as sns
import numpy as np
import matplotlib.pylab as plt


fig, ax = plt.subplots(1, 1, figsize=(12, 8))
# Since most of the transactions are 0$, I will remove these when plotting
# the distribution.
sns.distplot(np.log(target_df.loc[lambda df: df[TARGET_COL] >0, 
                                  TARGET_COL]), ax=ax)
ax.set_xlabel("Log of transaction revenue ($)")
# The same thing as above but this time aggregated using the 
# visitor unique id.
fig, ax = plt.subplots(1, 1, figsize=(12, 8))
# Since most of the transactions are 0$, I will remove these when plotting
# the distribution.

def _log_sum_agg(g):
    """ Take the natural logarithm of the aggregated sum
    (+1 to avoid -inf for a 0 sum).
    """
    return np.log(g.sum() + 1)

grouped_target_a = (target_df.groupby(VISITOR_ID_COL)
                             .agg({TARGET_COL: _log_sum_agg})
                             .values)
sns.distplot(grouped_target_a[grouped_target_a > np.log(1)], ax=ax)
ax.set_xlabel("Log of sum of transaction revenue ($)")
DATE_COL = "date"
TMS_GMT_COL = "tms_gmt"
# Here, I parse the DATE_COL to extract year, month, and day information 
# (using there positions). Then, I build the TMS_GMT column (using pandas' 
# to_datetime function) and extract additional calendar features: 
# day of week, week of year, and day of year. 
# Notice that I drop DATE_COL and TMS_GMT columns since these
# aren't numerical columns.
date_df = (train_df[[DATE_COL]].assign(year=lambda df: df[DATE_COL].astype(str)
                                                                   .str[0:4]
                                                                   .astype(int),
                                       month=lambda df: df[DATE_COL].astype(str)
                                                                    .str[4:6]
                                                                    .astype(int),
                                       day=lambda df: df[DATE_COL].astype(str)
                                                                  .str[6:8]
                                                                  .astype(int))
                               .drop(DATE_COL, axis=1)
                               .assign(tms_gmt=lambda df: pd.to_datetime(df))
                               .assign(dow=lambda df: df[TMS_GMT_COL].dt.dayofweek,
                                       woy=lambda df: df[TMS_GMT_COL].dt.week,
                                       doy=lambda df: df[TMS_GMT_COL].dt.day)
                               .drop(TMS_GMT_COL, axis=1))
date_df.sample(5)
records = []
GEO_COL = "geoNetwork"
for index, row in train_df[GEO_COL].iteritems():
    parsed_row = eval(row)
    records.append(parsed_row)

geo_df = pd.DataFrame(records)
geo_df.sample(2).T
GEO_COLS_TO_KEEP = ["country", "continent"]
engineered_train_df = (geo_df.loc[:, GEO_COLS_TO_KEEP]
                             .pipe(pd.get_dummies)
                             .pipe(pd.merge, date_df, 
                                   left_index=True,
                                   right_index=True)
                             .pipe(pd.merge, 
                                   train_df[[VISITOR_ID_COL]],
                                   left_index=True,
                                   right_index=True))
engineered_train_df.sample(2).T
from sklearn.linear_model import Lasso
from sklearn.model_selection import KFold
# For reproducibility
SEED = 314
CV = 5
# Resources are limited! 
N_SAMPLES = 10000
kf = KFold(CV, random_state=SEED)


benchmark = Lasso(random_state=SEED)

df = engineered_train_df.sample(N_SAMPLES).drop(VISITOR_ID_COL, axis=1)

# TODO: Do some cleaning and refactoring of the CV computation. 
# Also check the grouping step...
# LASSO warnings are annoying. :)
import warnings
warnings.simplefilter("ignore")


cv_rmse = []
for train_index, test_index in kf.split(df):
    train_features_df = df.iloc[train_index, :]
    test_features_df = df.iloc[test_index, :]
    train_target_s = target_df.loc[train_index, TARGET_COL]
    test_target_df = target_df.iloc[test_index, :].reset_index(drop=True)
    benchmark.fit(train_features_df, train_target_s)
    test_target_df.loc[:, "predictions"] = benchmark.predict(test_features_df)
    grouped_df  = (test_target_df.groupby(VISITOR_ID_COL)
                                 .agg({"predictions": _log_sum_agg, 
                                       TARGET_COL: _log_sum_agg})
                                 .reset_index())
    rmse = ((grouped_df["predictions"] - grouped_df[TARGET_COL]) ** 2).mean() ** 0.5
    cv_rmse.append(rmse)

cv_rmse = np.array(cv_rmse)
cv_rmse
"The mean CV RMSE for the benchmark is: {}".format(cv_rmse.mean())