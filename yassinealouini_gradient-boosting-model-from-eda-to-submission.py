# The usual imports :)
import pandas as pd
import numpy as np
import missingno as msno
import pandas_profiling as pdp
import lightgbm as lgb
import warnings
import datetime
# No one likes warnings :p
warnings.simplefilter('ignore')
# Some constants
# This is the augmented train produced by the following kernel:
# https://www.kaggle.com/yassinealouini/start-here-features-engineering-pipeline
TRAIN_DATA_PATH = "../input/augmented_train.csv"
TEST_DATA_PATH = "../input/augmented_test.csv"
DTYPE = {"fullVisitorId": str, "sessionId":str}
# The "campaignCode" column is only available in the train dataset.
# The "targetingCriteria" isn't useful at all (only NaN or {} values). 
TO_DROP_COLS = ["campaignCode", "targetingCriteria"]
ID_COLS = ["fullVisitorId", "sessionId", "visitId"]
# Will only need this column for identifying the visitors. 
ID_COL = "fullVisitorId"
TARGET_COL = "transactionRevenue"
ID_COLS_TO_DROP = list(set(ID_COLS) - set([ID_COL]))
TODAY = datetime.date.today()
OUTPUT_DATA_PATH = f"submission_{TODAY}.csv"
train_df = (pd.read_csv(TRAIN_DATA_PATH, low_memory=False, dtype=DTYPE)
              .drop(TO_DROP_COLS, axis=1))
# Will only use this to prepare the submission file at the end. 
test_df = (pd.read_csv(TEST_DATA_PATH, low_memory=False, dtype=DTYPE)
             .drop(TO_DROP_COLS, axis=1, errors="ignore"))
train_test_cols_diff = list(set(train_df.columns) - set(test_df.columns))
print(f"The train dataset shape: {train_df.shape}")
print(f"The train dataset shape: {test_df.shape}")
print("The only difference between test and train dasets should be the target.")
print(f"It is: {train_test_cols_diff}. The target column is: {TARGET_COL}.")
train_df.sample(2).T
msno.matrix(train_df)
# I fill missing values with -1 for now in order to get the profile report. This will change in the 
# upcoming sections.
pdp.ProfileReport(train_df.fillna(-1))
for col in ID_COLS:
    print(train_df[col].sample(5))
    print(24 * "*")
cols_diff = (train_df["sessionId"] != 
             (train_df["fullVisitorId"] + "_" + train_df["visitId"].astype(str))).sum()
print(f"The are {cols_diff} rows different between 'sessionId' and the concatenation of 'fullVisitorId' and 'visitId' columns")

for col in ID_COLS:
    print(f"The {col} column has {train_df[col].nunique()} unique values")
# The maximum number of visits for a user: 
train_df.groupby("fullVisitorId")["visitId"].count().max()
def get_cat_cols(df):
    # I remove the ID_COLS
    all_cat_cols = df.select_dtypes(include=["object"]).columns
    return list(set(all_cat_cols) - set(ID_COLS))
CAT_COLS = get_cat_cols(train_df)
print(f"There are {len(CAT_COLS)} categorical columns. \nThese are: {'|'.join(CAT_COLS)}")
msno.matrix(train_df.loc[:, CAT_COLS])
# TODO: What happens when a new category appears during test and wasn't 
# available during train?

# The training median number of uniques for categorical columns is 33.5
CAT_THRESHOLD = 30

class CategoricalTransformer(object):
    


    def __init__(self, cat_threshold=CAT_THRESHOLD):
        self.factors_mapping = None
        self.cat_threshold = cat_threshold
    
    def fit_transform(self, s):
        """ Fit and transform a categorical Series. 
        This will also transform the test categorical series.
        """
        cat_total = s.nunique(dropna=False)
        most_common = s.value_counts(dropna=False).index[0]
        print(f"Categorical transformation for: {s.name}")
        if cat_total > self.cat_threshold:
            print(f"Most common vs rest")
            # Most common category vs other categories
            return pd.Series((s == most_common).astype(int), 
                             name=f"is_{s.name}_{str(most_common).lower()}")
        elif cat_total > 2:
            print(f"Label encoding")
            # Factorization + mapping for test set
            if self.factors_mapping is None:
                labels, uniques = pd.factorize(s)
                self.factors_mapping = {v:k for k, v in enumerate(uniques)}
                return pd.Series(labels, name=col)
            else:
                return s.replace(self.factors_mapping)
        else:
            # Binary
            print(f"One-hot encoding")
            return (s == most_common).astype(int)        
    
train_cat_data = []
test_cat_data = []
for col in CAT_COLS:
    cat_tr = CategoricalTransformer()
    train_cat_s = cat_tr.fit_transform(train_df[col])
    train_cat_data.append(train_cat_s)
    test_cat_s = cat_tr.fit_transform(test_df[col])
    test_cat_data.append(test_cat_s)
def list_of_s_to_df(data):
    return pd.concat(data, axis=1, keys=[s.name for s in data])

train_cat_df = list_of_s_to_df(train_cat_data)
test_cat_df = list_of_s_to_df(test_cat_data)
train_cat_df.sample(2).T
test_cat_df.sample(2).T
processed_train_df = pd.concat([train_df.drop(CAT_COLS + ID_COLS_TO_DROP, axis=1), 
                                train_cat_df], axis=1)
processed_test_df = pd.concat([test_df.drop(CAT_COLS + ID_COLS_TO_DROP, axis=1), 
                               test_cat_df], axis=1)
processed_train_df['bounces'].value_counts(dropna=False)
processed_train_df['newVisits'].value_counts(dropna=False)
processed_train_df['page'].value_counts(dropna=False)
FILL_MISSINGS_COLS = ["page", "bounces", "newVisits"]
processed_train_df[FILL_MISSINGS_COLS] = processed_train_df[FILL_MISSINGS_COLS].fillna(0.0)
processed_test_df[FILL_MISSINGS_COLS] = processed_test_df[FILL_MISSINGS_COLS].fillna(0.0)
processed_train_df.sample(2).T
msno.matrix(processed_train_df)
pdp.ProfileReport(processed_train_df)
SEED = 314
lgb_reg = lgb.LGBMRegressor(random_state=SEED)
lgb_reg.fit(processed_train_df.drop([TARGET_COL, ID_COL], axis=1), 
            processed_train_df[TARGET_COL])
from lightgbm.plotting import plot_importance
plot_importance(lgb_reg)
plot_importance(lgb_reg, importance_type="gain")
# As mentioned in the previous sections, some categories haven't been seen during
# training. For now, I replace them with -1. 
# TODO: Improve this process.
NOT_SEEN_OPERATING_SYSTEM_CAT = ["Tizen", "Playstation Vita", "OS/2", "SymbianOS"]
processed_test_df["operatingSystem"] = processed_test_df["operatingSystem"].replace(NOT_SEEN_OPERATING_SYSTEM_CAT, -1)
NOT_SEEN_SLOT_CAT = ["Google Display Network"]
processed_test_df["slot"] = processed_test_df["slot"].replace(NOT_SEEN_SLOT_CAT, -1)
NOT_SEEN_AD_CAT = ["Content"]
processed_test_df["adNetworkType"] = processed_test_df["adNetworkType"].replace(NOT_SEEN_AD_CAT, -1)
processed_test_df = processed_test_df.fillna(-1)
test_predictions_a = lgb_reg.predict(processed_test_df.drop(ID_COL, axis=1))
submission_df = pd.DataFrame({TARGET_COL: test_predictions_a, 
                              ID_COL: processed_test_df[ID_COL]})
neg_predictions_total  = (test_predictions_a < 0).sum()
print(f"There are {neg_predictions_total} predicted values < 0.")
submission_df.loc[lambda df: df[TARGET_COL] < 0, TARGET_COL] = 0.0
TRANSFORMED_TARGET_COL = "PredictedLogRevenue"
processed_submission_df = (submission_df.groupby(ID_COL)
                                        .agg(lambda g: np.log1p(g.sum()))
                                        .reset_index()
                                        .rename(columns={TARGET_COL: TRANSFORMED_TARGET_COL}))
processed_submission_df.sample(5)
# A sanity check for the number of ID_COL unique values and number of rows 
# in the final test DataFrame.
assert processed_submission_df.shape[0] == processed_test_df[ID_COL].nunique()
processed_submission_df.to_csv(OUTPUT_DATA_PATH, index=False)