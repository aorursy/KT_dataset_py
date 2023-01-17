import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.svm import SVC

from sklearn.model_selection import GridSearchCV

from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.impute import SimpleImputer

from sklearn_pandas import DataFrameMapper

from sklearn.metrics import classification_report

from sklearn.ensemble import VotingClassifier, RandomForestClassifier



import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
FEATURES = ["MEMBERSHIP_TERM_YEARS", "ANNUAL_FEES", "MEMBER_MARITAL_STATUS", "MEMBER_GENDER", "MEMBER_ANNUAL_INCOME", "MEMBER_OCCUPATION_CD", "MEMBERSHIP_PACKAGE", "MEMBER_AGE_AT_ISSUE", "ADDITIONAL_MEMBERS", "PAYMENT_MODE"]

LABEL = "MEMBERSHIP_STATUS"
# Kaggle の Club Data Set を利用しています

def load_dataset(path_train="/kaggle/input/club-data-set/club_churn_train.csv",

                 path_test="/kaggle/input/club-data-set/club_churn_test.csv",

                 path_real_y="/kaggle/input/club-data-set/real_y_test_2.csv"):

    """Load training and test data with target label."""

    

    train = pd.read_csv(path_train)

    test = pd.read_csv(path_test)

    real_y = pd.read_csv(path_real_y)

    

    real_y = real_y.loc[:, ["Unnamed: 0.1", "MEMBERSHIP_STATUS"]]

    test = pd.merge(test, real_y, left_on="Unnamed: 0", right_on="Unnamed: 0.1")

    

    train.AGENT_CODE = train.AGENT_CODE.astype("object")

    test.AGENT_CODE = test.AGENT_CODE.astype("object")

    

    train.MEMBER_OCCUPATION_CD.fillna(-1, inplace=True)

    test.MEMBER_OCCUPATION_CD.fillna(-1, inplace=True)

    

    train.MEMBER_OCCUPATION_CD = train.MEMBER_OCCUPATION_CD.apply(lambda x: int(x))

    test.MEMBER_OCCUPATION_CD = test.MEMBER_OCCUPATION_CD.apply(lambda x: int(x))

        

    train.START_DATE = pd.to_datetime(train.START_DATE, format="%Y%m%d")

    test.START_DATE = pd.to_datetime(test.START_DATE, format="%Y%m%d")

    

    f = lambda x: np.nan if np.isnan(x) else pd.to_datetime(int(x), format="%Y%m%d")

    train.END_DATE = train.END_DATE.apply(f)

    test.END_DATE = test.END_DATE.apply(f)

    

    train.drop(columns=["Unnamed: 0"], inplace=True)

    test.drop(columns=["Unnamed: 0", "Unnamed: 0.1"], inplace=True)

    

    return train, test

    
train, test = load_dataset()
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)

sns.countplot(x=LABEL, data=train, order=["INFORCE", "CANCELLED"])

plt.title("Training data")

sns.despine()

plt.subplot(1, 2, 2)

sns.countplot(x=LABEL, data=test, order=["INFORCE", "CANCELLED"])

plt.title("Test data")

plt.xlabel("")

sns.despine()
x_train = train[FEATURES]

y_train = train[LABEL].apply(lambda x: 0 if x == "INFORCE" else 1)

x_test = test[FEATURES]

y_test = test[LABEL].apply(lambda x: 0 if x == "INFORCE" else 1)
# 前処理

# 勢い余って数値を StandardScaler でスケーリングしているけど実際は不要（決定木のため）

# カテゴリ変数もダミー変数にするより整数値への置換の方が一般的（間違えてダミー変数にした）

# そこそこ性能でたのでそのままにしています

mapper = DataFrameMapper([

    (['MEMBERSHIP_TERM_YEARS'], StandardScaler()), 

    (['ANNUAL_FEES'], [SimpleImputer(strategy="most_frequent"),  StandardScaler()]),

    (['MEMBER_MARITAL_STATUS'], [SimpleImputer(strategy="constant", fill_value="NA"), OneHotEncoder()]),

    (['MEMBER_GENDER'], [SimpleImputer(strategy="constant", fill_value="NA"), OneHotEncoder()]),

    (['MEMBER_ANNUAL_INCOME'], [SimpleImputer(strategy="most_frequent"),  StandardScaler()]),

    (['MEMBER_OCCUPATION_CD'], [OneHotEncoder()]),

    (['MEMBERSHIP_PACKAGE'], [OneHotEncoder()]),

    (['MEMBER_AGE_AT_ISSUE'], StandardScaler()),

    (['ADDITIONAL_MEMBERS'], StandardScaler()),

    (['PAYMENT_MODE'], [OneHotEncoder()])], input_df=True, df_out=True, default=None

)
x_train_transformed = mapper.fit_transform(x_train)

x_test_transformed = mapper.transform(x_test)
x_train_transformed
x_test_transformed
%%time



clf1 = RandomForestClassifier(random_state=42, n_jobs=-1, n_estimators=100)

clf1.fit(x_train_transformed, y_train)

print(classification_report(y_test, clf1.predict(x_test_transformed)))
%%time

param_grid = {"max_features": ["sqrt", "log2"], 

              "class_weight": [None, "balanced"],

              "min_samples_leaf": [0.005, 0.01, 0.02, 0.03]

             }

clf2 = GridSearchCV(

    estimator=RandomForestClassifier(random_state=42, n_jobs=-1, n_estimators=100),

    param_grid=param_grid,

    scoring="accuracy",

    n_jobs=-1

)

clf2.fit(x_train_transformed, y_train)

print(clf2.best_estimator_)

print(classification_report(y_test, clf2.predict(x_test_transformed)))
def train_weak_leaners(x_train, y_train, classifier,

                       grid_param: dict or None = None,

                       sample_rate: dict = {0: 3/7, 1: 1.0},

                       random_seeds: list = [0, 7, 42, 8, 2020, 777, 5],

                       cv=5, n_jobs=-1, scoring="accuracy", verbose=0):

    """Train weak leaners with iterative down sampling."""

    

    label_column = "__LABEL__"  # temporary column

    

    df = pd.DataFrame(x_train)

    df[label_column] = y_train

    fitted_classifiers = {}

    for i, seed in enumerate(random_seeds):

        # down-sampling with fixed random seed and sampling rate by each label

        df_sample = pd.DataFrame()

        for label, rate in sample_rate.items():

            df_sample = pd.concat([

                df_sample, 

                df[df[label_column] == label].sample(frac=rate, random_state=seed)])

        y_train_sample = df_sample[label_column]

        x_train_sample = df_sample.drop(columns=[label_column])

        clf = GridSearchCV(estimator=classifier, 

                           param_grid=grid_param, 

                           n_jobs=n_jobs, 

                           scoring=scoring,

                           verbose=verbose)

        clf.fit(x_train_sample, y_train_sample)

        fitted_classifiers[f"seed={seed}"] = clf

    return fitted_classifiers   
%%time



clfs = train_weak_leaners(x_train_transformed.copy(),

                          y_train.copy(),

                          RandomForestClassifier(random_state=42,

                                                 n_jobs=-1,

                                                 n_estimators=100),

                          grid_param=param_grid)

%%time

estimators = []

for name, estimator in clfs.items():

    estimators.append((name, estimator.best_estimator_))

print(estimators)

clf3 = VotingClassifier(estimators=estimators, voting="hard", n_jobs=-1)

clf3.fit(x_train_transformed, y_train)

print(classification_report(y_test, clf3.predict(x_test_transformed.copy())))
%%time

estimators = []

for name, estimator in clfs.items():

    estimators.append((name, estimator.best_estimator_))

print(estimators)

clf4 = VotingClassifier(estimators=estimators, voting="soft", n_jobs=-1)

clf4.fit(x_train_transformed, y_train)

print(classification_report(y_test, clf4.predict(x_test_transformed.copy())))