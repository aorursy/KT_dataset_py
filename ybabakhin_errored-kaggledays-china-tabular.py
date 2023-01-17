import warnings

warnings.simplefilter(action="ignore")



import os

import random

import lightgbm as lgb

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import seaborn as sns

from sklearn import preprocessing

from sklearn.metrics import f1_score

from sklearn.model_selection import StratifiedKFold



# Seed Everything

seed = 13

random.seed(seed)

os.environ["PYTHONHASHSEED"] = str(seed)

np.random.seed(seed)
from sklearn.model_selection import KFold





class TargetEncoder:

    def __init__(self, target, alpha=5):

        self.target = target

        self.alpha = alpha



    def fit_transform(self, train, categorical):

        self.train = train

        self.categorical = categorical



        # Create 5-fold cross-validation

        kf = KFold(n_splits=5, random_state=1, shuffle=True)

        train_feature = np.zeros(len(train))



        # For each folds split

        for train_index, test_index in kf.split(train):

            cv_train, cv_test = train.iloc[train_index], train.iloc[test_index]



            # Calculate out-of-fold statistics and apply to cv_test

            cv_test_feature = self._test_mean_target_encoding(cv_train, cv_test)



            # Save new feature for this particular fold

            train_feature[test_index] = cv_test_feature

        return train_feature



    def transform(self, test):

        

        # Get test target-encoded feature

        test_feature = self._test_mean_target_encoding(self.train, test)



        return test_feature



    def _test_mean_target_encoding(self, train, test):

        # Calculate global mean on the train data

        global_mean = train[self.target].mean()



        # Group by the categorical feature and calculate its properties

        train_groups = train.groupby(self.categorical)

        category_sum = train_groups[self.target].sum()

        category_size = train_groups.size()



        # Calculate smoothed mean target statistics

        train_statistics = (category_sum + global_mean * self.alpha) / (category_size + self.alpha)



        # Apply statistics to the test data and fill new categories

        test_feature = test[self.categorical].map(train_statistics).fillna(global_mean)

        return test_feature.values





from sklearn.metrics import f1_score



def evaluate_macroF1_lgb(truth, predictions):  

    # Follows the discussion in https://github.com/Microsoft/LightGBM/issues/1483

    pred_labels = predictions.reshape(len(np.unique(truth)),-1).argmax(axis=0)

    f1 = f1_score(truth, pred_labels, average="macro")

    

    return ("macroF1", f1, True)
def read_data(train_path, test_path):

    train = pd.read_csv(train_path)

    test = pd.read_csv(test_path)

    

    return train, test
def preprocess_data(train, test):

    

    # Concatenate train and test data together

    data = pd.concat([train, test], sort=False)



    # Drop duplicate and useless features

    features_to_drop = ["area2", "tamhog", "hhsize", "agesq"]

    features_to_drop += [x for x in data.columns if "SQB" in x]

    data.drop(features_to_drop, axis=1, inplace=True)



    # Transform original One-Hot-encoded features to a single column

    for ohe in [

        "pared",

        "piso",

        "techo",

        "abastagua",

        "sanitario",

        "energcocinar",

        "elimbasu",

        "epared",

        "etecho",

        "eviv",

        "lugar",

        "tipovivi",

        "electricity",

    ]:

        if ohe != "electricity":

            ohe_cols = [x for x in train.columns if x.startswith(ohe)]

        else:

            ohe_cols = ["public", "planpri", "noelec", "coopele"]



        data[ohe] = np.where(

            data[ohe_cols].sum(axis=1) == 0, "NEW_CAT", data[ohe_cols].idxmax(axis=1)

        )

        data.drop(ohe_cols, axis=1, inplace=True)



    # Fill in the missing data

    data.fillna(-999, inplace=True)



    train = data[: len(train)]

    test = data[-len(test) :]



    return train, test
def generate_features(train, test):

    data = pd.concat([train, test], sort=False)



    # Some feature engineering from: https://www.kaggle.com/gaxxxx/exploratory-data-analysis-lightgbm

    data["adult"] = data["hogar_adul"] - data["hogar_mayor"]

    data["dependency_count"] = data["hogar_nin"] + data["hogar_mayor"]

    data["dependency"] = np.where(data["adult"] == 0, 1, data["dependency_count"] / data["adult"])

    data["child_percent"] = data["hogar_nin"] / data["hogar_total"]

    data["elder_percent"] = data["hogar_mayor"] / data["hogar_total"]

    data["adult_percent"] = data["hogar_adul"] / data["hogar_total"]



    data["rent_per_bedroom"] = data["v2a1"] / data["bedrooms"]

    data["male_per_bedroom"] = data["r4h3"] / data["bedrooms"]

    data["female_per_bedroom"] = data["r4m3"] / data["bedrooms"]

    data["bedrooms_per_person_household"] = data["hogar_total"] / data["bedrooms"]



    data["escolari_age"] = data["escolari"] / data["age"]



    # Groupping features by a household (ID is idhogar)

    aggr_mean_list = ["rez_esc", "dis", "male", "female"]

    aggr_mean_list += [f"estadocivil{x}" for x in range(1, 8)]

    aggr_mean_list += [f"parentesco{x}" for x in range(2, 13)]

    aggr_mean_list += [f"instlevel{x}" for x in range(1, 10)]



    other_list = ["escolari", "age", "escolari_age"]



    for item in aggr_mean_list:

        data[item + "_mean"] = data.groupby("idhogar")[item].transform("mean")



    for item in other_list:

        for function in ["mean", "std", "min", "max", "sum"]:

            data[item + "_" + function] = (

                data.groupby("idhogar")[item].transform(function).fillna(0)

            )



    train = data[: len(train)]

    test = data[-len(test) :]



    return train, test
# Read the data

train, test = read_data(

    train_path="../input/costa-rican-household-poverty-prediction/train.csv",

    test_path="../input/costa-rican-household-poverty-prediction/test.csv",

)
train.head()
# Preprocess the data

train, test = preprocess_data(train, test)
train.head()
# Generate some features

train, test = generate_features(train, test)
# Keep only the heads of the households

train = train[train["parentesco1"] == 1]



# Transform target variables to the labels

target_encoder = preprocessing.LabelEncoder()

y = target_encoder.fit_transform(train["Target"])



# Drop all the ID variables

X = train.drop(["Id", "idhogar", "parentesco1"], axis=1)

X_test = test.drop(["Id", "idhogar", "parentesco1", "Target"], axis=1)
params = {

    "learning_rate": 0.1,

    "objective": "multiclass",

    "metric": "multi_logloss",

    "n_estimators": 1000,

    "class_weight": "balanced",

    "colsample_bytree": 0.9,

    "subsample": 0.8,

    "subsample_freq": 1,

    "num_class": 4,

    "lambda_l2": 1,

}
# Stratified K-Fold

num_folds = 5

skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=1)



# Initialize variables

y_preds = np.zeros((len(X_test), 4))

val_scores = []



for fold_n, (train_index, valid_index) in enumerate(skf.split(X, y)):

    

    # Get cross-validation split

    X_train = X.iloc[train_index]

    X_valid = X.iloc[valid_index]

    

    y_train = y[train_index]

    y_valid = y[valid_index]

    

    # Transform all the categorical features into Target-Encoded

    for f in X_train.columns:

        if X_train[f].dtype == "object" and f not in ["Id", "idhogar", "Target"]:

            te = TargetEncoder(target="Target", alpha=5)



            X_train[f] = te.fit_transform(X_train, f)

            X_valid[f] = te.transform(X_valid)

            X_test[f] = te.transform(X_test)

            

    X_train = X_train.drop(["Target"], axis=1)

    X_valid = X_valid.drop(["Target"], axis=1)



    # Train LightGBM model

    clf = lgb.LGBMClassifier(**params)

    clf = clf.fit(

        X_train,

        y_train,

        eval_set=[(X_valid, y_valid)],

        verbose=100,

        early_stopping_rounds=200,

    )



    # Make validation predictions

    y_pred_valid = clf.predict(X_valid)

    

    importances = pd.DataFrame({"feature": X_train.columns, "importance": clf.feature_importances_})



    # Evaluate the validation score

    score = f1_score(y_valid, y_pred_valid, average="macro")

    val_scores.append(score)

    print(f"Fold {fold_n}. F1 Score: {score:.5f}\n")

    

    # Make predictions on the test set (summing up the folds)

    y_preds += clf.predict_proba(X_test) / num_folds



print("Overall F1 Score: {:.3f}".format(np.mean(val_scores) + np.std(val_scores)))
importances


sns.barplot(data=importances.sort_values("importance", ascending=False).head(10), x="importance", y="feature")
y_preds = np.argmax(y_preds, axis=1)



submission = pd.DataFrame(

    {

        "Id": test["Id"],

        "Target": target_encoder.inverse_transform(y_preds).astype(int),

    }

)

submission.to_csv("submission.csv", index=False)
submission.head()
train.Target.value_counts()
submission.Target.value_counts()