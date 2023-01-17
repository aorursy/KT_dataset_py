import numpy as np # linear algebra

import pandas as pd

from sklearn.metrics import make_scorer, roc_auc_score,classification_report

from sklearn.metrics import roc_auc_score

from sklearn.ensemble import RandomForestClassifier

from sklearn.utils import class_weight

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import StratifiedShuffleSplit
train =  pd.read_csv("../input/tabular-melonoma/trainmod.csv")

test = pd.read_csv("../input/tabular-melonoma/test_sub.csv")

locing = ["l"+str(i) for i in range(10)]

colors_table = ["Color"+str(canal)+str(znach) for znach in range(3) for canal in range(3)]
def label_e(dataframe):

    dataframe.loc[dataframe["sex"].isnull(),["sex"]] = "male"

    dataframe.loc[dataframe["age_approx"].isnull(),["age_approx"]] = 50

    dataframe.loc[dataframe["anatom_site_general_challenge"].isnull(),["anatom_site_general_challenge"]] = "torso"

    dataframe["split"] = 0



    dataframe.loc[dataframe["age_approx"]<=40,["split"]] = 1

    dataframe.loc[(dataframe["age_approx"]>40) & (dataframe["age_approx"]<=76),["split"]] = 2

    dataframe.loc[dataframe["age_approx"]>76,["split"]] = 3

    patient_id = LabelEncoder()

    sex = LabelEncoder()

    # age_approx = LabelEncoder()

    anatom_site_general_challenge = LabelEncoder()



    patient_id.fit(dataframe["patient_id"].unique())

    sex.fit(dataframe["sex"].unique())

    # age_approx.fit(train["age_approx"].unique())

    anatom_site_general_challenge.fit(dataframe["anatom_site_general_challenge"].unique())



    dataframe["patient_id"] = patient_id.transform(dataframe["patient_id"])

    dataframe["sex"] = sex.transform(dataframe["sex"])

    # train["age_approx"] = age_approx.transform(train["age_approx"])

    dataframe["anatom_site_general_challenge"] = anatom_site_general_challenge.transform(dataframe["anatom_site_general_challenge"])
label_e(train)

label_e(test)
train.head()
train_c = train.copy()

train_split = 0

train_val_split = 0



split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=6)

for train_index, test_index in split.split(train_c,train_c["target"]):

    train_split = train_c.loc[train_index].copy()

    train_val_split = train_c.loc[test_index].copy()

    train_split.drop(["split"], axis=1, inplace=True)

    train_val_split.drop(["split"], axis=1, inplace=True)

locing2 = np.hstack((locing,["age_approx","veil","width","height","globuli","patient_id","anatom_site_general_challenge","sex"]))

locing2 = np.hstack((locing2,colors_table))

train_x = train_split[locing2]

train_y = train_split["target"]

val_x = train_val_split[locing2]

val_y = train_val_split["target"]
CW = class_weight.compute_class_weight('balanced',

                                                 np.unique(train["target"]),

                                                 train["target"])

clases = [0,1]

class_weights = dict(zip(clases,CW))

class_weights
tree = RandomForestClassifier(n_estimators=69, max_depth=50, min_samples_split=9,  min_samples_leaf=12, class_weight=class_weights)

tree.fit(train_x,train_y)

y_pred1 = tree.predict_proba(val_x)

print(make_scorer(roc_auc_score, needs_proba=True)(tree, val_x, val_y))
test_x = test[locing2]

test_x.head()
test_pred = tree.predict_proba(test_x)

prediction = pd.DataFrame(test_pred,columns=["t","target"])

test["target"] = prediction["target"]

submission = test[["image_name","target"]]

submission.to_csv("submit.csv", index=False, line_terminator="\n")
submission.head()