# Import modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import catboost as cat
from catboost import Pool, cv, CatBoostClassifier, CatBoostRegressor
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import warnings
from sklearn.metrics import (confusion_matrix, precision_recall_curve, auc,
                             roc_curve, recall_score, classification_report, f1_score,
                             precision_recall_fscore_support)
from sklearn.model_selection import train_test_split
warnings.filterwarnings("ignore")
%matplotlib inline
pd.options.display.max_columns = None
pd.options.display.max_rows = None
# Training dataset
train = pd.read_csv("/kaggle/input/widsdatathon2020/training_v2.csv")
# Testing dataset
test = pd.read_csv("/kaggle/input/widsdatathon2020/unlabeled.csv")
test.head()
train.head()
train.shape
train.info()
train.isnull().sum(axis=0)/len(train.index) * 100
plt.figure(figsize=(24,6))
(train.isnull().sum(axis=0)/len(train.index) * 100).plot(kind='bar')
plt.show()
train.describe()
dictionary = pd.read_csv("/kaggle/input/widsdatathon2020/WiDS Datathon 2020 Dictionary.csv")
dictionary.head()
dictionary.Category.value_counts()
def selectCategory(df,category):
    """
    This function will return the dataset
    with information about the specified
    category
    """
    return df[df.Category==category]
cols = selectCategory(dictionary,"GOSSIS example prediction")
cols
print(cols["Description"].values[0])
if "pred" not in train.columns:
    print("Column not present")
train.shape, dictionary.shape
cols = selectCategory(dictionary,"APACHE grouping")
cols
train[["apache_3j_bodysystem","apache_2_bodysystem"]].head(10)
train[["apache_3j_bodysystem","apache_2_bodysystem"]][train["apache_3j_bodysystem"] != train["apache_2_bodysystem"]].head(20)
# Columns to drop
toDrop = []
toDrop.append("apache_2_bodysystem")
cols = selectCategory(dictionary,"APACHE prediction")
cols
for i in cols.Description.values:
    print(i)
sns.heatmap(train[["apache_4a_hospital_death_prob","apache_4a_icu_death_prob"]].corr(),annot=True)
toDrop.append("apache_4a_hospital_death_prob")
cols = selectCategory(dictionary,"identifier")

cols
for i in cols.Description.values:
    print(i)
train.encounter_id.nunique(),train.hospital_id.nunique(),train.patient_id.nunique(),len(train.index)
toDrop.append("encounter_id")
toDrop.append("patient_id")
cols = selectCategory(dictionary,"APACHE comorbidity")

cols
for col in cols["Variable Name"].values:
    train[col] = train[col].fillna(0)
cols = selectCategory(dictionary,"labs blood gas")

cols
for i in range(len(cols.index)):
    print("{}: {}".format(cols["Variable Name"].values[i],cols.Description.values[i]))
plt.figure(figsize=(10,10))
sns.heatmap(abs(train[cols["Variable Name"]].corr()),annot=True)
plt.show()
for col in cols["Variable Name"].values:
    if col.startswith("d1_"):
        print("Column to delete: {}".format(col))
        toDrop.append(col)
cols = selectCategory(dictionary,"demographic")

cols
train.hospital_death.value_counts(normalize=True).plot(kind='bar')
sns.distplot(train["age"])
sns.distplot(train[train["hospital_death"]==0]["bmi"],hist=False)
sns.distplot(train[train["hospital_death"]==1]["bmi"],hist=False)
def bmiCategory(bmi):
    if bmi < 18.5:
        return "underweight"
    elif bmi < 24.9:
        return "normal"
    elif bmi < 29.9:
        return "overweight"
    else:
        return "obese"

train["bmi_category"] = train["bmi"].apply(bmiCategory)
test["bmi_category"] = test["bmi"].apply(bmiCategory)
train["bmi_category"].value_counts(normalize=True).plot(kind='bar')
sns.catplot(x="bmi_category",y="hospital_death",kind="bar",data=train)
toDrop.append("bmi")
train.elective_surgery.value_counts(normalize=True).plot(kind='bar')
sns.countplot(x="elective_surgery",hue="hospital_death",data=train)
train.groupby("elective_surgery")["hospital_death"].sum()/train.groupby("elective_surgery")["hospital_death"].count() * 100
sns.catplot(x="elective_surgery",y="hospital_death",kind="bar",data=train)
train["ethnicity"].value_counts(normalize=True).plot(kind='bar')
sns.catplot(x="ethnicity",y="hospital_death",kind="bar",data=train,aspect=1.8)
plt.show()
toDrop.append("ethnicity")
train["gender"].value_counts(normalize=True).plot(kind='bar')
sns.catplot(x="gender",y="hospital_death",kind="bar",data=train)
plt.show()
toDrop.append("gender")
sns.distplot(train[train["hospital_death"]==0]["height"],hist=False)
sns.distplot(train[train["hospital_death"]==1]["height"],hist=False)
toDrop.append("height")
train["hospital_admit_source"].value_counts(normalize=True).plot(kind='bar')
sns.catplot(x="hospital_admit_source",y="hospital_death",kind="bar",data=train,aspect=1.8)
plt.xticks(rotation=90)
plt.show()
train["icu_admit_source"].value_counts(normalize=True).plot(kind='bar')
sns.catplot(x="icu_admit_source",y="hospital_death",kind="bar",data=train,aspect=1.8)
plt.xticks(rotation=90)
plt.show()
if "icu_admit_type" not in train.columns:
    print("Column not found")
train.icu_id.nunique()
train["icu_stay_type"].value_counts(normalize=True).plot(kind='bar')
sns.catplot(x="icu_stay_type",y="hospital_death",kind="bar",data=train,aspect=1.8)
plt.xticks(rotation=90)
plt.show()
toDrop.append("icu_stay_type")
train["icu_type"].value_counts(normalize=True).plot(kind='bar')
sns.catplot(x="icu_type",y="hospital_death",kind="bar",data=train,aspect=1.8)
plt.xticks(rotation=90)
plt.show()
sns.boxplot(train["pre_icu_los_days"])
sns.distplot(train["pre_icu_los_days"])
sns.distplot(train[train["hospital_death"]==0]["pre_icu_los_days"])
sns.distplot(train[train["hospital_death"]==1]["pre_icu_los_days"])
train["readmission_status"].value_counts(normalize=True).plot(kind='bar')
toDrop.append("readmission_status")
sns.distplot(train[train["hospital_death"]==0]["weight"],hist=False)
sns.distplot(train[train["hospital_death"]==1]["weight"],hist=False)
toDrop.append("weight")
cols = selectCategory(dictionary,"APACHE covariate")

cols
plt.figure(figsize=(20,20))
sns.heatmap(abs(train[cols["Variable Name"]].corr()),annot=True)
plt.show()
toDrop.append("ventilated_apache")
toDrop.append("gcs_eyes_apache")
toDrop.append("gcs_verbal_apache")
toDrop.append("apache_3j_diagnosis")
cols = selectCategory(dictionary,"vitals")

plt.figure(figsize=(20,20))
sns.heatmap(abs(train[cols["Variable Name"]].corr()),annot=True)
plt.show()
toDrop.append("h1_sysbp_noninvasive_max")
toDrop.append("h1_sysbp_noninvasive_min")
toDrop.append("h1_mbp_noninvasive_max")
toDrop.append("h1_mbp_noninvasive_min")
toDrop.append("h1_diasbp_noninvasive_max")
toDrop.append("h1_diasbp_noninvasive_min")
cols = selectCategory(dictionary,"labs")

plt.figure(figsize=(20,20))
sns.heatmap(abs(train[cols["Variable Name"]].corr()),annot=True)
plt.show()
for col in cols["Variable Name"].values:
    if col.startswith("d1_"):
        print("Column to delete: {}".format(col))
        toDrop.append(col)
print("Total columns to drop: {}".format(len(toDrop)))
# Drop columns
train.drop(toDrop,axis=1,inplace=True)
train.shape
for col in train.columns:
    print(col)
categoricals_features = ["hospital_id", "hospital_admit_source",
                         "icu_admit_source","icu_id","icu_type",
                         "apache_3j_bodysystem","bmi_category"]
for cat in categoricals_features:
    print("{}: {}".format(cat,train[cat].nunique()))
# This cell is taken from 
# https://www.kaggle.com/jayjay75/wids2020-lgb-starter-adversarial-validation

# categorical feature need to be transform to numeric for mathematical purpose.
# different technics of categorical encoding exists here we will rely on our model API to deal with categorical
# still we need to encode each categorical value to an id , for this purpose we use LabelEncoder

print('Transform all String features to category.\n')
for usecol in categoricals_features:
    train[usecol] = train[usecol].astype('str')
    test[usecol] = test[usecol].astype('str')
    
    #Fit LabelEncoder
    le = LabelEncoder().fit(
            np.unique(train[usecol].unique().tolist()+
                      test[usecol].unique().tolist()))

    #At the end 0 will be used for null values so we start at 1 
    train[usecol] = le.transform(train[usecol])+1
    test[usecol]  = le.transform(test[usecol])+1
    
    train[usecol] = train[usecol].replace(np.nan, 0).astype('int').astype('category')
    test[usecol]  = test[usecol].replace(np.nan, 0).astype('int').astype('category')
train[categoricals_features].isna().sum(axis=0)
X_train = train.drop("hospital_death",axis=1)
y_train = train["hospital_death"]
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.20, random_state=2020)
# function to evaluate the score of our model
def eval_auc(pred,real):
    false_positive_rate, recall, thresholds = roc_curve(real, pred)
    roc_auc = auc(false_positive_rate, recall)
    return roc_auc
train_pool = Pool(data=X_train,label = y_train,cat_features=categoricals_features)
test_pool = Pool(data=X_test,label = y_test,cat_features=categoricals_features)

model_basic = CatBoostClassifier(iterations=100,custom_metric='AUC:hints=skip_train~false', 
                                 metric_period=2,task_type="GPU")
model_basic.fit(train_pool,plot=True,eval_set=test_pool)
print(model_basic.get_best_score())
# model = CatBoostClassifier(eval_metric='AUC:hints=skip_train~false',task_type="GPU",
#                           depth=9, iterations=800, learning_rate = 0.1,
#                           custom_metric = 'AUC:hints=skip_train~false')

# train_pool = Pool(data=X_train,label = y_train,cat_features=categoricals_features)
# test_pool = Pool(data=X_test,label = y_test,cat_features=categoricals_features)
# model.fit(train_pool,plot=True,eval_set=test_pool)

# Uncomment this cell to perform grid search
# The results I found are:

# Best model params: 
# {'depth': 9, 'iterations': 800, 'learning_rate': 0.1, 'custom_metric': 'AUC:hints=skip_train~false'}

# grid = {'learning_rate': [0.06, 0.08, 0.1, 0.12, 0.14],
#         'depth': [7, 9, 11, 13],
#         "iterations": [400, 600, 800, 1000],
#        "custom_metric":['Logloss:hints=skip_train~false', 'AUC:hints=skip_train~false']}

grid = {'learning_rate': [0.08, 0.1],
        'depth': [7, 9],
        "iterations": [600,800]}#,
       #"custom_metric":['Logloss:hints=skip_train~false', 'AUC:hints=skip_train~false']}

X_train = train.drop("hospital_death",axis=1)
y_train = train["hospital_death"]
train_pool = Pool(data=X_train,label = y_train,cat_features=categoricals_features)

# model = CatBoostClassifier(eval_metric='AUC:hints=skip_train~false',task_type="GPU")
model = CatBoostClassifier(eval_metric='AUC:hints=skip_train~false',task_type="GPU",
                          custom_metric='AUC:hints=skip_train~false')

grid_search_result = model.grid_search(grid, 
                                       train_pool,
                                       plot=True,
                                       refit = True,
                                       partition_random_seed=2020
                                    )

# print("Best model params: \n",grid_search_result["params"])
print(model.get_best_score())
feature_importances = model.get_feature_importance(train_pool)
feature_names = X_train.columns
for score, name in sorted(zip(feature_importances, feature_names), reverse=True):
    if score > 0.05:
        print('{0}: {1:.2f}'.format(name, score))
testEncounterId = test.encounter_id
test.drop(toDrop,axis=1,inplace=True)
test["hospital_death"] = model.predict(test.drop(["hospital_death"],axis=1),prediction_type='Probability')[:,1]
test["encounter_id"] = testEncounterId
test[["encounter_id","hospital_death"]].to_csv("submission.csv",index=False)
