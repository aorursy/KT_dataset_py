%matplotlib inline

import shap

import lime

import lightgbm as lgb

import matplotlib.pyplot as plt

import sklearn

from sklearn.externals.six import StringIO

from sklearn import preprocessing, metrics, model_selection

from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder, OneHotEncoder, LabelBinarizer 

from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, recall_score, precision_score, f1_score, precision_recall_curve

from sklearn.model_selection import KFold, StratifiedKFold, GridSearchCV, train_test_split

from sklearn import linear_model

from sklearn.linear_model import LogisticRegression

from datetime import datetime, date, timezone, timedelta

import seaborn as sns

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os, gc

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        

sns.set()
df_train = pd.read_csv('/kaggle/input/titanic/train.csv')

df_test = pd.read_csv('/kaggle/input/titanic/test.csv')

display(df_train.info())

display(df_train.head())
sns.distplot(df_train['Age'].dropna(),bins=range(0,100,10),kde=False)
sns.pairplot(df_train)
print('Train columns with null values:\n', df_train.isnull().sum())
display(df_train["Embarked"].value_counts())
display(df_train["Cabin"].str[0].value_counts())
display(df_train["Fare"].describe())

plt.hist(df_train["Fare"])

plt.show()

plt.hist(df_train["Fare"][df_train["Fare"]<=100])
def feature_engineering(df):

    # Null Value Handling

    df["Age"].fillna(df["Age"].median(),inplace=True)

    df["Embarked"].fillna(df['Embarked'].mode()[0], inplace = True)

    df = df.fillna(-1)

    

    # Feature Encoding

    df["Sex"] = df["Sex"].map({'male':1,'female':0}).fillna(-1).astype(int)

    df["Embarked"] = df["Embarked"].map({'S':0,'C':1,'Q':2}).astype(int)

    df["Cabin"] = df["Cabin"].str[0].map({'T':0,'G':1,'F':2,'E':3,'D':4,'C':5,'B':6,'A':7}).fillna(-1).astype(int)

    

    # Binning

    bins_age = np.linspace(0, 100, 10)

    df["AgeBin"] = np.digitize(df["Age"], bins=bins_age)

    

    df["FareBin"] = 0

    df["FareBin"][(df["Fare"]>=0)&(df["Fare"]<10)] = 1

    df["FareBin"][(df["Fare"]>=10)&(df["Fare"]<20)] = 2

    df["FareBin"][(df["Fare"]>=20)&(df["Fare"]<30)] = 3

    df["FareBin"][(df["Fare"]>=30)&(df["Fare"]<40)] = 4

    df["FareBin"][(df["Fare"]>=40)&(df["Fare"]<50)] = 5

    df["FareBin"][(df["Fare"]>=50)&(df["Fare"]<100)] = 6

    df["FareBin"][(df["Fare"]>=100)] = 7



    # Create New Features (Optional)

    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1

    df['Title'] = -1

    df['Title'][df["Name"].str.contains("Mr")] = 0

    df['Title'][df["Name"].str.contains("Master")] = 1

    df['Title'][df["Name"].str.contains("Miss")] = 2

    df['Title'][df["Name"].str.contains("Mrs")] = 3

    

    # Drop unsed columns

    del df["Age"]

    del df["Fare"]

    del df["Ticket"]

    

    return df
df_train_fe = feature_engineering(df_train)

df_test_fe = feature_engineering(df_test)



display(df_train_fe.head())
display(df_train_fe["FareBin"].value_counts())
#correlation heatmap of dataset

def correlation_heatmap(df):

    _ , ax = plt.subplots(figsize =(14, 12))

    colormap = sns.diverging_palette(220, 10, as_cmap = True)

    

    _ = sns.heatmap(

        df.corr(), 

        cmap = colormap,

        square=True, 

        cbar_kws={'shrink':.9 }, 

        ax=ax,

        annot=True, 

        linewidths=0.1,vmax=1.0, linecolor='white',

        annot_kws={'fontsize':12 }

    )

    

    plt.title('Pearson Correlation of Features', y=1.05, size=15)



correlation_heatmap(df_train_fe)
exclude_columns = [

    'Name',

    'Ticket',

    'PassengerId',

    'Survived'

]



evals_result = {}

features = [c for c in df_train_fe.columns if c not in exclude_columns]

target = df_train_fe['Survived']

print(len(target))



gc.collect()



X_train, X_test, y_train, y_test = train_test_split(df_train_fe[features], target, test_size=0.2, random_state=440)



param = {   

    'boost': 'gbdt',

    'learning_rate': 0.008,

    'feature_fraction':0.20,

    'bagging_freq':1,

    'bagging_fraction':1,

    'max_depth': -1,

    'num_leaves':17,

    'lambda_l2': 0.9,

    'lambda_l1': 0.9,

    'max_bin':200,

    'metric':{'auc','binary_logloss'},

#    'metric':{'binary_logloss'},

    'tree_learner': 'serial',

    'objective': 'binary',

    'verbosity': 1,

}



oof = np.zeros(len(df_train_fe))

predictions = np.zeros(len(df_test_fe))

feature_importance_train = pd.DataFrame()



lgb_train = lgb.Dataset(X_train, y_train)

lgb_valid = lgb.Dataset(X_test, y_test)

num_round = 10000

clf = lgb.train(param, lgb_train, num_round, valid_sets = [lgb_train, lgb_valid],

      verbose_eval=100, early_stopping_rounds = 1000, evals_result = evals_result)

oof = clf.predict(X_test, num_iteration=clf.best_iteration)



## Prediction

predictions = clf.predict(df_test_fe[features], num_iteration=clf.best_iteration)



# Visualize Metrics

axL = lgb.plot_metric(evals_result, metric='auc')

axL.set_title('AUC')

axL.set_xlabel('Iterations')

axL.set_ylim(0,1.1)

axR = lgb.plot_metric(evals_result, metric='binary_logloss')        

axR.set_title('Binary_Logloss')

axR.set_xlabel('Iterations')

plt.show()



# Importance

fold_importance_train = pd.DataFrame()

fold_importance_train["feature"] = features

fold_importance_train["importance"] = clf.feature_importance()

fold_importance_train["fold"] = 1

feature_importance_train = pd.concat([feature_importance_train, fold_importance_train], axis=0)



precisions, recalls, thresholds = precision_recall_curve(y_test, oof)



fig = plt.figure(figsize=(14,4))



### Threshold vs Precision/Recall

ax = fig.add_subplot(1,2,1)

ax.plot(thresholds, precisions[:-1], "b--", label="Precision")

ax.plot(thresholds, recalls[:-1], "g--", label="Recall")

ax.set_title("Threshold vs Precision/Recall")

ax.set_xlabel("Threshold")

ax.legend(loc="center left")

ax.set_ylim([0,1.1])

ax.grid()

fig.show()



### Precision-Recall Curve

ax = fig.add_subplot(1,2,2)

ax.step(recalls, precisions, color='b', alpha=0.2, where='post')

ax.fill_between(recalls, precisions, step='post', alpha=0.2, color='b')

ax.set_title('Precision-Recall Curve')

ax.set_xlabel('Recall')

ax.set_ylabel('Precision')

ax.set_ylim([0.0, 1.05])

ax.set_xlim([0.0, 1.0])

ax.grid()

fig.show()



### スコア分布

df_train_result = pd.DataFrame()

df_train_result['Actual Result'] = y_test

df_train_result['Prediction Score'] = oof

df_train_result = df_train_result.sort_values('Prediction Score',ascending=False).sort_index(ascending=False)

df_target = df_train_result[df_train_result['Actual Result']==1]

df_nontarget = df_train_result[df_train_result['Actual Result']==0]



### Show Importance

cols = (feature_importance_train[["feature","importance"]]

        .groupby("feature")

        .mean()

        .sort_values(by="importance", ascending=False)[:1000].index)

best_features = feature_importance_train.loc[feature_importance_train.feature.isin(cols)]



plt.figure(figsize=(14,5))

sns.barplot(x="importance", y="feature", 

            data=best_features.sort_values(by="importance",ascending=False))

plt.title('LightGBM Features (averaged over folds)')

plt.tight_layout()

submit = pd.DataFrame({ 'PassengerId' : df_test_fe["PassengerId"], 'Survived': np.int32(predictions >= 0.5) })

submit.to_csv('submission.csv', index=False)
shap.initjs()

explainer = shap.TreeExplainer(clf)

shap_values = explainer.shap_values(X_train)

shap.force_plot(explainer.expected_value[1], shap_values[1][0,:], X_train.iloc[0,:])
shap.force_plot(explainer.expected_value[1], shap_values[1][1,:], X_train.iloc[1,:])
shap.force_plot(explainer.expected_value[1], shap_values[1][2,:], X_train.iloc[2,:])
shap.force_plot(base_value=explainer.expected_value[1], shap_values=shap_values[1], features=X_train.columns)
shap.decision_plot(explainer.expected_value[1], shap_values[1][0,:], X_train.iloc[0,:])
shap.summary_plot(shap_values, X_train)

shap.summary_plot(shap_values, X_train, plot_type='bar')
shap.dependence_plot("AgeBin",shap_values[1], X_train)
shap.dependence_plot("Sex",shap_values[1], X_train)
shap.dependence_plot("Pclass",shap_values[1], X_train)
shap.dependence_plot("FareBin",shap_values[1], X_train)
shap.dependence_plot("Title",shap_values[1], X_train)
import lime

import lime.lime_tabular



def predict_fn(x):

    preds = clf.predict(x, num_iteration=clf.best_iteration).reshape(-1,1)

    p0 = 1 - preds

    return np.hstack((p0, preds))



explainerLime = lime.lime_tabular.LimeTabularExplainer(

    X_train.values,

    mode='classification',

    feature_names=features,

   class_names=["NotSurvived", "Survived"],

   verbose=True

    )



np.random.seed(1)

i = 0

exp = explainerLime.explain_instance(X_train[features].values[i], predict_fn, num_features=10)

exp.show_in_notebook(show_all=True)
shap.force_plot(explainer.expected_value[1], shap_values[1][0,:], X_train.iloc[0,:])