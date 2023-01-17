# python utilities

import random

import os



# general data science

import numpy as np

import pandas as pd

%matplotlib inline

import matplotlib.pyplot as plt

import seaborn as sns



# ---------- scikit-learn ------------

# preprocessing

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, OrdinalEncoder

from sklearn.compose import ColumnTransformer



# models

from sklearn.naive_bayes import GaussianNB, CategoricalNB

from sklearn.linear_model import LogisticRegression

from sklearn.base import BaseEstimator # for custom estimators

from sklearn.svm import SVC

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier



# model selection

from sklearn.model_selection import StratifiedKFold

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV



# model evaluation

from sklearn.metrics import classification_report, precision_recall_fscore_support, confusion_matrix, roc_auc_score, roc_curve

from sklearn.metrics import fbeta_score, make_scorer

# -------------------------------------



# imbalanced-learn

from imblearn.pipeline import Pipeline # if using imblearn's sampling we must use this over sklearn's Pipeline 

from imblearn.over_sampling import SMOTE, RandomOverSampler # oversampling

from imblearn.under_sampling import EditedNearestNeighbours, TomekLinks, NearMiss, RandomUnderSampler # undersampling



import warnings  

warnings.filterwarnings('ignore')



def seed_everything(seed = 42):

    random.seed(seed)

    np.random.seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)

    

seed_everything()
# load the data

data = pd.read_csv('../input/telco-customer-churn/WA_Fn-UseC_-Telco-Customer-Churn.csv')

data.shape
data.head()
data.info()
X = data.drop("Churn", axis=1)

y = data["Churn"]

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, test_size=0.2)
df_train = pd.concat([X_train, y_train], axis=1)
df_train.shape
df_train.head()
df_train.isna().mean()
df_train["TotalCharges"][df_train["customerID"] == "2775-SEFEE"].values[0]
df_train.apply(lambda x: x==' ', axis=1).mean()
df_train = df_train[df_train["TotalCharges"] != ' ']

df_train["TotalCharges"] = df_train["TotalCharges"].astype('float64')
df_train[["tenure", "TotalCharges", "MonthlyCharges"]].describe()
df_train.nunique().sort_values(ascending=False)
df_train = df_train.drop("customerID", axis=1)
low_unq_feats = df_train.columns[df_train.nunique()<10]

for feat in low_unq_feats:

    print(feat, df_train[feat].unique())
df_train["SeniorCitizen"] = df_train["SeniorCitizen"].replace({0:"No",1:"Yes"})
df_train.dtypes
sns.countplot("Churn", data=df_train)
(df_train["Churn"]=="No").sum(), (df_train["Churn"]=="Yes").sum()
cat_feats = df_train.columns[df_train.dtypes == 'object'][:-1] # :-1 to remove churn

cat_feats
fig, ax = plt.subplots(4,4,figsize=(14,14))

ax = ax.flatten()

for i,feat in enumerate(cat_feats):

    plt.sca(ax[i])

    df_unq = df_train[feat].value_counts().sort_values(ascending=False)



    sns.barplot(df_unq.index, df_unq.values, order=df_unq.index)

    

    plt.xlabel(str(feat), color='red', fontsize=14)

    plt.xticks(rotation=45, ha='right')

    

plt.tight_layout(h_pad=2)

plt.show()
df_train["Churn"] = df_train["Churn"].replace({"No":0, "Yes":1})
fig, ax = plt.subplots(4,4,figsize=(14,14))

ax = ax.flatten()

for i,feat in enumerate(cat_feats):

    plt.sca(ax[i])

    sns.barplot(x=feat, y="Churn", data=df_train)

                

    plt.xlabel(str(feat), color='red', fontsize=14)

    plt.xticks(rotation=45, ha='right')

    

plt.tight_layout(h_pad=2)

plt.show()
import plotly.graph_objects as go



for feat in cat_feats:

    df_train[feat] = df_train[feat].replace({"No":0,"Yes":1,"No internet service":0,"No phone service":0})

df_train["gender"] = df_train["gender"].replace({"Female":0,"Male":1})

df_train["InternetService"] = df_train["InternetService"].replace({"DSL":1,"Fiber optic":1})

df_train["Contract"] = df_train["Contract"].replace({"One year":1, "Two year":1, "Month-to-month":0})

df_train["PaymentMethod"] = df_train["PaymentMethod"].replace({"Mailed check":0, "Bank transfer (automatic)":1, 

                                                               "Electronic check":1, "Credit card (automatic)":1})





# based on the plot at https://plotly.com/python/radar-chart/

fig = go.Figure()



fig.add_trace(go.Scatterpolar(

      r=df_train.loc[df_train["Churn"]==0,cat_feats].mean().tolist(),

      theta=cat_feats.tolist(),

      fill='toself',

      name='No churn'

))



fig.add_trace(go.Scatterpolar(

      r=df_train.loc[df_train["Churn"]==1,cat_feats].mean().tolist(),

      theta=cat_feats.tolist(),

      fill='toself',

      name='Churn'

))



fig.update_layout(

  polar=dict(

    radialaxis=dict(

      visible=True,

      range=[0, 1]

    )),

  showlegend=False

)



fig.show()
num_feats = ["tenure", "TotalCharges", "MonthlyCharges"]

df_train[num_feats].describe()
fig, ax = plt.subplots(1,3,figsize=(14,6))

for i,feat in enumerate(num_feats):

    plt.sca(ax[i])

    sns.boxplot(x="Churn", y=feat, data=df_train, ax=ax[i])

    plt.ylabel("")

    plt.title(feat, color="red", fontsize=14)

    

plt.tight_layout()

plt.show()
df_train[num_feats].corr()
def tidy_up(df):

    df = df[df["TotalCharges"] != ' ']

    df["TotalCharges"] = df["TotalCharges"].astype('float64')

    

    df["SeniorCitizen"] = df["SeniorCitizen"].replace({0:"No",1:"Yes"})

    

    df["Churn"] = df["Churn"].replace({"No":0, "Yes":1})

    

    df.drop("customerID", axis=1, inplace=True)

    

    X = df.drop("Churn", axis=1)

    y = df["Churn"]

    

    return X,y



X_train, y_train = tidy_up(pd.concat([X_train, y_train], axis=1))

X_test, y_test = tidy_up(pd.concat([X_test, y_test], axis=1))
# define categorical features & initialize encoder

cat_feats = X_train.columns[X_train.dtypes == 'object']

onehot_encoder = OneHotEncoder() 
# define numeric features & initialize scaler

num_feats = ["tenure", "TotalCharges", "MonthlyCharges"]

scaler = StandardScaler()
# fit CategoricalNB to categorical features

ord_encoder = OrdinalEncoder()

X_train_c = X_train[cat_feats]

X_train_c = ord_encoder.fit_transform(X_train_c)

nb_c = CategoricalNB()

nb_c.fit(X_train_c, y_train)



# fit GaussianNB to numeric features

X_train_n = X_train[num_feats]

nb_n = GaussianNB()

nb_n.fit(X_train_n, y_train)



# get predicted class probabilities, P(Y=1|X),from each model. Then stack predictions and train another GaussianNB. 

train_preds_c = nb_c.predict_proba(X_train_c)[:,1]

train_preds_n = nb_n.predict_proba(X_train_n)[:,1]

train_preds_cn = np.vstack((train_preds_c, train_preds_n)).T

nb_cn = GaussianNB()

nb_cn.fit(train_preds_cn, y_train)



# test set predictions

X_test_c = X_test[cat_feats]

X_test_c = ord_encoder.transform(X_test_c)

test_preds_c = nb_c.predict(X_test_c)

X_test_n = X_test[num_feats]

test_preds_n = nb_n.predict(X_test_n)

test_preds_cn = np.vstack((test_preds_c, test_preds_n)).T

test_preds = nb_cn.predict(test_preds_cn)



# evaluate model

print("Train accuracy...")

print(classification_report(y_train, nb_cn.predict(train_preds_cn)))

print("Test accuracy...")

print(classification_report(y_test, test_preds))
ct = ColumnTransformer([('cat_feats', onehot_encoder, cat_feats),

                        ('num_feats', scaler, num_feats)])



model = LogisticRegression(penalty="l1", solver="liblinear")



# no undersampling/oversampling

pipe = Pipeline([("preprocessing", ct),

                 ("logreg", model)])



kf = StratifiedKFold(n_splits=5)



grid = GridSearchCV(pipe, param_grid={'logreg__C': [0.01, 0.1, 1, 10]}, cv=kf, scoring='f1')

grid.fit(X_train, y_train)
grid.best_estimator_["logreg"]
predict = grid.best_estimator_.predict(X_test)

print(classification_report(y_test, predict))
def brief_classification_report(y_test, predict):

    rep = np.array(precision_recall_fscore_support(y_test, predict))

#     print("       precision       recall            f1")

#     print("0\t", "\t\t".join(["%0.02f" % x for x in rep[:-1,0]]) )

    print("1\t", "\t\t".join(["%0.02f" % x for x in rep[:-1,1]]))



sm_names = ["Edited NN", "Tomek Links", "Random Undersampling", "Near-Miss", "SMOTE", "Random Oversampling"] 

sms = [EditedNearestNeighbours(), TomekLinks(), RandomUnderSampler(), NearMiss(), SMOTE(), RandomOverSampler()]



ct = ColumnTransformer([('cat_feats', onehot_encoder, cat_feats),

                        ('num_feats', scaler, num_feats)])



model = LogisticRegression(penalty="l1", solver="liblinear")



print("       precision       recall            f1")

for sm_name, sm in zip(sm_names, sms):

    

    pipe = Pipeline([("preprocessing", ct),

                     ("sampling", sm),

                    ("logreg", model)])



    kf = StratifiedKFold(n_splits=5)



    grid = GridSearchCV(pipe, param_grid={'logreg__C': [0.01, 0.1, 1, 10]}, cv=kf, scoring='f1')

    grid.fit(X_train, y_train)

    

    predict = grid.best_estimator_.predict(X_test)

    print(sm_name)

    brief_classification_report(y_test, predict)
# This is a custom estimator. Code courtesty of:  https://stackoverflow.com/a/53926097/7638741 .

class ClfSwitcher(BaseEstimator):

    def __init__(

        self, 

        estimator = SVC(),

    ):

        """

        A Custom BaseEstimator that can switch between classifiers.

        :param estimator: sklearn object - The classifier

        """ 



        self.estimator = estimator





    def fit(self, X, y=None, **kwargs):

        self.estimator.fit(X, y)

        return self





    def predict(self, X, y=None):

        return self.estimator.predict(X)





    def predict_proba(self, X):

        return self.estimator.predict_proba(X)





    def score(self, X, y):

        return self.estimator.score(X, y)
ct = ColumnTransformer([('cat_feats', onehot_encoder, cat_feats),

                    ('num_feats', scaler, num_feats)])



pipe = Pipeline([("preprocessing", ct),

                 ("sampling", RandomOverSampler()),

                 ("clf", ClfSwitcher())])



parameters = [

    {

        'clf__estimator': [SVC()],

        'clf__estimator__C': [0.1, 1, 10, 20],

        'clf__estimator__kernel': ['rbf', 'poly']

    },

    {

        'clf__estimator': [KNeighborsClassifier()],

        'clf__estimator__n_neighbors':[3,5,10,20]

    },

    {

        'clf__estimator': [RandomForestClassifier()],

        'clf__estimator__n_estimators': [100,200], 

        'clf__estimator__max_depth': [15,30], 

        'clf__estimator__max_features': [5,10],

        'clf__estimator__min_samples_leaf': [4,8]

    },

]



kf = StratifiedKFold(n_splits=5)



grid = GridSearchCV(estimator = pipe, param_grid=parameters, cv=kf, scoring='f1')



grid.fit(X_train, y_train)
def format_cv_results(search):

    df = pd.concat([pd.DataFrame(grid.cv_results_["params"]),pd.DataFrame(grid.cv_results_["mean_test_score"], columns=["Score"])],axis=1)

    df = df.sort_values("Score", ascending=False)

    return df.fillna(value="")

df_res = format_cv_results(grid)

df_res
predict = grid.best_estimator_.predict(X_test)

print(classification_report(y_test, predict))
fpr, tpr, thresholds = roc_curve(y_test, grid.predict_proba(X_test)[:,1])

roc_auc = roc_auc_score(y_test, predict)



plt.plot(fpr, tpr, lw=1, label='AUC = %0.2f'%(roc_auc))

plt.plot([0, 1], [0, 1], '--k', lw=1)

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('ROC curve')

plt.legend(loc="lower right", frameon = True).get_frame().set_edgecolor('black')
ct = ColumnTransformer([('cat_feats', onehot_encoder, cat_feats),

                        ('num_feats', scaler, num_feats)])



model = XGBClassifier(learning_rate=0.02, 

                    n_estimators=200,

                    booster = 'gbtree',

                    objective='binary:logistic')



pipe = Pipeline([("preprocessing", ct),

                ("sampling", RandomOverSampler()),

                ("xgb", model)])



tuned_parameters = {

        'xgb__min_child_weight': [1, 5, 10],

        'xgb__gamma': [0.5, 1, 1.5, 2, 5, 10],

        'xgb__subsample': [0.6, 0.8, 1.0],

        'xgb__colsample_bytree': [0.6, 0.8, 1.0],

        'xgb__max_depth': [3, 5, 8]

        }



kf = StratifiedKFold(n_splits=5)



grid = RandomizedSearchCV(estimator = pipe, 

                                   param_distributions=tuned_parameters, 

                                   cv=kf,

                                   n_iter=20, 

                                   scoring='f1', 

                                   n_jobs=-1, 

                                   verbose=3)



grid.fit(X_train, y_train)
preds = grid.best_estimator_.predict(X_test)

print(classification_report(y_test, preds))
fpr, tpr, thresholds = roc_curve(y_test, grid.best_estimator_.predict_proba(X_test)[:,1])

roc_auc = roc_auc_score(y_test, preds)



plt.plot(fpr, tpr, lw=1, label='AUC = %0.2f'%(roc_auc))

plt.plot([0, 1], [0, 1], '--k', lw=1)

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('ROC curve')

plt.legend(loc="lower right", frameon = True).get_frame().set_edgecolor('black')
feat_names = []

for feat in cat_feats:

    for level in X_train[feat].unique():

        feat_names.append("%s_%s" % (feat,level))

feat_names.extend(num_feats)



importances = grid.best_estimator_["xgb"].feature_importances_

importances_dict = {f:i for f,i in zip(feat_names, importances)}



n = 20 # only plot a few 

importances = pd.DataFrame.from_dict(importances_dict, orient='index').rename(columns={0: 'Gini-importance'}).head(n)



importances.sort_values(by='Gini-importance', ascending=False).plot(kind='bar', rot=45, figsize=(14,6), fontsize=14)

plt.xticks(ha='right')

plt.show()
import pickle

with open('customer-churn_XGBoost','wb') as f:

    pickle.dump(grid.best_estimator_, f)