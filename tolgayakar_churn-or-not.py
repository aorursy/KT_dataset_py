
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

import warnings
warnings.filterwarnings('ignore')
from pandas_profiling import ProfileReport
churn = pd.read_csv("../input/churn-predictions-personal/Churn_Predictions.csv")
df = churn.copy()
df.head()
df.info()
df.drop("RowNumber",axis = 1,inplace = True)
df.head(20)
df[(df["Age"] == 18)]
#df = df[~((df["HasCrCard"] == 0) & (df["Balance"] > 0))]
#df["LegalAgeLimit"] = df["Age"] - df["Tenure"]
df = df[~((df["Age"]-df["Tenure"] < 18) & (df["HasCrCard"] == 1))]
df.shape
df[df.Tenure == 0]
df["Exited"].value_counts().plot.pie(autopct = "%.1f");
df.head()
df['Geography'].value_counts()
df[df["Exited"] == 1]
f, ax = plt.subplots(figsize= [15,10])
sns.heatmap(df.corr(), annot=True, fmt=".2f", ax=ax )
ax.set_title("Correlation Matrix", fontsize=20)
plt.show()
df[df["Age"] == 18]
df.head()
df = df[df["Age"] > 18]
df.shape
df.head()
df["IsActiveMember"].value_counts()
df["Agecat"] = pd.qcut(df["Age"] ,5)
df.groupby("Agecat")["Exited"].value_counts()
df[df["Balance"] == 0.0]["Exited"].value_counts()
df.groupby(["Geography","Gender"])["Exited"].value_counts()
df.head()
df[df["Age"] > 90]
df.Age.max()
import plotly.express as px
fig = px.bar(df,y = "Exited", x = "Age" , color = "Geography")
fig.show()
df.groupby(["Gender"])["Exited"].value_counts().plot.barh();
df.head()
df.groupby("Gender")["Exited"].value_counts()
df.groupby("IsActiveMember")["Exited"].value_counts()
df.groupby("Gender")["Exited"].value_counts().plot.pie(autopct = "%.1f");
df.isnull().sum()
df["Agecat"].value_counts()
df.head()
dummies = pd.get_dummies(df[["Gender","Geography"]],drop_first = True)
dummies.head()
df = pd.concat([df,dummies],axis = 1)
df.head()
df.drop(["Surname","Geography","Gender"],axis =1 , inplace = True)
df.drop("CustomerId",axis = 1,inplace = True)
df.drop("Agecat",axis = 1,inplace = True)


df.head()
from sklearn.preprocessing import StandardScaler,MinMaxScaler,RobustScaler,Normalizer

scaler = RobustScaler()
X = df.drop("Exited",axis= 1)
y = df["Exited"]
X.head()
dummies = X[["NumOfProducts","HasCrCard","IsActiveMember","Gender_Male","Geography_Germany","Geography_Spain"]]
X = X.drop(X[["NumOfProducts","HasCrCard","IsActiveMember","Gender_Male","Geography_Germany","Geography_Spain"]],axis = 1)
cols = X.columns
index = X.index
X =scaler.fit_transform(X)
X = pd.DataFrame(X, columns = cols ,index =index)

X.head()
X = pd.concat([X,dummies],axis =1)
X.head()
print("X:",X.shape,"y:",y.shape)
import numpy as np
import pandas as pd 
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score, mean_squared_error, r2_score, roc_auc_score, roc_curve, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import KFold
#All scores of models before tuned
def before_tuned():
    models = []
    models.append(('LR', LogisticRegression()))
    models.append(('KNN', KNeighborsClassifier()))
    models.append(('CART',DecisionTreeClassifier()))
    models.append(('RF', RandomForestClassifier()))
    models.append(('SVM', SVC(gamma='auto')))
    models.append(('XGB', GradientBoostingClassifier()))
    models.append(("LightGBM", LGBMClassifier()))

    # evaluate each model in turn
    results = []
    names = []

    for name, model in models:

            X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2 ,random_state = 12345)
            

            
            model_f = model.fit(X_train,y_train)
            y_pred = model_f.predict(X_test)
            test_r = accuracy_score(y_test,y_pred)
            results.append(test_r)
            names.append(name)
            msg = "%s: %f " % (name, test_r)
            print(msg)

#some predicts before model is tuned.
before_tuned()
def feature_importances():
    models2 = []
    models2.append(('CART', DecisionTreeClassifier( random_state = 12345)))
    models2.append(('RF', RandomForestClassifier( random_state = 12345)))
    models2.append(('XGB', GradientBoostingClassifier( random_state = 12345)))
    models2.append(("LightGBM", LGBMClassifier( random_state = 12345)))
    for name, model in models2:
            X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2 ,random_state = 12345)
            base = model.fit(X_train,y_train)
            y_pred = base.predict(X_test)
            acc_score = accuracy_score(y_test, y_pred)
            feature_imp = pd.Series(base.feature_importances_,
                            index=X.columns).sort_values(ascending=False)

            sns.barplot(x=feature_imp, y=feature_imp.index)
            plt.xlabel('Feature Importance Scores')
            plt.ylabel('Features')
            plt.title(name)
            plt.show()
feature_importances()
cart = DecisionTreeClassifier(random_state = 12345)
cart_params = {"max_depth": [2,3,4,5,10,20, 100, 1000],
              "min_samples_split": [2,10,5,30,50,10]}
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2 ,random_state = 12345)
gs_cv = GridSearchCV(cart, 
                     cart_params, 
                     cv = 10, 
                     n_jobs = -1, 
                     verbose = 2).fit(X_train, y_train)

cart_tuned = DecisionTreeClassifier(**gs_cv.best_params_).fit(X_train,y_train)

y_pred = cart_tuned.predict(X_test)
accuracy_score(y_test,y_pred)
rf = RandomForestClassifier(random_state = 12345)
rf_params = {"n_estimators" :[100,200,500,1000], 
             "max_features": [3,5,7], 
             "min_samples_split": [2,5,10,30],
            "max_depth": [3,5,8,None]}
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2 ,random_state = 12345)
gs_cv = GridSearchCV(rf, 
                     rf_params, 
                     cv = 10, 
                     n_jobs = -1, 
                     verbose = 2).fit(X_train, y_train)

rf_tuned = RandomForestClassifier(**gs_cv.best_params_).fit(X_train,y_train)

y_pred = rf_tuned.predict(X_test)
accuracy_score(y_test,y_pred)
xgb = GradientBoostingClassifier(random_state = 12345)
xgb_params = {"max_depth": [2,3,4,5,8],
             "n_estimators": [100,200,500,1000]}
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2 ,random_state = 12345)
gs_cv = GridSearchCV(xgb, 
                     xgb_params, 
                     cv = 10, 
                     n_jobs = -1, 
                     verbose = 2).fit(X_train, y_train)

xgb_tuned = GradientBoostingClassifier(**gs_cv.best_params_).fit(X_train,y_train)

y_pred = xgb_tuned.predict(X_test)
accuracy_score(y_test,y_pred)
lgbm = LGBMClassifier(random_state = 12345)
lgbm_params = {"learning_rate": [0.01, 0.03, 0.05, 0.1, 0.5],
              "n_estimators": [500, 1000, 1500],
              "max_depth":[3,5,8]}
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2 ,random_state = 12345)
gs_cv = GridSearchCV(lgbm, 
                     lgbm_params, 
                     cv = 10, 
                     n_jobs = -1, 
                     verbose = 2).fit(X_train, y_train)

lgbm_tuned = LGBMClassifier(**gs_cv.best_params_).fit(X_train,y_train)

y_pred = lgbm_tuned.predict(X_test)
accuracy_score(y_test,y_pred)
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, precision_recall_curve, roc_curve, auc, average_precision_score
y_pred_prob = lgbm_tuned.predict_proba(X_test)[:,1]
fig, ax = plt.subplots()
fpr, tpr, thresholds_roc = roc_curve(y_test, y_pred_prob)
roc_auc = auc(fpr,tpr)
ax.plot(fpr,tpr, label = " area = {:0.2f}".format(roc_auc))
ax.plot([0,1], [0,1], 'r', linestyle = "--", lw = 2)
ax.set_xlabel("False Positive Rate", fontsize = 10)
ax.set_ylabel("True Positive Rate", fontsize = 10)
ax.set_title("ROC Curve", fontsize = 18)
ax.legend(loc = 'best')

close_default = np.argmin(np.abs(thresholds_roc - 0.5))
ax.plot(fpr[close_default], tpr[close_default], 'o', markersize = 8)
plt.tight_layout()