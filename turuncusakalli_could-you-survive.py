import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

warnings.filterwarnings("ignore")



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
train_df = pd.read_csv("/kaggle/input/titanic/train.csv")

test_df = pd.read_csv("/kaggle/input/titanic/test.csv")

test_PassengerId = test_df["PassengerId"]
train_df.shape
train_df.head()
train_df.info()
train_df.describe()
def bar_plot(column_name):



    # count number of categorical variable(value/sample)

    number_of_values=train_df[column_name].value_counts()



    # visualize

    plt.figure(figsize=(9,3))

    plt.bar(number_of_values.index,number_of_values)

    plt.xticks(number_of_values.index, number_of_values.index.values)

    plt.ylabel("Frequency")

    plt.title(column_name)

    plt.show()

    print("{}: \n {}".format(column_name,number_of_values))
cat_variables_sense=["Survived", "Sex", "Pclass", "Embarked", "SibSp", "Parch"]

for i in cat_variables_sense:

    bar_plot(i)
cat_variables_nonsense = ["Cabin", "Name", "Ticket"]

cat_variables= cat_variables_sense + cat_variables_nonsense
def hist_plot(column_name):

    

    plt.figure(figsize=(9,3))

    plt.hist(train_df[column_name],bins=50)

    plt.xlabel("column_name")

    plt.ylabel("Frequency")

    plt.title("{} distribution with hist".format(column_name))

    plt.show()
num_variables=["Fare", "Age","PassengerId"]

for i in num_variables:

    hist_plot(i)
train_df[["Pclass","Survived"]].groupby("Pclass").mean().sort_values(by="Survived",ascending=False)
train_df[["Sex","Survived"]].groupby("Sex").mean().sort_values(by="Survived",ascending=False)
train_df[["SibSp","Survived"]].groupby("SibSp").mean().sort_values(by="Survived",ascending=False)
train_df[["Parch","Survived"]].groupby("Parch").mean().sort_values(by="Survived",ascending=False)
def outlier_detection(df,feature):

    df[feature]=sorted(df[feature])

    Q1, Q3 = np.percentile(df[feature] , [25,75])

    IQR = Q3 - Q1

    lower_range = Q1 - (1.5 * IQR)

    upper_range = Q3 + (1.5 * IQR)

    outliers=[]

    for i in df[feature]:

        if (i < lower_range) | (i > upper_range):

            outliers.append(i)

    print(len(outliers)," values detected and fixed of", feature, "feature.")

    df[feature][df.loc[:,feature]<lower_range]=lower_range

    df[feature][df.loc[:,feature]>upper_range]=upper_range

    

    return df[feature]
for col in num_variables:

    train_df[col]= outlier_detection(train_df,col) # Main DF CHANGED

    test_df[col]=outlier_detection(test_df,col) # Main DF CHANGED
train_df_len = len(train_df)

train_df = pd.concat([train_df,test_df],axis = 0).reset_index(drop = True)  # Main DF CHANGED
import missingno as msno

msno.matrix(train_df);
train_df.isnull().sum()
import statistics as stats

train_df["Embarked"] = train_df["Embarked"].fillna(stats.mode(train_df["Embarked"])) # Main DF CHANGED

train_df[train_df["Embarked"].isnull()]
train_df["Fare"] = train_df["Fare"].fillna(stats.mode(train_df["Fare"]))
train_df["Sex"] = [1 if i == "male" else 0 for i in train_df["Sex"]] # Main DF CHANGED

sns.heatmap(train_df[["Age","Sex","SibSp","Parch","Pclass"]].corr(), annot = True, cmap = "coolwarm")

plt.show()
index_nan_age = list(train_df["Age"][train_df["Age"].isnull()].index)

for i in index_nan_age:

    age_pred = (train_df["Age"][((train_df["SibSp"] == train_df.iloc[i]["SibSp"]) &(train_df["Parch"] == train_df.iloc[i]["Parch"]) & \

                                 (train_df["Pclass"] == train_df.iloc[i]["Pclass"]))].median())

    

    # sometimes some values can't fill with upper method. so we will impute them with that column's mean

    age_med = train_df["Age"].median()

    if not np.isnan(age_pred):

        train_df["Age"].iloc[i] = age_pred # Main DF CHANGED

    else:

        train_df["Age"].iloc[i] = age_med # Main DF CHANGED

train_df[train_df["Age"].isnull()]
len(train_df.Cabin), train_df.Cabin.isnull().sum()
train_df.drop(labels = ["Cabin"], axis = 1, inplace = True) # Main DF CHANGED
g = sns.factorplot(x = "SibSp", y = "Survived", data = train_df, kind = "bar", size = 6)

g.set_ylabels("Survived Probability")

plt.show()
g = sns.factorplot(x = "Parch", y = "Survived", kind = "bar", data = train_df, size = 6)

g.set_ylabels("Survived Probability")

plt.show()
train_df["FamilyMembers"]=train_df["Parch"]+train_df["SibSp"]+1 # Main DF CHANGED

train_df.drop(columns=["Parch","SibSp"], axis=1, inplace=True)

train_df
g = sns.factorplot(x = "FamilyMembers", y = "Survived", data = train_df, kind = "bar")

g.set_ylabels("Survival")

plt.show()
train_df["FamilyMembers"]=[1 if i < 4.5 else 0 for i in train_df["FamilyMembers"]] # Main DF CHANGED

train_df["FamilyMembers"].head()
sns.countplot(x = "FamilyMembers", data = train_df)

plt.show()
g = sns.factorplot(x = "Pclass", y = "Survived", data = train_df, kind = "bar", size = 6)

g.set_ylabels("Survived Probability")

plt.show()
train_df.Pclass.dtypes
train_df["Pclass"] = train_df["Pclass"].astype("category") 

train_df = pd.get_dummies(train_df, columns= ["Pclass"]) # Main DF CHANGED

train_df.head()
g = sns.FacetGrid(train_df, col = "Survived")

g.map(sns.distplot, "Age", bins = 25)

plt.show()
train_df["Title"] = [i.split(".")[0].split(",")[-1].strip() for i in train_df["Name"]] # Main DF CHANGED
train_df["Title"].head()
sns.countplot(x="Title", data = train_df)

plt.xticks(rotation = 60)

plt.show()
# convert to categorical

train_df["Title"] = train_df["Title"].replace(["Lady","the Countess","Capt","Col","Don","Dr","Major","Rev","Sir","Jonkheer","Dona"],"other") # Main DF CHANGED

train_df["Title"] = ( [0 if i == "Master" else 1 if i == "Miss" or i == "Ms" or \

                       i == "Mlle" or i == "Mrs" else 2 if i == "Mr" else 3 for i in train_df["Title"]] )  # Main DF CHANGED

train_df["Title"].head()
sns.countplot(x="Title", data = train_df)

plt.xticks(rotation = 60)

plt.show()
train_df.drop(labels = ["Name"], axis = 1, inplace = True) # Main DF CHANGED
train_df = pd.get_dummies(train_df,columns=["Title"]) # Main DF CHANGED

train_df.head()
sns.countplot(x = "Embarked", data = train_df);
train_df=pd.get_dummies(train_df, columns=["Embarked"]) # Main DF CHANGED

train_df
train_df.Ticket.head(15)
tickets = []

for i in list(train_df.Ticket):

    if not i.isdigit():

        tickets.append(i.replace(".","").replace("/","").strip().split(" ")[0])

    else:

        tickets.append("X")

train_df["Ticket"] = tickets # Main DF CHANGED

train_df["Ticket"].unique()
from category_encoders import BinaryEncoder

encoder=BinaryEncoder(cols=["Ticket"])

train_df=encoder.fit_transform(train_df) # Main DF CHANGED

train_df
sns.countplot(x="Sex", data=train_df);
train_df["Sex"] = train_df["Sex"].astype("category")

train_df = pd.get_dummies(train_df, columns=["Sex"]) # Main DF CHANGED

train_df.head()
train_df.drop(labels = ["PassengerId"], axis = 1, inplace = True) # Main DF CHANGED
test = train_df[train_df_len:]

test.drop(labels = ["Survived"],axis = 1, inplace = True) # Main DF CHANGED
from sklearn.model_selection import train_test_split

train = train_df[:train_df_len] # Main DF CHANGED

X_train = train.drop(labels = "Survived", axis = 1) # Main DF CHANGED

y_train = train["Survived"] # Main DF CHANGED

X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size = 0.33, random_state = 42) # Main DF CHANGED



print( "X_train", len(X_train), "\nX_test", len(X_test), "\ny_train", len(y_train), "\ny_test", len(y_test), "\ntest", len(test) )
def models(X_train,Y_train):

    

    #use logistic regression

    from sklearn.linear_model import LogisticRegression

    log=LogisticRegression(random_state=42)

    log.fit(X_train,Y_train)

    

    #use KNeighbors

    from sklearn.neighbors import KNeighborsClassifier

    knn=KNeighborsClassifier(n_neighbors=5,metric="minkowski",p=2)

    knn.fit(X_train,Y_train)

    

    #use SVC (linear kernel)

    from sklearn.svm import SVC

    svc_lin=SVC(kernel="linear",random_state=42,probability=True)

    svc_lin.fit(X_train,Y_train)

    

    #use SVC (RBF kernel)

    svc_rbf=SVC(kernel="rbf",random_state=42,probability=True)

    svc_rbf.fit(X_train,Y_train)

    

    #use GaussianNB

    from sklearn.naive_bayes import GaussianNB

    gauss=GaussianNB()

    gauss.fit(X_train,Y_train)

    

    #use Decision Tree

    from sklearn.tree import DecisionTreeClassifier

    tree=DecisionTreeClassifier(criterion="entropy",random_state=42)

    tree.fit(X_train,Y_train)

    

    #use Random Forest Classifier

    from sklearn.ensemble import RandomForestClassifier

    forest=RandomForestClassifier(n_estimators=10,criterion="entropy",random_state=42)

    forest.fit(X_train,Y_train)

    

    # use Hist Gradient Boosting Classifier

    from sklearn.experimental import enable_hist_gradient_boosting

    from sklearn.ensemble import HistGradientBoostingClassifier

    histgrad=HistGradientBoostingClassifier()

    histgrad.fit(X_train,y_train)

    

    # use GBM

    from sklearn.ensemble import GradientBoostingClassifier

    gbm=GradientBoostingClassifier()

    gbm.fit(X_train,y_train)

    

    # use XGBoost

    #!pip install xgboost

    from xgboost import XGBClassifier

    xgboost=XGBClassifier()

    xgboost.fit(X_train,y_train)

    

    # use LightGBM

    #!conda install -c conda-forge lightgbm

    from lightgbm import LGBMClassifier

    lightgbm=LGBMClassifier()

    lightgbm.fit(X_train,y_train)



    #print the training scores for each model

    print('[0] Logistic Regression Training Score:',log.score(X_train,Y_train))

    print('\n[1] K Neighbors Training Score:',knn.score(X_train,Y_train))

    print('\n[2] SVC Linear Training Score:',svc_lin.score(X_train,Y_train))

    print('\n[3] SVC RBF Training Score:',svc_rbf.score(X_train,Y_train))

    print('\n[4] Gaussian Training Score:',gauss.score(X_train,Y_train))

    print('\n[5] Decision Tree Training Score:',tree.score(X_train,Y_train))

    print('\n[6] Random Forest Training Score:',forest.score(X_train,Y_train))

    print('\n[7] Hist Gradient Boosting Training Score:',histgrad.score(X_train,Y_train))

    print('\n[8] Gradient Boosting Training Score:',gbm.score(X_train,Y_train))

    print('\n[9] XGBoost Training Score:',xgboost.score(X_train,Y_train))

    print('\n[10] Light GBM Training Score:',lightgbm.score(X_train,Y_train))

    

    return log,knn,svc_lin,svc_rbf,gauss,tree,forest,histgrad,gbm,xgboost,lightgbm
log,knn,svc_lin,svc_rbf,gauss,tree,forest,histgrad,gbm,xgboost,lightgbm=models(X_train,y_train)
models=[log,knn,svc_lin,svc_rbf,gauss,tree,forest,histgrad,gbm,xgboost,lightgbm]

models_name=["log","knn","svc_lin","svc_rbf","gauss","tree","forest","histgrad","gbm","xgboost","lightgbm"]
before_tune_accuracy_score={}

def accuracy_score_calculator(model,model_name):

    from sklearn.metrics import accuracy_score

    y_pred=model.predict(X_test)

    before_tune_accuracy_score[model_name]=accuracy_score(y_test,y_pred)
before_tune_roc_score={}

def roc_score_calculator(model,model_name):

    from sklearn.metrics import roc_auc_score

    y_pred=model.predict_proba(X_test)[:,1]

    before_tune_roc_score[model_name]=roc_auc_score(y_test,y_pred)
for i in range(len(models)):

    roc_score_calculator(models[i],models_name[i])

    accuracy_score_calculator(models[i],models_name[i])
size=np.arange(len(models))

plt.bar(size-0.2, before_tune_roc_score.values(), color='g', width=0.4, tick_label=models_name)

plt.bar(size+0.2, before_tune_accuracy_score.values(),color='b', width=0.4, tick_label=models_name)

plt.legend(["Before Roc Score", "Before Accuracy Score"]);
from sklearn.metrics import plot_roc_curve

plt.figure(figsize=(10,10))

ax = plt.gca()

log_disp = plot_roc_curve(log, X_test, y_test, ax=ax, alpha=0.8)

knn_disp = plot_roc_curve(knn, X_test, y_test, ax=ax, alpha=0.8)

svc_lin_disp = plot_roc_curve(svc_lin, X_test, y_test, ax=ax, alpha=0.8)

svc_rbf_disp = plot_roc_curve(svc_rbf, X_test, y_test, ax=ax, alpha=0.8)

gauss_disp = plot_roc_curve(gauss, X_test, y_test, ax=ax, alpha=0.8)

tree_disp = plot_roc_curve(tree, X_test, y_test, ax=ax, alpha=0.8)

forest_disp = plot_roc_curve(forest, X_test, y_test, ax=ax, alpha=0.8)

histgrad_disp = plot_roc_curve(histgrad, X_test, y_test, ax=ax, alpha=0.8)

gbm_disp = plot_roc_curve(gbm, X_test, y_test, ax=ax, alpha=0.8)

xgboost_disp = plot_roc_curve(xgboost, X_test, y_test, ax=ax, alpha=0.8)

lightgbm_disp = plot_roc_curve(lightgbm, X_test, y_test, ax=ax, alpha=0.8)

plt.legend(loc = 'lower right', prop={'size': 16})

plt.show()
from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.ensemble import GradientBoostingClassifier



classifier = [ SVC(random_state = 42) , LogisticRegression(random_state = 42), 

              GradientBoostingClassifier(random_state = 42) ]



gbm_param_grid = { "loss":["deviance"], "learning_rate": [0.01, 0.05, 0.1, 0.2], 

                  "min_samples_split": np.linspace(0.1, 0.5, 12), "min_samples_leaf": np.linspace(0.1, 0.5, 12), 

                  "max_depth":[3,5,8], "subsample":[0.5, 0.8, 0.9, 1.0], "n_estimators":[10] }



svc_param_grid = {"kernel" : ["linear"], "probability":[True], "gamma": [0.001, 0.01, 0.1, 1] }



logreg_param_grid = {"C":np.logspace(-3,3,7), "penalty": ["l1","l2"], "solver": ['liblinear'] }



classifier_param = [svc_param_grid, logreg_param_grid, gbm_param_grid ]
from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import StratifiedKFold



cv_result = []

best_estimators = []

for i in range(len(classifier)):

    clf = GridSearchCV(classifier[i], param_grid=classifier_param[i], cv = StratifiedKFold(n_splits = 10), scoring = "accuracy", n_jobs = -1, verbose=1)

    clf.fit(X_train,y_train)

    cv_result.append(clf.best_score_)

    best_estimators.append(clf.best_estimator_)

    print(cv_result[i])
cv_results = pd.DataFrame({"Cross Validation Means":cv_result, 

                           "ML Models":["LogisticRegression", "GradientBoostingClassifier", "SVC"]})



g = sns.barplot("Cross Validation Means", "ML Models", data = cv_results)

g.set_xlabel("Mean Accuracy")

g.set_title("Cross Validation Scores");
from sklearn.metrics import accuracy_score

from sklearn.ensemble import VotingClassifier

votingC = VotingClassifier(estimators = [("dt",best_estimators[0]),

                                        ("rfc",best_estimators[1]),

                                        ("lr",best_estimators[2])],

                                        voting = "soft", n_jobs = -1)

votingC = votingC.fit(X_train, y_train)

accuracy_score(votingC.predict(X_test),y_test)
test_survived = pd.Series(votingC.predict(test), name = "Survived").astype(int)

results = pd.concat([test_PassengerId, test_survived],axis = 1)

results.to_csv("titanic.csv", index = False)