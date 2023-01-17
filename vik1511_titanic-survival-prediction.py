# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from scipy import stats

import seaborn as sns

from collections import Counter



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

import warnings

warnings.filterwarnings('ignore')



print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
traindf = pd.read_csv("../input/train.csv")

testdf  = pd.read_csv("../input/test.csv")

Ids = testdf.PassengerId
target = traindf['Survived']

all_data = [traindf,testdf]
numerical_features = [col for col in traindf.drop("PassengerId",axis=1)._get_numeric_data().columns ]

catg_cols = traindf.select_dtypes(include=["object"]).columns.values
outlier_indices=[]

indices =[]

for col in numerical_features:

    q1,q3 = np.percentile(traindf[col],[25,75])

    IQR = q3-q1

    upper_bound = q3+1.5*IQR

    lower_bound = q1-1.5*IQR

    data = traindf[(traindf[col]>upper_bound)] | traindf[traindf[col]<lower_bound]

    outlier_indices.extend(data.index)

outlier_indices = Counter(outlier_indices)

outlier_indices = [k for k,v in outlier_indices.items() if v>2]

outlier_indices

traindf.loc[outlier_indices]
traindf = traindf.drop(outlier_indices,axis=0).reset_index(drop=True)

target = target.drop(outlier_indices,axis=0).reset_index(drop=True)
portion_survived   = (traindf[traindf['Survived']==1]["Survived"].count()/traindf.shape[0])*100

portion_not_survived = (traindf[traindf['Survived']==0]["Survived"].count()/traindf.shape[0])*100

fig = plt.figure(figsize=(3,4))

sns.countplot(traindf['Survived'])

plt.title("Survival")

plt.show()

survival_df  =pd.DataFrame({"person_survived":"%0.2f"%portion_survived,"person not survived":"%0.2f"%portion_not_survived},index=["survival(%)"])

survival_df
plt.figure(figsize=(6,4))

sns.barplot(traindf["Sex"],y=traindf["Survived"])

plt.ylabel("Survival prob")

plt.show()

traindf[["Sex","Survived"]].groupby("Sex").mean()
classp1_pass = (traindf[traindf["Pclass"]==1].reset_index(drop=True))

classp2_pass = (traindf[traindf["Pclass"]==2].reset_index(drop=True))

classp3_pass= (traindf[traindf["Pclass"]==3].reset_index(drop=True))

proportion_p1 = (classp1_pass[classp1_pass["Survived"]==1]["Survived"].count()/len(classp1_pass))*100

proportion_p2 = (classp2_pass[classp2_pass["Survived"]==1]["Survived"].count()/len(classp2_pass))*100

proportion_p3 = (classp3_pass[classp3_pass["Survived"]==1]["Survived"].count()/len(classp3_pass))*100

survivalC = pd.DataFrame({"Survival(%)":[proportion_p1,proportion_p2,proportion_p3],"Total_passenger":[len(classp1_pass),len(classp2_pass),len(classp3_pass)]},index=["classp1","classp2","classp3"])

fig,ax = plt.subplots(1,2,figsize=(10,5))

sns.barplot(x='Pclass',y='Survived',data=traindf,ax=ax[0])

sns.barplot(x='Pclass',y='Survived',hue="Sex",data=traindf,ax=ax[1])

ax[0].set_title("Pclass distribution")

ax[0].set_ylabel("Survival prob")

ax[1].set_ylabel("Survival prob")

plt.show()

survivalC

for data in [traindf,testdf]:

    data["Sex"] = data["Sex"].apply(lambda x: 1 if x=="male" else 0)

    data["Pclass"] = data["Pclass"].astype(str)
for data in [traindf,testdf]:

    data["total_members"] = data["SibSp"]+data["Parch"]+1

    data['isAlone'] = data["total_members"].apply(lambda x: 1 if x==1 else 0)
plt.figure()

sns.barplot(x = traindf["isAlone"],y=traindf["Survived"])

plt.ylabel("Survival Prob")

plt.show()

print(traindf["Age"].describe())

print("----------------------------")

print(testdf["Age"].describe())
traindf[["Pclass","Age"]].groupby("Pclass").hist()

traindf[["Pclass","Age"]].groupby("Pclass").mean().iloc[1]
import random

for data in [traindf,testdf]:

    st = int(data["Age"].std())

    mean = int(data["Age"].mean())

    data["Age"] = data["Age"].fillna(random.randint(mean-st,mean+st))
plt.figure()

sns.catplot(x='Pclass',y="Age",hue="Survived",data=traindf,kind='violin')

plt.show()

plt.figure()

sns.catplot(x='Survived',y='Age',data=traindf,kind="violin")

plt.show()
for data in [traindf,testdf]:

    data["Age"] = data["Age"].apply(lambda x:"1" if x<=10 else("2" if (x>10 and x<30) else "3"))
traindf["Age"].sample(2)
for data in [traindf,testdf]:

    data["Embarked"] = data["Embarked"].fillna(data["Embarked"].mode())
traindf.info()
traindf.describe()
plt.figure(figsize=(25,6))

sns.catplot(x="Survived",y="Fare",data=traindf,kind='violin')
for data in [traindf,testdf]:

    median = data["Fare"].median()

    bins = pd.IntervalIndex.from_tuples([(0, 10), (10, 50),(50,1000)],closed="left")

    data["Fare"] = data["Fare"].fillna(median)

    data["Fare"] = pd.cut(data["Fare"],bins,labels=[1,2,3])

    
traindf.describe()
drop_columns = ["PassengerId","Name","SibSp","Parch","Ticket","Cabin"]
for data in [traindf,testdf]:

    data.drop(drop_columns,axis=1,inplace=True)
traindf = traindf.drop("Survived",axis=1)
for data in [traindf,testdf]:

    data["total_members"] = data["total_members"].astype(str)
data = pd.concat([traindf,testdf],axis=0)

data = pd.get_dummies(data)

traindf = data[:traindf.shape[0]]

testdf = data[traindf.shape[0]:]
trainX = traindf.values

trainY = target.values

testX = testdf.values

from sklearn.model_selection import train_test_split

trainx,testx,trainy,testy = train_test_split(trainX,trainY,test_size=0.3,random_state=42)
from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import AdaBoostClassifier

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.ensemble import BaggingClassifier

from sklearn.ensemble import ExtraTreesClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.svm import SVC

from xgboost import XGBClassifier

classifier = []

lr = LogisticRegression()

rf = RandomForestClassifier()

abc = AdaBoostClassifier()

gbc = GradientBoostingClassifier()

bc = BaggingClassifier()

etc = ExtraTreesClassifier()

xgbc = XGBClassifier()

svm = SVC()

dtc = DecisionTreeClassifier()

classifier.extend([lr,abc,rf,gbc,bc,etc,xgbc,svm,dtc])
from sklearn.model_selection import KFold,cross_val_score,cross_validate

def scoreModel(n_cv,model):

        kf = KFold(n_cv,random_state=42,shuffle=True).get_n_splits()

        cv_score = cross_validate(model,trainX,trainY,cv=kf)

        return cv_score
cv_results_train=[]

cv_results_test = []

model_param = []

model_time = []

cv_std=[]

for model in classifier:

    score = scoreModel(10,model)

    model_param.append((model.get_params()))

    cv_results_train.append(score["train_score"].mean())

    cv_results_test.append(score["test_score"].mean())

    model_time.append(score["fit_time"].mean())

    cv_std.append(score["test_score"].std())
df = pd.DataFrame({"ML algorithms":[model.__class__.__name__ for model in classifier],"Parameters":model_param,"Accuracy mean(train)":cv_results_train,"Accuracy mean(test)":cv_results_test,

                  "fit time(sec)":model_time,"Cv_std":cv_std})

df.sort_values(by="Accuracy mean(test)",ascending=False).reset_index(drop=True)
plt.figure(figsize=(10,7))

sns.barplot(x='Accuracy mean(test)',y="ML algorithms",data=df,orient="V",**{"xerr":cv_std})
from sklearn.model_selection import GridSearchCV

cv_split = KFold(7,random_state=42,shuffle=True).get_n_splits()

dtc_params={"criterion":["gini","entropy"], "min_samples_split":[2,3,4],"min_samples_leaf":[1,2],"random_state":[0],"max_depth":[100,200,None]}

rf_params={"criterion":["gini","entropy"],"bootstrap" :[True,False],"random_state":[0],"n_estimators":[50,100,300]}

abc_params={"n_estimators":[50,100,300],"learning_rate":[0.01,0.03,0.1,0.3,1],"algorithm":["SAMME.R"],"random_state":[0]}

gbc_params={"loss":["deviance"],"learning_rate":[0.01,0.03,0.1,0.3,1],"n_estimators":[50,300,500],"random_state":[0],"criterion":["friedman_mse"]}

bc_params={"n_estimators":[50,100,300],"oob_score":[True,False],"random_state":[0]}

xgbc_param = {"booster":["dart","gbtree"],"learning_rate":[0.1,0.3,1],"max_depth":[2,3,4],"n_estimators":[100,300],"seed":[0]}
gridSearch = []

gs_dtc = GridSearchCV(dtc,param_grid = dtc_params,scoring="accuracy",cv=cv_split,verbose=1,n_jobs=-1)

gs_rf = GridSearchCV(rf,param_grid=rf_params,scoring="accuracy",n_jobs=-1,cv=cv_split,verbose=1)

gs_abc = GridSearchCV(abc,param_grid=abc_params,scoring="accuracy",n_jobs=-1,cv=cv_split,verbose=1)

gs_gbc = GridSearchCV(gbc,param_grid=gbc_params,scoring="accuracy",n_jobs=-1,cv=cv_split,verbose=1)

gs_bc = GridSearchCV(bc,param_grid=bc_params,scoring="accuracy",n_jobs=-1,cv=cv_split,verbose=1)

gs_xgbc = GridSearchCV(xgbc,param_grid = xgbc_param,scoring="accuracy",n_jobs=-1,cv=cv_split,verbose=1)

gridSearch.extend([gs_dtc,gs_rf,gs_abc,gs_gbc,gs_bc,gs_xgbc])
best_params=[]

best_train_score = []

best_test_score = []

std_train = []

std_test = []

best_index =0 

for gs in gridSearch:

    gs.fit(trainx,trainy)

    best_params.append((gs.best_params_))

    train_score = gs.cv_results_["mean_train_score"][gs.best_index_]

    test_score = gs.cv_results_["mean_test_score"][gs.best_index_]

    best_train_score.extend([train_score])

    best_test_score.extend([test_score])

    std_train.append([gs.cv_results_['std_train_score'][gs.best_index_]])

    std_test.append(gs.cv_results_['std_test_score'][gs.best_index_])
grid_df = pd.DataFrame({"MLA":[model.estimator.__class__.__name__ for model in gridSearch],"Best_params":best_params,

                        "Best train accuracy":best_train_score,"Best Test accuracy":best_test_score,

                        "std_train":std_train,"std_test":std_test}

                      )

print("Performance of models after model tuning ")

grid_df.sort_values(by="Best Test accuracy",ascending=False).reset_index(drop=True)
i =0

for model in gridSearch:

    model.estimator.set_params(**best_params[i])

    i = i+1
from sklearn.ensemble import VotingClassifier

estimators = [

        ("GB",gbc),

        ("RF",rf),

        ("xgb",xgbc),

        ("ABC",abc)

]
vote_hard = VotingClassifier(estimators,voting="soft",weights=[2,1,3,1])

vot_hard_cv = cross_validate(vote_hard,trainx,trainy,cv=cv_split,scoring="accuracy")

vote_hard.fit(trainx,trainy)
vote_hard.score(trainx,trainy)

vot_hard_cv["test_score"]
gs_gbc.best_params_
g = GradientBoostingClassifier(criterion= 'friedman_mse',

 learning_rate= 0.01,

 loss='deviance',

 n_estimators=500,

 random_state= 0)
xgbc.fit(trainx,trainy)

g.fit(trainx,trainy)

g.score(testx,testy)
xgbc.score(testx,testy)
pred = g.predict(testX)
df_res = pd.DataFrame({"PassengerId":Ids,"Survived":pred})
output = df_res.to_csv("Output.csv",index=False)