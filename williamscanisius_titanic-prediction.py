#importing necessary packages
import pandas as pd
import numpy as np 
import matplotlib
%matplotlib inline
from matplotlib import pyplot as plt
import seaborn as sb
train= pd.read_csv("../input/titanic/train.csv")
train.head()
train.isnull().sum()
train.columns
train = train.drop(train.columns[[0,3,6,7,8,10,11]], axis = 1)
train.head()
#train.dtypes
train["Sex"]= train["Sex"].astype("category")
def sex(x):
    if x =="male":
        return 1
    else:
        return 0
train["Sex"] = train["Sex"].apply(sex)
train.select_dtypes(include=("int64","float64")).describe()
#we do have missing value for Age her will be handdle befor regression
train[["Survived","Sex"]].describe()
#Crossing Freatures : pivot tables
pd.pivot_table(train, "Survived", index="Pclass", columns="Sex", aggfunc=np.mean, margins=True,margins_name="PropTotal")
pd.pivot_table(train, "Age", index="Pclass", columns="Sex", aggfunc=np.mean, margins=True,margins_name="PropTotal")
#We notice here that survied man are 3 year old than survived women. Also The survived pepople age decrease as
#Pclass went from 1 to 3. That me show that people in class 1 have best conditions than the remain. This
#sound normal as theh best the class is the cheaper the Fare class is(take a look on it down) 
pd.pivot_table(train,"Fare" ,["Pclass"], aggfunc=np.mean)
#we can notice that class 1 fare is 84/13 times cheaper than class 3 fare
#looking at the survied dist by Pcal or sex
import plotly.express as px
fig = px.box(train, x="Survived", y="Age", color="Sex",facet_col="Pclass")
fig.update_layout(autosize=True)
fig.show()
#As the Pclass is function of the fare i decide to only keep Pclass in ma modelisation
#Choosing features for model
df = train.drop(train.columns[4], axis = 1)
df.head()
df.describe()
#How missing value  in Age
age_null = df[df['Age'].isnull()]
#so we have among 891 records thsi is more than 10%
age_null.shape[0]
df["Age"] = df["Age"].fillna(df["Age"].mean(skipna=True))#fil na with mean(Age)
df.describe()
#creating the target and de labels
df_labels = df.drop(df.columns[0], axis = 1)
df_labels.head()
df_target = df.iloc[:,0]
df_target.head()
#We first logisticRegression from Sklearn to build and fit the model
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
model1 = LogisticRegression()
model1.fit(df_labels,df_target)
#our model is build and we can how test it performance and predict values of test dataset
#First of all we have to select out labels on test dataset
test= pd.read_csv("../input/titanic/test.csv")
test.head()
test.describe()
test["Age"] = test["Age"].fillna(test["Age"].mean(skipna=True))#fil na with mean(Age)
x_test = test.iloc[:,[0,1,3,4]]
x_test.head()
x_test["Sex"] = x_test["Sex"].apply(sex)
x_test.head()
x_for_pred = x_test.iloc[:,[0,1,2,3]]
x_for_pred.head()
target_predic = model1.predict(x_for_pred.iloc[:,[1,2,3]])
target_predic
full_pred_data = x_for_pred
full_pred_data["survived"] = target_predic
full_pred_data.head()
#Description of my prediction
pd.pivot_table(full_pred_data, "survived", ["Pclass"], aggfunc=np.mean)
pd.pivot_table(full_pred_data, "survived", ["Sex"], aggfunc=np.mean)
prediction = full_pred_data.iloc[:,[0,4]]
prediction.head()
prediction.set_index("PassengerId")
model1.score(df_labels,df_target)
from sklearn.model_selection import GridSearchCV
from sklearn import linear_model
log = linear_model.LogisticRegression(penalty ='l2',solver = 'newton-cg',
                                      max_iter = 1000,l1_ratio=None,
                                      multi_class ='ovr')
hyperparameters = dict(C=np.logspace(0, 4, 10))
clf = GridSearchCV(log, hyperparameters,cv = 5, verbose=0)

clf.fit(df_labels,df_target)
clf.estimator
target_predic2 =clf.predict(x_for_pred.iloc[:,1:])
# Create regularization penalty space
penalty = ['l1', 'l2',"elasticnet","none"]

# Create regularization hyperparameter space
C = np.logspace(0, 4, 10)

#choosen solver
solver = ['newton-cg', 'lbfgs', 'liblinear', 'saga']
# Create hyperparameter options
hyperparameters = dict(cv = 5,C=np.logspace(0, 4, 10), penalty=['l1', 'l2'])
log2 = LogisticRegressionCV(Cs=10, fit_intercept=True, cv=5, dual=False, penalty=["l1","l2","elasticnet","none"], scoring='f1', 
                     solver=['liblinear','newton-cg', 'lbfgs', 'sag', 'saga'],
                     tol=0.0001, max_iter=1000, class_weight=None, n_jobs=None, verbose=0, 
                     refit=True, intercept_scaling=1.0, multi_class='auto', random_state=True, l1_ratios=None)
log2 = LogisticRegressionCV()
target_predic3 = log2.fit(df_labels,df_target)
#first prediction count
prediction["survived"].value_counts()
target_predic3 = log2.predict(x_for_pred.iloc[:,1:])

full_pred_data2 = x_for_pred
full_pred_data2["survived"] = target_predic2
prediction2 = full_pred_data2.iloc[:,[0,4]]
prediction2.head()
prediction2["survived"].value_counts()