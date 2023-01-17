# load pandas, numpy, matplotlib, seaborn, re module

import pandas as pd

import numpy as np



# Visualisation

import re

import seaborn as sns

import matplotlib.pyplot as plt

import matplotlib as mpl

import matplotlib.pyplot as plt

import matplotlib.pylab as pylab

import seaborn as sns



# Configure visualisations

%matplotlib inline

sns.set_style( 'white' )

sns.set_palette("muted")
train=pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")

combined = train.append(test)

train.shape, test.shape, combined.shape
Target = train.Survived

fig, ax = plt.subplots(1,2,figsize=(10,5),)

sns.countplot(x=Target,ax=ax[0])

train.Survived.value_counts().plot(kind="pie")

# In training set, the more passenger were dead.
train.info()
Missing = combined.isnull().sum()

Existed = combined.notnull().sum()

Missing = pd.concat([Missing,Missing+Existed],keys=["Missing","Total"],axis=1).sort_values(by="Missing",ascending=False)

Missing[Missing.Missing>0]
# factors used to collect key features which has contribution to target.

factors={}
factors["Survived"]=1

factors["PassengerId"]=1

#append feature to factors one by one

# two must be selected: Survived, PassengerID. They are submssion required feature.
train.select_dtypes(include=["number"]).describe()
columns = train.select_dtypes(include=["number"]).columns

fig,ax =plt.subplots(2,len(columns),figsize=(21,10))

for i in range(len(columns)):

    col=columns[i]

    data = train[col]

    data = data[data.notnull()]

    #data.hist(ax=ax[0,i])

    #print(data.name)

    sns.distplot(data,ax=ax[0,i],kde=False,label=None)

    ax[0,i].set_xlabel("")

    ax[0,i].set_xticklabels([])

    #sns.regplot(x=train[col], y=train.Survived,ax=ax[1,i]);

    if col!="Survived":

        sns.regplot(x=train[col], y=train.Survived,ax=ax[1,i],logistic=True,marker="+");

    else:

        sns.regplot(x=train[col], y=train.Survived,ax=ax[1,i],marker="+");

#plt.subplots_adjust(hspace=0,wspace=2)
# following columns identified as key features

# Pasenger ID was excluded as it looks like average distrubition. It hardly has any trends with the Survived. 

# Fare looks a bit of strange when higher than 200. It will be further explored later.



columns = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare'] 



for i in columns:

    factors[i]=1

factors
train.select_dtypes(include=["O"]).describe()
columns = ["Survived","Sex","Embarked"]



fig,ax =plt.subplots(2,len(columns),figsize=(10,5))



for i in range(len(columns)):

    col=columns[i]

    data = train[col]

    data = data[data.notnull()]

    

    sns.countplot(x=col, data=train,ax=ax[0,i])

    ax[0,i].set_xlabel("")

    ax[0,i].set_xticklabels([])

    if col!="Survived":

        sns.pointplot(x=train[col],y=train.Survived,ax=ax[1,i]);

    else:

        pass

plt.subplots_adjust(hspace=0.5,wspace=0.5)
# following columns identified as key factors

columns = ["Sex","Embarked"] 



for i in columns:

    factors[i]=1

len(factors) ,factors
# Thanks for kaggle turorial 

# Titanic Data Science Solutions Python Notebook and other online resources. I saved time on this pieces.



combined["t_titles"] = combined.Name.str.extract("([A-Za-z]+)\.",expand = False)



# a map of more aggregated titles

title_map = { 

            "Capt":       "Officer",

            "Col":        "Officer",

            "Major":      "Officer",

            "Jonkheer":   "Royalty",

            "Don":        "Royalty",

            "Sir" :       "Royalty",

            "Dr":         "Officer",

            "Rev":        "Officer",

            "the Countess":"Royalty",

            "Dona":       "Royalty",

            "Mme":        "Mrs",

            "Mlle":       "Miss",

            "Ms":         "Mrs",

            "Mr" :        "Mr",

            "Mrs" :       "Mrs",

            "Miss" :      "Miss",

            "Master" :    "Master",

            "Lady" :      "Royalty",

            "Countess":   "Royalty"

            }

combined["t_titles"] =combined["t_titles"].map(title_map)

combined["t_titles"].value_counts()
if combined.t_titles.isnull().sum()==0:

    factors["t_titles"]=1

    print(len(factors),"\n",factors)

else:

    print("Missing data.")
#print(combined.Ticket.head())

t_pre =combined.Ticket.str.extract("(?P<Pre>[A-Za-z/.]+[0-9]?)",expand=False)

t_pre = t_pre.str.replace("[/.]","")

sns.pointplot(x=t_pre,y=combined.Survived) 
t_pre[t_pre.isnull()]="NA"  #fill the pre with "NA" for whose ticket no pre information.



if t_pre.isnull().sum()==0:

    combined["t_pre"] = t_pre

    factors["t_pre"]=1

    print(len(factors),"\n",factors)

else:

    print("missing data")

    print(combined[combined["t_pre"].isnull()].head(3))
t_num =combined.Ticket.str.extract("(?P<Num>[0-9]{3,10})",expand=False)

t_num =t_num.fillna("9999") #there is no ticket number 9999, so use it as special number for not ticket number passenger.

fig,ax =plt.subplots(1,1,figsize=(5,5))

sns.regplot(t_num.astype(int),combined.Survived)

t_num.astype(int).sort_values(ascending=False).head()
if t_num.isnull().sum()==0:

    combined["t_num"]=t_num.astype(int)

    factors["t_num"]=1

    print(len(factors),"\n",factors)

else:

    print("Missing Data")

    print(combined[combined["t_num"].isnull().head(3)])
#two differenct ways to extract the number length & value info. I choiced both feature. 

#In later featuring importance, the num_start show high important. one might be removed if further polish needed.



t_num_log10 = np.log10(combined["t_num"]).astype("int") # the length calculated by log10 with higher confidence  

#t_num_len = t_num.str.len()  

t_num_start=t_num.str.get(0)

#t_num_start2=t_num.str.slice(0,1)



#cols = [t_num_log,t_num_len,t_num_start,t_num_start2]

cols = [t_num_log10,t_num_start]



fig,ax=plt.subplots(2,len(cols),figsize=(10,5))

Y = combined.Survived

for i in range(len(cols)):

    col=cols[i]

    sns.regplot(col.astype("int"),Y,ax=ax[0,i])    

    sns.pointplot(col,Y,ax=ax[1,i])

mask1 = t_num_start.isnull()

mask2 = t_num_log10.isnull()

if mask1.sum()==0 and mask2.sum()==0 :

    combined["t_num_start"]=t_num_start

    combined["t_num_log10"]=t_num_log10

    factors["t_num_start"]=1

    factors["t_num_log10"]=1

    print(len(factors),"\n",factors)

else:

    print("Missing Data")

    print(combined.loc[mask1])

    print(combined.loc[mask2])

    

    
t_nShare = combined.Ticket.value_counts()

t_nShare=combined.Ticket.replace(t_nShare.index.values,t_nShare.tolist())

fig,ax = plt.subplots(1,2,figsize = (10,5))

sns.pointplot(x=t_nShare,y=combined.Survived,ax=ax[0])

sns.regplot(x=t_nShare,y=combined.Survived,ax=ax[1])
mask = t_nShare.isnull()



if mask.sum()==0:

    combined["t_nShare"]=t_nShare

    factors["t_nShare"]=1

    print(len(factors),"\n",factors)

else:

    print("Missing Data")

    print(combined.loc[mask])

    

    
FamilySize = combined["SibSp"] + combined["Parch"]



fig,ax = plt.subplots(1,2,figsize = (10,5))

sns.pointplot(x=FamilySize,y=combined.Survived,ax=ax[0])

sns.regplot(x=FamilySize,y=combined.Survived,ax=ax[1])
mask = FamilySize.isnull()

if mask.sum()==0:

    combined["FamilySize"]=FamilySize

    factors["FamilySize"]=1

    print(len(factors),"\n",factors)

else:

    print("Missing Data")

    print(combined.loc[mask])  

mask = combined.Fare.isnull()

combined[mask]

# use R2 coefficient to determine to feature contribution to Fare

(combined.corr()**2).sort_values(by="Fare",ascending=False).Fare


pd.pivot_table(combined[combined.t_nShare==1],columns="Pclass",values="Fare",index=["t_nShare","Parch"],aggfunc="median")
mask1 = combined.t_titles == "Mr" 

mask2 = combined.Pclass ==3

mask3 = combined.Parch == 0

mask4 = combined.t_nShare ==1

mask = mask1 & mask2 & mask3 & mask4



fill_fare =combined[mask].Fare.median()

fill_fare
Fare=combined["Fare"].fillna(fill_fare)



mask = Fare.isnull()

if mask.sum()==0:

    combined["Fare"]=Fare

    factors["Fare"]=1

    print(len(factors),"\n",factors)

else:

    print("Missing Data")

    print(combined.loc[mask,"Fare"])

    
combined[combined.Fare.isnull()]
f_Single=combined.Fare/combined.t_nShare



fig,ax = plt.subplots(1,2,figsize = (10,5))

sns.distplot(f_Single,ax=ax[0],kde=False)

sns.regplot(x=f_Single,y=combined.Survived,ax=ax[1],logistic=True)
mask = f_Single.isnull()



if mask.sum()==0:

    combined["f_Single"]=f_Single

    factors["f_Single"]=1

    print(len(factors),"\n",factors)

else:

    print("Missing Data")

    print(combined.loc[mask])

    
# too much of missing, set all missing as "NA"

c_cabin = combined.Cabin.fillna("NA")

c_pre =c_cabin.str.extract("(?P<Pre>[A-Za-z/.])",expand=False)
fig,ax=plt.subplots(1,2,figsize=(10,5))

sns.pointplot(x=c_pre,y=combined.Survived,ax=ax[0])

sns.barplot(x=c_pre,y=combined.Survived,ax=ax[1])
mask = c_pre.isnull()



if mask.sum()==0:

    combined["c_pre"]=c_pre

    factors["c_pre"]=1

    print(len(factors),"\n",factors)

else:

    print("Missing Data")

    print(combined.loc[mask])
print(combined[combined.Embarked.isnull()])

mask1 =combined.Embarked.notnull()

mask2 = combined.f_Single >=40

#mask3 =combined.c_pre=="B" # too much of missing value, do not use it.

mask4 = combined.t_num_start =="1"

mask5 = combined.Pclass ==1

mask6 = combined.t_pre =="NA"

mask = mask1&mask2&mask4 & mask5 & mask6



sns.violinplot(x="t_num_log10",y="Embarked",data=combined[mask],logistic=False)
Embarked=combined["Embarked"].fillna("NA")



mask = Embarked.isnull()



if mask.sum()==0:

    combined["Embarked"]=Embarked

    factors["Embarked"]=1

    print(len(factors),"\n",factors)

else:

    print("Missing Data")

    print(combined.loc[mask])

AgeR2 = combined.corr().Age **2  # R2 = square of corr. R2 is the deterination coefficent

print(AgeR2.sort_values(ascending=False))
pd.pivot_table(data=combined, columns=["t_titles"],index=["Pclass","SibSp"],values="Age")
g=sns.FacetGrid(data=combined,col="t_titles",row="Pclass")

g.map(sns.distplot, "Age")
group_key = ["Pclass","Embarked","t_titles"] # Remove f_single as more missing value returned.

fill_Age_mean = lambda g: g.fillna(g.mean())

fill_Age_result=combined.groupby(group_key).Age.transform(fill_Age_mean)

mask = fill_Age_result.isnull()



if mask.sum()==0:

    combined["Age"]=fill_Age_result

    factors["Age"]=1

    print(len(factors),"\n",factors)

else:

    print("Missing Data")

    print(combined.loc[mask])
np.log10(50)
from sklearn import preprocessing

le = preprocessing.LabelEncoder()
tmp = combined[list(factors.keys())]

cols = tmp.select_dtypes(include=["O"]).columns
for i in cols[:]:

    print(i)

    le.fit(tmp[i])

    Encoded=le.transform(tmp[i])

    combined[i]=Encoded

combined.head()
combined[list(factors.keys())].info()
day ="2017_9_21"

version="2"

fname = "PreProcess"+ day +"_v"+version+".h5"

combined_pre = combined[list(factors.keys())]

combined_pre.to_hdf(fname,"pre")

combined_pre.shape, pd.read_hdf(fname,"pre").shape
import pandas as pd

import numpy as np



import scipy.stats as stats

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



# Going to use these 5 base models for the stacking

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier

from sklearn.svm import SVC

from sklearn.cross_validation import KFold;

from sklearn.model_selection import cross_val_score



import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.tools as tls

import re

import sklearn

import xgboost as xgb

import warnings

warnings.filterwarnings('ignore')

combined =pd.read_hdf("PreProcess2017_9_21_v2.h5","pre")

train=combined[:891]

test=combined[891:]



drop_col=["Survived","PassengerId"]

X_train=combined.drop(drop_col,axis=1)[:891]

X_test =combined.drop(drop_col,axis=1)[891:]

y_train =combined["Survived"][:891]

PassengerId = combined["PassengerId"][891:]

x_train_cols=X_train.columns

print(combined.shape,X_train.shape,X_test.shape,y_train.shape,PassengerId.shape)

print(x_train_cols)
from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import chi2, f_classif,mutual_info_classif

#x_train, x_test, y_train

sel = SelectKBest(mutual_info_classif,k=14)

# k less than 10, lead to mean socre descrease

# k = all, the standard dieviation is a bit of high. some feature might be noise

x_train = sel.fit_transform(X_train, y_train)

mask =sel.get_support()

x_test=X_test.loc[:,mask]

print(x_train.shape,x_test.shape)
print("select features:",x_train_cols[mask])

print("Dropped Features:",x_train_cols[~mask])

#improve in future
#Standardization, or mean removal and variance scaling

from sklearn import preprocessing



x_train = preprocessing.scale(x_train)

x_test =preprocessing.scale(x_test)

x_train.shape,x_test.shape
# Some useful parameters which will come in handy later on

ntrain = train.shape[0]

ntest = test.shape[0]

SEED = 42 # for reproducibility

NFOLDS = 5 # set folds for out-of-fold prediction

kf = KFold(ntrain, n_folds= NFOLDS)



# Class to extend the Sklearn classifier

class SklearnHelper(object):

    def __init__(self, clf, seed=0, params=None):

        params['random_state'] = seed

        self.clf = clf(**params)



    def train(self, x_train, y_train):

        self.clf.fit(x_train, y_train)



    def predict(self, x):

        return self.clf.predict(x)

    

    def fit(self,x,y):

        return self.clf.fit(x,y)

    

    def feature_importances(self,x,y):

        return self.clf.fit(x,y).feature_importances_

    

# Class to extend XGboost classifer
def get_oof(clf, x_train, y_train, x_test):

    oof_train = np.zeros((ntrain,))

    oof_test = np.zeros((ntest,))

    oof_test_skf = np.empty((NFOLDS, ntest))



    for i, (train_index, test_index) in enumerate(kf):

        x_tr = x_train[train_index]

        y_tr = y_train[train_index]

        x_te = x_train[test_index]



        clf.train(x_tr, y_tr)



        oof_train[test_index] = clf.predict(x_te)

        oof_test_skf[i, :] = clf.predict(x_test)



    oof_test[:] = oof_test_skf.mean(axis=0)

    return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)



from sklearn.model_selection import learning_curve



def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,

                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5), scoring='accuracy'):

    plt.figure(figsize=(5,5))

    plt.title(title)

    if ylim is not None:

        plt.ylim(*ylim)

    plt.xlabel("Training examples")

    plt.ylabel(scoring)

    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=cv, scoring=scoring,

                                                            n_jobs=n_jobs, train_sizes=train_sizes)

    train_scores_mean = np.mean(train_scores, axis=1)

    train_scores_std = np.std(train_scores, axis=1)

    test_scores_mean = np.mean(test_scores, axis=1)

    test_scores_std = np.std(test_scores, axis=1)

    plt.grid()



    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,

                     train_scores_mean + train_scores_std, alpha=0.1,

                     color="r")

    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,

                     test_scores_mean + test_scores_std, alpha=0.1, color="g")

    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",

             label="Training score")

    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",

             label="Cross-validation score")



    plt.legend(loc="best")

    return plt
# Put in our parameters for said classifiers

# Random Forest parameters

rf_params = {}

# Extra Trees Parameters

et_params = {}

# AdaBoost parameters

ada_params = {}

# Gradient Boosting parameters

#Gradient Boosting {'loss': 'exponential', 'max_depth': 5, 'max_features': 'sqrt', 'min_samples_leaf': 1, 'n_estimators': 100} 

gb_params = {}

# Support Vector Classifier parameters 

svc_params = {}
from sklearn.model_selection import train_test_split

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import classification_report



RF  = RandomForestClassifier()

Ada = AdaBoostClassifier()

GB  = GradientBoostingClassifier()

ET  = ExtraTreesClassifier()

SVM = SVC()

XGB = xgb.XGBClassifier()



ada_tuned_parameters =[{"n_estimators":[500],"algorithm":["SAMME", "SAMME.R"],"learning_rate":[1]}]

gb_tuned_parameters =[{"loss":["deviance", "exponential"],'n_estimators': [100,500], 'max_features': ["sqrt","log2"],

                       'max_depth': [5,7],'min_samples_leaf':[1]}]

et_tuned_parameters = [{'n_estimators': [100,500],"criterion":["gini","entropy"],"max_features":["auto","log2",None],"max_depth":[5,15,None]}]

rf_tuned_parameters = [{'n_estimators': [100,500],"criterion":["gini","entropy"],"max_features":["auto","log2",None],"max_depth":[5,15,None]}]

svm_tuned_parameters = [{'kernel': ['rbf'],'C': [1,10],'gamma': ["auto", 1e-2, 1e-3],"class_weight":["balanced",None]}]

xgb_tuned_parameters = [{'n_estimators': [500],"max_depth":[5],"learning_rate":[0.025],"gamma":[0.01,0.1],"min_child_weight":[7,9]}]



scores = ['f1']



estimators={"Ada Boost":Ada, 

            "Gradient Boosting":GB, 

            "Extra Trees":ET,

            "Support VectorMachine":SVM,

            "XGBooting":XGB}



px_train, px_test, py_train, py_test = train_test_split(x_train, y_train, test_size=0.25, random_state=0)



for score in scores:

    print("# Tuning hyper-parameters for %s" % score)  

    

    clf0 = GridSearchCV(Ada, ada_tuned_parameters, cv=5,scoring='%s_macro' % score)

    clf0.fit(x_train, y_train)

    print("Ada\n",clf0.best_params_,"\n\n")

    ada_params=clf0.best_params_

    

    clf1 = GridSearchCV(SVM, svm_tuned_parameters, cv=5,scoring='%s_macro' % score)

    clf1.fit(x_train, y_train)

    print("SVM\n",clf1.best_params_,"\n\n")

    svc_params=clf1.best_params_

    

    clf2 = GridSearchCV(GB, gb_tuned_parameters, cv=5,scoring='%s_macro' % score)

    clf2.fit(x_train, y_train)

    print("Gradient Boosting",clf2.best_params_,"\n\n")

    

    gb_params = clf2.best_params_

    

    clf3 = GridSearchCV(ET, et_tuned_parameters, cv=5,scoring='%s_macro' % score)

    clf3.fit(x_train, y_train)

    print("Extree",clf3.best_params_,"\n\n")

    et_params = clf3.best_params_

    

    clf4 = GridSearchCV(XGB, xgb_tuned_parameters, cv=5,scoring='%s_macro' % score)

    clf4.fit(x_train, y_train)

    print("XGB",clf4.best_params_,"\n\n")

    xgb_params = clf4.best_params_



   

    clf5 = GridSearchCV(RF, rf_tuned_parameters, cv=5,scoring='%s_macro' % score)

    clf5.fit(x_train, y_train)

    print("Random Forest",clf5.best_params_,"\n\n")

    rf_params = clf5.best_params_
# Create 5 objects that represent our 4 models

rf = SklearnHelper(clf=RandomForestClassifier, seed=SEED, params=rf_params)

et = SklearnHelper(clf=ExtraTreesClassifier, seed=SEED, params=et_params)

ada = SklearnHelper(clf=AdaBoostClassifier, seed=SEED, params=ada_params)

gb = SklearnHelper(clf=GradientBoostingClassifier, seed=SEED, params=gb_params)

svc = SklearnHelper(clf=SVC, seed=SEED, params=svc_params)




# Create our OOF train and test predictions. These base results will be used as new features

et_oof_train, et_oof_test = get_oof(et, x_train, y_train, x_test) # Extra Trees

rf_oof_train, rf_oof_test = get_oof(rf,x_train, y_train, x_test) # Random Forest

ada_oof_train, ada_oof_test = get_oof(ada, x_train, y_train, x_test) # AdaBoost 

svc_oof_train, svc_oof_test = get_oof(svc,x_train, y_train, x_test) # Support Vector Classifier

gb_oof_train, gb_oof_test = get_oof(gb,x_train, y_train, x_test) # Gradient Boost



print("Training is complete")


predictions_ada=ada.clf.predict(x_test)

predictions_svc=svc.clf.predict(x_test)

predictions_et=et.clf.predict(x_test)

predictions_rf=rf.clf.predict(x_test)

predictions_gb=gb.clf.predict(x_test)

print("Prediction is completed")
from sklearn.metrics import classification_report



gbm=XGB.set_params(**xgb_params).fit(x_train, y_train)

predictions_xgb = gbm.predict(x_test)



y_train_pred_xgb = gbm.predict(x_train)



target =["Dead","Survived"]

result = classification_report(y_train, y_train_pred_xgb, target_names=target)



print(result)
et_score=et.clf.score(x_train,y_train)

rf_score=rf.clf.score(x_train,y_train)

ada_score=ada.clf.score(x_train,y_train)

gb_score=gb.clf.score(x_train,y_train)

svc_score=svc.clf.score(x_train,y_train)

xgb_score = gbm.score(x_train,y_train)



score_df = pd.DataFrame( [{

     'Random Forest score': rf_score,

     'Extra Trees score': et_score,

      'AdaBoost score': ada_score,

    'Gradient Boost score': gb_score,

    "Support Vector Machine":svc_score,

    "XGBoost":xgb_score

    }])

score_df
base_predictions_train = pd.DataFrame( {'RandomForest': rf_oof_train.ravel(),

     'ExtraTrees': et_oof_train.ravel(),

     'AdaBoost': ada_oof_train.ravel(),

      'GradientBoost': gb_oof_train.ravel(),   

     "XGB":y_train_pred_xgb

    })

base_predictions_train.head(10)
#this code could be removed in future after ....xgb/vote class handled



vc_x_train = np.concatenate(( et_oof_train, rf_oof_train, ada_oof_train, gb_oof_train, svc_oof_train), axis=1)

vc_x_test = np.concatenate(( et_oof_test, rf_oof_test, ada_oof_test, gb_oof_test, svc_oof_test), axis=1)

vc_x_train.shape,vc_x_test.shape
from sklearn.ensemble import VotingClassifier



clf_vc = VotingClassifier(estimators=[('ada', ada.clf), ('RF',rf.clf),

                                      ('Gradient Boost', gb.clf),('Support Vector Machine', svc.clf), ('XGBoost', gbm)], 

                          voting='hard')

clf_vc = clf_vc.fit(vc_x_train, y_train)

predictions_vc =clf_vc.predict(vc_x_test)
from sklearn.metrics import classification_report

#local = pd.read_csv("Stacking20170915-1.csv").Survived.values

#local.shape, predictions.shape

vc_score = clf_vc.score(vc_x_train, y_train)
base_predictions_train = pd.DataFrame( {

    'RandomForest': rf_oof_train.ravel(),

     'ExtraTrees': et_oof_train.ravel(),

     'AdaBoost': ada_oof_train.ravel(),

      'GradientBoost': gb_oof_train.ravel(),

      "Voting"  :clf_vc.predict(vc_x_train),

      "XGB":y_train_pred_xgb

                                        

    })

base_predictions_train.head(10)


# Ada Boost

plot_learning_curve(ada.clf, "Ada Boost", x_train, y_train, cv=5)



# Extra Tree

plot_learning_curve(et.clf, "Extra Tree", x_train, y_train, cv=5)



# Gradient Boost

plot_learning_curve(gb.clf, "Gradient Boost", x_train, y_train, cv=5)



# Random Forest score

plot_learning_curve(rf.clf, "Random Forest", x_train, y_train, cv=5)



# Support Vector Machine

plot_learning_curve(svc.clf, "Support Vector Machine", x_train, y_train, cv=5)



# XGBosst

plot_learning_curve(gbm, "XGBoost", x_train, y_train, cv=5)



#Ensemble

plot_learning_curve(clf_vc, "VotingClassifier", x_train, y_train, cv=5)
score_df = pd.DataFrame( [{

     'Random Forest score': rf_score,

     'Extra Trees score': et_score,

      'AdaBoost score': ada_score,

    'Gradient Boost score': gb_score,

    "Support Vector Machine":svc_score,

      "XGBoost":xgb_score,

    "Voting":vc_score

    }])

score_df.sort_values(by=0,axis=1)
tmp_clfs={'Random Forest': rf.clf,

     'Extra Trees': et.clf,

      'AdaBoost': ada.clf,

    'Gradient Boost': gb.clf,

    "Support Vector Machine":svc.clf,

    "XGB": gbm,

    "Voting":clf_vc

    }

cvs_all=[]

for i,j in tmp_clfs.items():

    score=cross_val_score(j, x_train,y_train, cv=5,scoring="f1")

    cvs={}

    cvs["Estimator"]=i

    cvs["Score_mean"]=score.mean()

    cvs["Score_std"]=score.std()

    cvs["Score_low2z"]=score.mean()-score.std()*3

    cvs["Score_high2z"]=score.mean()+score.std()*3

    cvs_all.append(cvs)

pd.DataFrame(cvs_all).sort_values(by="Score_low2z",ascending=False)
# Generate Submission File 

StackingSubmission = pd.DataFrame({ 'PassengerId': PassengerId,

                            'Survived': predictions_vc.astype(int) })

StackingSubmission.to_csv("../input/parameter_tune_vc.csv", index=False)