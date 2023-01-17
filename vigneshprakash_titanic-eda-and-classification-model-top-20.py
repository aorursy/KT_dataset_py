import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
%matplotlib inline
pd.set_option("display.max_columns",None)
import warnings 
warnings.filterwarnings('ignore')
df_train=pd.read_csv('../input/titanic/train.csv')
df_test=pd.read_csv('../input/titanic/test.csv')
df_train.head()
df_train.info()
df_train.describe()
df_train.describe(include='object')
df_train.drop(columns=["Cabin","Ticket",],inplace=True)
df_train.head(10)
df_test.drop(columns=["Cabin","Ticket",],inplace=True)
df_test.head()
df_train.isnull().sum()
df_train["Age"] = df_train["Age"].fillna(df_train["Age"].mean())
df_train.head()
df_train.dropna(inplace = True)
df_train.head()
df_test.isnull().sum()
df_test["Age"] = df_test["Age"].fillna(df_test["Age"].mean())
df_test['Fare'] = df_test['Fare'].fillna(df_test['Fare'].median()) 
df_test.isnull().sum()
df_train['title']=list(map(lambda x : x.split(',')[1].split('.')[0].lstrip(),df_train['Name']))
df_train.title.unique()
df_test['title']=list(map(lambda x : x.split(',')[1].split('.')[0].lstrip(),df_test['Name']))
df_test.title.unique()
column=df_train.describe().columns
plt.figure(figsize=(20,10))
df_train.boxplot(column=['Survived','Pclass','Age','SibSp','Parch','Fare'])
sns.countplot(df_train['Survived'])
sns.barplot(x="Sex", y="Survived", data=df_train)

#print percentages of females vs. males that survive
print("Percentage of females who survived:", df_train["Survived"][df_train["Sex"] == 'female'].value_counts(normalize = True)[1]*100)

print("Percentage of males who survived:", df_train["Survived"][df_train["Sex"] == 'male'].value_counts(normalize = True)[1]*100)
# Create subplot
plt.subplots(figsize = (8,5))
sns.barplot(x = "Pclass", y = "Survived", data=df_train, linewidth=2)
plt.title("Passenger Class Distribution - Survived vs Non-Survived", fontsize = 10)
plt.xlabel("Socio-Economic class", fontsize = 10);
plt.ylabel("% of Passenger Survived", fontsize = 10);
labels = ['1st', '2nd', '3rd']
val = [0,1,2] 
plt.xticks(val, labels);


#print percentages of 1st vs. 2nd and 3rd class
print("Percentage of 1st class who survived:", df_train["Survived"][df_train["Pclass"] == 1].value_counts(normalize = True)[1]*100)

print("Percentage of 2nd class who survived:", df_train["Survived"][df_train["Pclass"] == 2].value_counts(normalize = True)[1]*100)

print("Percentage of 3rd class who survived:", df_train["Survived"][df_train["Pclass"] == 3].value_counts(normalize = True)[1]*100)
#create a subplot
f,ax=plt.subplots(1,2,figsize=(10,5))

# create bar plot using groupby
df_train[['Sex','Survived']].groupby(['Sex']).mean().plot.bar(color=['#a85ee0'],ax=ax[0])
ax[0].set_title('Survived vs Sex')

# create count plot
sns.countplot('Sex',hue='Survived',data=df_train,ax=ax[1])
ax[1].set_title('Sex:Survived vs Dead')
plt.show()
# create subplot plot
f,ax=plt.subplots(1,2,figsize=(10,5))

# create bar plot using groupby
df_train['Pclass'].value_counts().plot.bar(color=['#080035','#0F006B','#8B80C7'],ax=ax[0])
ax[0].set_title('Number Of Passengers By Pclass')
ax[0].set_ylabel('Count')

# create count plot
sns.countplot('Pclass',hue='Survived',data=df_train,ax=ax[1])
ax[1].set_title('Pclass:Survived vs Dead')
plt.show()
# create subplot plot

f,ax=plt.subplots(1,2,figsize=(18,8))

# create violinplot plot using groupby

sns.violinplot("Pclass","Age", hue="Survived", data=df_train,split=True,ax=ax[0])
ax[0].set_title('Pclass and Age vs Survived')
ax[0].set_yticks(range(0,110,10))
sns.violinplot("Sex","Age", hue="Survived", data=df_train,split=True,ax=ax[1])
ax[1].set_title('Sex and Age vs Survived')
ax[1].set_yticks(range(0,110,10))
plt.show()
# create subplot plot
f,ax=plt.subplots(2,2,figsize=(20,8))

# create Bar (count) plot for Embarked vs. No. Of Passengers Boarded
sns.countplot('Embarked',data=df_train,ax=ax[0,0],color="#b4bf82")
ax[0,0].set_title('No. Of Passengers Boarded')

# create Bar (count) plot for Embarked vs. Male-Female Split
sns.countplot('Embarked',hue='Sex',data=df_train,ax=ax[0,1])
ax[0,1].set_title('Male-Female Split for Embarked')

# create Bar (count) plot for Embarked vs Survived
sns.countplot('Embarked',hue='Survived',data=df_train,ax=ax[1,0])
ax[1,0].set_title('Embarked vs Survived')

# create Bar (count) plot for Embarked vs Pclass
sns.countplot('Embarked',hue='Pclass',data=df_train,ax=ax[1,1])
ax[1,1].set_title('Embarked vs Pclass')
plt.subplots_adjust(wspace=0.2,hspace=0.5)
plt.show()
sns.boxplot(x="Pclass", y="Age", data=df_train)
sns.stripplot(x="Pclass", y="Age", data=df_train, jitter=True, edgecolor="gray")
tab = pd.crosstab(df_train['Sex'], df_train['Survived'])
print(tab)

dummy = tab.div(tab.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True)
dummy = plt.xlabel('Port embarked')
dummy = plt.ylabel('Percentage')
sns.lmplot(x='Age', y='Fare', hue='Survived', 
           data=df_train.loc[df_train['Survived'].isin([1,0])], 
           fit_reg=False)
sns.heatmap(df_train.corr(),cmap="ocean",annot=True)
df_train.boxplot('Age','title',figsize=(15,8))
df_train.info()
sns.pairplot(df_train[['Survived','Pclass','Age','SibSp','Parch','Fare']], kind="scatter", hue="Survived", palette="Set2")
df_train.head()
df_test.head()
# so we have titles for Nobels like Master, Capt...and others for regular people..
# so lets replace Nobels people by Dummy value 1 and regular people by Dummy value 0
title_mapping = {'Mr': 0, 'Mrs': 0, 'Miss': 0, 'Master' : 1,'Don': 1, 'Rev' : 1,'Dr' : 1,'Mme': 0, 'Ms': 0, 'Major': 1,
 'Lady': 1, 'Sir': 1, 'Mlle': 0, 'Col': 1, 'Capt': 1, 'Countess': 1, 'Jonkheer': 1,'Dona': 1,}

df_train['title'] = df_train['title'].map(title_mapping)
df_train['title'] = df_train['title'].fillna(0)
df_train['title']=df_train['title'].astype('int')
    
print(df_train['title'].unique())
df_train['title'].value_counts()
title_mapping = {'Mr': 0, 'Mrs': 0, 'Miss': 0, 'Master' : 1,'Don': 1, 'Rev' : 1,'Dr' : 1,'Mme': 0, 'Ms': 0, 'Major': 1,
 'Lady': 1, 'Sir': 1, 'Mlle': 0, 'Col': 1, 'Capt': 1, 'Countess': 1, 'Jonkheer': 1,'Dona': 1,}

df_test['title'] = df_test['title'].map(title_mapping)
df_test['title'] = df_test['title'].fillna(0)

    
print(df_test['title'].unique())
df_train=pd.get_dummies(data=df_train,columns=['Sex','Embarked'],drop_first=True)
df_test=pd.get_dummies(data=df_test,columns=['Sex','Embarked'],drop_first=True)
df_train.drop(columns=['PassengerId','Name'],inplace=True)
df_test.drop(columns=['Name'],inplace=True)
df_train.head()
df_test.head()
import statsmodels.api as sm
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report,roc_auc_score,log_loss,f1_score

x=df_train.drop(columns=['Survived'])
y=df_train['Survived']
x.head()
from sklearn.preprocessing import StandardScaler
ss=StandardScaler()
X=ss.fit_transform(x)
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion='entropy',random_state=0,class_weight='balanced')

from sklearn.linear_model import LogisticRegression
LR=LogisticRegression(class_weight='balanced')

from sklearn.naive_bayes import GaussianNB
nb=GaussianNB()

from sklearn.metrics import confusion_matrix,accuracy_score,roc_auc_score,roc_curve
from sklearn.ensemble import RandomForestClassifier
rfc=RandomForestClassifier(n_estimators=100,random_state=0,class_weight='balanced')

from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier()

from sklearn.ensemble import AdaBoostClassifier,BaggingClassifier,GradientBoostingClassifier
gb=GradientBoostingClassifier(random_state=0)
bc=BaggingClassifier(base_estimator=knn,random_state=0)

import lightgbm as lgb
lgbm=lgb.LGBMClassifier(random_state=0)

from xgboost import XGBClassifier
classifier = XGBClassifier()
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint as sp_randint

rfc_tunned=RandomForestClassifier(n_estimators=100,random_state=0)
params={'n_estimators':sp_randint(1,1000),
        'max_features':sp_randint(1,9),
        'max_depth': sp_randint(2,50),
        'min_samples_split':sp_randint(2,80),
        'min_samples_leaf':sp_randint(1,80),
        'criterion':['gini','entropy']}

rsearch_rfc=RandomizedSearchCV(rfc_tunned,params,cv=3,scoring='accuracy',n_jobs=-1,random_state=0)

rsearch_rfc.fit(X,y)
rsearch_rfc.best_params_
rfc_tunned=RandomForestClassifier(**rsearch_rfc.best_params_,random_state=0)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import RandomizedSearchCV,GridSearchCV
from scipy.stats import randint as sp_randint

knn_tunned=KNeighborsClassifier()

params={'n_neighbors':sp_randint(1,20),'p':sp_randint(1,30)}

rsearch_knn=RandomizedSearchCV(knn_tunned,params,cv=3,scoring='accuracy',n_jobs=-1,random_state=0)
rsearch_knn.fit(X,y)
rsearch_knn.best_params_
knn_tunned=KNeighborsClassifier(**rsearch_knn.best_params_)
from scipy.stats import randint as sp_randint
from scipy.stats import uniform as sp_uniform 

lgbm_tunned=lgb.LGBMClassifier(random_state=0)
params={'n_estimators':sp_randint(1,1000),
       'max_depth': sp_randint(2,80),
        'learning_rate':sp_uniform(0.001,0.05),
        'num_leaves':sp_randint(2,50)
       }

rsearch_lgbm=RandomizedSearchCV(lgbm_tunned,param_distributions=params,cv=3,scoring='accuracy',n_iter=200,n_jobs=-1,random_state=0)

rsearch_lgbm.fit(X,y)
rsearch_lgbm.best_params_
lgbm_tunned=lgb.LGBMClassifier(**rsearch_lgbm.best_params_,random_state=0)
from scipy.stats import randint as sp_randint
from scipy.stats import uniform as sp_uniform 

gb_tuned=GradientBoostingClassifier(random_state=0)
params= {'learning_rate':[0.15,0.1,0.05,0.01,0.005,0.001], 
         'n_estimators':sp_randint(2,1500),
         'max_depth':sp_randint(1,10),
        'min_samples_split':sp_randint(2,100), 
         'min_samples_leaf':sp_randint(1,10),
        'max_features':sp_randint(1,9),
        'subsample':[0.7,0.75,0.8,0.85,0.9,0.95,1]}

rsearch_gb=RandomizedSearchCV(gb_tuned,param_distributions=params,cv=3,n_iter=200,n_jobs=-1,random_state=0)

rsearch_gb.fit(X,y)
rsearch_gb.best_params_
gb_tuned=GradientBoostingClassifier(**rsearch_gb.best_params_,random_state=0)
models=[]
models.append(('Logistic',LR))
models.append(('Decision Tree',dt))
models.append(('Naive Bayes',nb))
models.append(('Random Forest',rfc))
models.append(('Random Forest Tunned',rfc_tunned))
models.append(('KNN',knn))
models.append(('KNN Tunned',knn_tunned))
models.append(('Bagging',bc))
models.append(('Gradient Boost',gb))
models.append(('Gradient Boost Tunned',gb_tuned))
models.append(('LGBM',lgbm))
models.append(('LGBM Tunned',lgbm_tunned))
models.append(('XGB',classifier))
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from sklearn.model_selection import train_test_split

results=[]
Var=[]
names=[]
for name,model in models:
    #kfold=model_selection.KFold(shuffle=True,n_splits=10,random_state=0)
    cv_results=cross_val_score(model,X,y,cv=10,scoring='roc_auc')
    results.append(np.mean(cv_results))
    Var.append(np.var(cv_results))
    names.append(name)

r_df=pd.DataFrame({'Model':names,'ROC-AUC':results,'Variance Error':Var})
print(r_df)
df_test.head()
df_test2=df_test.drop('PassengerId',axis=1)
gb.fit(X,y)
y_test_pred = gb.predict(df_test2)
submission = pd.DataFrame({
        "PassengerId": df_test["PassengerId"],
        "Survived": y_test_pred
    })

submission.to_csv('gender_submission.csv', index=False)