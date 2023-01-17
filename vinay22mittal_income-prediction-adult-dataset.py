import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

import warnings                        # To ignore any warnings

warnings.filterwarnings("ignore")
df_test=pd.read_csv("../input/adult-test.csv")

df_train=pd.read_csv("../input/adult-training.csv")

print(df_test.shape)

print(df_train.shape)
df_train.head()
df_test.head()
column=["age","workclass","fnlwgt","education","education_num","marital_status","occupation","relationship","race","gender",

        "capital_gain","capital_loss","hours_per_week","native_country","income_bracket"]

len(column)
df_train.columns=column

df_train.columns
df_test=pd.read_csv("../input/adult-test.csv",names=column)

df_test=df_test.drop(0)

df_test.head()
df_test_org=df_test.copy()

df_train_org=df_train.copy()
df_train.dtypes
df_test.dtypes
df_test["age"]=df_test["age"].astype("int64")
def aa(x):

    #print(x)

    return x==' ?'
#df_test.apply(aa).sum()
#df_train.apply(aa)
#df_train.apply(aa).mean()
for variable in ["workclass","occupation","native_country"]:

    group=df_train.groupby([variable])

    group[variable].count().plot.bar()

    plt.show()
df_train["workclass"].mode()
df_train["workclass"].replace(' ?'," Private",inplace=True)
df_train["native_country"].mode()
df_train["native_country"].replace(' ?'," United-States",inplace=True)
df_train["occupation"].mode()
df_train["occupation"].replace(' ?'," Prof-specialty",inplace=True)
#df_train.apply(aa).mean()
# test data set

df_test["workclass"].replace(' ?'," Private",inplace=True)

df_test["native_country"].replace(' ?'," United-States",inplace=True)

df_test["occupation"].replace(' ?'," Prof-specialty",inplace=True)

#df_test.apply(aa).mean()
for col in df_train.columns:

    if df_train[col].dtype!= 'int64':

        #print(col)

        df_train[col]=df_train[col].apply(lambda x: x.replace(" ",""))

        df_test[col]=df_test[col].apply(lambda x: x.replace(" ",""))

        df_test[col]=df_test[col].apply(lambda x: x.replace(".",""))
for col in df_train:

    if df_train[col].dtype!='int64':

        print(col, end=": ")

        print(df_train[col].unique())
plt.hist(df_train["age"])
#Normalization of train

df_train["age"]=(df_train["age"]-df_train["age"].min())/(df_train["age"].max()-df_train["age"].min())

plt.hist(df_train["age"])
#Normalization of train

df_test["age"]=(df_test["age"]-df_test["age"].min())/(df_test["age"].max()-df_test["age"].min())

plt.hist(df_test["age"])
df_train["workclass"].value_counts()
# never worked and without pay both are similer class so we mearge them

df_train["workclass"].replace("Never-worked","Without-pay",inplace=True)

df_test["workclass"].replace("Never-worked","Without-pay",inplace=True)
sns.distplot(df_train["fnlwgt"])
df_train["fnlwgt"].describe()
#it have large SD so we should take log for resuce SD

df_train["fnlwgt"]=np.log(df_train["fnlwgt"])

sns.distplot(df_train["fnlwgt"])
# in test

df_test["fnlwgt"]=np.log(df_test["fnlwgt"])

sns.distplot(df_test["fnlwgt"])
df_train["education"].value_counts().plot.bar()
Edu=pd.crosstab(df_train["education"],df_train["income_bracket"])

Edu.plot(kind='bar',figsize=(10,5))
df_train["education"]=df_train["education"].apply(lambda x: "Primary" if x in ['1st-4th','5th-6th','7th-8th','9th','10th','11th','12th'] else x)
Edu=pd.crosstab(df_train["education"],df_train["income_bracket"])

Edu.plot(kind='bar',figsize=(10,5))
# chnage is test

df_test["education"]=df_test["education"].apply(lambda x: "Primary" if x in ['1st-4th','5th-6th','7th-8th','9th','10th','11th','12th'] else x)
#df_train["education_num"]=df_train["education_num"].astype("int64")

#df_test["education_num"]=df_test["education_num"].astype("int64")
Edu=pd.crosstab(df_train["education_num"],df_train["income_bracket"])

Edu.plot(kind='bar',figsize=(10,5))
mar=pd.crosstab(df_train["marital_status"],df_train["income_bracket"])

mar.plot(kind='bar',figsize=(10,5))

print(df_train["marital_status"].value_counts())
occ=pd.crosstab(df_train["occupation"],df_train["income_bracket"])

occ.plot(kind='bar',figsize=(10,5))

print(df_train["occupation"].value_counts())
rel=pd.crosstab(df_train["relationship"],df_train["income_bracket"])

rel.plot(kind='bar',figsize=(10,5))

print(df_train["relationship"].value_counts())
race=pd.crosstab(df_train["race"],df_train["income_bracket"])

race.plot(kind='bar',figsize=(10,5))

print(df_train["race"].value_counts())
sex=pd.crosstab(df_train["gender"],df_train["income_bracket"])

sex.plot(kind='bar',figsize=(10,5))

print(df_train["gender"].value_counts())
plt.hist(df_train["hours_per_week"])

plt.show()



plt.hist(df_train["capital_gain"])

plt.show()



plt.hist(df_train["capital_loss"])

plt.show()
#capital gain train Normalization

df_train["capital_gain"]=(df_train["capital_gain"]-df_train["capital_gain"].min())/(df_train["capital_gain"].max()-df_train["capital_gain"].min())

plt.hist(df_train["capital_gain"])
#test

df_test["capital_gain"]=(df_test["capital_gain"]-df_test["capital_gain"].min())/(df_test["capital_gain"].max()-df_test["capital_gain"].min())
#capital_loss normalization train

df_train["capital_loss"]=(df_train["capital_loss"]-df_train["capital_loss"].min())/(df_train["capital_loss"].max()-df_train["capital_loss"].min())

plt.hist(df_train["capital_loss"])
#test

df_test["capital_loss"]=(df_test["capital_loss"]-df_test["capital_loss"].min())/(df_test["capital_loss"].max()-df_test["capital_loss"].min())
#hour_per_week normalization train

df_train["hours_per_week"]=(df_train["hours_per_week"]-df_train["hours_per_week"].min())/(df_train["hours_per_week"].max()-df_train["hours_per_week"].min())

plt.hist(df_train["hours_per_week"])
# test

df_test["hours_per_week"]=(df_test["hours_per_week"]-df_test["hours_per_week"].min())/(df_test["hours_per_week"].max()-df_test["hours_per_week"].min())

plt.hist(df_test["hours_per_week"])
df_train["fnlwgt"]=(df_train["fnlwgt"]-df_train["fnlwgt"].min())/(df_train["fnlwgt"].max()-df_train["fnlwgt"].min())

plt.hist(df_train["fnlwgt"])

plt.show()



df_test["fnlwgt"]=(df_test["fnlwgt"]-df_test["fnlwgt"].min())/(df_test["fnlwgt"].max()-df_test["fnlwgt"].min())

plt.hist(df_test["fnlwgt"])

plt.show()

df_train["education_num"]=(df_train["education_num"]-df_train["education_num"].min())/(df_train["education_num"].max()-df_train["education_num"].min())

plt.hist(df_train["education_num"])

plt.show()





df_test["education_num"]=(df_test["education_num"]-df_test["education_num"].min())/(df_test["education_num"].max()-df_test["education_num"].min())

plt.hist(df_test["education_num"])

plt.show()
country=pd.crosstab(df_train["native_country"],df_train["income_bracket"])

country.plot(kind='bar',figsize=(10,5))

print(df_train["native_country"].value_counts())
def native(country):

    if country in ['England', 'Germany', 'Canada', 'Italy', 'France', 'Greece', 'Philippines']:

        return 'Western'

    elif country in ['Mexico', 'Puerto-Rico', 'Honduras', 'Jamaica', 'Columbia', 'Laos', 'Portugal', 'Haiti',

                     'Dominican-Republic', 'El-Salvador', 'Guatemala', 'Peru', 

                     'Trinadad&Tobago', 'Outlying-US(Guam-USVI-etc)', 'Nicaragua', 'Vietnam', 'Holand-Netherlands','Cuba']:

        return 'Poor'

    elif country in ['India', 'Iran', 'Cambodia', 'Taiwan', 'Japan', 'Yugoslavia', 'China', 'Hong']:

        return 'Eastern'

    elif country in ['South', 'Poland', 'Ireland', 'Hungary', 'Scotland', 'Thailand', 'Ecuador']:

        return 'Poland_country'

    elif country in ['United-States']:

        return 'US'

    else: 

        return country 
df_train["native_country"]=df_train["native_country"].apply(native)

df_test["native_country"]=df_test["native_country"].apply(native)
country=pd.crosstab(df_train["native_country"],df_train["income_bracket"])

country.plot(kind='bar',figsize=(10,5))

print(df_train["native_country"].value_counts())
df_train.head()
df_test.head()
from sklearn import preprocessing
catagorial_df_train=df_train.select_dtypes("object")

catagorial_df_train.head()
# apply Label encoder to df_categorical

le=preprocessing.LabelEncoder()
catagorial_df_train=catagorial_df_train.apply(le.fit_transform)
catagorial_df_train.head()
#concat catagorial_df_train to df_train

train=df_train.drop(catagorial_df_train.columns,axis=1)

train=pd.concat([train,catagorial_df_train],axis=1)
catagorial_df_test=df_test.select_dtypes("object")

catagorial_df_test=catagorial_df_test.apply(le.fit_transform)

test=df_test.drop(catagorial_df_train.columns,axis=1)

test=pd.concat([test,catagorial_df_test],axis=1)

test.head()
train.head()
train.dtypes
train["income_bracket"]=train["income_bracket"].astype("int64")
X_trainD=train.drop("income_bracket",axis=1)

Y_trainD=train["income_bracket"]



X_testD=test.drop("income_bracket",axis=1)

Y_testD=test["income_bracket"]
print(X_trainD.shape)

print(Y_trainD.shape)

print(X_testD.shape)

print(Y_testD.shape)
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
accuracy_list_train=[]

accuracy_list_test=[]

for i in range(1,10):

    dt_default = DecisionTreeClassifier(criterion='gini',max_depth=i)

    dt_default.fit(X_trainD, Y_trainD)

    Y_trainD_pred=dt_default.predict(X_trainD)

    Y_testD_pred=dt_default.predict(X_testD)

    accuracy_list_train.append(accuracy_score(Y_trainD_pred,Y_trainD))

    accuracy_list_test.append(accuracy_score(Y_testD_pred,Y_testD))

plt.plot(accuracy_list_train)

plt.title("train")

plt.plot(accuracy_list_test)

plt.show()
dt_default = DecisionTreeClassifier(max_depth=4)

dt_default.fit(X_trainD, Y_trainD)
Y_trainD_pred=dt_default.predict(X_trainD)

Y_testD_pred=dt_default.predict(X_testD)
accuracy_score(Y_trainD_pred,Y_trainD)
confusion_matrix(Y_trainD_pred,Y_trainD)
accuracy_score(Y_testD_pred,Y_testD)
# Confusion Matrix

confusion_matrix(Y_testD_pred,Y_testD)
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=10,criterion='entropy',max_depth=5)  # n_estimators : no. of tree
rf.fit(X_trainD, Y_trainD)
Y_trainD_pred=rf.predict(X_trainD)

Y_testD_pred=rf.predict(X_testD)

print(accuracy_score(Y_trainD_pred,Y_trainD))

print(accuracy_score(Y_testD_pred,Y_testD))
from sklearn.model_selection import cross_val_score

acc=cross_val_score(RandomForestClassifier(n_estimators=10,criterion='entropy',max_depth=5),X_trainD,Y_trainD,cv=5).mean()
print(acc)
acc_list=[]

for i in range(1,10):

    acc = cross_val_score(RandomForestClassifier(n_estimators=i,max_depth=5),X_trainD,Y_trainD,cv=5).mean()

    acc_list.append(acc)
plt.plot(acc_list)

plt.show()
print(np.argmax(acc_list))
rf = RandomForestClassifier(n_estimators=30,max_depth=5,criterion='entropy')

rf.fit(X_trainD,Y_trainD)

print(rf.score(X_trainD,Y_trainD))

print(rf.score(X_testD,Y_testD))
df_train["income_bracket"]=df_train["income_bracket"].apply(lambda x: 0 if x =="<=50K" else 1)
df_train["income_bracket"]=df_train["income_bracket"].astype("object")
df_test["income_bracket"]=df_test["income_bracket"].apply(lambda x: 0 if x =="<=50K" else 1)

df_test["income_bracket"]=df_test["income_bracket"].astype("object")
catagorial_df_train=df_train.select_dtypes("object")

catagorial_df_train.drop("income_bracket",axis=1,inplace=True)
#train

train_dummies=df_train.drop(catagorial_df_train.columns,axis=1)

train_dummies=pd.concat([train_dummies,pd.get_dummies(catagorial_df_train)],axis=1)

train_dummies.head()
train_dummies.income_bracket=train_dummies["income_bracket"].astype("uint8")
len(test_dummies.columns)
#test

catagorial_df_test=df_test.select_dtypes("object")

catagorial_df_test.drop("income_bracket",axis=1,inplace=True)

test_dummies=df_test.drop(catagorial_df_test.columns,axis=1)

test_dummies=pd.concat([test_dummies,pd.get_dummies(catagorial_df_test)],axis=1)

test_dummies.head()
test_dummies.income_bracket=test_dummies["income_bracket"].astype("uint8")
X_train=train_dummies.drop("income_bracket",axis=1)

Y_train=train_dummies["income_bracket"]



X_test=test_dummies.drop("income_bracket",axis=1)

Y_test=test_dummies["income_bracket"]
print(X_train.shape)

print(Y_train.shape)

print(X_test.shape)

print(Y_test.shape)
from sklearn.linear_model import LogisticRegression
logistic_model=LogisticRegression()

logistic_model.fit(X_train,Y_train)
Y_test_pred=logistic_model.predict(X_test)
accuracy_score(Y_test_pred,Y_test)
Y_train_pred=logistic_model.predict(X_train)

accuracy_score(Y_train,Y_train_pred)