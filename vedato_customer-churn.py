# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#!pip install ycimpute

#from ycimpute.imputer import iterforest,EM



from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA

from sklearn.preprocessing import LabelEncoder

from sklearn import datasets, metrics, model_selection, svm

import missingno as msno



from fancyimpute import KNN

from sklearn.preprocessing import OrdinalEncoder



import numpy as np

import pandas as pd 

import statsmodels.api as sm

import statsmodels.formula.api as smf

import seaborn as sns

from sklearn.preprocessing import scale 

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

from sklearn.metrics import roc_auc_score,roc_curve

import statsmodels.formula.api as smf

import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.naive_bayes import GaussianNB

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import GradientBoostingClassifier

from xgboost import XGBClassifier

from lightgbm import LGBMClassifier

from catboost import CatBoostClassifier



from warnings import filterwarnings

filterwarnings('ignore')



pd.set_option('display.max_columns', None)

import gc
encoder=OrdinalEncoder()

imputer=KNN()



def encode(data):

    '''function to encode non-null data and replace it in the original data'''

    #retains only non-null values

    nonulls = np.array(data.dropna())

    #reshapes the data for encoding

    impute_reshape = nonulls.reshape(-1,1)

    #encode date

    impute_ordinal = encoder.fit_transform(impute_reshape)

    #Assign back encoded values to non-null values

    data.loc[data.notnull()] = np.squeeze(impute_ordinal)

    return data
df=pd.read_csv("/kaggle/input/churn-modelling/Churn_Modelling.csv")

df.head()
df.info()
df.nunique()
df.select_dtypes(include="object").nunique()
for i in df.select_dtypes(include="object"):

    print(df.select_dtypes(include="object")[i].value_counts())
data=df

data.head()
df['Gender']=LabelEncoder().fit_transform(df['Gender'])

df['Surname']=LabelEncoder().fit_transform(df['Surname'])

dms= pd.get_dummies(df[['Geography']])

df=df.drop('Geography', axis=1)

df=pd.concat([df, dms], axis=1)

df.head()
df.isnull().sum().any()
df.corrwith(df["Exited"], method="spearman")
plt.subplots(figsize=(20,15))

sns.heatmap(df.corr(), annot=True);
df[["CreditScore","Age","Tenure","Balance","NumOfProducts","EstimatedSalary"]].describe().T
sns.scatterplot(y= df.Age, x=df.EstimatedSalary, hue=df.Exited);
sns.scatterplot(y= df.Age, x=df.Balance, hue=df.Exited);
sns.scatterplot(x=df.Tenure,y=df.NumOfProducts, hue=df.Exited);
sns.boxplot(df.CreditScore)

plt.show()

sns.distplot(df["CreditScore"]);
df["CreditScore"].describe()
'''

# how to eliminate outliers



Q1=df["CreditScore"].quantile(.25)

Q3=df["CreditScore"].quantile(.75)

IQR=Q3-Q1

print(Q1)

print(Q3)

print(IQR)



lower_threshold=Q1-1.5*IQR

lower_threshold



zz=pd.DataFrame(df["CreditScore"]<lower_threshold)

zz.loc[zz["CreditScore"]==True].index



for i in zz.loc[zz["CreditScore"]==True].index:

    df["CreditScore"][i]= 383

    print(df["CreditScore"][i])

    

'''
sns.boxplot(df.Age)

plt.show()

sns.distplot(df["Age"]);
df.Age.describe()
sns.boxplot(df.Tenure)

plt.show()

sns.distplot(df["Tenure"]);
sns.boxplot(df.Balance)

plt.show()

sns.distplot(df["Balance"]);
sns.distplot(df["EstimatedSalary"]);
sns.boxplot(df.NumOfProducts)

plt.show()

sns.distplot(df["NumOfProducts"]);
df["NumOfProducts"].describe()
sns.boxplot(df.EstimatedSalary)

plt.show()

sns.distplot(df.EstimatedSalary);
# Age Segmentation



df.Age
x=df[["Age"]]

bins=[0,20,30,40,50,60,70,120]

labels=["below 20","20-29","30-39","40-49","50-59","60-69","70+"]

x["age_segment"]=pd.cut(x["Age"], bins, labels=labels, include_lowest=True)

df["age_segment"]=x["age_segment"]
df["age_segment"]=LabelEncoder().fit_transform(df["age_segment"])

df
df.Age.std()/df.Age.mean()
# Coef.of variance



columns=["CreditScore",

        "Age",

        "Tenure",

        "Balance",

        "EstimatedSalary"]



for i in columns:

    df["VC_"+i]=df[i].std()/df[i].mean()



df
Ktrain, Ktest = train_test_split(df, test_size=0.30, random_state=4)

y_Ktest=Ktest["Exited"]

X_Ktest=Ktest.drop(["Exited"], axis=1)
y=Ktrain['Exited']

X=Ktrain.drop(['Exited',"RowNumber","CustomerId","Surname"], axis=1).astype('float64')

X_train, X_test,y_train,y_test = train_test_split(X,y, test_size=0.30, random_state=4)
def compML (df, y, algorithm):

    

    y=df[y]

    X=df.drop(['Exited',"RowNumber","CustomerId","Surname"], axis=1).astype('float64')

    X_train, X_test,y_train,y_test=train_test_split(X,y, test_size=0.25, random_state=4)

    

    model=algorithm().fit(X_train, y_train)

    y_pred=model.predict(X_test)

    accuracy= accuracy_score(y_test, y_pred)

    #return accuracy

    model_name= algorithm.__name__

    print(model_name,": ",accuracy)
models = [LogisticRegression,

          KNeighborsClassifier,

          GaussianNB,

          SVC,

          DecisionTreeClassifier,

          RandomForestClassifier,

          GradientBoostingClassifier,

          LGBMClassifier,

          XGBClassifier,

          #CatBoostClassifier

         ]
df
for x in models:

    compML(df,"Exited",x)
clf=GradientBoostingClassifier().fit(X_train, y_train)

y_pred=clf.predict(X_test)

accuracy_score(y_test, y_pred)
clf
GBM_params = {"loss":[ 'deviance', 'exponential'],

             "min_samples_split":[2,3],

             "n_estimators":[100,200,500],

             "min_samples_leaf":[1,2],

             }
GBM_cv_model = GridSearchCV(clf, 

                            GBM_params, 

                            cv=10, n_jobs=-1, 

                            verbose=2).fit(X_train, y_train)
GBM_cv_model.best_params_
clf_tuned = GradientBoostingClassifier(learning_rate= 0.1,

                                       max_depth= 3,

                                       n_estimators= 100,

                                       subsample= 1).fit(X_train, y_train)

y_pred=clf.predict(X_test)

accuracy_score(y_test, y_pred)
Importance = pd.DataFrame({'Importance':clf_tuned.feature_importances_*100},

                         index = X_train.columns)



Importance.sort_values(by = 'Importance',

                      axis = 0,

                      ascending = True).plot(kind = 'barh',

                                            color = '#d62728',

                                            figsize=(10,6), 

                                            edgecolor='white')

plt.xlabel('Variable Importance')

plt.gca().legend_ = None
print(X.shape)

print(X_Ktest.shape)

print(y_train.shape)

print(y_Ktest.shape)

y_Ktest.head()
#Ktest_Exited=Ktest["Exited"]

#Ktest=Ktest.drop(["Exited"], axis=1)

X_Ktest= X_Ktest.drop(["RowNumber","CustomerId","Surname"], axis=1).astype('float64')
# X_Ktest.drop(['predictions'], axis=1, inplace=True)
predictions= clf.predict(X_Ktest)
real_test_y=pd.DataFrame(y_Ktest)

real_test_y["predictions"]=predictions



real_test_y.loc[:,"predictions"]=round(real_test_y.loc[:,"predictions"] ).astype(int)



real_test_y.head()
accuracy_score(real_test_y.loc[:,"Exited"],real_test_y.loc[:,"predictions"] )