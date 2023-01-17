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
#load libraries

import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns

from scipy.stats import norm

from sklearn.preprocessing import StandardScaler

from scipy import stats

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import MinMaxScaler

%matplotlib inline



import os

print(os.listdir("../input"))



from sklearn import linear_model

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import Perceptron

from sklearn.linear_model import SGDClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC, LinearSVC

from sklearn.naive_bayes import GaussianNB
df = pd.read_csv("/kaggle/input/predicting-pulsar-starintermediate/pulsar_data_train.csv")
df.head()
#let's check for missing/ unique values

def dfaux (df):

    cant = df.isnull().sum()

    df_aux = pd.DataFrame(index = df.columns, data =

                         {'type': df.dtypes,

                          'unique_values': df.nunique(),

                          'have_null?': df.isnull().any(),

                          'how many?' : cant,

                          'per' : cant/df.shape[0]*100 })

    return df_aux
dfaux(df)
df.columns
#replacing null values

df[" Excess kurtosis of the integrated profile"] = df[" Excess kurtosis of the integrated profile"].replace(np.NaN, df[" Excess kurtosis of the integrated profile"].mean())
#replacing null values

df[" Standard deviation of the DM-SNR curve"] = df[" Standard deviation of the DM-SNR curve"].replace(np.NaN, df[" Standard deviation of the DM-SNR curve"].mean())

df[" Skewness of the DM-SNR curve"] = df[" Skewness of the DM-SNR curve"].replace(np.NaN, df[" Skewness of the DM-SNR curve"].mean())
#lets check again

dfaux(df)
df.shape
df.info()
df.describe()
#correlation

corr = df.corr()

sns.heatmap(data=df.corr(),annot=True,cmap="coolwarm",linewidths=1,fmt=".2f",linecolor="gray")
#Correlation with target variable

cor_target = abs(corr["target_class"])#Selecting highly correlated features

relevant_features = cor_target[cor_target>0]

relevant_features.nlargest(n=12)
sns.pairplot(data=df,

             palette="husl",

             hue="target_class",

             vars=[" Mean of the integrated profile",

                   " Excess kurtosis of the integrated profile",

                   " Skewness of the integrated profile",

                   " Mean of the DM-SNR curve",

                   " Excess kurtosis of the DM-SNR curve",

                   " Skewness of the DM-SNR curve"])



plt.suptitle("PairPlot of Data Without Std. Dev. Fields",fontsize=18)



plt.tight_layout()

plt.show()   # pairplot without standard deviaton fields of data
#feature scaling

features = df.drop("target_class", axis=1)

scaler = MinMaxScaler(feature_range=(0,1))

fscaled = scaler.fit_transform(features)
#separating input and target variable

X = df.drop("target_class", axis=1)

Y = df["target_class"]
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state = 0 )
#1. SGD

sgd = linear_model.SGDClassifier(max_iter=5, tol=None)

sgd.fit(x_train, y_train)

y_pred = sgd.predict(x_test)

acc_sgd = round(sgd.score(x_train, y_train)*100, 2)
print(acc_sgd)
#2. Random Forest

rforest = RandomForestClassifier(n_estimators = 100)

rforest.fit(x_train, y_train)

y_pred2 = rforest.predict(x_test)



acc_rf = round(rforest.score(x_train, y_train)* 100, 2)

print(acc_rf)
#3. Logistic Regression

logreg = LogisticRegression()

logreg.fit(x_train, y_train)

y_pred3 = logreg.predict(x_test)



acc_lr = round(logreg.score(x_train, y_train) * 100, 2)

print(acc_lr)
#4. KNN

knn = KNeighborsClassifier(n_neighbors = 4)

knn.fit(x_train, y_train)

y_pred4 = knn.predict(x_test)

acc_knn = round(knn.score(x_train, y_train)*100 ,2)

print(acc_knn)
#5. GNB

gauss = GaussianNB()

gauss.fit(x_train, y_train)

y_pred5 = gauss.predict(x_test)

acc_gauss = round(gauss.score(x_train, y_train)*100,2)

print(acc_gauss)
#6. Decision Tree

dt = DecisionTreeClassifier()

dt.fit(x_train, y_train)

y_pred6 = dt.predict(x_test)

acc_dt = round(dt.score(x_train, y_train)*100,2)

print(acc_dt)
importances = pd.DataFrame({'feature':x_train.columns, 'importance':np.round(dt.feature_importances_, 3)})

importances = importances.sort_values('importance', ascending=False).set_index('feature')

importances.head(15)
importances.plot.bar()
#confusion matrix

from sklearn.metrics import precision_score, recall_score



from sklearn.model_selection import cross_val_predict

from sklearn.metrics import confusion_matrix

predictions = cross_val_predict(dt, x_train, y_train, cv=3)

confusion_matrix(y_train, predictions)



print("Precision:", precision_score(y_train, predictions))

print("Recall:",recall_score(y_train, predictions))
#f1 score

from sklearn.metrics import f1_score

f1_score(y_train, predictions)
#roc-auc

y_scores = dt.predict_proba(x_train)
y_scores= y_scores[:,1]

from sklearn.metrics import roc_auc_score

rascore  = roc_auc_score(y_train, y_scores)

print(rascore)