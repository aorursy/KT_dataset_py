# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline 

import seaborn as sns

from sklearn import metrics

from sklearn.preprocessing import StandardScaler

from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import train_test_split

import seaborn as sns

sns.set("talk","darkgrid",font_scale=1,font="sans-serif",color_codes=True)

import statsmodels.api as sm

import scipy.stats as stats

from sklearn.metrics import precision_recall_curve

from sklearn.metrics import average_precision_score
df = pd.read_csv(r"../input/pima-indians-diabetes-database/diabetes.csv")

df.head()
fig, ax = plt.subplots(figsize=(10,10))

sns.heatmap(df.isnull())

plt.title("Detecting missing values")

plt.show()
dfcorr = df.corr()

fig, ax = plt.subplots(figsize=(10,10))

sns.heatmap(dfcorr, annot=True, annot_kws={"size":12}, cmap="coolwarm")

plt.title("Correlation matrix")

plt.show()
dfcov = df.cov()

fig, ax = plt.subplots(figsize=(10,10))

sns.heatmap(dfcov, annot=True, annot_kws={"size":12}, cmap="coolwarm")

plt.title("Covariance matrix")

plt.show()
df.describe().transpose()
x = df[['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']]

y = df.iloc[::,-1]

x_constant = sm.add_constant(x)

model = sm.OLS(y,x_constant).fit()

model.predict(x_constant)

model.summary()
x = df[['Pregnancies','Glucose','BloodPressure','DiabetesPedigreeFunction']]

y = df.iloc[::,-1]

x_constant = sm.add_constant(x)

model = sm.OLS(y,x_constant).fit()

model.predict(x_constant)

model.summary()
x_trainDT, x_testDT, y_trainDT, y_testDT = train_test_split(x,y,test_size=0.2,random_state=0)

scalerDT = StandardScaler()

x_trainDT = scalerDT.fit_transform(x_trainDT)

x_testDT = scalerDT.transform(x_testDT)

DT = DecisionTreeClassifier()

DT.fit(x_trainDT,y_trainDT)
y_predDT = DT.predict(x_testDT)

pd.DataFrame({"Actual_Outcome":y_testDT, "Predicted_Outcome":y_predDT})
from sklearn.metrics import classification_report

reportDT = pd.DataFrame(classification_report(y_testDT, y_predDT, output_dict=True)).transpose()

reportDT
y_predDT_proba = DT.predict_proba(x_testDT)[::,1]

fprDT, tprDT, _ = metrics.roc_curve(y_testDT, y_predDT_proba)

aucDT =  metrics.roc_auc_score(y_testDT, y_predDT_proba)

fig, ax = plt.subplots(figsize=(10,10))

plt.plot(fprDT, tprDT, label = "Decision Tree , auc " + str(aucDT), color="green")

plt.plot([0,1],[0,1],alpha=0.8,lw=2, color="red")

plt.xlim([0.00,1.01])

plt.ylim([0.00,1.01])

plt.title("Validation ROC Curve - Decision Tree")

plt.xlabel("Specificity")

plt.ylabel("Sensitivity")

plt.legend(loc=4)

plt.show()
precisionrecallDT,recallDT,thresholdDT = precision_recall_curve(y_testDT,y_predDT)

apsDT = average_precision_score(y_testDT,y_predDT)

fig, ax = plt.subplots(figsize=(10,10))

plt.plot(precisionrecallDT,recallDT, label = "Decision Tree , aps " + str(apsDT), color="green")

plt.title("Precision-Recall Curve - Decision Tree")

plt.xlabel("Precision")

plt.ylabel("Recall")

plt.legend(loc=4)

plt.axhline(y=0.5, color="red", lw=2)

plt.show()