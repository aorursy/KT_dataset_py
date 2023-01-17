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

from sklearn.naive_bayes import GaussianNB

from sklearn.model_selection import train_test_split

import seaborn as sns

sns.set("talk","darkgrid",font_scale=1,font="sans-serif",color_codes=True)

import statsmodels.api as sm

import scipy.stats as stats

from sklearn.metrics import classification_report

df = pd.read_csv(r"../input/pima-indians-diabetes-database/diabetes.csv")

df.head()
x = df[['Pregnancies','Glucose','BloodPressure','DiabetesPedigreeFunction']]

y = df.iloc[::,-1]
x_trainNB, x_testNB, y_trainNB, y_testNB = train_test_split(x,y,test_size=0.2,random_state=0)

scalerNB = StandardScaler()

x_trainNB = scalerNB.fit_transform(x_trainNB)

x_testNB = scalerNB.transform(x_testNB)

NB = GaussianNB()

NB.fit(x_trainNB,y_trainNB)
y_predNB = NB.predict(x_testNB)

pd.DataFrame({"Actual_Outcome":y_testNB, "Predicted_Outcome":y_predNB})
reportNB = pd.DataFrame(classification_report(y_testNB, y_predNB, output_dict=True)).transpose()

reportNB
y_predNB_proba = NB.predict_proba(x_testNB)[::,1]

fprNB, tprNB, _ = metrics.roc_curve(y_testNB, y_predNB_proba)

aucNB =  metrics.roc_auc_score(y_testNB, y_predNB_proba)

fig, ax = plt.subplots(figsize=(10,10))

plt.plot(fprNB, tprNB, label = "Naive Bayes , auc " + str(aucNB), color="orange")

plt.plot([0,1],[0,1],alpha=0.8,lw=2, color="red")

plt.xlim([0.00,1.01])

plt.ylim([0.00,1.01])

plt.title("Validation ROC Curve - Naive Bayes")

plt.xlabel("Specificity")

plt.ylabel("Sensitivity")

plt.legend(loc=4)

plt.show()
from sklearn.metrics import precision_recall_curve

from sklearn.metrics import average_precision_score

precisionrecallNB,recallNB,thresholdNB = precision_recall_curve(y_testNB,y_predNB)

apsNB = average_precision_score(y_testNB,y_predNB)

fig, ax = plt.subplots(figsize=(10,10))

plt.plot(precisionrecallNB,recallNB, label = "Naive Bayes , aps " + str(apsNB), color="orange")

plt.title("Precision-Recall Curve -Naive Bayes")

plt.xlabel("Precision")

plt.ylabel("Recall")

plt.legend(loc=4)

plt.axhline(y=0.5, color="red", lw=2)

plt.show()