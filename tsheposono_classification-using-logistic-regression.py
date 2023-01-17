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

from pylab import rcParams

plt.rcParams["figure.figsize"] = [10,10]

from sklearn import metrics

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split, learning_curve

import seaborn as sns

sns.set("talk","ticks",font_scale=1,font="sans-serif",color_codes=True)

import statsmodels.api as sm

import scipy.stats as stats

import warnings

warnings.filterwarnings("ignore")
df = pd.read_csv(r"../input/pima-indians-diabetes-database/diabetes.csv")

df.head()
sns.heatmap(df.isnull())

plt.title("Detecting missing values")

plt.show()
dfcorr = df.corr()

sns.heatmap(dfcorr, annot=True, annot_kws={"size":12}, cmap="coolwarm")

plt.title("Correlation matrix")

plt.show()
dfcov = df.cov()

sns.heatmap(dfcov, annot=True, annot_kws={"size":12}, cmap="coolwarm")

plt.title("Covariance matrix")

plt.show()
df.describe().transpose()
sns.catplot(x="Age", y="Pregnancies",data=df,height=6, aspect=3, kind="swarm")

plt.title("Age vs Pregnancies")

plt.show()
sns.catplot(x="Age", y="BloodPressure",data=df,height=6, aspect=3)

plt.title("Age vs BloodPressure")

plt.show()
sns.catplot(x="Pregnancies", y="BloodPressure",data=df,height=6, aspect=3)

plt.title("Pregnancies vs BloodPressure")

plt.show()
sns.pairplot(df)
x = df[['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']]

y = df.iloc[::,-1]

x_constant = sm.add_constant(x)

model = sm.Logit(y,x_constant).fit()

model.predict(x_constant)

model.summary()
x = df[['Pregnancies','Glucose','BloodPressure','DiabetesPedigreeFunction']]

y = df.iloc[::,-1]

x_constant = sm.add_constant(x)

model = sm.Logit(y,x_constant).fit()

model.predict(x_constant)

model.summary()
x_trainlogreg, x_testlogreg, y_trainlogreg, y_testlogreg = train_test_split(x,y,test_size=0.2,random_state=0)

scalerlogreg = StandardScaler()

x_trainlogreg = scalerlogreg.fit_transform(x_trainlogreg)

x_testlogreg = scalerlogreg.transform(x_testlogreg)

logreg = LogisticRegression()

logreg.fit(x_trainlogreg,y_trainlogreg)
y_predlogreg = logreg.predict(x_testlogreg)

pd.DataFrame({"Actual_Outcome":y_testlogreg, "Predicted_Outcome":y_predlogreg})
from sklearn.metrics import classification_report

reportlogreg = pd.DataFrame(classification_report(y_testlogreg, y_predlogreg, output_dict=True)).transpose()

reportlogreg
logregcmat = pd.DataFrame(metrics.confusion_matrix(y_testlogreg, y_predlogreg), columns = ("Predicted","Predicted"), index=["Actual","Actual"])

logregcmat
y_predlogreg_proba = logreg.predict_proba(x_testlogreg)[::,1]

fprlogreg, tprlogreg, _ = metrics.roc_curve(y_testlogreg, y_predlogreg_proba)

auclogreg =  metrics.roc_auc_score(y_testlogreg, y_predlogreg_proba)

fig, ax = plt.subplots(figsize=(10,10))

plt.plot(fprlogreg, tprlogreg, label = "Logistic Regression , auc " + str(auclogreg), color="navy")

plt.plot([0,1],[0,1],alpha=0.8,lw=2, color="red")

plt.xlim([0.00,1.01])

plt.ylim([0.00,1.01])

plt.title("Validation ROC Curve - Logistic Regression")

plt.xlabel("Specificity")

plt.ylabel("Sensitivity")

plt.legend(loc=4)

plt.show()
precisionrecalllogreg,recalllogreg,thresholdlogreg = metrics.precision_recall_curve(y_testlogreg,y_predlogreg)

apslogreg = metrics.average_precision_score(y_testlogreg,y_predlogreg)

fig, ax = plt.subplots(figsize=(10,10))

plt.plot(precisionrecalllogreg,recalllogreg, label = "Logistic Regression , aps " + str(apslogreg), color="navy")

plt.title("Precision-Recall Curve - Logistic Regression")

plt.xlabel("Precision")

plt.ylabel("Recall")

plt.legend(loc=4)

plt.axhline(y=0.5, color="red", lw=2)

plt.show()
trainsizelogreg, trainscorelogreg, testscorelogreg = learning_curve(LogisticRegression(),x,y,cv=10,n_jobs=-1,scoring="accuracy",train_sizes=np.linspace(0.1,1.0,50))

trainscorelogreg_mean = np.mean(trainscorelogreg,axis=1)

trainscorelogreg_std = np.std(trainscorelogreg,axis=1)

testscorelogreg_mean = np.mean(testscorelogreg,axis=1)

testscorelogreg_std = np.std(testscorelogreg,axis=1)

plt.plot(trainsizelogreg, trainscorelogreg_mean, label ="Training score")

plt.plot(trainsizelogreg,testscorelogreg_mean,label="Cross-validation score")

plt.title("Learning Curve - Logistic Regression")

plt.xlabel("Training Set Size")

plt.ylabel("Accuracy Score")

plt.show()