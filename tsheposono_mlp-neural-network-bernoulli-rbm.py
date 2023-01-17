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
import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

sns.set("talk","whitegrid",font_scale=1,font="sans-serif",color_codes=True)

from sklearn import metrics

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split, learning_curve

from sklearn.neural_network import MLPClassifier

from pylab import rcParams

plt.rcParams["figure.figsize"] = [10,10]

import warnings

warnings.filterwarnings("ignore")

df = pd.read_csv(r"../input/pima-indians-diabetes-database/diabetes.csv")

df.head()
sns.heatmap(df.isnull())

plt.title("Detect Missing Values")

plt.show()
dfcorr = df.corr()

sns.heatmap(dfcorr, annot=True, annot_kws={"size":12}, cmap="coolwarm")

plt.title("Correlation Matrix")

plt.show()
dfcov = df.cov()

sns.heatmap(dfcov, annot=True, annot_kws={"size":12}, cmap="coolwarm")

plt.title("Covariance Matrix")

plt.show()
x = df[["Pregnancies","Glucose","BloodPressure","SkinThickness","Insulin","BMI","DiabetesPedigreeFunction","Age"]]

y = df.iloc[::,-1]
x_trainMLP, x_testMLP, y_trainMLP, y_testMLP = train_test_split(x,y,test_size=0.2,random_state=0)

scalerMLP = StandardScaler()

x_trainMLP = scalerMLP.fit_transform(x_trainMLP)

x_testMLP = scalerMLP.transform(x_testMLP)

MLP = MLPClassifier()

MLP.fit(x_trainMLP,y_trainMLP)
y_predMLP = MLP.predict(x_testMLP)

pd.DataFrame({"Actual":y_testMLP, "Predicted":y_predMLP})
y_testMLP.mean()
1 - y_testMLP.mean()
classificationreportMLP = pd.DataFrame(metrics.classification_report(y_testMLP,y_predMLP,output_dict=True)).transpose()

classificationreportMLP
cmatMLP = pd.DataFrame(metrics.confusion_matrix(y_testMLP,y_predMLP),columns =("Positive","Negative"),index=["Positive","Negative"])

cmatMLP
y_predMLP_proba = MLP.predict_proba(x_testMLP)[::,1]

fprMLP,tprMLP,_ = metrics.roc_curve(y_testMLP,y_predMLP_proba)

aucMLP = metrics.roc_auc_score(y_testMLP,y_predMLP_proba)

plt.plot(fprMLP,tprMLP,label="MLP Neural Network , auc " +str(aucMLP),color="black")

plt.plot([0,1],[0,1],color="red")

plt.xlim([0.00,1.01])

plt.ylim([0.00,1.01])

plt.xlabel("Specificty")

plt.ylabel("Sensitivity")

plt.title("Validation ROC Curve - Multi-Layer Percepron Neural Networks")

plt.legend(loc=4)

plt.show()
precisionMLP, recallMLP, thresholdMLP = metrics.precision_recall_curve(y_testMLP,y_predMLP)

apsMLP = metrics.average_precision_score(y_testMLP,y_predMLP)

plt.plot(precisionMLP, recallMLP,label="MLP Neural Network , aps " +str(apsMLP),color="black")

plt.axhline(y=0.5,color="red")

plt.xlabel("Precision")

plt.ylabel("Recall")

plt.title("Precision-Recall - Multi-Layer Percepron Neural Networks")

plt.legend(loc=4)

plt.show()
trainsizeMLP, trainscoreMLP, testscoreMLP = learning_curve(MLPClassifier(),x,y,cv=10, n_jobs=-1,scoring="accuracy",train_sizes=np.linspace(0.1,1.0,50))

trainscoreMLP_mean = np.mean(trainscoreMLP,axis=1)

trainscoreMLP_std = np.std(trainscoreMLP,axis=1)

testscoreMLP_mean = np.mean(testscoreMLP,axis=1)

testscoreMLP_std = np.std(testscoreMLP,axis=1)

plt.plot(trainsizeMLP,trainscoreMLP_mean, color="red", label="Training score")

plt.plot(trainsizeMLP,testscoreMLP_mean, color="black", label="Cross-Validation score")

plt.xlabel("Training Size Set")

plt.ylabel("Accuracy Score")

plt.title("Learning Curve - Multi-Layer Percepron Neural Networks")

plt.show()
from sklearn.linear_model import LogisticRegression

from sklearn.neural_network import BernoulliRBM

from sklearn.pipeline import Pipeline

x_trainRBM, x_testRBM, y_trainRBM, y_testRBM = train_test_split(x,y,test_size=0.2,random_state=0)

scalerRBM = StandardScaler()

x_trainRBM = scalerRBM.fit_transform(x_trainRBM)

x_testRBM = scalerRBM.transform(x_testRBM)

logreg = LogisticRegression()

RBM = BernoulliRBM()

classifier = Pipeline(steps=([("rbm",RBM),("logreg",logreg)]))

classifier.fit(x_trainRBM,y_trainRBM)
y_predRBM = classifier.predict(x_testRBM)

pd.DataFrame({"Actual":y_testRBM, "Predicted":y_predRBM})
y_testRBM.mean()
1 - y_testRBM.mean()
cmatRBM = pd.DataFrame(metrics.confusion_matrix(y_testRBM,y_predRBM),columns =("Positive","Negative"),index=["Positive","Negative"])

cmatRBM

y_predRBM_proba = classifier.predict_proba(x_testRBM)[::,1]

fprRBM,tprRBM,_ = metrics.roc_curve(y_testRBM,y_predRBM_proba)

aucRBM = metrics.roc_auc_score(y_testRBM,y_predRBM_proba)

plt.plot(fprRBM,tprRBM,label="Bernoulli RBM Neural Network , auc " +str(aucRBM),color="gray")

plt.plot([0,1],[0,1],color="red")

plt.xlim([0.00,1.01])

plt.ylim([0.00,1.01])

plt.xlabel("Specificty")

plt.ylabel("Sensitivity")

plt.title("Validation ROC Curve - Bernoulli Restricted Boltzman")

plt.legend(loc=4)

plt.show()
precisionRBM, recallRBM, thresholdRBM = metrics.precision_recall_curve(y_testRBM,y_predRBM)

apsRBM = metrics.average_precision_score(y_testRBM,y_predRBM)

plt.plot(precisionRBM, recallRBM,label="Bernoulli RBM Neural Network, aps " +str(apsRBM),color="gray")

plt.axhline(y=0.5,color="red")

plt.xlabel("Precision")

plt.ylabel("Recall")

plt.title("Precision-Recall - Bernoulli Restricted Boltzman")

plt.legend(loc=4)

plt.show()