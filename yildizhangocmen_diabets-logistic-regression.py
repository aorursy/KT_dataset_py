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
import statsmodels.api as sm
import seaborn as sns 
import matplotlib.pyplot as plt 
from sklearn.preprocessing import scale , StandardScaler
from sklearn.model_selection import train_test_split ,GridSearchCV ,cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score ,mean_squared_error ,r2_score , roc_auc_score ,roc_curve ,classification_report
from sklearn.linear_model import LogisticRegression 



#Reading CSV File 
df =pd.read_csv("../input/pima-indians-diabetes-database/diabetes.csv")
df.head()
df["Outcome"].value_counts()
# Observing the descriptive statistics of independent variables.
df.describe().T
#Since the maximum values are high, it can be said that there are outliers in pregnancies.
y =df["Outcome"]
X = df.drop(["Outcome"], axis =1)
# we called dependent y ,other variables x independent variable
# also "outcome" variable done drop
y.head()
X.head()
log_model = LogisticRegression(solver = "liblinear").fit(X,y)
#fixed value 
log_model.intercept_
#coefficient of independent variables

log_model.coef_
log_model.predict(X)[0:10]
#Evaluation of the success of the established model.
#The predicted values are called y_pred

y_pred = log_model.predict(X)
#We will be evaluating errors using the complexity matrix.
#y = real variable 
#y_predict = predict variable
confusion_matrix(y,y_pred)
#Acccuracy_score = successful predictions / all occasions
accuracy_score(y,y_pred)
#Other method classification report
print(classification_report(y,y_pred))
log_model.predict_proba(X)[0:10]
logit_roc_auc = roc_auc_score(y,log_model.predict(X))
fpr,tpr,thresholds = roc_curve(y,log_model.predict_proba(X)[:,1])
plt.figure()
plt.plot(fpr,tpr ,label = "AUC ( area = % 0.2f)"% logit_roc_auc) #labels ,#grafiğin üzerine değer yansıtmak için kullanılan kısım
plt.plot ([0,1],[0,1],"r--")
plt.xlim([0.0 , 1.0])
plt.ylim([0.0 , 1.05])          # eksen ayarlamaları
plt.xlabel("False Positive Rate ")
plt.ylabel( "True Positive Rate")
plt.title("Receiver operating characteristic")
plt.legend(loc = "lower right")
plt.savefig("Log_ROC")
plt.show()

#fpr false positive rate
#tpr true positive rate 