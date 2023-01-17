#import libraries

import pandas as pd

import numpy as np

import statsmodels.api as sm

import scipy.stats as st

import matplotlib.pyplot as plt

import seaborn as sn

from sklearn.metrics import confusion_matrix

from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import roc_auc_score

import matplotlib.mlab as mlab

%matplotlib inline
#load data

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        

def read_file(file):

    x = open(file,'r', encoding = 'utf-8') #Opens the text file into variable x but the variable cannot be explored yet

    y = x.read() #By now it becomes a huge chunk of string that we need to separate line by line

    content = y.splitlines() #The splitline method converts the chunk of string into a list of strings

    return content

df = pd.read_csv('../input/framingham-heart-study-dataset/framingham.csv')
df.head()
df.isnull().sum()
df.drop(['education'],axis=1,inplace=True)

df.head()
count=0

for i in df.isnull().sum(axis=1):

    if i>0:

        count=count+1

print('Total number of rows with missing values is ', count)

print('since it is only',round((count/len(df.index))*100), 'percent of the entire dataset the rows with missing values are excluded.')
df.dropna(axis=0,inplace=True)
df.TenYearCHD.value_counts()
sn.countplot(x='TenYearCHD',data=df)
df['male'].value_counts().head(10).plot.barh() # Top 10 Dates on which the most number of messages were sent

plt.xlabel('count')

plt.ylabel('Gender')
df.describe()
df.shape
plt.figure(figsize = (10, 10))

sn.heatmap(df.corr(), annot = True)

plt.show()
#scatterplot

sn.set()

cols = ['age','male','cigsPerDay','totChol','sysBP','glucose','TenYearCHD']

sn.pairplot(df[cols], size = 2.5)

plt.show();
import sklearn



new_features=df[['age','male','cigsPerDay','totChol','sysBP','glucose','TenYearCHD']]

x=new_features.iloc[:,:-1]

y=new_features.iloc[:,-1]

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.20,random_state=5)
from sklearn.linear_model import LogisticRegression

logreg=LogisticRegression()

logreg.fit(x_train,y_train)

y_pred=logreg.predict(x_test)
sklearn.metrics.accuracy_score(y_test,y_pred)
from sklearn.metrics import confusion_matrix

cm=confusion_matrix(y_test,y_pred)

conf_matrix=pd.DataFrame(data=cm,columns=['Predicted:0','Predicted:1'],index=['Actual:0','Actual:1'])

plt.figure(figsize = (8,5))

sn.heatmap(conf_matrix, annot=True,fmt='d',cmap="YlGnBu")
TN=cm[0,0]

TP=cm[1,1]

FN=cm[1,0]

FP=cm[0,1]

sensitivity=TP/float(TP+FN)

specificity=TN/float(TN+FP)





print('The acuuracy of the model = TP+TN/(TP+TN+FP+FN) = ',(TP+TN)/float(TP+TN+FP+FN),'\n',



'The Missclassification = 1-Accuracy = ',1-((TP+TN)/float(TP+TN+FP+FN)),'\n',



'Sensitivity or True Positive Rate = TP/(TP+FN) = ',TP/float(TP+FN),'\n',



'Specificity or True Negative Rate = TN/(TN+FP) = ',TN/float(TN+FP),'\n',



'Positive Predictive value = TP/(TP+FP) = ',TP/float(TP+FP),'\n',



'Negative predictive Value = TN/(TN+FN) = ',TN/float(TN+FN),'\n',



'Positive Likelihood Ratio = Sensitivity/(1-Specificity) = ',sensitivity/(1-specificity),'\n',



'Negative likelihood Ratio = (1-Sensitivity)/Specificity = ',(1-sensitivity)/specificity)
from sklearn.preprocessing import binarize

for i in range(1,5):

    cm2=0

    y_pred_prob_yes=logreg.predict_proba(x_test)

    y_pred2=binarize(y_pred_prob_yes,i/10)[:,1]

    cm2=confusion_matrix(y_test,y_pred2)

    print ('With',i/10,'threshold the Confusion Matrix is ','\n',cm2,'\n',

            'with',cm2[0,0]+cm2[1,1],'correct predictions and',cm2[1,0],'Type II errors( False Negatives)','\n\n',

          'Sensitivity: ',cm2[1,1]/(float(cm2[1,1]+cm2[1,0])),'Specificity: ',cm2[0,0]/(float(cm2[0,0]+cm2[0,1])),'\n\n\n')
from sklearn.metrics import roc_curve

fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob_yes[:,1])

plt.plot(fpr,tpr)

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.0])

plt.title('ROC curve for Heart disease classifier')

plt.xlabel('False positive rate (1-Specificity)')

plt.ylabel('True positive rate (Sensitivity)')

plt.grid(True)
#auc

sklearn.metrics.roc_auc_score(y_test,y_pred_prob_yes[:,1])