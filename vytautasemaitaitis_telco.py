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
from sklearn.model_selection import train_test_split
from sklearn import linear_model
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
from sklearn.datasets import make_classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from pylab import rcParams
df = pd.read_csv('../input/telco-customer-churn/WA_Fn-UseC_-Telco-Customer-Churn.csv')
df.head()
uniq = pd.DataFrame({'Diferent_value_count': df.nunique(), 'DTypes': df.dtypes})
print ("Rows     : " ,df.shape[0])
print ("Columns  : " ,df.shape[1])
print ("\nFeatures : \n" ,df.columns.tolist())
print ("\nMissing values :  ", df.isnull().sum().values.sum())
print ("\nUnique values :  \n", uniq)
df[df['TotalCharges'] ==' ']
df.loc[df['TotalCharges'] ==' ','TotalCharges'] = df['MonthlyCharges'] 
df[df['TotalCharges'] ==' ']
df["TotalCharges"] = df["TotalCharges"].astype(float)
df['Contract_term'] = df["Contract"].replace({'Month-to-month':0, 'One year':1, 'Two year':1})
df.drop(['customerID', 'Contract'], axis = 1, inplace = True)
df['MultipleLines'] = df['MultipleLines'].replace({'No phone service':0, 'No' : 0, 'Yes' : 1})
replace_col_val = [ 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
for i in replace_col_val : 
    df[i]  = df[i].replace({'No internet service' : 'No'})
replace_col_val2 = ['Partner', 'Dependents', 'PhoneService', 'OnlineSecurity',
                   'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
                   'StreamingMovies', 'PaperlessBilling', 'Churn']
for i in replace_col_val2 : 
    df[i]  = df[i].replace({'Yes' : 1, 'No': 0})
df.head()
colors = ['gold', 'yellowgreen']
explode = (0.1, 0)
labels = 'Yes', 'No'
No, Yes= df['Churn'].value_counts(sort = True)
sizes = [Yes, No]
rcParams['figure.figsize'] = 5,5
plt.pie(sizes, explode=explode, labels=labels, colors=colors,
        autopct='%1.1f%%', shadow=True, startangle=140,)
plt.title('Percent of churn in customer')
plt.show()
print('Churn Yes -', Yes)
print('Churn No -', No)
fig, axis = plt.subplots(1,2, figsize = (10,10))
colors = ['gold', 'yellowgreen']
explode = (0.1, 0)
labels = 'Yes', 'No'
axis[0].set_title('Percent of churn in Male')
axis[1].set_title('Percent of churn in Female')

No, Yes= df[df['gender']== 'Male'].Churn.value_counts()
sizes = [Yes, No]
axis[0].pie(sizes, explode=explode, labels=labels, colors=colors,
        autopct='%1.1f%%', shadow=True, startangle=140,)

No, Yes= df[df['gender']!= 'Male'].Churn.value_counts()
sizes = [Yes, No]
axis[1].pie(sizes, explode=explode, labels=labels, colors=colors,
        autopct='%1.1f%%', shadow=True, startangle=140,)

plt.show()
colum = ['SeniorCitizen', 'Partner', 'Dependents', 'MultipleLines',
       'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
       'StreamingTV', 'StreamingMovies', 'PaperlessBilling',
       'Contract_term']
i = 0
j= 0
fig, axis = plt.subplots(6, 4,figsize = (20,30))
colors = ['gold', 'yellowgreen']
explode = (0.1, 0)
labels = 'Yes', 'No'
for r in colum:
    if i == 6:
        pass
    else:
        No, Yes= df[df[r]== 1].Churn.value_counts()
        sizes = [Yes, No]
        axis[i,j].pie(sizes, explode=explode, labels=labels, colors=colors,
        autopct='%1.1f%%', shadow=True, startangle=140)
        axis[i,j].set_title(r + ' is Yes')
        No, Yes= df[df[r]== 0].Churn.value_counts()
        sizes = [Yes, No]
        j+=1
        axis[i,j].pie(sizes, explode=explode, labels=labels, colors=colors,
        autopct='%1.1f%%', shadow=True, startangle=140)
        axis[i,j].set_title(r + ' is No')
        if j < 3:
            j+=1
        else:
            i += 1
            j = 0
df.groupby('Churn').Contract_term.value_counts()
df_Contract_term = df[df['Contract_term'] == 0]
df_Contract_term.drop(['Contract_term'], axis = 1, inplace = True)
df_Contract_term.head()
colum2 = ['SeniorCitizen', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
       'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
       'StreamingTV', 'StreamingMovies', 'PaperlessBilling']
i = 0
j= 0
fig, axis = plt.subplots(6, 4,figsize = (20,30))
colors = ['gold', 'yellowgreen']
explode = (0.1, 0)
labels = 'Yes', 'No'
for r in colum2:
    if i == 6:
        pass
    else:
        No, Yes= df_Contract_term[df_Contract_term[r]== 1].Churn.value_counts()
        sizes = [Yes, No]
        axis[i,j].pie(sizes, explode=explode, labels=labels, colors=colors,
        autopct='%1.1f%%', shadow=True, startangle=140)
        axis[i,j].set_title(r + ' is Yes')
        No, Yes= df_Contract_term[df_Contract_term[r]== 0].Churn.value_counts()
        sizes = [Yes, No]
        j+=1
        axis[i,j].pie(sizes, explode=explode, labels=labels, colors=colors,
        autopct='%1.1f%%', shadow=True, startangle=140)
        axis[i,j].set_title(r + ' is No')
        if j < 3:
            j+=1
        else:
            i += 1
            j = 0
df_Contract_term.drop(['gender'], axis = 1, inplace = True)
df3 = df_Contract_term
df3.describe()
df3[['tenure','MonthlyCharges','TotalCharges','Churn']].corr()
sns.pairplot(df_Contract_term[['tenure','Churn']], hue='Churn', height=5)
sns.pairplot(df_Contract_term[['MonthlyCharges','Churn']], hue='Churn', height=5)
sns.pairplot(df_Contract_term[['TotalCharges','Churn']], hue='Churn', height=5)
plt.figure(figsize = (15,7))
sns.countplot(df_Contract_term['tenure'])
sns.pairplot(df_Contract_term[['MonthlyCharges','Churn']], hue='Churn', height=5)
sns.pairplot(df_Contract_term[['tenure','MonthlyCharges','TotalCharges','Churn']], hue='Churn')
bins = [0, 4, 36, 500]
labels = ['<=4','5 - 36','>36']
df3['tenure_group'] = pd.cut(df3['tenure'], bins=bins, labels=labels)
matrix = np.triu(df3.corr())
plt.figure(figsize=(30,10))
cmap = sns.diverging_palette(500, 1)
sns.heatmap(df3.corr().round(2), mask=matrix, vmin=-0.7, vmax=0.7, annot=True, 
            cmap=cmap,
            square=True)
df3_dummies = pd.get_dummies(df3, drop_first=True)
df3 = df3_dummies
matrix = np.triu(df3.corr())
plt.figure(figsize=(30,10))
cmap = sns.diverging_palette(500, 1)
sns.heatmap(df3.corr().round(2), mask=matrix, vmin=-0.7, vmax=0.7, annot=True, 
            cmap=cmap,
            square=True)
df3.drop(['TotalCharges'], axis = 1, inplace = True)
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import statsmodels.formula.api as smf
X = df3.drop('Churn', axis = 1)
y = df3['Churn']
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 42)
logistic = LogisticRegression()
logistic.fit(X_train, y_train)
predictions = logistic.predict(X_test)
predictions
conf_matrix = metrics.confusion_matrix(y_test, predictions)
conf_matrix
fig, ax = plt.subplots()
ax= plt.subplot()
sns.heatmap(conf_matrix,annot=True, ax = ax, cmap='YlGnBu', fmt='d')
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual') 
ax.set_title('Confusion Matrix');
ax.xaxis.set_ticklabels(['0 - Not churn', '1 - Churn'])
ax.yaxis.set_ticklabels(['0 - Not churn', '1 - Churn'])
Accuracy = metrics.accuracy_score(y_test, predictions)
Precision = metrics.precision_score(y_test, predictions)
recall = metrics.recall_score(y_test, predictions)
F1 = 2 * (Precision * recall) / (Precision + recall)
def plot_roc_curve(fpr, tpr):
    plt.plot(fpr, tpr, color='orange', label='ROC')
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.show()
probs = logistic.predict_proba(X_test)
probs = probs[:, 1]
auc = roc_auc_score(y_test, probs)
print('AUC: %.2f' % auc)
fpr, tpr, thresholds = roc_curve(y_test, probs)
plot_roc_curve(fpr, tpr)
print('Accuracy :', Accuracy.round(3))
print('Precision :', Precision.round(3))
print('recall :', recall.round(3))
print('F-1 :', F1.round(3))
print('Area under the curve :', auc.round(3))
df3.columns = ['SeniorCitizen', 'Partner', 'Dependents', 'tenure', 'PhoneService', 'MultipleLines',
       'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
       'StreamingMovies', 'PaperlessBilling', 'MonthlyCharges',
       'Churn', 'InternetService_Fiber_optic', 'InternetService_No',
       'PaymentMethod_Credit_card_automatic',
       'PaymentMethod_Electronic_check', 'PaymentMethod_Mailed_check','tenure_group_5_36', 'tenure_group_36']
est2 = smf.logit('Churn ~ SeniorCitizen + Partner+Dependents +tenure+PhoneService+MultipleLines + OnlineSecurity + OnlineBackup +DeviceProtection+ TechSupport + StreamingTV + StreamingMovies + PaperlessBilling + MonthlyCharges+InternetService_Fiber_optic + InternetService_No + PaymentMethod_Credit_card_automatic+PaymentMethod_Electronic_check+PaymentMethod_Mailed_check+tenure_group_5_36+tenure_group_36',df3).fit()
est2.summary()
df3.drop(['Partner', 'Dependents', 'PhoneService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
          'TechSupport', 'StreamingTV',
          'StreamingMovies', 'MonthlyCharges',
          'InternetService_No', 'PaymentMethod_Credit_card_automatic','PaymentMethod_Mailed_check', 'tenure_group_36'], axis = 1, inplace = True)
X = df3.drop('Churn', axis = 1)
y = df3['Churn']
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 42)
logistic = LogisticRegression()
logistic.fit(X_train, y_train)
predictions = logistic.predict(X_test)
predictions
conf_matrix = metrics.confusion_matrix(y_test, predictions)
conf_matrix
fig, ax = plt.subplots()
ax= plt.subplot()
sns.heatmap(conf_matrix,annot=True, ax = ax, cmap='YlGnBu', fmt='d')
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual') 
ax.set_title('Confusion Matrix');
ax.xaxis.set_ticklabels(['0 - Not churn', '1 - Churn'])
ax.yaxis.set_ticklabels(['0 - Not churn', '1 - Churn'])
Accuracy = metrics.accuracy_score(y_test, predictions)
Precision = metrics.precision_score(y_test, predictions)
recall = metrics.recall_score(y_test, predictions)
F1 = 2 * (Precision * recall) / (Precision + recall)
def plot_roc_curve(fpr, tpr):
    plt.plot(fpr, tpr, color='orange', label='ROC')
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.show()
probs = logistic.predict_proba(X_test)
probs = probs[:, 1]
auc = roc_auc_score(y_test, probs)
print('AUC: %.2f' % auc)
fpr, tpr, thresholds = roc_curve(y_test, probs)
plot_roc_curve(fpr, tpr)
print('Accuracy :', Accuracy.round(3))
print('Precision :', Precision.round(3))
print('recall :', recall.round(3))
print('F-1 :', F1.round(3))
print('Area under the curve :', auc.round(3))
df3.columns
est2 = smf.logit('Churn ~ SeniorCitizen+tenure+MultipleLines+PaperlessBilling+InternetService_Fiber_optic+PaymentMethod_Electronic_check+tenure_group_5_36',df3).fit()
est2.summary()
