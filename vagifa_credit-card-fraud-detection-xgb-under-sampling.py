import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
sns.set_style('darkgrid')

df = pd.read_csv('/kaggle/input/creditcardfraud/creditcard.csv')
df.head()
df.info()
df.describe()
sns.distplot(df['Time'],bins=30)

plt.title('Histogram of Time feature')

plt.show()
sns.distplot(df['Amount'],bins=10)

plt.title('Histogram of Amount feature')

plt.show()
plt.figure(figsize=(10,10))

sns.heatmap(df.corr())

plt.show()
sns.countplot(df['Class'])

plt.show()
from sklearn.model_selection import StratifiedKFold

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()



df['Time'] = scaler.fit_transform(df['Time'].values.reshape(-1,1))

df['Amount'] = scaler.fit_transform(df['Amount'].values.reshape(-1,1))
X = df.drop('Class',axis=1)

y = df['Class']
kf = StratifiedKFold(n_splits=10,random_state=42,shuffle=True)
for train,test in kf.split(X,y):

    X_train,X_test = X.loc[train],X.loc[test],

    y_train,y_test = y.loc[train],y.loc[test]
from xgboost import XGBClassifier
xgb = XGBClassifier(n_estimators=200,learning_rate=0.3,gamma=0.2)
xgb.fit(X_train,y_train)
pred = xgb.predict(X_test)
from sklearn.metrics import classification_report,confusion_matrix,roc_auc_score,roc_curve

from imblearn.metrics import classification_report_imbalanced
def evaluate(prediction,test,imb=False):

    sns.heatmap(confusion_matrix(prediction,test),annot=True)

    plt.show()

    if imb == True:

        print(classification_report(prediction,test))

    else:

        print(classification_report_imbalanced(test,prediction))

    score = roc_auc_score(test,prediction)

    print('AUC Score: ' + str(score))

    fp,tp,_ = roc_curve(test,prediction)



    plt.figure(figsize=(8,8))

    plt.plot(fp,tp,linestyle='dashed',c='orange',label='XGBoostClassifier')

    plt.plot([0,1],[0,1],'r--')

    plt.xlabel('False Positive Rate')

    plt.ylabel('True Positive Rate')

    plt.title('ROC Visualisation')

    plt.legend()

    plt.show()
evaluate(pred,y_test)
from imblearn.under_sampling import RandomUnderSampler
rus = RandomUnderSampler(sampling_strategy=1)



X_rus,y_rus = rus.fit_sample(X,y)
rus_df = pd.concat([X_rus,y_rus],axis=1)

sns.countplot(rus_df['Class'])

plt.show()
undersample_X = rus_df.drop('Class',axis=1)

undersample_y = rus_df['Class']



for train,test in kf.split(undersample_X,undersample_y):

    X_train_rus,X_test_rus = undersample_X.loc[train],undersample_X.loc[test],

    y_train_rus,y_test_rus = undersample_y.loc[train],undersample_y.loc[test]
from sklearn.ensemble import RandomForestClassifier



rf = RandomForestClassifier(n_estimators=200,max_features=2)
rf.fit(undersample_X,undersample_y)
pred_rf = rf.predict(X_test)
evaluate(pred_rf,y_test,imb=True)