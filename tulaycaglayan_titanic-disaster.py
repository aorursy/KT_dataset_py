# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# for drawing graphs 
import matplotlib.pyplot as plt 
import seaborn as sns 

# Classification algorithms 
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
dt_train = pd.read_csv("../input/train.csv")
dt_test = pd.read_csv("../input/test.csv")

print("-" * 5, "train data:", "-" * 5)
dt_train.info()
print("-" * 5, "test data:", "-" * 5)
dt_test.info()
#get passenger Id of test data 
test_PassengerId = dt_test.PassengerId.values 
dt_train.head(5)
# combine these datasets to run certain operations on both datasets together
combined =[dt_train, dt_test]

for dataset in combined:
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
    dataset.drop(['PassengerId', 'Name','Ticket','Cabin'], axis=1, inplace=True)
    dataset['Fare'] = dataset['Fare'].fillna(0)
    dataset['Sex'] = [1 if each =='female' else 0 for each in dataset.Sex ]
    dataset['Embarked'] = [1 if each =='C' else 2 if each =='Q' else 3 if each =='S' else 0 for each in dataset.Embarked ]
    dataset['Age'] = dataset['Age'].fillna((dataset['Age'].mean()))
    dataset['Age'] = [1 if each <17 else 2 if each >= 18 and each <= 35 else 3 if each >= 36 and each <= 45 else 4 if each >= 36 and each <= 55 else 5 if each >= 56 and each <= 65 else 6 if each >65 else 0 for each in dataset.Age ]
    
#Convert Title to Integer
dt_train.Title.unique()
for dataset in combined:
    dataset['Title'] = [1 if each in('Mr','Sir') else 2 if each in('Mrs','Miss','Lady','Ms') else 3 if each in('Capt','Dr','Major') else 4 for each in dataset.Title ]
dt_train.head(10)
plt.subplots(figsize=(8,5))
ax = sns.countplot(dt_train['Age'],hue=dt_train['Survived'],order=[1,2,3,4,5,6])
plt.title('Distribution of Age range survived')
plt.xlabel('Age range')
plt.ylabel('Survived count')
for p in ax.patches:
        ax.annotate(p.get_height(), (p.get_x()+0.1, p.get_height()+50))
        
plt.show()
fig1, ax1 = plt.subplots(figsize=(8,5))
sf = dt_train['Pclass'].value_counts() #Produces Pandas Series
explode =()
for i in range(len(sf.index)):
    if i == 0:
        explode += (0.1,)
    else:
        explode += (0,)

ax1.pie(sf.values, explode=explode,labels=sf.index, autopct='%1.1f%%', shadow=True, startangle=90)
# Equal aspect ratio ensures that pie is drawn as a circle
ax1.axis('equal')  
plt.tight_layout()
plt.legend()
plt.show()
# I want to train test split on train data 
y = dt_train.Survived.values
X = dt_train.drop(['Survived'],axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)
logreg = LogisticRegression()
logreg.fit(X_train, y_train)

y_pred = logreg.predict(X_test)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))
print('weights are : ', logreg.coef_)
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
logit_roc_auc = roc_auc_score(y_test, logreg.predict(X_test))
fpr, tpr, thresholds = roc_curve(y_test, logreg.predict_proba(X_test)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()
y_pred = logreg.predict(dt_test)
survived_test = (y_pred == 0).sum()
not_survived_test = (y_pred == 1).sum()

df_result = pd.DataFrame(y_pred, index=test_PassengerId)
df_result

print('With probabilty of %0.81, ',survived_test,' of test data passengers are survived and ',not_survived_test,' of test data passengers are not survived')