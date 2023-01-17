import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
loans = pd.read_csv('../input/loan-data/loan_data.csv')
loans.info()
loans.describe()
loans.head()
plt.figure(figsize=(10,6))
loans[loans['credit.policy']==1]['fico'].hist(alpha=0.5,color='blue',
                                              bins=30,label='Credit.Policy=1')
loans[loans['credit.policy']==0]['fico'].hist(alpha=0.5,color='red',
                                              bins=30,label='Credit.Policy=0')
plt.legend()
plt.xlabel('FICO')
plt.figure(figsize=(10,4))
loans[loans['not.fully.paid']==1]['fico'].hist(color='blue', bins=30, label='not fully paid = 1', alpha=0.5)
loans[loans['not.fully.paid']==0]['fico'].hist(color='red', bins=30, label='not fully paid = 0', alpha=0.5)
plt.legend()
plt.xlabel('Not Fully Paid')
sns.set_style('darkgrid')
sns.countplot(x='purpose', hue='not.fully.paid', data=loans, palette='Set1')
sns.jointplot(data=loans, x='fico', y='int.rate', color='green')
sns.lmplot(x='fico', y='int.rate', data=loans, hue='credit.policy', col='not.fully.paid')
loans.info()
cat_feats = ['purpose']
final_data = pd.get_dummies(data=loans,columns=cat_feats,drop_first=True)
final_data.head()
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(final_data.drop('not.fully.paid', axis=1), final_data['not.fully.paid'], test_size=0.33, random_state=42)
from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier()
dtree.fit(X_train, y_train)
predictions = dtree.predict(X_test)
from sklearn.metrics import confusion_matrix, classification_report
print(classification_report(y_test, predictions))
print(confusion_matrix(y_test, predictions))
from sklearn.ensemble import RandomForestClassifier
rdf = RandomForestClassifier()
rdf.fit(X_train, y_train)
predicts = rdf.predict(X_test)
print(classification_report(y_test, predicts))
confusion_matrix(y_test, predicts)