import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
#loading the dataset 
data=pd.read_csv('../input/turnover.csv')
#checking the head of the our data
data.head()
data.info()
data.describe()
sns.countplot(data['left'],hue=data['Work_accident'])
sns.distplot(data['average_montly_hours'],color='red',bins=50,kde=False)
fig,axes=plt.subplots(figsize=(10,6))
sns.heatmap(data.corr(),annot=True,cmap='plasma')
sns.pairplot(data)
data.info()
final_data=pd.get_dummies(data,columns=['sales','salary'],drop_first=True)
final_data.info()
final_data.head()
from sklearn.cross_validation import train_test_split
X=final_data.drop('left',axis=1)
y=final_data['left']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
from sklearn.tree import DecisionTreeClassifier
dtc_model=DecisionTreeClassifier()
dtc_model.fit(X_train,y_train)
dtc_predictions=dtc_model.predict(X_test)
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
print('Classfication Report - \n',classification_report(y_test,dtc_predictions))
print('\n')
print('confusion_matrix  - \n',confusion_matrix(y_test,dtc_predictions))

print('Accuracy of the Model -',accuracy_score(y_test,dtc_predictions))
