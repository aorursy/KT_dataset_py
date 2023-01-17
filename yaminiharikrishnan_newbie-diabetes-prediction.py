import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
data=pd.read_csv('../input/pima-indians-diabetes-database/diabetes.csv')
data.head()
Features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness','Insulin','BMI','DiabetesPedigreeFunction', 'Age']

X = data[Features]   

y = data.Outcome 
data.describe() 
data.info() 
data.isnull()
data.isnull().sum()
plt.rcParams['figure.figsize'] = (40, 41)

plt.style.use('dark_background')



sns.pairplot(data, hue = 'Outcome', palette = 'husl')

plt.title('Pair plot for the data', fontsize = 40)

plt.show()
plt.rcParams['figure.figsize'] = (15, 15)



sns.heatmap(data.corr(), annot = True)

plt.title('Correlation Plot')

plt.show()
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_scaled = sc.fit_transform(X)

X_scaled
pd.DataFrame(X_scaled)
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(X_scaled, y, test_size = 0.33,random_state=42) 

x_train.shape
from sklearn import linear_model

from sklearn.linear_model import LogisticRegression

clf = LogisticRegression()



clf.fit(x_train,y_train)

y_pred=clf.predict(x_test)
print(y_pred)
from sklearn.metrics import accuracy_score

print(accuracy_score(clf.predict(x_train), y_train))

print(accuracy_score(y_pred, y_test))
from sklearn.metrics import confusion_matrix 

confusion_matrix(y_pred, y_test)
from sklearn.metrics import classification_report

classification_report(y_pred, y_test)