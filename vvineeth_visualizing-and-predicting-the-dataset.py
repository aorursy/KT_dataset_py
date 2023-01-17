import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
data = pd.read_csv('../input/advertising/advertising.csv')
data.head()
sns.set_style('whitegrid')
data['Age'].hist(bins = 30)
sns.jointplot(x = 'Age', y = 'Area Income', data = data)
sns.jointplot(x = 'Age', y = 'Daily Time Spent on Site', data = data, kind = 'kde')
sns.jointplot(x = 'Daily Time Spent on Site', y = 'Daily Internet Usage', data = data, color = 'green')
sns.pairplot(data, hue = 'Clicked on Ad')
#Model building and Predictions
data.info()
from sklearn.model_selection import train_test_split

X = data[['Daily Time Spent on Site', 'Age', 'Area Income','Daily Internet Usage',  'Male']]
y = data['Clicked on Ad']
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size =  0.4, random_state= 101)
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()

# Model Fitting

model.fit(train_X, train_y)
prediction = model.predict(test_X)
# Model Evaluation
from sklearn.metrics import classification_report
print(classification_report(test_y, prediction))
from sklearn.metrics import confusion_matrix

confusion_matrix(test_y, prediction)
