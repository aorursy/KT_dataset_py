import pandas as pd

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.grid_search import GridSearchCV

from sklearn.metrics import classification_report, confusion_matrix

from sklearn.svm import SVC
df = pd.read_csv('../input/voice.csv')

df.head()
df.info()
X = df.drop('label', axis = 1)

y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 45)
model = SVC()
model.fit(X_train, y_train)
prediction = model.predict(X_test)
print(classification_report(y_test,prediction))

print(confusion_matrix(y_test, prediction))
param_grid = {'C':[1,10,100,1000],'gamma':[1,0.1,0.001,0.0001], 'kernel':['linear','rbf']}
grid = GridSearchCV(SVC(),param_grid,refit = True, verbose=2)
grid.fit(X_train,y_train)
grid.best_params_
predic = grid.predict(X_test)
print(classification_report(y_test,predic))

print(confusion_matrix(y_test, predic))