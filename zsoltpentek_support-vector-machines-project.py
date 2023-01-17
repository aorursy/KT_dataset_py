import seaborn as sns
import pandas as pd
iris = pd.read_csv('../input/iris-flower-dataset/IRIS.csv')
iris.head()
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
sns.pairplot(data=iris, hue='species')
setosa = iris[iris['species']=='setosa']
sns.kdeplot( setosa['sepal_width'], setosa['sepal_length'],
                 cmap="plasma", shade=True, shade_lowest=False)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(iris.drop('species', axis=1), iris['species'])
from sklearn.svm import SVC
svc = SVC()
svc.fit(X_train, y_train)
predictions = svc.predict(X_test)
from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))
from sklearn.model_selection import GridSearchCV
param_grid = {'C':[0.1,1,10,100], 'gamma':[0.1,0.01, 0.001, 1]}
grid = GridSearchCV(svc, param_grid, verbose=3)
grid.fit(X_train, y_train)
pred = grid.predict(X_test)
confusion_matrix(y_test, pred)
print(classification_report(y_test, pred))