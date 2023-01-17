

import pandas as pd 

iris = pd.read_csv('../input/Iris.csv')
iris.head()
del iris["Id"]
iris.describe()
from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score



features = iris[["SepalLengthCm",	"SepalWidthCm",	"PetalLengthCm",	"PetalWidthCm"]]

op = iris[["Species"]]

op.head()
features_train, features_test, op_train,op_test = train_test_split(features, op, test_size=0.3)
tree = DecisionTreeClassifier()

tree.fit(features_train,op_train)

op_pred_test = tree.predict(features_test)
print("Accuracy", accuracy_score(op_pred_test,op_test))