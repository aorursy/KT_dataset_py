import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

iris = pd.read_csv("../input/Iris.csv")

iris.head()
del iris['Id']

iris.head()
iris["Species"].value_counts()
from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

features = iris[[0,1,2,3]]

op = iris[[4]]

print(features.head())

print(op.head())
features_train, features_test, op_train, op_test = train_test_split(features,op,test_size=0.3)
tree = DecisionTreeClassifier()

tree.fit(features_train,op_train)

op_pred_tree = tree.predict(features_test)
print("Accuracy",accuracy_score(op_test,op_pred_tree))
x1=pd.DataFrame([{'SepalLengthCm': 4.9, 'SepalWidthCm': 3.0, 'PetalLengthCm': 1.4, 'PetalWidthCm': 0.2}])

test_pred = tree.predict(x1)

print(test_pred)