import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

iris = pd.read_csv("../input/Iris.csv")

iris.head()
del iris["Id"]
iris["Species"].value_counts()
from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score



features = iris[[0,1,2,3]]

op = iris[[4]]
features_train, features_test, op_train, op_test = train_test_split(features, op, test_size =0.3)
tree = DecisionTreeClassifier()

tree.fit(features_train,op_train)

op_pred_tree = tree.predict(features_test)
print("Accuracy:", accuracy_score(op_pred_tree,op_test))
x1 = pd.DataFrame([{'S1':1.5,'S2':2.0,'P1':4.2,'P2':3.5}])

op1 = tree.predict(x1)

print(op1)