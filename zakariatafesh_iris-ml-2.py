import pandas as pd

from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import classification_report, accuracy_score
iris_file = '../input/iris-data-2/iris.csv'

data_iris = pd.read_csv(iris_file)

data_iris = data_iris.dropna(axis=0)

data_iris.columns

features = ['sepal.length', 'sepal.width', 'petal.length', 'petal.width']

X = data_iris[features]

y = data_iris.variety

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1 , test_size = 0.2)

iris_model = DecisionTreeClassifier(random_state=1)

# Fit Model

iris_model.fit(train_X, train_y)



# Make validation predictions and calculate mean absolute error

val_predictions = iris_model.predict(val_X)

#val_mae = mean_absolute_error(val_predictions, val_y)

#print("Validation MAE: {:,.0f}".format(val_mae))

print(accuracy_score(y_true= val_y ,y_pred= val_predictions))
print(classification_report(val_y, val_predictions))