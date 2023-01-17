from numpy import loadtxt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

dataset = loadtxt('../input/pima-indians-diabetes.csv', delimiter=",")
# split data into X and y (featuers and target)
X = dataset[:,0:8]
Y = dataset[:,8]

# split data into train and test sets
seed = 7
test_size = 0.33
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)

# fit model no training data
model = XGBClassifier()
model.fit(X_train, y_train)
# make predictions for test data
y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]
# evaluate predictions
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report
expected = y_test
print("XgBoost Accuracy", accuracy_score(expected,y_pred))
print(classification_report(expected, y_pred,target_names=['No', 'Yes']))
print(confusion_matrix(expected, y_pred))
