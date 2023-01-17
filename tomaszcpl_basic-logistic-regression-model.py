import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
data = pd.read_csv('../input/Iris.csv')
data.head()
data.describe()
cols = ["SepalLengthCm", "SepalWidthCm","PetalLengthCm","PetalWidthCm"]

X = data[cols]
y = data['Species']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
print((logreg.score(X_test, y_test)))