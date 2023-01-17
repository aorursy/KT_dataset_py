import numpy as np

import pandas as pd

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import classification_report
data = pd.read_csv("../input/mushrooms.csv")
data.head()
data.describe()
data = data.apply(LabelEncoder().fit_transform)
X = data.iloc[:,1:]

y = data.iloc[:, 0]
X.columns
cols = ['cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor',

       'gill-attachment', 'gill-spacing', 'gill-size', 'gill-color',

       'stalk-shape', 'stalk-root', 'stalk-surface-above-ring',

       'stalk-surface-below-ring', 'stalk-color-above-ring',

       'stalk-color-below-ring', 'veil-type', 'veil-color', 'ring-number',

       'ring-type', 'spore-print-color', 'population', 'habitat']
for col in cols:

    X[col] = X[col].astype("category")
X.describe()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
rfc = RandomForestClassifier(n_estimators=200)
rfc.fit(X_train, y_train)
p = rfc.predict(X_test)
print(classification_report(y_test, p))