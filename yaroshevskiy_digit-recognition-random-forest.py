import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn import ensemble, cross_validation, tree

import math



data = pd.read_csv("../input/train.csv")

test_data = pd.read_csv("../input/test.csv")



X = data.iloc[:, 1:]

y = data.iloc[:, 0]

X.head()
rfClassifier = ensemble.RandomForestClassifier(n_estimators = 180, 

                                               max_features = int(math.sqrt(X.shape[1])))

cross_validation.cross_val_score(rfClassifier, X, y, scoring = 'accuracy', cv = 3).mean()
predicted = rfClassifier.fit(X, y).predict(test_data)
pd.DataFrame({"ImageId": list(range(1, len(test_data) + 1)), 

              "Label": predicted}).to_csv('submission_rf.csv', index=False, header=True)