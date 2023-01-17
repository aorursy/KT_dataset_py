import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import numpy as np

from sklearn.decomposition import PCA

from sklearn.svm import SVC

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.cross_validation import cross_val_score
data = pd.read_csv("../input/train.csv")

X = data.iloc[:, 1:]

y = data.iloc[:, 0]

data.shape
pca = PCA(n_components=35, whiten=True)

pca.fit(X)

red_X = pca.transform(X)
gbc = SVC()

gbc.fit(red_X, y)


test_data = pd.read_csv("../input/test.csv")

predicted = gbc.predict(pca.transform(test_data))
pd.DataFrame({"ImageId": list(range(1, len(test_data)+1)), 

              "Label": predicted}).to_csv('submission_gbc.csv', index=False, header=True)