import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sbs

sbs.set()
train_set = pd.read_csv('../input/train.csv')

test_set = pd.read_csv('../input/test.csv')



train_set.head()
print(train_set.iloc[3, 0])

image_array = np.asfarray(train_set.iloc[3, 1:]).reshape((28, 28))

plt.imshow(image_array, cmap = 'Greys', interpolation = 'None')
X_train = train_set.iloc[:, 1:].values

y_train = train_set.label.values
from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier(criterion='entropy', n_estimators=10, max_depth=3, random_state=0)

classifier.fit(X_train, y_train)
classifier.score(X_train, y_train)
test_set.head()
pred = classifier.predict(test_set)

pred
sub_lines = []



for i in range(0, len(pred)):

    sub_lines.append([i + 1, pred[i]])

    

submission = pd.DataFrame(sub_lines, columns=['ImageId', 'Label'])

submission.to_csv('submission.csv', index=False)

submission.head()