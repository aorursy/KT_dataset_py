import numpy as np

import pandas as pd

from sklearn import datasets

from sklearn.model_selection import train_test_split

from catboost import CatBoostClassifier, Pool



train = pd.read_csv('files/train.csv')
Y = train.pop('label').values

X = train.values
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.03, random_state=0)



train_pool = Pool(data=X_train, label=y_train)

test_pool = Pool(data=X_test, label=y_test)





model = CatBoostClassifier(iterations=100, random_seed=0, loss_function="MultiClass", verbose=True)

model.fit(train_pool, use_best_model=True, eval_set=test_pool)
model.score(X_test, y_test)
test = pd.read_csv('files/test.csv')
results = model.predict(test.values)
out_file = open("files/predictions.csv", "w")

out_file.write("ImageId,Label\n")

for i in range(len(results)):

    out_file.write(str(i+1) + "," + str(int(results[i])) + "\n")

out_file.close()