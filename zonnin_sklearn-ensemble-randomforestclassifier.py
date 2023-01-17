import numpy as np

import pandas as pd

from pandas import DataFrame

from sklearn.ensemble import RandomForestClassifier



df = pd.read_csv('../input/train.csv')

Y = df['label']

X = df.drop('label', axis=1)

x_test = pd.read_csv('../input/test.csv')



model = RandomForestClassifier()



model.fit(X, Y)



predicted = model.predict(x_test)



submission = DataFrame(predicted, columns=['Label'], 

                       index=np.arange(1, 28001))

submission.index.names = ['ImageId']

submission