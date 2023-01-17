import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('../input/tv_sleep.csv', index_col='child')
print(data)
plt.scatter(data.tv_time, data.deep_sleep)
_ = plt.show()
from sklearn import gaussian_process as gauss, tree, svm

features = data[['tv_time']]
model = svm.SVR(gamma='auto', degree=3)
model = model.fit(features, data.deep_sleep)

def predict(tv_time):
    return model.predict([[tv_time]])
test = pd.DataFrame(
    data=[(tv_time, predict(tv_time)[0]) for tv_time in np.linspace(0,4, num=9)],
    columns=['tv_time', 'deep_sleep']
)
print(test)
plt.scatter(test.tv_time, test.deep_sleep)
_ = plt.show()