import numpy as np

import pandas as pd

from sklearn import metrics

from patsy import dmatrices

import matplotlib.pyplot as plt
data = pd.read_csv('../input/HR_comma_sep.csv')
data
pd.crosstab(data.left, data.salary).plot(kind = 'bar')

plt.show()
data.dtypes
y, X = dmatrices('left~satisfaction_level+last_evaluation+number_project+average_montly_hours+time_spend_company+Work_accident+promotion_last_5years+C(sales)+C(salary)', data, return_type='dataframe')
# 第一列加上截距

X
X = np.asmatrix(X)
y = np.ravel(y)
X
y
X.shape[:] # 14999(组数据) * 19（个特征）
y.shape[:] # 14999(个结果) * 1
# X归一化，第一列截距不计算

for i in range(1, X.shape[1]):

    xmin = X[:, i].min()

    xmax = X[:, i].max()

    X[:, i] = (X[:, i] - xmin) / (xmax - xmin)
X
# 随机产生一组w：19(个特征) * 1

w = np.random.randn(X.shape[1], 1).ravel()

# 定义学习率 learning rate

r = 1
# p = p(yi|xi) = 1 / (1 + exp(-w * xi))

# maximum likelihood：

# l(w) = sum(yi * ln(p) + (1 - yi) * (1 - p)) 

# delta(w) = sum((p(yi|xi) - yi) * xij)

# w = w - delta(w)
# 500 rounds

for i in range(500):

    prob = (1. / (1 + np.exp(-np.matmul(X, w)))).ravel()

    prob = np.array(prob.transpose())

    prob_y = list(zip(prob, y))

    loss = -sum([np.log(p) if y == 1 else np.log(1 - p) for p, y in prob_y])

    err_rate = 0

    for j in range(len(y)):

        if ((prob[j] > 0.5 and y[j] == 0) or (prob[j] <= 0.5 and y[j] == 1)):

            err_rate += 1

    err_rate /= len(y)

    if i % 5 == 0:

        print("i = " + str(i) + ", loss = " + str(loss) + ", error rate = " + str(err_rate))

    delta_w = np.zeros(X.shape[1])

    for j in range(len(y)):

        delta_w += (prob[j] - y[j]) * np.asarray(X[j, :]).ravel()

    delta_w /= len(y)

    w -= r * delta_w