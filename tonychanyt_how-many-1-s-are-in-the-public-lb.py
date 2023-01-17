import numpy as np

import pandas as pd

from matplotlib import pyplot as plt

from sklearn.metrics import log_loss
l = []

p = [0.369] * 1000

print(p)

for r in range(1, 1000):

    y = [1]*r + [0]*(1000-r)

    #print(y)

    t=log_loss(y, p)

    #print(t)

    l.append(t)

    #print(l)

l = np.array(l)

x = np.arange(0.1, 100, 0.1)
plt.plot(x, l, '_')

plt.title('Log Loss vs. Pct. Positve with Constant Prediction of 0.37')

plt.xlabel('% Positve in LB')

plt.ylabel('Log Loss for Constant Prediction 0.37')

plt.grid()

plt.show()
test = pd.read_csv('../input/test.csv')

sub = test[['test_id']].copy()

sub['is_duplicate'] = 0.174

sub.to_csv('constant_sub.csv', index=False)