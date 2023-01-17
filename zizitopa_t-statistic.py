import numpy as np

import pandas as pd

import random

import math
random.seed(42)

np.random.seed(42)

mu = random.randint(20, 50)



sigma = 5
array = sigma * np.random.randn(10000) + mu
m_0 = 30
n = 20
X_20 = np.random.choice(array, 20)

X_20
X_bar = X_20.mean()

X_bar
t = (X_bar - m_0) * math.sqrt(n) /  sigma

t
m_1 = X_bar
X_20_1 = np.random.choice(array, 20)

X_20_1
X_bar_1 = X_20_1.mean()

X_bar_1
t_1 = (X_bar_1 - m_1) * math.sqrt(n) /  sigma

t_1
print('real mu = ' + str(mu))

print('my mu = ' + str(m_1))
random.seed(42)

np.random.seed(42)



mu_2 = random.randint(10, 80)

sigma_2 = random.randint(5, 10)
array_2 = sigma_2 * np.random.randn(10000) + mu_2
m_2 = 70
X_20_2 = np.random.choice(array_2, 20)

X_20_2
X_bar_2 = X_20_2.mean()

X_bar_2
s_X = math.sqrt(((X_20_2 - X_bar_2) ** 2).sum() / (n - 1))

s_X
t = (X_bar_2 - m_2) * math.sqrt(20) / s_X

t
m_3 = X_bar_2
X_20_3 = np.random.choice(array_2, 20)

X_20_3
X_bar_3 = X_20_3.mean()

X_bar_3
s_X = math.sqrt(((X_20_3 - X_bar_3) ** 2).sum() / (n - 1))

s_X
t = (X_bar_3 - m_3) * math.sqrt(20) / s_X

t
print('real mu = ' + str(mu_2))

print('my mu = ' + str(m_3))
sigma_2