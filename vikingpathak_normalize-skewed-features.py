import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from sklearn.datasets import load_boston
boston = load_boston()
X = boston['data']
y = boston['target']
cols = boston['feature_names']
df = pd.DataFrame(X, columns = cols)
df['Price'] = y
df.head()
df.skew().idxmax()
sns.kdeplot(df.CRIM)
df.CRIM.min()
log_CRIM = np.log(df.CRIM)
log_CRIM.skew()
sns.kdeplot(log_CRIM)
sqrt_CRIM = np.sqrt(df.CRIM)
sqrt_CRIM.skew()
sns.kdeplot(sqrt_CRIM)
from scipy import stats

boxcox_CRIM = stats.boxcox(df.CRIM)
boxcox_CRIM = pd.Series(boxcox_CRIM[0])
boxcox_CRIM.skew()
sns.kdeplot(boxcox_CRIM)