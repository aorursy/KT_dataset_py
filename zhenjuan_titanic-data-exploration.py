import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style('whitegrid')

%matplotlib inline



from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB



# Any results you write to the current directory are saved as output.
df_train = pd.read_csv('../input/train.csv', dtype={"Age": np.float64},)

df_test = pd.read_csv('../input/test.csv', dtype={"Age": np.float64},)
df_train.head(5)
df_test.head(5)
df_train.info()

print("---------------------------------")

df_test.info()
