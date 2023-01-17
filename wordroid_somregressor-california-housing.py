!pip install git+https://github.com/darecophoenixx/wordroid.sblo.jp
from som import som
%matplotlib inline

from IPython.display import SVG

from keras.utils.vis_utils import model_to_dot
import random



import numpy as np

import pandas as pd

from sklearn import datasets

from sklearn.datasets import load_digits

from sklearn import preprocessing

from sklearn.decomposition import PCA

from sklearn.metrics import f1_score, classification_report, confusion_matrix, roc_auc_score

from sklearn.model_selection import train_test_split



from keras.utils import to_categorical



import matplotlib.pyplot as plt

import seaborn as sns
from sklearn.datasets.california_housing import fetch_california_housing
cal_housing = fetch_california_housing()
X_train, X_test, y_train, y_test = train_test_split(cal_housing.data,

                                                    cal_housing.target,

                                                    test_size=0.2,

                                                    random_state=1)
names = cal_housing.feature_names
X_train[:3]
sns.distplot(X_train[:,0])
sns.distplot(np.log(X_train[:,0]))
sns.distplot(X_train[:,1])
sns.distplot(X_train[:,2])
sns.distplot(np.log(X_train[:,2]))
sns.distplot(X_train[:,3])
sns.distplot(np.log(X_train[:,3]))
sns.distplot(X_train[:,4])
sns.distplot(np.log(X_train[:,4]))
sns.distplot(X_train[:,5])
sns.distplot(np.log(X_train[:,5]))
sns.distplot(X_train[:,6])
sns.distplot(X_train[:,7])
sns.distplot(y_train)
sns.distplot(np.log(y_train))
y_train
import sklearn

sklearn.__version__
pt = preprocessing.PowerTransformer()
pt.fit(np.c_[cal_housing.target, cal_housing.data])
Xy_sc = pt.transform(np.c_[cal_housing.target, cal_housing.data])

Xy_sc.shape
pt.inverse_transform(Xy_sc)[:3]
df = pd.DataFrame(Xy_sc)

sns.pairplot(df)
y_sc = Xy_sc[:,0]

y_sc.shape
X_sc = Xy_sc[:,1:]

X_sc.shape
sobj = som.SomRegressor((20, 30), it=(15,1500), r2=(1.5,0.5), verbose=2, alpha=1.5)

sobj
sobj.fit(X_sc, y_sc)
lw = 2

plt.plot(np.arange(len(sobj.sksom.meanDist)), sobj.sksom.meanDist, label="mean distance to closest landmark",

             color="darkorange", lw=lw)

plt.legend(loc="best")
sobj.predict(X_sc)
np.c_[y_sc, sobj.predict(X_sc)]
df = pd.DataFrame(np.c_[y_sc, sobj.predict(X_sc)])

df.columns = ['col1', 'col2']

df.head()

sns.lmplot("col1", "col2", data=df, fit_reg=False, size=8, scatter_kws={'alpha': 0.5})
df = pd.DataFrame(np.c_[cal_housing.target, pt.inverse_transform(np.c_[sobj.predict(X_sc), X_sc])[:,0]])

df.columns = ['col1', 'col2']

df.head()

sns.lmplot("col1", "col2", data=df, fit_reg=False, size=8, scatter_kws={'alpha': 0.5})
img = som.conv2img(sobj.sksom.landmarks_, (20, 30), target=[0,1,2])

plt.figure(figsize=(10, 10))

plt.imshow(img)
img = som.conv2img(sobj.sksom.landmarks_, (20, 30), target=[3,4,5])

plt.figure(figsize=(10, 10))

plt.imshow(img)
img = som.conv2img(sobj.sksom.landmarks_, (20, 30), target=[6,7,8])

plt.figure(figsize=(10, 10))

plt.imshow(img)
df1= pd.DataFrame(sobj.sksom.landmarks_)

df1['cls'] = 'K'

df1.head()

df2 = pd.DataFrame(Xy_sc)

df2['cls'] = 'X'

df2.head()

df = pd.concat([df2, df1], axis=0)

df.head()

df.shape

sns.pairplot(df, markers=['s', '.'], hue='cls', plot_kws={'alpha': 0.3}, diag_kind='hist')