%matplotlib inline
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn import cluster, datasets, mixture

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns

from keras.layers import Input, Embedding, LSTM, GRU, Dense, Dropout, Lambda, \
    Conv1D, Conv2D, Conv3D, \
    Conv2DTranspose, \
    AveragePooling1D, \
    MaxPooling1D, MaxPooling2D, MaxPooling3D, \
    GlobalAveragePooling1D, \
    GlobalMaxPooling1D, GlobalMaxPooling2D, \
    LocallyConnected1D, LocallyConnected2D, \
    concatenate, Flatten, Average, Activation, \
    RepeatVector, Permute, Reshape, Dot, \
    multiply, dot, add, \
    PReLU, \
    Bidirectional, TimeDistributed, \
    SpatialDropout1D, \
    BatchNormalization
from keras.models import Model, Sequential
from keras import losses
from keras.callbacks import BaseLogger, ProgbarLogger, Callback, History
from keras.wrappers.scikit_learn import KerasClassifier
from keras import regularizers
from keras import initializers
from keras.metrics import categorical_accuracy
from keras.constraints import maxnorm, non_neg
from keras.optimizers import RMSprop
from keras.utils import to_categorical
from keras import backend as K
from keras_ex.gkernel import GaussianKernel, GaussianKernel2, GaussianKernel3
n_samples = 1500

X, y = datasets.make_moons(n_samples=n_samples, noise=.08, random_state=0)
df = pd.DataFrame(X)
df.columns = ["col1", "col2"]
df['cls'] = y

df.head()
sns.lmplot("col1", "col2", hue="cls", data=df, fit_reg=False, height=8)
def make_modelz():
    inp = Input(shape=(2,), name='inp')
    oup = inp
    
    oup = Dense(3, activation='sigmoid', name='hidden_1')(oup)
    oup = Dense(3, activation='sigmoid', name='hidden_2')(oup)
    oup = Dense(1, activation='sigmoid', name='classifier')(oup)
    
    model = Model(inp, oup)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return {
        'model': model,
    }

models = make_modelz()
model = models['model']
model.summary()
model.fit(X, y, verbose=0,
          batch_size=32,
          epochs=150)
y_pred = model.predict(X)

df = pd.DataFrame(X)
df.columns = ["col1", "col2"]
df['cls'] = (0.5<y_pred[:,0]).astype(int)
df.head()
sns.lmplot("col1", "col2", hue="cls", data=df, fit_reg=False)
h = .01
x_min, x_max = X[:, 0].min() - .1, X[:, 0].max() + .1
y_min, y_max = X[:, 1].min() - .1, X[:, 1].max() + .1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
y_pred = model.predict(np.c_[xx.ravel(), yy.ravel()], batch_size=1024)

cm = plt.cm.coolwarm
cm_bright = ListedColormap(['#FF0000', '#0000FF'])
y_pred = y_pred.reshape(xx.shape)
plt.figure(figsize=(10, 8))
plt.contourf(xx, yy, y_pred, 100, cmap=cm, alpha=1)
plt.scatter(X[:,0], X[:,1], c=y, cmap=cm_bright, edgecolors='k')
num_landmark = 30

def make_modelz():
    inp = Input(shape=(2,), name='inp')
    oup = inp
    
    oup = GaussianKernel3(num_landmark, 2, name='gkernel1')(oup)
    oup = Dense(1, activation='sigmoid', name='classifier')(oup)
    model = Model(inp, oup)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return {
        'model': model,
    }

models = make_modelz()
model = models['model']
model.summary()
wgt = model.get_layer('gkernel1').get_weights()[0]
df = pd.DataFrame(np.vstack([X, wgt]))
df.columns = ["col1", "col2"]
df['cls'] = [str(ee) for ee in y] + ['Landmark']*wgt.shape[0]
sns.lmplot("col1", "col2", markers=['.', '.', 'o'], hue="cls", data=df, fit_reg=False, height=6)
init_wgt = model.layers[1].get_weights()

init_wgt[1][0] = np.log(1/(2*2*0.1))
model.layers[1].set_weights(init_wgt)
model.fit(X, y, verbose=0,
          batch_size=32,
          epochs=150)
y_pred = model.predict(X)

df = pd.DataFrame(X)
df.columns = ["col1", "col2"]
df['cls'] = (0.5<y_pred[:,0]).astype(int)
df.head()
sns.lmplot("col1", "col2", hue="cls", data=df, fit_reg=False)
h = .01
x_min, x_max = X[:, 0].min() - .1, X[:, 0].max() + .1
y_min, y_max = X[:, 1].min() - .1, X[:, 1].max() + .1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
y_pred = model.predict(np.c_[xx.ravel(), yy.ravel()], batch_size=1024)

cm = plt.cm.coolwarm
cm_bright = ListedColormap(['#FF0000', '#0000FF'])
y_pred = y_pred.reshape(xx.shape)
plt.figure(figsize=(10, 8))
plt.contourf(xx, yy, y_pred, 100, cmap=cm, alpha=1)
plt.scatter(X[:,0], X[:,1], c=y, cmap=cm_bright, edgecolors='k')
wgt = model.get_layer('gkernel1').get_weights()[0]
df = pd.DataFrame(np.vstack([X, wgt]))
df.columns = ["col1", "col2"]
df['cls'] = [str(ee) for ee in y] + ['Landmark']*wgt.shape[0]
sns.lmplot("col1", "col2", markers=['.', '.', 'o'], hue="cls", data=df, fit_reg=False, height=6)
