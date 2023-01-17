# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
from keras import models
from keras import layers
from keras import regularizers
from keras.callbacks import ReduceLROnPlateau
from keras import optimizers

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('../input/voice.csv')
print('head entries:\n', df.head())
print('dataset info:\n', df.info())
print('#null:\n', df.isnull().sum())
print('col. name: ', df.columns)
import seaborn as sns
ax = plt.axes()
sns.heatmap(df.corr(), vmax=0.8, ax=ax, linewidths=0.25, square=True, linecolor='black')
ax.set_title('Orignal feature correlations')

# feature separation
g = sns.PairGrid(df, hue='label', vars=["meanfun", "sfm", 'IQR'])
g = g.map(plt.scatter, s=4)
plt.show()

#grid = sns.FacetGrid(df, row="label", col="meanfun", margin_titles=True)
#grid.map(plt.hist, "sfm");

from sklearn.feature_selection import SelectKBest, f_classif

def select_kbest_clf(df, target, k=5):
    """
    Selecting K-Best features for classification
    - df: A pandas dataFrame with the training data
    - target: target variable name in DataFrame
    - k: desired number of features from the data
    
    returns feature_scores: scores for each feature 
    """
    feat_selector = SelectKBest(f_classif, k=k)
    _= feat_selector.fit(df.drop(target, axis=1), df[target])
    
    feat_scores = pd.DataFrame()
    feat_scores["F Score"] = feat_selector.scores_
    feat_scores["P Value"] = feat_selector.pvalues_
    feat_scores["Support"] = feat_selector.get_support()
    feat_scores["Attribute"] = df.drop(target, axis=1).columns
    
    return feat_scores
    
k=select_kbest_clf(df, 'label', k=5).sort_values(['F Score'],ascending=False)
plt.figure()
k1=sns.barplot(x=k['F Score'],y=k['Attribute'])
k1.set_title('Feature Importance')
# increase dataset size by perturbing the most important feature
for i in range(3):
    copy = df
    
    copy['meanfun'] = copy['meanfun'] + np.random.uniform(0, 1e-2)
    
    df = df.append(copy, ignore_index=True)
    # print("shape of df after {0}th intertion of this loop is {1}".format(i, df.shape))

df.apply(np.random.permutation)
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA

# features vs. label
X = df.iloc[:, :-1].values
y = df.iloc[:, -1]

# Encode label category: male -> 1, female -> 0
gender_encoder = LabelEncoder()
y = gender_encoder.fit_transform(y)

# redundant features are identified by investigating feature correlation
X2 = np.delete(np.array(X), [7, 11, 17] , 1)  
X = X2

print(X.shape)

# feature normalization
scaler = StandardScaler()

#X = scaler.fit_transform(X)
#scaler = MinMaxScaler()
for i in range(X.shape[1]):
    vec = scaler.fit_transform(X2[:, i].reshape([-1, 1]))
    #print(vec.shape)
    X[:, i] = vec.reshape(-1)
plt.hist(X[:, 2])
plt.show()

# dimension reduction via PCA
#pca = PCA(n_components=17)
#pca.fit(X2)
#X2 = pca.transform(X2) 

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.2, random_state=1, stratify=y)
nDim = X_train.shape[1]    # number of features used for classification

# DNN architecture 
nUnitL1 = 128              
nUnitL2 = 64
nUnitL3 = 32
batch_size = 32

model = models.Sequential()
model.add(layers.Dense(nUnitL1, activation='relu', kernel_initializer='normal', kernel_regularizer=regularizers.l2(0.001), input_shape=(nDim,)))
#model.add(layers.Dropout(0.25))
model.add(layers.Dense(nUnitL2, activation='relu', kernel_initializer='normal', kernel_regularizer=regularizers.l2(0.001)))
#model.add(layers.Dropout(0.25))
model.add(layers.Dense(nUnitL3, activation='relu', kernel_initializer='normal', kernel_regularizer=regularizers.l2(0.001)))
model.add(layers.Dense(1, activation='sigmoid'))

#model.compile(optimizer=optimizers.RMSprop(lr=0.001),
              
model.compile(optimizer=optimizers.RMSprop(lr=0.0002),
              loss='binary_crossentropy',
              metrics=['accuracy'])

reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5,
                              patience=5, min_lr=0.00001)
hist = model.fit(X_train, y_train, epochs=50,
          batch_size=batch_size, 
          verbose=1,callbacks=[reduce_lr], 
          validation_data = [X_test, y_test])
results = model.evaluate(X_test, y_test)

loss_values = hist.history['loss']
val_loss_values = hist.history['val_loss']

epochs = range(1, 51)
plt.plot(epochs, loss_values, 'bo', label='Training loss')
plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()
print('final test resutls: ', results)

# result evaluation
y_pred = model.predict(X_test)
ypred = np.zeros((y_pred.shape)).astype(int)
ypred[np.where(y_pred > 0.5)] = 1
print(y_pred.shape, y_test.shape)
#print(ypred[:10], y_test[:10])
falsePos = np.array(np.where(~np.equal(ypred.reshape(-1), y_test))).reshape([-1,1])
#print(falsePos[:10])
print(len(falsePos))

maxPos = np.array(np.where(y_pred==np.max(y_pred))).reshape(-1)
print(maxPos[0])
print(X_test[maxPos[0], :])
for pos in falsePos:
    print(y_pred[pos])
    #print(X_test[pos, :])
#plt.figure()
#plt.plot(y_pred, 'b.')
#plt.plot(y_test, 'r*')