import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from random import randint
from sklearn import svm
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
%matplotlib inline
path = '../input/'
df_train = pd.read_csv(path + 'train.csv')
df_test = pd.read_csv(path + 'test.csv')
print(df_train.shape)
print(df_train.head())
rand_num = randint(0, 41999)
rand_label = df_train.iloc[rand_num]
#print(rand_label[1:])
mat_label = rand_label[1:].values.reshape(28,28)
print("The label of the pixel is ---- " + str(rand_label[0]))
plt.imshow(mat_label, cmap='gray')
plt.show()
na = df_train.isnull().sum()
na = na[na > 0]
print(na)
df_sample = df_train.iloc[:20000]
X = (df_sample.iloc[:,1:].values).astype('float32')
labels = df_sample['label'].values.astype('int32')
y = np_utils.to_categorical(labels)

scale = np.max(X)
X /= scale

mean = np.std(X)
X -= mean

input_dim = X.shape[1]
nb_classes = y.shape[1]
#X_train, X_test, y_train, y_test = train_test_split(X, labels, random_state=42)
#clf = svm.SVC()
#clf = clf.fit(X_train, y_train)
#clf.score(X_test, y_test)
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
model = Sequential()
model.add(Dense(128, input_dim=input_dim))
model.add(Activation('relu'))
model.add(Dropout(0.15))
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.15))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics = ['accuracy'])
model.fit(X, y, epochs=60, batch_size=32, verbose=2)
X_test = df_test.values.astype('float32')
X_test /= scale
X_test -= mean
preds = model.predict_classes(X_test, verbose=0)
pd.DataFrame({"ImageId": list(range(1,len(preds)+1)), "Label": preds}).to_csv('ans.csv', index=False, header=True)
from keras.utils import plot_model
import pydot
plot_model(model, to_file='model.png')