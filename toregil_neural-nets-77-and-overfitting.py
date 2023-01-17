import numpy as np # linear algebra

np.set_printoptions(precision=0, suppress=True)

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



from sklearn.preprocessing import Imputer, StandardScaler

from sklearn.decomposition import PCA

from sklearn.model_selection import train_test_split

from sklearn.manifold import TSNE

from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score



import matplotlib.pyplot as plt

%matplotlib inline
from keras.models import Sequential, Model

from keras.optimizers import Adam

from keras.regularizers import l2

from keras.layers import Dense, BatchNormalization, Dropout

from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

import keras.backend as K
with open('../input/diabetes.csv') as f:

    print('\n'.join(f.readline().split(',')[:-1]))
raw_data = np.loadtxt('../input/diabetes.csv', skiprows=1, delimiter=',')

raw_feat = raw_data[:,:-1]

raw_labels = raw_data[:,-1] 
x_train, x_test, y_train, y_test = train_test_split(raw_feat, raw_labels, 

                                                    test_size=0.3, random_state=700)
imp = Imputer(missing_values = 0) #replace zero values by mean

clean_feat = raw_feat.copy()

clean_feat[:,1:] = imp.fit_transform(raw_feat[:,1:]) #We don't want to do this for pregnancies

print(raw_feat[:8,4], '\n', clean_feat[:8,4])
#We must do this again for the train test, or we would get data leakage from the test set

#in taking the mean.

x_train = x_train.copy()

x_train[:,1:] = imp.fit_transform(x_train[:,1:])

x_test[:,1:] = imp.transform(x_test[:,1:])
scaler = StandardScaler() #scale to zero mean and unit variance

clean_feat = scaler.fit_transform(clean_feat)

x_train = scaler.fit_transform(x_train)

x_test = scaler.transform(x_test)
pca = PCA(n_components=7)

clean_pca = pca.fit_transform(clean_feat)

plt.scatter(clean_pca[:,0], clean_pca[:,1], color=np.where(raw_labels>0.5,'r','g'))
tsne = TSNE(n_iter=3000)

tsne_data = tsne.fit_transform(clean_pca)

plt.scatter(tsne_data[:,0], tsne_data[:,1], color=np.where(raw_labels>0.5,'r','g'))
pca_train = pca.fit_transform(x_train)

pca_test = pca.transform(x_test)
model = Sequential()

model.add(Dense(32, activation='relu', input_dim = x_train.shape[1]))

model.add(Dense(64, activation='relu'))

model.add(BatchNormalization())

model.add(Dropout(0.5))

model.add(Dense(256, activation='relu'))

model.add(Dropout(0.75))

model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer=Adam(1e-4), loss='binary_crossentropy', metrics=['accuracy'])

hist = model.fit(x_train, y_train, validation_data=(x_test, y_test), 

          epochs=50, verbose=0, batch_size=8)

print(model.evaluate(x_train, y_train, verbose=0))

print(model.evaluate(x_test, y_test, verbose=0))
plt.plot(hist.history['loss'], color='b')

plt.plot(hist.history['val_loss'], color='r')

plt.show()

plt.plot(hist.history['acc'], color='b')

plt.plot(hist.history['val_acc'], color='r')

plt.show()
y_true = (raw_labels + 0.5).astype("int")

y_pred = (model.predict(clean_feat) + 0.5).astype("int").reshape(-1,)

color = np.where(y_true * y_pred == 1, 'g', 'r')

color[np.where(y_true * (1-y_pred) == 1)[0]] = 'b'

color[np.where(y_pred * (1-y_true) == 1)[0]] = 'y'
for i in range(4):

    for j in range(i+1,4):

        plt.scatter(clean_pca[:,i], clean_pca[:,j], color=color)

        plt.show()
model.compile(optimizer=Adam(3e-4), loss='binary_crossentropy', metrics=['accuracy'])

hist = model.fit(clean_feat, raw_labels, 

          epochs=500, verbose=0, batch_size=8)

print(model.evaluate(x_train, y_train, verbose=0))
y_true = (raw_labels + 0.5).astype("int")

y_pred = (model.predict(clean_feat) + 0.5).astype("int").reshape(-1,)

color = np.where(y_true * y_pred == 1, 'g', 'r')

color[np.where(y_true * (1-y_pred) == 1)[0]] = 'b'

color[np.where(y_pred * (1-y_true) == 1)[0]] = 'y'
for i in range(4):

    for j in range(i+1,4):

        plt.scatter(clean_pca[:,i], clean_pca[:,j], color=color)

        plt.show()
model = Sequential()

model.add(Dense(8, activation='relu', kernel_regularizer=l2(.1), input_dim = x_train.shape[1]))

model.add(Dense(8, activation='relu', kernel_regularizer=l2(.05)))

model.add(BatchNormalization())

model.add(Dense(16, activation='relu'))

model.add(Dropout(0.75))

model.add(Dense(1, activation='sigmoid'))
#To keep the best weights, in case of overfitting

callback1 = ModelCheckpoint('tiny.h5', monitor='val_loss', 

                           save_best_only=True, save_weights_only=True)



#To reduce learning rate over time

callback2 = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=100, 

                              mode='min', epsilon=0.05, min_lr=1e-8)



#To adjust for the fact y_train has twice as many 0's as 1's

#Does not seem to improve results in this case

#sample_weight = (1 + y_train) 
# Label smoothing, sometimes improves net training by avoiding 

# areas where the activation function has flat gradient.

y_train = 0.9 * y_train + 0.05  

y_test = 0.9 * y_test + 0.05
def float_accuracy(y_true, y_pred):

    """

    Equivalent to Keras' built-in binary_accuracy, but can be used with label smoothing.

    """

    return K.mean(K.equal(K.round(y_true), K.round(y_pred)), axis=-1)
model.compile(optimizer=Adam(1e-4), loss='binary_crossentropy', metrics=[float_accuracy])



hist = model.fit(x_train, y_train, validation_data=(x_test, y_test),

                 epochs=1000, verbose=0, batch_size=8, 

                 callbacks = [callback1, callback2])

print(model.evaluate(x_train, y_train, verbose=0))

print(model.evaluate(x_test, y_test, verbose=0))
plt.plot(hist.history['loss'], color='b')

plt.plot(hist.history['val_loss'], color='r')

plt.show()

plt.plot(hist.history['float_accuracy'], color='b')

plt.plot(hist.history['val_float_accuracy'], color='r')

plt.show()
model.load_weights('tiny.h5')

print(model.evaluate(x_train, y_train, verbose=0))

print(model.evaluate(x_test, y_test, verbose=0))
threshold = 0.5

y_true = np.where(y_test > 0.5, 1, 0).astype("int")

y_pred = np.where(model.predict(x_test) > threshold, 1, 0).astype("int").reshape(-1,)

cm = confusion_matrix(y_true, y_pred)

pos = np.sum(cm[0])

print(cm)
threshold = 0.6

y_true = np.where(y_test > 0.5, 1, 0).astype("int")

y_pred = np.where(model.predict(x_test) > threshold, 1, 0).astype("int").reshape(-1,)

cm = confusion_matrix(y_true, y_pred)

print(cm)
y_score = model.predict(x_test)

fpr, tpr, thresholds = roc_curve(y_true, y_score)

plt.plot(thresholds, 1.-fpr)

plt.plot(thresholds, tpr)

plt.show()

crossover_index = np.min(np.where(1.-fpr <= tpr))

crossover_cutoff = thresholds[crossover_index]

crossover_specificity = 1.-fpr[crossover_index]

print("Crossover at {0:.2f} with specificity {1:.2f}".format(crossover_cutoff, crossover_specificity))
plt.plot(fpr, tpr)

plt.show()

print("ROC area under curve is {0:.2f}".format(roc_auc_score(y_true, y_score)))