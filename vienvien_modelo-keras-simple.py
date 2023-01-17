# Inspiración:

#https://www.kaggle.com/sudalairajkumar/simple-exploration-baseline-santander-value

#https://www.kaggle.com/mortido/keras-simple-model
# Librerías

import numpy as np

import pandas as pd

from sklearn.model_selection import train_test_split
# Muestra archivos disponibles

import os

print(os.listdir("../input"))
# Carga bases disponibles

train_df = pd.read_csv('../input/train.csv')

test_df = pd.read_csv('../input/test.csv')
print ("Training set:")

n_data  = len(train_df)

n_features = train_df.shape[1]

print ("Number of Records: {}".format(n_data))

print ("Number of Features: {}".format(n_features))



# testing set

print ("\nTesting set:")

n_data  = len(test_df)

n_features = test_df.shape[1]

print ("Number of Records: {}".format(n_data))

print ("Number of Features: {}".format(n_features))
train_df.head(n=10)
test_df.head(n=10)
train_df.info()
test_df.info()
#### Check if there are any NULL values in Train Data

print("Total Train Features with NaN Values = " + str(train_df.columns[train_df.isnull().sum() != 0].size))

if (train_df.columns[train_df.isnull().sum() != 0].size):

    print("Features with NaN => {}".format(list(train_df.columns[train_df.isnull().sum() != 0])))

    train_df[train_df.columns[train_df.isnull().sum() != 0]].isnull().sum().sort_values(ascending = False)
#### Check if there are any NULL values in Test Data

print("Total Test Features with NaN Values = " + str(test_df.columns[test_df.isnull().sum() != 0].size))

if (test_df.columns[test_df.isnull().sum() != 0].size):

    print("Features with NaN => {}".format(list(test_df.columns[test_df.isnull().sum() != 0])))

    test_df[test_df.columns[test_df.isnull().sum() != 0]].isnull().sum().sort_values(ascending = False)
x_train = train_df.drop(["ID", "target"], axis=1)

y_train = np.log1p(train_df["target"].values)



x_test = test_df.drop(["ID"], axis=1)
# check and remove constant columns

colsToRemove = []

for col in x_train.columns:

    if x_train[col].std() == 0: 

        colsToRemove.append(col)

        

# remove constant columns in the training set

x_train.drop(colsToRemove, axis=1, inplace=True)



# remove constant columns in the test set

x_test.drop(colsToRemove, axis=1, inplace=True) 



print("Removed `{}` Constant Columns\n".format(len(colsToRemove)))

#print(colsToRemove)
# Check and remove duplicate columns

colsToRemove = []

colsScaned = []

dupList = {}



columns = x_train.columns



for i in range(len(columns)-1):

    v = x_train[columns[i]].values

    dupCols = []

    for j in range(i+1,len(columns)):

        if np.array_equal(v, x_train[columns[j]].values):

            colsToRemove.append(columns[j])

            if columns[j] not in colsScaned:

                dupCols.append(columns[j]) 

                colsScaned.append(columns[j])

                dupList[columns[i]] = dupCols

                

# remove duplicate columns in the training set

x_train.drop(colsToRemove, axis=1, inplace=True) 



# remove duplicate columns in the testing set

x_test.drop(colsToRemove, axis=1, inplace=True)



print("Removed `{}` Duplicate Columns\n".format(len(dupList)))

print(dupList)




# Ajusta distribución de variables

#x_train = np.log1p(x_train)

#x_test = np.log1p(x_test)



# Junta bases para obtener estadísticos de media y desviación estándar

x_total = pd.concat((x_test, x_train), axis=0).replace(0,  np.nan)



# Escala valores

x_train = (x_train -x_total.mean()) / x_total.std()

x_test = (x_test -x_total.mean()) /x_total.std()



# Escala variable objetvo
# Genera base de entrenamiento y validación

dev_x, val_x, dev_y, val_y = train_test_split(x_train, y_train, test_size = 0.2, random_state = 45)
# Definición del modelo

from keras import models

from keras import layers

from keras import regularizers



model = models.Sequential()

model.add(layers.Dense(2000, kernel_regularizer=regularizers.l2(0.05), activation='linear', input_shape=(x_train.shape[1],)))

model.add(layers.BatchNormalization())

model.add(layers.Activation('tanh'))

model.add(layers.Dropout(0.5))



model.add(layers.Dense(1024, kernel_regularizer=regularizers.l2(0.05), activation='linear'))

model.add(layers.BatchNormalization())

model.add(layers.LeakyReLU(alpha=0.3))

model.add(layers.Dropout(0.5))



model.add(layers.Dense(512, kernel_regularizer=regularizers.l2(0.05), activation='linear'))

model.add(layers.BatchNormalization())

model.add(layers.LeakyReLU(alpha=0.3))



model.add(layers.Dense(32, activation='relu'))

model.add(layers.BatchNormalization())



model.add(layers.Dense(1))
import keras.backend as K

from keras.optimizers import Adam



# Función de pérdida-metrica

def root_mean_squared_error(y_true, y_pred):

        return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1)) 

    

# Compilando el modelo

model.compile(optimizer=Adam(lr=0.01),

              loss=root_mean_squared_error,

              metrics=[root_mean_squared_error])
import keras

batch_size = 256

epochs = 100



lr_scheduler = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=1, verbose=1, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0.00001)

es = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.00005, patience=5, verbose=0, mode='auto')



# Entrenando el modelo

history = model.fit(#dev_x,

                    #dev_y,

                    x_train,

                    y_train,

                    epochs=epochs,

                    batch_size=batch_size,

                    #callbacks=[lr_scheduler, es],

                    validation_data=(val_x, val_y))
history1 = model.fit(#dev_x,

                    #dev_y,

                    x_train,

                    y_train,

                    epochs=epochs,

                    batch_size=batch_size,

                    #callbacks=[lr_scheduler, es],

                    validation_data=(val_x, val_y))
history_dict = history.history

#history_dict.keys()

history_dict
# Muestra el gráfico del entrenamiento

import matplotlib.pyplot as plt

history_dict = history.history

loss_values = history_dict['loss']

val_loss_values = history_dict['val_loss']

epochs = range(1,101)

plt.plot(epochs, loss_values, 'bo', label='Training loss')

plt.plot(epochs, val_loss_values, 'b', label='Validation loss')

plt.title('Training and validation loss')

plt.xlabel('Epochs')

plt.ylabel('Loss')

plt.legend()

plt.show()
plt.clf()

rmse_values = history_dict['root_mean_squared_error']

print(len(rmse_values))

val_rmse_values = history_dict['val_root_mean_squared_error']

print(len(epochs))

epochs = range(1, 101)

plt.plot(epochs, rmse_values, 'bo', label='Training rmse')

plt.plot(epochs, val_rmse_values, 'red', label='Training val_rmse_values')

plt.title('Training and validation rmse')



plt.xlabel('Epochs')

plt.ylabel('Rmse')

plt.legend()

plt.show()
#import matplotlib.pyplot as plt

#import seaborn as sns

#%matplotlib inline

#plt.figure(figsize=(12,8))

#sns.distplot( np.log1p(train_df["target"].values), bins=1413, kde=False)

#plt.xlabel('Target', fontsize=12)

#plt.title("Log of Target Histogram", fontsize=14)

#plt.show()
y_pre = model.predict(x_test.iloc[0:3])

y_pre
model.summary()
pred_keras = np.expm1(model.predict(x_test))

pred_keras
sub = pd.read_csv('../input/sample_submission.csv')

sub["target"] = pred_keras



print(sub.head(20))

sub.to_csv('keras_modelo_v3.csv', index=False)
y_train
#import gc

#gc.collect()