# TensorFlow 
from tensorflow.keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, MaxPooling2D, Dropout, average, concatenate
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import to_categorical, plot_model
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.datasets import mnist

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import warnings
warnings.filterwarnings('ignore')
df = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')
df.shape, df_test.shape
train = df[:38000].as_matrix()
test = df[38000:].as_matrix()
df_test = df_test.as_matrix()
train.shape, test.shape
X_train = train[:,1:].reshape((train.shape[0],28,28,1)) / 255
y_train = train[:,0]
y_train = to_categorical(y_train, num_classes=10) 
X_valid = df_test.reshape((28000,28,28,1))
X_train.shape, y_train.shape, X_valid.shape
X_test = test[:,1:].reshape((test.shape[0],28,28,1)) / 255
y_test = test[:,0]
y_test = to_categorical(y_test, num_classes=10) 
X_test.shape, y_test.shape
#X_train[X_train < 0.3] = 0
#X_test[X_test < 0.3] = 0
input_shape = (28,28,1) 
classes = 10

model = Sequential()
model.add(Conv2D(14, kernel_size=3, activation='relu', padding='same', input_shape=input_shape))
model.add(Conv2D(14, kernel_size=3, activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
model.add(Dropout(0.25))

model.add(Conv2D(64, kernel_size=3, activation='relu', padding='same'))
model.add(Conv2D(64, kernel_size=3, activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
model.add(Dropout(0.25))

model.add(Conv2D(128, kernel_size=3, activation='relu', padding='same'))
model.add(Conv2D(128, kernel_size=3, activation='relu', padding='same'))
model.add(Conv2D(128, kernel_size=3, activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
model.add(Dropout(0.25))

model.add(Conv2D(256, kernel_size=3, activation='relu', padding='same'))
model.add(Conv2D(256, kernel_size=3, activation='relu', padding='same'))
model.add(Conv2D(256, kernel_size=3, activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(classes, activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=["accuracy"])
model.summary()
datagen = ImageDataGenerator(rotation_range=10, zoom_range=0.1, width_shift_range=0.1, height_shift_range=0.1)  
annealer = LearningRateScheduler(lambda x: 1e-3 * 0.95 ** x)
callback = [ModelCheckpoint('best_model.hdf5', monitor='val_acc', save_best_only=True)]
model_fit = model.fit_generator(datagen.flow(X_train, 
                                             y_train,
                                             batch_size=128),
                                epochs=30, 
                                steps_per_epoch=X_train.shape[0] // 128,
                                validation_data=(X_test, y_test), 
                                callbacks=[annealer], verbose=1)

plt.plot(model_fit.history['acc'], label='Аккуратность на обучающем наборе')
plt.plot(model_fit.history['val_acc'], label='Аккуратность на проверочном наборе')
plt.xlabel('Эпоха')
plt.ylabel('Ошибка')
plt.legend()
plt.show()

plt.plot(model_fit.history['loss'], label='Ошибка на обучающем наборе')
plt.plot(model_fit.history['val_loss'], label='Ошибка на проверочном наборе')
plt.xlabel('Эпоха')
plt.ylabel('Ошибка')
plt.legend()
plt.show()
pred = model.predict(X_valid)
results = np.argmax(pred, axis=1)
plt.figure(figsize=(10,40))
plt.subplots_adjust(top=0.5)
for i in range(100):
    plt.subplot(10,10, i + 1)
    plt.title('label:{}'.format(results[i]))
    plt.imshow(X_valid[i].reshape(28, 28))
def write_to_submission_file(predicted_labels, out_file, train_num=0,
                    target='Label', index_label="ImageId"):
    # turn predictions into data frame and save as csv file
    predicted_df = pd.DataFrame(predicted_labels, 
                                index = np.arange(train_num + 1, train_num + 1 + predicted_labels.shape[0]),
                                columns=[target])
    predicted_df.to_csv(out_file, index_label=index_label)
results
write_to_submission_file(results, out_file='submission.csv')
print("Сохраняем сеть")
# Сохраняем сеть для последующего использования
# Генерируем описание модели в формате json
model_json = model.to_json()
json_file = open("cvd_model.json", "w")
# Записываем архитектуру сети в файл
json_file.write(model_json)
json_file.close()
# Записываем данные о весах в файл
model.save_weights("cvd_model.h5")
print("Сохранение сети завершено")

