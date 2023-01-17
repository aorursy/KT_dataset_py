import numpy as np
import pandas as pd
train = pd.read_csv('../input/digit-recognizer/train.csv')
test = pd.read_csv('../input/digit-recognizer/test.csv')
len(train), len(test)
train.columns
train['label'].unique()
import matplotlib.pyplot as plt
tr_sample = train.drop('label', axis=1).values.reshape(-1,28,28)[0]
ts_sample = test.values.reshape(-1,28,28)[0]
test['pixel345'].value_counts()
fig, ax = plt.subplots(1, 2, figsize=(16,8))

ax[0].imshow(tr_sample, cmap='gray')
ax[0].set(title='Train', xticks=[], yticks=[])

ax[1].imshow(ts_sample, cmap='gray')
ax[1].set(title='Test', xticks=[], yticks=[]);
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
ts_sample.min(), ts_sample.max()
x_test = test.values.reshape(-1,28,28,1)
x_test = x_test/255

x_train_full = train.drop('label', axis=1).values.reshape(-1,28,28,1)
y_train_full = train['label'].values

x_train, x_val, y_train, y_val = train_test_split(x_train_full, y_train_full, test_size=0.2, random_state=42)
batch_size=64

img_gen = ImageDataGenerator(rescale=1/255, 
                             rotation_range=30, 
                             zoom_range=.1,
                             shear_range=.1,
                             width_shift_range=.25,
                             height_shift_range=.25)

train_gen = img_gen.flow(x_train, y_train, 
                         batch_size=batch_size)

valid_gen = img_gen.flow(x_val, y_val, 
                         batch_size=batch_size,
                         shuffle=False)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense, Conv2D, MaxPool2D, Dropout, BatchNormalization, AveragePooling2D, GlobalAveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
early_stop = EarlyStopping(monitor='val_loss', patience=5, mode='min',restore_best_weights=True)

check_point = ModelCheckpoint('digit_reg_mnist_z.h5', monitor='val_accuracy', save_best_only=True)

lr_plateau = ReduceLROnPlateau(monitor='val_accuracy', 
                               patience=2,
                               factor=.2, 
                               min_lr=1e-6)
model = Sequential()

model.add(Conv2D(32, kernel_size=(3,3), input_shape=(28,28,1), padding='same'))
model.add(BatchNormalization(momentum=.9, epsilon=1e-5))
model.add(Activation('relu'))

model.add(Conv2D(64, kernel_size=(3,3), padding='same'))
model.add(BatchNormalization(momentum=.9, epsilon=1e-5))
model.add(Activation('relu'))

model.add(MaxPool2D(pool_size=(2,2)))

model.add(Conv2D(64, kernel_size=(3,3), padding='same'))
model.add(BatchNormalization(momentum=.9, epsilon=1e-5))
model.add(Activation('relu'))

model.add(Conv2D(128, kernel_size=(3,3), padding='same'))
model.add(BatchNormalization(momentum=.9, epsilon=1e-5))
model.add(Activation('relu'))

# model.add(MaxPool2D(pool_size=(2,2)))
model.add(AveragePooling2D(pool_size=(2,2)))

model.add(Conv2D(128, kernel_size=(3,3), padding='same'))
model.add(BatchNormalization(momentum=.9, epsilon=1e-5))
model.add(Activation('relu'))

model.add(Conv2D(128, kernel_size=(3,3), padding='same'))
model.add(BatchNormalization(momentum=.9, epsilon=1e-5))
model.add(Activation('relu'))

model.add(MaxPool2D(pool_size=(2,2)))

model.add(Conv2D(256, kernel_size=(3,3), padding='same'))
model.add(BatchNormalization(momentum=.9, epsilon=1e-5))
model.add(Activation('relu'))

model.add(Conv2D(256, kernel_size=(3,3), padding='same'))
model.add(BatchNormalization(momentum=.9, epsilon=1e-5))
model.add(Activation('relu'))

model.add(GlobalAveragePooling2D())
model.add(Dense(10, activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adamax',
              metrics=['accuracy'])

model.summary()
model.fit(train_gen,
          epochs=100,
          steps_per_epoch=250,
          validation_data=valid_gen,
          callbacks=[lr_plateau, early_stop])
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
eval_df = pd.DataFrame(model.history.history)
length = len(eval_df)
fig, ax = plt.subplots(1, 2, figsize=(15,6))

eval_df[['loss','val_loss']].plot(ax=ax[0])
ax[0].set(title='Loss', xlabel='Epoch', xticks=range(0,length,2))

eval_df[['accuracy','val_accuracy']].plot(ax=ax[1])
ax[1].set(title='Accuracy', xlabel='Epoch', xticks=range(0,length,2));
pred = np.argmax(model.predict(valid_gen), axis=1)
pred
plt.figure(figsize=(15,10))
sns.heatmap(confusion_matrix(y_val, pred), annot=True, fmt='d', lw=.3, cmap='YlGnBu')
plt.title('Confusion Matrix')
plt.xlabel('Prediction')
plt.ylabel('True Label')
print(classification_report(y_val, pred))
x_test.shape
lr_plateau = ReduceLROnPlateau(monitor='accuracy', 
                               patience=2,
                               factor=.2, 
                               min_lr=1e-6)

full_train_gen = img_gen.flow(x_train_full, y_train_full, 
                              batch_size=batch_size)
model.fit(full_train_gen,
          epochs=22,
          steps_per_epoch=250,
          callbacks=[lr_plateau])
real_pred = np.argmax(model.predict(x_test), axis=1) # Prediction on real test data
submit_df = pd.read_csv('../input/digit-recognizer/sample_submission.csv')
submit_df['Label'] = real_pred
submit_df.to_csv('submission.csv', index=False)