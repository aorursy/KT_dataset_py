import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout, BatchNormalization
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn.model_selection import train_test_split

image_size = 100
df = pd.read_csv('../input/labels.csv')
df.head()
df.describe()
df[df.isnull()]['breed'].count()
df['breed'].value_counts().sort_values(axis=0).plot(kind='barh', 
                                                    figsize=(10, 30), 
                                                    title='Dog breeds')
print('Minimum count eskimo_dog %s' % df[df['breed']=='eskimo_dog']['breed'].count())
print('Maximum count scottish_deerhound %s' % df[df['breed']=='scottish_deerhound']['breed'].count())
labels_df = pd.get_dummies(df['breed'])
labels = labels_df.values

print(labels.shape)
labels[0]
data_original = np.array([img_to_array(load_img('../input/train/%s.jpg'%img, target_size=(image_size, image_size), color_mode='grayscale')) for img in df['id'].values.tolist()]).astype(np.int16)
data_flipped = np.array(list(map(np.fliplr, data_original)))

print(data_original.shape, data_flipped.shape)
plt.imshow(data_original[0].reshape(image_size, image_size), cmap='gray')
plt.show()
plt.imshow(data_flipped[0].reshape(image_size, image_size), cmap='gray')
data = np.concatenate((data_original, data_flipped),axis=0)
data = np.true_divide(data, 255, dtype=np.float64)

labels = np.concatenate((labels, labels), axis=0)

print(data.shape, labels.shape)
plt.imshow(data[0].reshape(image_size, image_size), cmap='gray')
plt.show()
plt.imshow(data[10222].reshape(image_size, image_size), cmap='gray')
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.3)
print(x_train.shape, y_test.shape)

del data
del labels
del data_original
del data_flipped
model = Sequential()
model.add(Conv2D(32, (3, 3), 
                 padding='same', 
                 input_shape=(image_size, image_size, 1), 
                 activation='relu'))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.1))

model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.1))

model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.1))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(120, activation='softmax'))

model.compile(loss='categorical_crossentropy', 
              optimizer=optimizers.Adam(), 
              metrics=['accuracy'])
model_hist = model.fit(x_train, y_train,
                       epochs=5,
                       batch_size=16,
                       validation_data=(x_test, y_test))
plt.plot(model_hist.history['acc'])
plt.plot(model_hist.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
plt.plot(model_hist.history['loss'])
plt.plot(model_hist.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
sample_sub_df = pd.read_csv('../input/sample_submission.csv')
sample_sub_data = np.array([img_to_array(load_img('../input/test/%s.jpg'%img, target_size=(image_size, image_size), color_mode='grayscale')) for img in sample_sub_df['id'].values.tolist()]).astype(np.int16)
sample_sub_data = np.true_divide(sample_sub_data, 255, dtype=np.float64)
print(sample_sub_data.shape)
plt.imshow(sample_sub_data[0].reshape(image_size, image_size), cmap='gray')
predictions = model.predict(sample_sub_data, verbose=1)
submission_df = pd.DataFrame(predictions)
columns_names = labels_df.columns.values
submission_df.columns = columns_names
submission_df.insert(0, 'id', sample_sub_df['id'])

submission_df.to_csv('output.csv', index = False)

submission_df.head()