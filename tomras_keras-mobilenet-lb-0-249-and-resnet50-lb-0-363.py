import numpy as np
import pandas as pd
import glob
import matplotlib.pyplot as plt
%matplotlib inline 
import cv2
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications.mobilenet import MobileNet
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras import optimizers

img_size = 128 # image height and width
train_size = None # number of samples for training
test_size = None # number of samples for testing
files_paths = glob.glob('../input/train/*.jpg') # list of image files
files_labels = [[1, 0] if 'dog' in f else [0, 1] for f in files_paths] # labels

print(len(files_paths), len(files_labels))
def files_2_img_array(files_list): 
    '''Takes list of image files paths and return np array of images'''
    imgs = []
    for i in files_list:
        img = cv2.imread(i)
        img = cv2.resize(img, (img_size, img_size))
#         img = np.true_divide(img, 255, dtype=np.float64)
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        imgs.append(img)
    return np.array(imgs)

x = files_2_img_array(files_paths)
y = np.array(files_labels)
print(x.shape, y.shape)
n = 4
fig, axs = plt.subplots(nrows=n, ncols=n, sharex=True, sharey=True, figsize=(10, 10))
for i in range(n**2):
    ax = axs[i // n, i % n]
    ax.imshow(x[i])
    ax.set_title('label: %s, idx: %s' % (str(y[i]), str(i)))
    ax.axis('off')
plt.show()
x_flipped = np.array([np.fliplr(img) for img in x]) # performing flipping 
x = np.concatenate([x, x_flipped])
y = np.concatenate([y, y])

del x_flipped
plt.imshow(x[100])
plt.axis('off')
plt.show()

plt.imshow(x[25100])
plt.axis('off')
plt.show()
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

if train_size:
    x_train = x_train[:train_size]
    y_train = y_train[:train_size]
if test_size:
    x_test = x_test[:test_size]
    y_test = y_test[:test_size]

print(x_train.shape, x_test.shape)
mobile_net_model= MobileNet(weights='imagenet', include_top=False, input_shape=(img_size, img_size, 3))
mobile_net_model.trainable = False

model = Sequential()
model.add(mobile_net_model)
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(2, activation='softmax'))
model.summary()
model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.Adam(),
              metrics=['acc'])
history = model.fit(
    x_train,
    y_train,
    epochs=15,
    batch_size=100,
    validation_data=(x_test, y_test),
    verbose=1)
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
test_files_paths = glob.glob('../input/test/*.jpg')
xx = files_2_img_array(test_files_paths)
xx.shape
predictions = model.predict(xx)
predictions.shape
sub_ids = [i.split('/')[3].split('.')[0] for i in test_files_paths]
sub_labels = [i[0] for i in predictions]

submission_df = pd.DataFrame({'id': sub_ids, 'label': sub_labels})
submission_df.to_csv('output_mobilenet.csv', index = False)

submission_df.head()
res_net_model = ResNet50(weights='imagenet', include_top=False, input_shape=(img_size, img_size, 3))
res_net_model.trainable = False

model = Sequential()
model.add(res_net_model)
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(2, activation='softmax'))
model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.Adam(),
              metrics=['acc'])

history = model.fit(
    x_train,
    y_train,
    epochs=15,
    batch_size=100,
    validation_data=(x_test, y_test),
    verbose=1)
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
predictions = model.predict(xx)
predictions.shape
sub_labels = [i[0] for i in predictions]

submission_df = pd.DataFrame({'id': sub_ids, 'label': sub_labels})
submission_df.to_csv('output_resnet.csv', index = False)

submission_df.head()