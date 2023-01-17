import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix

import itertools



import keras

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization

from keras.optimizers import RMSprop

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import ReduceLROnPlateau

from keras.utils import plot_model

from IPython.display import Image

from keras.utils.np_utils import to_categorical
df1 = pd.read_csv('../input/digit-recognizer/train.csv')

df2 = pd.read_csv('../input/digit-recognizer/test.csv')
df1.head()
Y_train = df1["label"]

X_train = df1.drop(labels = ["label"],axis = 1).values 
fig = plt.figure(figsize=(20,20))

for i in range(6):

    ax = fig.add_subplot(1, 6, i+1, xticks=[], yticks=[])

    ax.imshow(X_train[i].reshape(28,28), cmap='gray')

    ax.set_title(str(Y_train[i]))
def visualize_input(img, ax):

    ax.imshow(img, cmap='gray')

    width, height = img.shape

    thresh = img.max()/2.5

    for x in range(width):

        for y in range(height):

            ax.annotate(str(round(img[x][y],2)), xy=(y,x),

                        horizontalalignment='center',

                        verticalalignment='center',

                        color='white' if img[x][y]<thresh else 'black')



fig = plt.figure(figsize = (12,12)) 

ax = fig.add_subplot(111)

visualize_input(X_train[9].reshape(28,28), ax)
g = sns.countplot(Y_train)
X_train = X_train/255.0

X_test = df2/255.0
X_train = X_train.reshape(-1,28,28,1)

X_test = X_test.values.reshape(-1,28,28,1)
Y_train = to_categorical(Y_train, num_classes = 10)
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = 0.1, random_state=7)
model = Sequential()



model.add(Conv2D(filters=32, kernel_size=(5,5), padding='Same', activation='relu', input_shape=(28,28,1)))

model.add(BatchNormalization())

model.add(Conv2D(filters=32, kernel_size=(5,5), padding='Same', activation='relu'))

model.add(BatchNormalization())

model.add(MaxPool2D(pool_size=(2,2)))

model.add(Dropout(0.25))



model.add(Conv2D(filters=64, kernel_size=(3,3), padding='Same', activation='relu'))

model.add(BatchNormalization())

model.add(Conv2D(filters=64, kernel_size=(3,3), padding='Same', activation='relu'))

model.add(BatchNormalization())

model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))

model.add(Dropout(0.25))



model.add(Flatten())

model.add(Dense(256, activation='relu'))

model.add(BatchNormalization())

model.add(Dense(256, activation='relu'))

model.add(BatchNormalization())

model.add(Dropout(0.5))

model.add(Dense(10, activation='softmax'))
plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)

Image("model.png")
epochs = 30

batch_size = 64
optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
model.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])
learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', 

                                            patience=2, 

                                            verbose=1, 

                                            factor=0.5, 

                                            min_lr=0.00001)
image_gen=ImageDataGenerator(rotation_range=10,width_shift_range=0.1,height_shift_range=0.1,shear_range=0.1,zoom_range=0.1,horizontal_flip=False,vertical_flip=False,fill_mode='nearest')
train_image_gen=image_gen.fit(X_train)
model.fit_generator(image_gen.flow(X_train, Y_train, batch_size=batch_size), epochs=epochs, validation_data = (X_val, Y_val), callbacks = [learning_rate_reduction])
metrics=pd.DataFrame(model.history.history)

metrics
metrics[['loss' , 'val_loss']].plot()

plt.show()
metrics[['accuracy' , 'val_accuracy']].plot()

plt.show()
np.random.seed(16)

random_selection=np.random.randint(0,4201,size=1)

random_sample=X_val[random_selection]

print('Prediction:')

print(model.predict_classes(random_sample.reshape(1,28,28,1))[0])

plt.imshow(random_sample.reshape(28,28),cmap='binary')

plt.show
np.random.seed(9)

random_selection=np.random.randint(0,4201,size=1)

random_sample=X_val[random_selection]

print('Prediction:')

print(model.predict_classes(random_sample.reshape(1,28,28,1))[0])

plt.imshow(random_sample.reshape(28,28),cmap='binary')

plt.show
np.random.seed(27)

random_selection=np.random.randint(0,4201,size=1)

random_sample=X_val[random_selection]

print('Prediction:')

print(model.predict_classes(random_sample.reshape(1,28,28,1))[0])

plt.imshow(random_sample.reshape(28,28),cmap='binary')

plt.show
def plot_confusion_matrix(cm, classes,

                          normalize=False,

                          title='Confusion matrix',

                          cmap=plt.cm.Blues):

    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title)

    plt.colorbar()

    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, rotation=45)

    plt.yticks(tick_marks, classes)



    if normalize:

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]



    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        plt.text(j, i, cm[i, j],

                 horizontalalignment="center",

                 color="white" if cm[i, j] > thresh else "black")



    plt.tight_layout()

    plt.ylabel('True label')

    plt.xlabel('Predicted label')



Y_pred = model.predict(X_val)

Y_pred_classes = np.argmax(Y_pred,axis = 1) 

Y_true = np.argmax(Y_val,axis = 1) 

confusion_mtx = confusion_matrix(Y_true, Y_pred_classes) 

plot_confusion_matrix(confusion_mtx, classes = range(10)) 
results = model.predict(X_test)
results = np.argmax(results,axis = 1)

results = pd.Series(results,name="Label")
submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)

submission.to_csv("submission.csv",index=False)