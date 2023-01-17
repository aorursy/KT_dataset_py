import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import keras
from keras.models import *
from keras.layers import *
from keras.losses import *
from keras.callbacks import *
from keras.applications.vgg16 import VGG16
from keras.applications.inception_v3 import InceptionV3
import skimage
from skimage.color import rgb2gray
import sklearn
from sklearn.metrics import *
from imageio import imread
import seaborn as sns
import matplotlib.pyplot as plt
import glob
test = pd.DataFrame()
test['file'] = glob.glob("../input/walk_or_run_test/test/run/*")+glob.glob("../input/walk_or_run_test/test/walk/*")
test['label'] = [ 1 for _ in glob.glob("../input/walk_or_run_test/test/run/*")]+[0 for _ in glob.glob("../input/walk_or_run_test/test/walk/*")]

train = pd.DataFrame()
train['file'] = glob.glob("../input/walk_or_run_train/train/run/*")+glob.glob("../input/walk_or_run_train/train/walk/*")
train['label'] = [ 1 for _ in glob.glob("../input/walk_or_run_train/train/run/*")]+[0 for _ in glob.glob("../input/walk_or_run_train/train/walk/*")]
train = train.sample(frac=1).reset_index(drop=True)
train.head()
def make_model():
    transfer_model = InceptionV3(include_top=False, weights='imagenet', input_tensor=None, input_shape=(None,None,3), pooling='avg', classes=1000)
    model = Sequential()
    model.add(InputLayer((None,None,3)))
    model.add(transfer_model)
    model.add(Dropout(0.5))
    model.add(Dense(1,activation='sigmoid'))
    return transfer_model, model

transfer_model, model = make_model()
model.summary()
def image_gen(files, labels, batch_size=10, randomized=False, random_seed=1):
    rng = np.random.RandomState(random_seed)
    img_batch = []
    label_batch = []
    while True:
        indices = np.arange(len(files))
        if randomized:
            rng.shuffle(indices)
        for index in indices:
            img = imread(files[index])[:,:,0:3]/255
            label = labels[index]
            img_batch.append(img)
            label_batch.append(label)
            if len(img_batch) == batch_size:
                yield np.array(img_batch), np.array(label_batch)
                img_batch = []
                label_batch = []
        
        if len(img_batch) > 0:
                yield np.array(img_batch), np.array(label_batch)
                img_batch = []
                label_batch = []
transfer_model.trainable=False
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['acc'])
batch_size=100
epochs=50
history = model.fit_generator(image_gen(train['file'], train['label'], batch_size=batch_size, randomized=True, random_seed=1),
                    steps_per_epoch=int(np.ceil(len(train)/batch_size)),
                    epochs=epochs,
                    validation_data=image_gen(test['file'], test['label'], batch_size=batch_size, randomized=True),
                    validation_steps=int(np.ceil(len(test)/batch_size)),
                    callbacks=[EarlyStopping(monitor='val_loss', patience=3, verbose=0), ModelCheckpoint(filepath='./weights.hdf5', monitor='val_loss', verbose=0, save_best_only=True)],
                    verbose=2,
                   )
model.load_weights('weights.hdf5')
def plot_history(history):
    plt.subplot(1,2,1)
    plt.plot(history.history['loss'], 'r')
    plt.plot(history.history['val_loss'], 'b')
    plt.subplot(1,2,2)
    plt.plot(history.history['acc'], 'r')
    plt.plot(history.history['val_acc'], 'b')
    
plot_history(history)
!rm -f *.hdf5

transfer_model.trainable=True
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['acc'])
batch_size=100
epochs=50
history = model.fit_generator(image_gen(train['file'], train['label'], batch_size=batch_size, randomized=True, random_seed=1),
                    steps_per_epoch=int(np.ceil(len(train)/batch_size)),
                    epochs=epochs,
                    validation_data=image_gen(test['file'], test['label'], batch_size=batch_size, randomized=True),
                    validation_steps=int(np.ceil(len(test)/batch_size)),
                    callbacks=[ModelCheckpoint(filepath='./weights.hdf5', monitor='val_loss', verbose=0, save_best_only=True)],
                    verbose=2,
                   )
model.load_weights('weights.hdf5')
plot_history(history)
model.load_weights('weights.hdf5')
predicted_prob = model.predict_generator(image_gen(test['file'], test['label'], batch_size=1, randomized=False),
                    steps=len(test))
predicted = np.round(predicted_prob)

truth = test['label'].values
print(classification_report(truth, predicted))
sns.heatmap(confusion_matrix(truth, predicted), annot=True)