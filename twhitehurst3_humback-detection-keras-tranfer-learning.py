import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import os

import matplotlib.image as mplimg

from matplotlib.pyplot import imshow

from IPython.display import SVG







from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import OneHotEncoder



from keras import layers

from keras.preprocessing import image

from keras.layers import Activation,Conv2D,Flatten,SeparableConv2D,Dense,Input,Dropout,BatchNormalization,GlobalMaxPooling2D,GlobalAveragePooling2D,MaxPooling2D,AveragePooling2D

from keras.models import Model,Sequential

from keras.applications.mobilenet import MobileNet,preprocess_input

from keras.applications import MobileNet

from keras.optimizers import SGD,Adam

from keras.utils.vis_utils import plot_model,model_to_dot

from keras.metrics import categorical_accuracy, top_k_categorical_accuracy, categorical_crossentropy

from keras.applications.vgg16 import VGG16

from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping,EarlyStopping,TensorBoard,ReduceLROnPlateau,CSVLogger,LearningRateScheduler
def show_final_history(history):

    fig, ax = plt.subplots(1, 2, figsize=(15,5))

    ax[0].set_title('loss')

    ax[0].plot(history.epoch, history.history["loss"], label="Train loss")

    ax[0].plot(history.epoch, history.history["val_loss"], label="Validation loss")

    ax[1].set_title('acc')

    ax[1].plot(history.epoch, history.history["acc"], label="Train acc")

    ax[1].plot(history.epoch, history.history["val_acc"], label="Validation acc")

    ax[0].legend()

    ax[1].legend()
def step_decay(epoch):

    initial_lrate = 0.1

    drop = 0.5

    epochs_drop = 5.0

    lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))

    return lrate
def collect_labels(y):

    values = np.array(y)

    label_encoder = LabelEncoder()

    integer_encoded = label_encoder.fit_transform(values)

    onehot_encoder = OneHotEncoder(sparse=False)

    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)

    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)

    y = onehot_encoded

    return y, label_encoder
def collect_images(data, m, dataset):

    X_train = np.zeros((m, 100, 100, 3))

    count = 0

    

    for fig in data['Image']:

        img = image.load_img("../input/"+dataset+"/"+fig, target_size=(100, 100, 3))

        x = image.img_to_array(img)

        x = preprocess_input(x)

        X_train[count] = x

        if (count%500 == 0):

            print("Collecting Image: ", count+1, ", ", fig)

        count += 1

    return X_train
os.listdir("../input/")

training_df = pd.read_csv("../input/train.csv")

training_df.head(5)
X = collect_images(training_df,training_df.shape[0],'train')

y,label_encoder = collect_labels(training_df['Id'])

X /= 255

y.shape
base_model = VGG16(include_top=False, weights='imagenet',input_shape=(100,100,3))



for layer in base_model.layers[:-12]:

        layer.trainable = False

        

for layer in base_model.layers:

    print(layer, layer.trainable)



model = Sequential()

model.add(base_model)

model.add(GlobalAveragePooling2D())

model.add(Dense(1024,activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(512,activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(5005,activation='softmax'))



model.summary()
checkpoint = ModelCheckpoint(

    './base.model',

    monitor='categorical_accuracy',

    verbose=1,

    save_best_only=True,

    mode='max',

    save_weights_only=False,

    period=1

)

earlystop = EarlyStopping(

    monitor='val_loss',

    min_delta=0.001,

    patience=30,

    verbose=1,

    mode='auto'

)

tensorboard = TensorBoard(

    log_dir = './logs',

    histogram_freq=0,

    batch_size=16,

    write_graph=True,

    write_grads=True,

    write_images=False,

)



csvlogger = CSVLogger(

    filename= "training_csv.log",

    separator = ",",

    append = False

)



lrsched = LearningRateScheduler(step_decay,verbose=1)



reduce = ReduceLROnPlateau(

    monitor='val_loss',

    factor=0.8,

    patience=5,

    verbose=1, 

    mode='auto',

    min_delta=0.0001, 

    cooldown=1, 

    min_lr=0.0001

)



callbacks = [checkpoint,tensorboard,earlystop,csvlogger,reduce]
opt = SGD(lr=1e-3,momentum=0.99)

opt1 = Adam(lr=2e-3)





model.compile(

    loss='categorical_crossentropy',

    optimizer=opt,

    metrics=['accuracy']

)



history = model.fit(

    X,

    y,

    epochs=5,

    batch_size=128,

    verbose=1,

)
plt.plot(history.history['acc'])

plt.title('Model accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.show()
test = os.listdir("../input/test/")

col = ['Image']

test_df = pd.DataFrame(test, columns=col)

test_df['Id'] = ''

X = collect_images(test_df, test_df.shape[0], "test")

X /= 255
predictions = model.predict(np.array(X), verbose=1)



for i, pred in enumerate(predictions):

    test_df.loc[i, 'Id'] = ' '.join(label_encoder.inverse_transform(pred.argsort()[-5:][::-1]))



test_df.to_csv('submission.csv', index=False)

test_df.head()