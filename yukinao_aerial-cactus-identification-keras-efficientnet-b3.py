import numpy as np

import pandas as pd

import os

import matplotlib.image as mpimg



import cv2

import seaborn as sns



import matplotlib.pyplot as plt

%matplotlib inline



from sklearn.metrics import confusion_matrix, cohen_kappa_score

from keras.models import Model

from keras import optimizers, applications

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import EarlyStopping, ReduceLROnPlateau

from keras.layers import Dense, Dropout, GlobalAveragePooling2D, Input



import warnings

sns.set(style="whitegrid")

warnings.filterwarnings("ignore")
DATA_DIR = '../input/aerial-cactus-identification'

TRAIN_DIR = DATA_DIR + '/train/train/'

TEST_DIR = DATA_DIR + '/test/test/'
train = pd.read_csv(DATA_DIR + '/train.csv')
print('Number of train samples: ', train.shape[0])

display(train.head())
train.tail()
print('Number of train samples: ', train.shape[0])

vc = train['has_cactus'].value_counts()

print(vc)
train['has_cactus'].value_counts().plot.bar()
sns.set_style("white")

count = 1

plt.figure(figsize=[10, 10])

for img_name in train['id'][:15]:

    img = cv2.imread("../input/aerial-cactus-identification/train/train/%s" % img_name)[...,[2, 1, 0]]

    HEIGHT, WIDTH, CHANEL = img.shape[:3]

    plt.subplot(5, 5, count)

    plt.imshow(img)

    plt.title("Image %s" % count)

    count += 1

    

plt.show()
#画像のサイズ、カラー

print(HEIGHT)

print(WIDTH)

print(CHANEL)
# Model parameters

BATCH_SIZE = 128

EPOCHS = 20

WARMUP_EPOCHS = 2

LEARNING_RATE = 1e-4

WARMUP_LEARNING_RATE = 1e-3 

N_CLASSES = train['has_cactus'].nunique()

ES_PATIENCE = 5

RLROP_PATIENCE = 3

DECAY_DROP = 0.5
# Preprocecss data

# 数字から文字に変換

train['has_cactus'] = train['has_cactus'].astype('str')

train.head()
train_datagen=ImageDataGenerator(

    rescale=1./255,

    rotation_range=30,

    horizontal_flip=True,

    vertical_flip=True,

    fill_mode='nearest',

    validation_split=0.2

)



train_generator=train_datagen.flow_from_dataframe(

    dataframe=train,

    directory=TRAIN_DIR,

    x_col="id",

    y_col="has_cactus",

    batch_size=BATCH_SIZE,

    class_mode="categorical",

    target_size=(HEIGHT, WIDTH),

    subset='training')



valid_generator=train_datagen.flow_from_dataframe(

    dataframe=train,

    directory=TRAIN_DIR,

    x_col="id",

    y_col="has_cactus",

    batch_size=BATCH_SIZE,

    class_mode="categorical",    

    target_size=(HEIGHT, WIDTH),

    subset='validation')
# Please check the your Setting, Internet botton = ON ,right side

!git clone https://github.com/Tony607/efficientnet_keras_transfer_learning

%cd ./efficientnet_keras_transfer_learning/
# imagenet's weights for fine tuning

from efficientnet import EfficientNetB3 as Net

from efficientnet import center_crop_and_resize, preprocess_input



conv_base = Net(weights='imagenet',include_top=False,input_shape=(HEIGHT,WIDTH,CHANEL))
#Add dense

from tensorflow.keras import models

from tensorflow.keras import layers



dropout_rate = 0.2

model = models.Sequential()

model.add(conv_base)

model.add(layers.GlobalMaxPooling2D(name="gap"))

model.add(layers.Dropout(dropout_rate, name="dropout_out"))

model.add(layers.Dense(N_CLASSES,activation="softmax", name="final_output"))
model.summary()
#freeze the EfficientNet's Weight

conv_base.trainable = False
%cd ../

!ls
from tensorflow.keras import optimizers



metric_list = ["accuracy"]

optimizer = optimizers.Adam(lr=WARMUP_LEARNING_RATE)

model.compile(optimizer=optimizer, loss="categorical_crossentropy",  metrics=metric_list)

model.summary()
%%time



STEP_SIZE_TRAIN = train_generator.n//train_generator.batch_size

STEP_SIZE_VALID = valid_generator.n//valid_generator.batch_size



history_warmup = model.fit_generator(generator=train_generator,

                              steps_per_epoch=STEP_SIZE_TRAIN,

                              validation_data=valid_generator,

                              validation_steps=STEP_SIZE_VALID,

                              epochs=WARMUP_EPOCHS,

                              verbose=1).history
for layer in model.layers:

    layer.trainable = True
optimizer = optimizers.Adam(lr=LEARNING_RATE)

model.compile(optimizer=optimizer, loss="categorical_crossentropy",  metrics=metric_list)

model.summary()
%%time



history_finetunning = model.fit_generator(generator=train_generator,

                              steps_per_epoch=STEP_SIZE_TRAIN,

                              validation_data=valid_generator,

                              validation_steps=STEP_SIZE_VALID,

                              epochs=EPOCHS,

                              #epochs=1,

                              verbose=1).history
history = {'loss': history_warmup['loss'] + history_finetunning['loss'], 

           'val_loss': history_warmup['val_loss'] + history_finetunning['val_loss'], 

           'acc': history_warmup['acc'] + history_finetunning['acc'], 

           'val_acc': history_warmup['val_acc'] + history_finetunning['val_acc']}



sns.set_style("whitegrid")

fig, (ax1, ax2) = plt.subplots(2, 1, sharex='col', figsize=(20, 14))



ax1.plot(history['loss'], label='Train loss')

ax1.plot(history['val_loss'], label='Validation loss')

ax1.legend(loc='best')

ax1.set_title('Loss')



ax2.plot(history['acc'], label='Train Accuracy')

ax2.plot(history['val_acc'], label='Validation accuracy')

ax2.legend(loc='best')

ax2.set_title('Accuracy')



plt.xlabel('Epochs')

sns.despine()

plt.show()
# 評価

complete_datagen = ImageDataGenerator(rescale=1./255)

complete_generator = complete_datagen.flow_from_dataframe(  

        dataframe=train,

        directory = TRAIN_DIR,

        x_col="id",

        target_size=(HEIGHT, WIDTH),

        batch_size=1,

        shuffle=False,

        class_mode=None)



STEP_SIZE_COMPLETE = complete_generator.n//complete_generator.batch_size

train_preds = model.predict_generator(complete_generator, steps=STEP_SIZE_COMPLETE)

train_preds = [np.argmax(pred) for pred in train_preds]
%%time

labels = ['0 - No', '1 - Yes']

cnf_matrix = confusion_matrix(train['has_cactus'].astype('int'), train_preds)

cnf_matrix_norm = cnf_matrix.astype('float') / cnf_matrix.sum(axis=1)[:, np.newaxis]

df_cm = pd.DataFrame(cnf_matrix_norm, index=labels, columns=labels)

plt.figure(figsize=(16, 7))

sns.heatmap(df_cm, annot=True, fmt='.2f', cmap="Blues")

plt.show()
!rm -rf ./efficientnet_keras_transfer_learning/
submit = pd.read_csv('../input/aerial-cactus-identification/sample_submission.csv')



test_datagen = ImageDataGenerator(rescale=1./255)



test_generator = test_datagen.flow_from_dataframe(  

        dataframe=submit,

        directory = TEST_DIR,

        x_col="id",

        target_size=(HEIGHT, WIDTH),

        batch_size=1,

        shuffle=False,

        class_mode=None)
# 予測

test_generator.reset()

STEP_SIZE_TEST = test_generator.n//test_generator.batch_size

preds = model.predict_generator(test_generator, steps=STEP_SIZE_TEST)

predictions = [np.argmax(pred) for pred in preds]
#提出

filenames = test_generator.filenames

results = pd.DataFrame({'id':filenames, 'has_cactus':predictions})

results.to_csv('submission.csv',index=False)

results.head(10)