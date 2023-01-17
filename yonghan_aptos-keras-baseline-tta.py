import os

import sys



import numpy as np

import pandas as pd

import cv2

import seaborn as sns



from math import ceil

from tqdm import tqdm



from PIL import Image

from matplotlib import pyplot as plt



from sklearn.model_selection import train_test_split



from keras.applications.inception_v3 import InceptionV3, preprocess_input

from keras.preprocessing.image import ImageDataGenerator

from keras.models import Sequential, Model

from keras.layers import Input, Dense, GlobalAveragePooling2D, Dropout

from keras.optimizers import RMSprop, Adam, SGD

from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
DATA_PATH = '../input/aptos2019-blindness-detection'



TRAIN_IMG_PATH = os.path.join(DATA_PATH, 'train_images')

TEST_IMG_PATH = os.path.join(DATA_PATH, 'test_images')

TRAIN_LABEL_PATH = os.path.join(DATA_PATH, 'train.csv')

TEST_LABEL_PATH = os.path.join(DATA_PATH, 'test.csv')



df_train = pd.read_csv(TRAIN_LABEL_PATH)

df_test = pd.read_csv(TEST_LABEL_PATH)



print('num of train images ', len(os.listdir(TRAIN_IMG_PATH)))

print('num of test images  ', len(os.listdir(TEST_IMG_PATH)))
plt.figure(figsize=(12, 6))

sns.countplot(df_train["diagnosis"])

plt.title("Number of data per each diagnosis")

plt.show()
df_train['diagnosis'] = df_train['diagnosis'].astype('str')

df_train = df_train[['id_code', 'diagnosis']]

if df_train['id_code'][0].split('.')[-1] != 'png':

    for index in range(len(df_train['id_code'])):

        df_train['id_code'][index] = df_train['id_code'][index] + '.png'

        

df_test = df_test[['id_code']]

if df_test['id_code'][0].split('.')[-1] != 'png':

    for index in range(len(df_test['id_code'])):

        df_test['id_code'][index] = df_test['id_code'][index] + '.png'



train_data = np.arange(df_train.shape[0])

train_idx, val_idx = train_test_split(train_data, train_size=0.8, random_state=2019)



X_train = df_train.iloc[train_idx, :]

X_val = df_train.iloc[val_idx, :]

X_test = df_test



print(X_train.shape)

print(X_val.shape)

print(X_test.shape)
num_classes = 5

img_size = (299, 299, 3)

nb_train_samples = len(X_train)

nb_validation_samples = len(X_val)

nb_test_samples = len(X_test)

epochs = 50

batch_size = 32



train_datagen = ImageDataGenerator(

    horizontal_flip=True,

    vertical_flip=True,

    width_shift_range=0.1,

    height_shift_range=0.1,

    brightness_range=[0.5, 1.5],

    rescale=1./255

)

val_datagen = ImageDataGenerator(

    rescale=1./255

)

# Apply TTA

test_datagen = ImageDataGenerator(

    horizontal_flip=True,

    vertical_flip=True,

    width_shift_range=0.1,

    height_shift_range=0.1,

    brightness_range=[0.5, 1.5],

    rescale=1./255

)



train_generator = train_datagen.flow_from_dataframe(

    dataframe=X_train, 

    directory=TRAIN_IMG_PATH,

    x_col='id_code',

    y_col='diagnosis',

    target_size=img_size[:2],

    color_mode='rgb',

    class_mode='categorical',

    batch_size=batch_size,

    seed=2019

)

validation_generator = val_datagen.flow_from_dataframe(

    dataframe=X_val, 

    directory=TRAIN_IMG_PATH,

    x_col='id_code',

    y_col='diagnosis',

    target_size=img_size[:2],

    color_mode='rgb',

    class_mode='categorical',

    batch_size=batch_size,

    shuffle=False,

    seed=2019

)

test_generator = test_datagen.flow_from_dataframe(

    dataframe=X_test,

    directory=TEST_IMG_PATH,

    x_col='id_code',

    y_col=None,

    target_size= img_size[:2],

    color_mode='rgb',

    class_mode=None,

    batch_size=batch_size,

    shuffle=False,

    seed=2019

)
def get_model(file_path, input_shape, num_classes):

    input_tensor = Input(shape=input_shape)

    base_model = InceptionV3(include_top=False,

                             weights=None,

                             input_tensor=input_tensor)

    base_model.load_weights(filepath=file_path)

    x = base_model.output

    x = GlobalAveragePooling2D()(x)

    x = Dense(1024, activation='relu')(x)

    x = Dropout(0.25)(x)

    output_tensor = Dense(num_classes, activation='softmax')(x)

    

    model = Model(inputs=input_tensor, outputs=output_tensor)

    

#     optimizer = Adam(lr=1e-4)

    optimizer = RMSprop(lr=1e-4)

    model.compile(

        loss='categorical_crossentropy',

        optimizer=optimizer,

        metrics=['accuracy']

    )

    

    return model
model_path = '../input/inceptionv3/'

weight_file = 'inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'

model = get_model(

    file_path=os.path.join(model_path, weight_file),

    input_shape=img_size,

    num_classes=num_classes

)

# model.summary()
LOG_DIR = './logs'

if not os.path.isdir(LOG_DIR):

    os.mkdir(LOG_DIR)

else:

    pass

CKPT_PATH = LOG_DIR + '/checkpoint-{epoch:02d}-{val_loss:.4f}.hdf5'



checkPoint = ModelCheckpoint(

    filepath=CKPT_PATH,

    monitor='val_loss',

    verbose=1,

    save_best_only=True,

    mode='min'

)

reduceLROnPlateau = ReduceLROnPlateau(

    monitor='val_loss',

    factor=0.1,

    patience=5,

    min_lr=0.000001,

    verbose=1,

    mode='min'

)

earlyStopping = EarlyStopping(

    monitor='val_loss',

    patience=15,

    verbose=1,

    mode='min'

)



history = model.fit_generator(

    train_generator,

    steps_per_epoch=ceil(nb_train_samples/batch_size),

    epochs=epochs,

    validation_data=validation_generator,

    validation_steps=ceil(nb_validation_samples/batch_size),

    callbacks=[checkPoint, reduceLROnPlateau, earlyStopping],

    verbose=2  # If occur error that 'Timeout waiting for IOPub output', set verbose to 0.

)
acc = history.history['acc']

val_acc = history.history['val_acc']

 

plt.plot(acc)

plt.plot(val_acc)

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.legend(['Train', 'Val'], loc='upper left')

plt.show()
loss = history.history['loss']

val_loss = history.history['val_loss']



plt.plot(loss)

plt.plot(val_loss)

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.legend(['Train', 'Val'], loc='upper left')

plt.show()
log_dir_list = os.listdir(LOG_DIR)

ckpt_list = []

for file in log_dir_list:

    if file.split('-')[0] == 'checkpoint':

        ckpt_list.append(file)



loss_list = []

for file in ckpt_list:

    file = file.split('-')[2]

    file = file[:-3]    # Remove extension name

    loss_list.append(file)

    

# The model with the lowest validation loss

loss = ckpt_list[loss_list.index(min(loss_list))]

best_model = LOG_DIR + '/' + loss

model.load_weights(best_model)
# Apply TTA

preds_tta = []

tta_steps = 10

for i in tqdm(range(tta_steps)):

    test_generator.reset()

    preds = model.predict_generator(

        generator=test_generator,

        steps =ceil(nb_test_samples/batch_size)

    )

    preds_tta.append(preds)
preds_mean = np.mean(preds_tta, axis=0)

predicted_class_indices = np.argmax(preds_mean, axis=1)
labels = (train_generator.class_indices)

labels = dict((v,k) for k,v in labels.items())

predictions = [labels[k] for k in predicted_class_indices]



submission = pd.read_csv(os.path.join(DATA_PATH, 'sample_submission.csv'))

submission['diagnosis'] = predictions

submission.to_csv("submission.csv", index=False)

submission.head()
plt.figure(figsize=(12, 6))

sns.countplot(submission["diagnosis"])

plt.title("Number of data per each diagnosis")

plt.show()
diagnosis_0 = 0

diagnosis_1 = 1

diagnosis_2 = 2

diagnosis_3 = 3

diagnosis_4 = 4

for idx in range(len(submission['diagnosis'])):

    if submission['diagnosis'][idx] == '0':

        diagnosis_0 += 1

    elif submission['diagnosis'][idx] == '1':

        diagnosis_1 += 1

    elif submission['diagnosis'][idx] == '2':

        diagnosis_2 += 1

    elif submission['diagnosis'][idx] == '3':

        diagnosis_3 += 1

    elif submission['diagnosis'][idx] == '4':

        diagnosis_4 += 1

print("  0 - No DR              {}".format(diagnosis_0))

print("  1 - Mild               {}".format(diagnosis_1))

print("  2 - Moderate           {}".format(diagnosis_2))

print("  3 - Severe             {}".format(diagnosis_3))

print("  4 - Proliferative DR   {}".format(diagnosis_4))