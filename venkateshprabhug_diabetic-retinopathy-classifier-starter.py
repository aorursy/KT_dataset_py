import os

import tensorflow as tf

import zipfile
import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
os.listdir('../input/resnet50')
train = pd.read_csv('../input/aptos2019-blindness-detection/train.csv')
train.head()
train.shape
train['id_code'] = train['id_code'] + '.png'
import cv2 as cv
plt.figure(figsize=[20,20])

pos = 1

for file in train['id_code'][:25]:

    img = cv.imread('../input/aptos2019-blindness-detection/train_images/'+file)

    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    img = cv.resize(img, (1024, 1024))

    plt.subplot(5, 5, pos)

    pos += 1

    plt.title(train['diagnosis'][pos])

    plt.imshow(img)

plt.show()
sns.countplot(x='diagnosis', data=train, palette="GnBu_d")
# class_0, class_1, class_2, class_3, class_4 = train['diagnosis'].value_counts()

# df_0 = train[train['diagnosis'] == '0']

# df_1 = train[train['diagnosis'] == '1']

# df_2 = train[train['diagnosis'] == '2']

# df_3 = train[train['diagnosis'] == '3']

# df_4 = train[train['diagnosis'] == '4']
train['diagnosis'].value_counts()
# df_0_u = df_0.sample(class_4)

# df_1_u = df_1.sample(class_4)

# df_2_u = df_2.sample(class_4)

# df_4_u = df_4.sample(class_4)
# df_u = pd.concat([df_0_u, df_1_u, df_2_u, df_3, df_4_u], axis=0)
# sns.countplot(x='diagnosis', data=df_u)
# df_u['diagnosis'] = df_u['diagnosis'].astype('str')

# # train['diagnosis'] = train['diagnosis'].astype('str')
seed = 10

batch_size = 32

img_size = 32

nb_epochs = 5
train = train.sample(frac=1, random_state=seed)
train['diagnosis'] = train['diagnosis'].astype('str')
x = train['id_code'].values

y = train['diagnosis'].values
from sklearn.model_selection import train_test_split
x_train, x_va, y_train, y_val = train_test_split(x, y, random_state = 0, test_size=0.2, stratify=y)
x_train.shape, x_val.shape, x_test.shape
train_df = pd.DataFrame({'image': x_train, 'class': y_train})

valid_df = pd.DataFrame({'image': x_val, 'class': y_val})
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
valid_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_dataframe(

  dataframe = train_df,

    directory = '../input/aptos2019-blindness-detection/train_images',

    target_size=(img_size,img_size),

    x_col='image',

    y_col='class',

    class_mode='categorical',

    color_mode='grayscale',

    batch_size=batch_size,

    shuffle=True

)
valid_generator = valid_datagen.flow_from_dataframe(

  dataframe = valid_df,

    directory = '../input/aptos2019-blindness-detection/train_images',

    target_size=(img_size, img_size),

    x_col='image',

    y_col='class',

    class_mode='categorical',

    color_mode='grayscale',

    batch_size=batch_size,

    shuffle=True

)
# local_weights_file = '../input/resnet50/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'


# pre_trained_model = tf.keras.applications.resnet50.ResNet50(input_shape = (256, 256, 1), 

#                                 include_top = False, 

#                                 weights = None)
# pre_trained_model.load_weights(local_weights_file)
# for layer in pre_trained_model.layers:

#   layer.trainable = False
# pre_trained_model.summary()
# last_layer = pre_trained_model.get_layer('mixed7')

# print('last layer output shape: ', last_layer.output_shape)

# last_output = last_layer.output
model = tf.keras.models.Sequential([

#     pre_trained_model,

#     tf.keras.layers.MaxPooling2D(),

#     tf.keras.layers.Dropout(0.5),

#     tf.keras.layers.Flatten(),

    tf.keras.layers.Conv2D(32, (6, 6), activation = tf.nn.relu, input_shape = (img_size, img_size, 1)),

    tf.keras.layers.MaxPool2D(2, 2),

    tf.keras.layers.Dropout(0.8),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(5, activation = tf.nn.softmax)

])
checkpoint = tf.keras.callbacks.ModelCheckpoint("model_1.h5", monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)



learning_rate_reduction = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_acc', 

                                            patience=2, 

                                            verbose=1, 

                                            factor=0.5, 

                                            min_lr=0.000001)
optimizer = tf.keras.optimizers.RMSprop(lr=0.001)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['acc'])
history = model.fit_generator(train_generator, 

                    epochs=nb_epochs, 

                    validation_data=valid_generator, 

                    callbacks=[learning_rate_reduction, checkpoint],

                    steps_per_epoch=100,

                    use_multiprocessing=True,

                    class_weight={0:0.1, 1: 0.2, 2:0.2, 3:0.3, 4:0.2})
# # Plot the loss and accuracy curves for training and validation 

# fig, ax = plt.subplots(2,1)

# ax[0].plot(history.history['loss'], color='b', label="Training loss")

# ax[0].plot(history.history['val_loss'], color='r', label="validation loss",axes =ax[0])

# legend = ax[0].legend(loc='best', shadow=True)



# ax[1].plot(history.history['acc'], color='b', label="Training accuracy")

# ax[1].plot(history.history['val_acc'], color='r',label="Validation accuracy")

# legend = ax[1].legend(loc='best', shadow=True)
import numpy as np
# train_preds = model.predict_generator(valid_generator)
# train_preds = [np.argmax(pred) for pred in train_preds]
# from sklearn.metrics import cohen_kappa_score
# print("Train Cohen Kappa score: %.3f" % cohen_kappa_score(train_preds, y_test.astype('int'), weights='quadratic'))
test = pd.read_csv('../input/aptos2019-blindness-detection/test.csv')
test_df = pd.DataFrame()
test_df['image'] = test['id_code'] + '.png'
test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_dataframe(

  dataframe = test_df,

    directory = '../input/aptos2019-blindness-detection/test_images',

    target_size=(img_size, img_size),

    x_col='image',

    y_col=None,

    class_mode=None,

    color_mode='grayscale',

    batch_size=batch_size

)
pred = model.predict_generator(test_generator)
import numpy as np
pred = np.argmax(pred, 1).reshape(-1, 1)
pred.shape, test.shape
pred = pd.DataFrame(pred)
submission = test.copy()
submission['diagnosis'] = pred
sns.countplot(x='diagnosis', data=submission)
submission.to_csv('submission.csv',index=False)