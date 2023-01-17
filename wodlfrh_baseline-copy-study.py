# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import gc

import os

import warnings

import seaborn as sns

import matplotlib.pyplot as plt

from tqdm import tqdm



from keras import backend as K

warnings.filterwarnings(action='ignore')



K.image_data_format()
DATA_PATH = '../input'

os.listdir(DATA_PATH)
TRAIN_IMG_PATH = os.path.join(DATA_PATH,'train')

TEST_IMG_PATH = os.path.join(DATA_PATH,'test')



df_train = pd.read_csv(os.path.join(DATA_PATH, 'train.csv'))

df_test = pd.read_csv(os.path.join(DATA_PATH, 'test.csv'))

df_class=pd.read_csv(os.path.join(DATA_PATH, 'class.csv'))
df_train.shape, df_test.shape, df_class.shape
df_train.head()
if set(list(df_train.img_file)) == set(os.listdir(TRAIN_IMG_PATH)):

    print("ok")

else : 

    print("is null")
if set(list(df_test.img_file)) == set(os.listdir(TEST_IMG_PATH)):

    print("ok")

else : 

    print("is null")
print("Number of Train Data : {}".format(df_train.shape[0]))

print("Number of Test Data : {}".format(df_test.shape[0]))
df_class.head()
print("Number of Target : {}".format(df_class.shape[0]))

print("Varierty of Train Data Target : {}".format(df_train['class'].nunique()))
plt.figure(figsize=(12,6))

sns.countplot(df_train['class'], order=df_train['class'].value_counts(ascending=True).index)
cntEachClass = df_train['class'].value_counts(ascending=False)

print("class of most count : {}".format(cntEachClass.index[0]))

print("num of most count : {}".format(cntEachClass[cntEachClass.index[0]]))
print("class of least count : {}".format(cntEachClass.index[-1]))

print("num of least count : {}".format(cntEachClass[cntEachClass.index[-1]]))
print("Mean : {}".format(cntEachClass.mean()))
cntEachClass.describe()
import PIL

from PIL import ImageDraw



tmp_imgs = df_train['img_file'][100:110]

plt.figure(figsize=(12,20))



for num, f_name in enumerate(tmp_imgs):

    img = PIL.Image.open(os.path.join(TRAIN_IMG_PATH, f_name))

    plt.subplot(5,2, num+1)

    plt.imshow(img)
def draw_rect(drawcontext, pos, outline=None, width=0):

    (x1, y1) = (pos[0], pos[1])

    (x2, y2) = (pos[2], pos[3])

    points = (x1,y1), (x2,y1), (x2,y2), (x1,y2), (x1,y1)

    drawcontext.line(points, fill=outline, width=width)
def make_boxing_img(img_name):

    if img_name.split('_')[0] == "train":

        PATH = TRAIN_IMG_PATH

        data = df_train

    elif img_name.split('_')[0] == "test":

        PATH = TEST_IMG_PATH

        data = df_test

    

    img = PIL.Image.open(os.path.join(PATH, img_name))

    pos = data.loc[data["img_file"] == img_name, \

                  ['bbox_x1','bbox_y1','bbox_x2','bbox_y2']].values.reshape(-1)

    draw = ImageDraw.Draw(img)

    draw_rect(draw, pos, outline='red',width=10)

    

    return img
f_name = "train_00102.jpg"



plt.figure(figsize=(20,10))

plt.subplot(1,2,1)



origin_img = PIL.Image.open(os.path.join(TRAIN_IMG_PATH, f_name))

plt.imshow(origin_img)



plt.subplot(1,2,2)

boxing = make_boxing_img(f_name)

plt.imshow(boxing)



plt.show()
from sklearn.model_selection import train_test_split



df_train['class'] = df_train['class'].astype('str')
df_train = df_train[['img_file','class']]

df_test = df_test[['img_file']]



its = np.arange(df_train.shape[0])

train_idx, val_idx = train_test_split(its, train_size = 0.8, random_state=42)



X_train = df_train.iloc[train_idx, :]

X_val = df_train.iloc[val_idx, :]



print(X_train.shape, X_val.shape, df_test.shape)
from keras.applications.resnet50 import ResNet50, preprocess_input

from keras.preprocessing.image import ImageDataGenerator



img_size = (224, 224)

nb_train_samples = len(X_train)

nb_validation_samples = len(X_val)

nb_test_samples = len(df_test)

epochs = 3

batch_size = 32



train_datagen = ImageDataGenerator(

    horizontal_flip = True,

    vertical_flip = False,

    zoom_range = 0.1,

    preprocessing_function = preprocess_input)



val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)



train_generator = train_datagen.flow_from_dataframe(

    dataframe = X_train,

    directory = '../input/train',

    x_col = 'img_file',

    y_col = 'class',

    target_size = img_size,

    color_mode = 'rgb',

    class_mode = 'categorical',

    batch_size=batch_size,

    seed = 42

)



validation_generator = val_datagen.flow_from_dataframe(

    dataframe = X_val,

    directory = '../input/train',

    x_col = 'img_file',

    y_col = 'class',

    target_size = img_size,

    color_mode = 'rgb',

    class_mode = 'categorical',

    batch_size=batch_size,

    shuffle=False

)



test_generator = test_datagen.flow_from_dataframe(

    dataframe = df_test,

    directory = '../input/test',

    x_col = 'img_file',

    y_col = None,

    target_size = img_size,

    color_mode = 'rgb',

    class_mode = None,

    batch_size=batch_size,

    shuffle=False

)
#resNet_model = ResNet50(include_top=False, input_shape = (224,224,3))
#resNet_model.summary()
"""

from keras.models import Sequential, Model

from keras.layers import Dense, Dropout, Flatten, Activation, Conv2D, GlobalAveragePooling2D



model = Sequential()

model.add(resNet_model)

model.add(GlobalAveragePooling2D())

model.add(Dense(196, activation='softmax', kernel_initializer='he_normal'))

model.summary()

"""
from keras import layers

from keras.models import Sequential, Model

from keras.layers import Dense, Dropout, Flatten, Activation, Conv2D, GlobalAveragePooling2D



model = Sequential()

model.add(layers.Conv2D(64, (5,5), activation='relu', input_shape=(224, 224, 3)))

model.add(layers.MaxPooling2D((2,2)))

model.add(layers.ZeroPadding2D(padding=1))

model.add(layers.Conv2D(256, (3,3), activation='relu'))

model.add(layers.MaxPooling2D((2,2)))

model.add(layers.ZeroPadding2D(padding=1))

model.add(layers.Conv2D(512, (3,3), activation='relu'))

model.add(layers.MaxPooling2D((2,2)))

model.add(layers.ZeroPadding2D(padding=1))

model.add(layers.Conv2D(512, (3,3), activation='relu'))

model.add(layers.GlobalAveragePooling2D())

model.add(layers.Dense(196, activation='softmax', kernel_initializer='he_normal'))

model.summary()
from sklearn.metrics import f1_score



def micro_f1(y_true, y_pred):

    return f1_score(y_true, y_pred, average='micro')



model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
def get_steps(num_samples, batch_size):

    if (num_samples % batch_size)>0:

        return (num_samples // batch_size) + 1

    else :

        return num_samples // batch_size
%%time

from keras.callbacks import ModelCheckpoint, EarlyStopping



filepath = "my_resnet_model_{val_acc:.2f}_{val_loss:.4f}.h5"



es = EarlyStopping(monitor='val_acc', min_delta=0, patience=3, verbose=1, mode='auto')



callbackList = [es]



history = model.fit_generator(

    train_generator,

    steps_per_epoch = get_steps(nb_train_samples, batch_size),

    epochs = epochs,

    validation_data = validation_generator,

    validation_steps = get_steps(nb_validation_samples, batch_size),

    callbacks = callbackList

)

gc.collect()
plt.plot(history.history['acc'])

plt.plot(history.history['val_acc'])

plt.show()
plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.show()
%%time

test_generator.reset()

prediction = model.predict_generator(

    generator = test_generator,

    steps = get_steps(nb_test_samples, batch_size),

    verbose=1

)
prediction
train_generator.class_indices
predicted_class_indices = np.argmax(prediction, axis=1)
labels = (train_generator.class_indices)
dict((v,k) for k,v in labels.items())
labels = dict((v,k) for k,v in labels.items())

predictions = [labels[k] for k in predicted_class_indices]
submission = pd.read_csv(os.path.join(DATA_PATH, 'sample_submission.csv'))

submission['class'] = predictions

submission.to_csv("submission.csv", index=False)

submission.head()