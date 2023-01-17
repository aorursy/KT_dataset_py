import numpy as np

import pandas as pd 

import os

import warnings

import PIL

from numba import jit

warnings.filterwarnings(action='ignore')

print(os.listdir("../input"))
DATA_PATH = '../input/2019-3rd-ml-month-with-kakr/'

WEIGHTS_PATH = '../input/southgg-ensemble-model-weights/'

os.listdir(DATA_PATH)
os.listdir(WEIGHTS_PATH)
TRAIN_IMG_PATH = os.path.join(DATA_PATH,'train')

TEST_IMG_PATH = os.path.join(DATA_PATH,'test')



df_train = pd.read_csv(os.path.join(DATA_PATH, 'train.csv'))

df_test = pd.read_csv(os.path.join(DATA_PATH, 'test.csv'))

df_class=pd.read_csv(os.path.join(DATA_PATH, 'class.csv'))
from keras.applications.xception import Xception, preprocess_input

from keras.models import Sequential, Model

from keras import optimizers

from keras.layers import Dense, Dropout, Flatten, Activation, Conv2D, GlobalAveragePooling2D

from keras.preprocessing.image import ImageDataGenerator

from skimage.color import rgb2hsv
from keras import backend as K



def recall_m(y_true, y_pred):

        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))

        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))

        recall = true_positives / (possible_positives + K.epsilon())

        return recall



def precision_m(y_true, y_pred):

        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))

        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))

        precision = true_positives / (predicted_positives + K.epsilon())

        return precision

    

def f1_m(y_true, y_pred):

    precision = precision_m(y_true, y_pred)

    recall = recall_m(y_true, y_pred)

    return 2*((precision*recall)/(precision+recall+K.epsilon()))
@jit

def get_random_eraser(input_img, p=0.5, s_l=0.02, s_h=0.4, r_1=0.3, r_2=1/0.3, v_l=0, v_h=255, pixel_level=False):



    img_h, img_w, img_c = input_img.shape

    p_1 = np.random.rand()



    if p_1 > p:

        return input_img



    while True:

        s = np.random.uniform(s_l, s_h) * img_h * img_w

        r = np.random.uniform(r_1, r_2)

        w = int(np.sqrt(s / r))

        h = int(np.sqrt(s * r))

        left = np.random.randint(0, img_w)

        top = np.random.randint(0, img_h)



        if left + w <= img_w and top + h <= img_h:

            break



    if pixel_level:

        c = np.random.uniform(v_l, v_h, (h, w, img_c))

    else:

        c = np.random.uniform(v_l, v_h)



    input_img[top:top + h, left:left + w, :] = c







    return input_img
@jit

def rgb_to_hsv_erg(img):

    cvt_img = rgb2hsv(img)

    erg_img = get_random_eraser(cvt_img)

    pre_img = preprocess_input(erg_img)

    return pre_img
@jit

def rgb_to_hsv(img):

    cvt_img = rgb2hsv(img)

    pre_img = preprocess_input(cvt_img)

    return pre_img
IMG_SIZE = (299,299)

def crop_boxing_img(img_name, margin=10):

    if img_name.split('_')[0] == "train":

        PATH = TRAIN_IMG_PATH

        data = df_train

    elif img_name.split('_')[0] == "test":

        PATH = TEST_IMG_PATH

        data = df_test

        

    img = PIL.Image.open(os.path.join(PATH, img_name))

    pos = data.loc[data["img_file"] == img_name, ['bbox_x1','bbox_y1','bbox_x2','bbox_y2']].values.reshape(-1)

    

    width , height = img.size

    x1 = max(0, pos[0] - margin)

    y1 = max(0, pos[1] - margin)

    x2 = min(pos[2] + margin, width)

    y2 = min(pos[3] + margin, height)

    

    if abs(pos[2] - pos[0]) > width or abs(pos[3] - pos[1]) > height:

        print(f'{img_name} is wrong bounding box, img size : {img.size}, bbox_x1 : {pos[0]}, bbox_x2 : {pos[2]}, bbox_y1 : {pos[1]}, bbox_y2 : {pos[3]}')

        

        return img

    return img.crop((x1,y1,x2,y2)).resize(IMG_SIZE)
TRAIN_CROP_PATH = "./train_crop"

TEST_CROP_PATH = "./test_crop"

!mkdir {TRAIN_CROP_PATH}

!mkdir {TEST_CROP_PATH}
for i, row in df_train.iterrows():

    PATH = TRAIN_CROP_PATH

    cropped = crop_boxing_img(row['img_file'])

    cropped.save(os.path.join(PATH, row['img_file']))
for i, row in df_test.iterrows():

    PATH = TEST_CROP_PATH

    cropped = crop_boxing_img(row['img_file'])

    cropped.save(os.path.join(PATH, row['img_file']))
batch_size = 32

img_size = (299, 299)
train_datagen = ImageDataGenerator(

    horizontal_flip = True,

    vertical_flip = False,

    zoom_range = 0.1,

    preprocessing_function = rgb_to_hsv)
test_datagen = ImageDataGenerator(

    preprocessing_function=rgb_to_hsv)
df_train['class'] = df_train['class'].astype('str')
train_generator = train_datagen.flow_from_dataframe(

            dataframe = df_train,

            directory = TRAIN_CROP_PATH,

            x_col = 'img_file',

            y_col = 'class',

            target_size = img_size,

            color_mode = 'rgb',

            class_mode = 'categorical',

            batch_size=batch_size,

            seed = 42

        )
test_generator = test_datagen.flow_from_dataframe(

    dataframe = df_test,

    directory = TEST_CROP_PATH,

    x_col = 'img_file',

    y_col = None,

    target_size = img_size,

    color_mode = 'rgb',

    class_mode = None,

    batch_size=batch_size,

    shuffle=False

)
def get_steps(num_samples, batch_size):

    if (num_samples % batch_size)>0:

        return (num_samples // batch_size) + 1

    else :

        return num_samples // batch_size
def get_model():

    Xceptionmodel = Xception(weights='imagenet', input_shape = (299,299,3), include_top = False)

    model = Sequential()

    model.add(Xceptionmodel)

    model.add(GlobalAveragePooling2D())

    model.add(Dense(1024, activation='relu'))

    model.add(Dropout(0.5))

    model.add(Dense(1024, activation='relu'))

    model.add(Dropout(0.5))

    model.add(Dense(196, activation='softmax', kernel_initializer='he_normal'))

    model.summary()

    

    optimizer = optimizers.RMSprop(lr=0.0003)

    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=[f1_m])

    

    return model
model_xception_names = []

for i in range(1,6):

    model_name = WEIGHTS_PATH + str(i) + '_xception.hdf5'

    print(model_name)

    model_xception_names.append(model_name)
xception_prediction = []

for i, name in enumerate(model_xception_names):

    model_xception = get_model()

    model_xception.load_weights(name)

    test_generator.reset()

    pred = model_xception.predict_generator(

        generator=test_generator,

        steps = get_steps(len(df_test), batch_size),

        verbose=1

    )

    xception_prediction.append(pred)



y_pred_xception = np.mean(xception_prediction, axis=0)
predicted_class_indices
predicted_class_indices = np.argmax(y_pred_xception, axis=1)

labels = (train_generator.class_indices)
labels = dict((v,k) for k,v in labels.items())

predictions = [labels[k] for k in predicted_class_indices]
submission = pd.read_csv(os.path.join(DATA_PATH, 'sample_submission.csv'))

submission['class'] = predictions

submission.to_csv("submission.csv", index=False)

submission.head()
!rm -rf {TRAIN_CROP_PATH}

!rm -rf {TEST_CROP_PATH}