import gc

import os 

import warnings

warnings.filterwarnings(action='ignore')

import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

from tqdm import tqdm # 진행 상태 표시as

from keras import backend as K

K.image_data_format() # 채널 first 인지 last인지 여부 판단

# Image visualization



import PIL

from PIL import ImageDraw
path = '../input/2019-3rd-ml-month-with-kakr'

os.listdir(path)

# 이미지 폴더 경로 

train_img_path = os.path.join(path,'train')

test_img_path = os.path.join(path,'test')

# csv 파일 경로

df_train = pd.read_csv(os.path.join(path,'train.csv'))

df_test = pd.read_csv(os.path.join(path,'test.csv'))

df_class = pd.read_csv(os.path.join(path,'class.csv'))
#crop



def crop_boxing_img(img_name, margin=-4, size=(224,224)):

    if img_name.split('_')[0] == 'train':

        PATH = train_img_path

        data = df_train

    else:

        PATH = test_img_path

        data = df_test



    img = PIL.Image.open(os.path.join(PATH, img_name))

    pos = data.loc[data["img_file"] == img_name, ['bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2']].values.reshape(-1)



    width, height = img.size

    x1 = max(0, pos[0] - margin)

    y1 = max(0, pos[1] - margin)

    x2 = min(pos[2] + margin, width)

    y2 = min(pos[3] + margin, height)



    return img.crop((x1, y1, x2, y2)).resize(size)
TRAIN_CROPPED_PATH = '../cropped_train'

TEST_CROPPED_PATH = '../cropped_test'
if (os.path.isdir(TRAIN_CROPPED_PATH) == False):

    os.mkdir(TRAIN_CROPPED_PATH)



if (os.path.isdir(TEST_CROPPED_PATH) == False):

    os.mkdir(TEST_CROPPED_PATH)



for i, row in df_train.iterrows():

    cropped = crop_boxing_img(row['img_file'])

    cropped.save(os.path.join(TRAIN_CROPPED_PATH, row['img_file']))



for i, row in df_test.iterrows():

    cropped = crop_boxing_img(row['img_file'])

    cropped.save(os.path.join(TEST_CROPPED_PATH, row['img_file']))
# Set Path of Preprocessed Train Images

TRAIN_IMG_PREP_PATH = os.path.join('..', 'train_prep')

if not os.path.exists(TRAIN_IMG_PREP_PATH):

    os.makedirs(TRAIN_IMG_PREP_PATH, exist_ok=True)



# Set Path of Preprocessed Test Images

TEST_IMG_PREP_PATH = os.path.join('..', 'test_prep')

if not os.path.exists(TEST_IMG_PREP_PATH):

    os.makedirs(TEST_IMG_PREP_PATH, exist_ok=True)
def img_he_pad(img_file_name, add_padding=True):

    if img_file_name.split('_')[0] == 'train':

        IMG_CROP_PATH = TRAIN_CROPPED_PATH

        data = df_train

    elif img_file_name.split('_')[0] == 'test':

        IMG_CROP_PATH = TEST_CROPPED_PATH

        data = df_test

        

    # --- Histogram Equalization --- #

    img = cv2.imread(os.path.join(IMG_CROP_PATH, img_file_name))

    img_y_cr_cb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)

    y, cr, cb = cv2.split(img_y_cr_cb)



    # Equalize y (CLAHE (Contrast Limited Adaptive Histogram Equalization))

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(6,6))

    y_eq = clahe.apply(y)

    img_y_cr_cb_eq = cv2.merge((y_eq, cr, cb))

    img_bgr_eq = cv2.cvtColor(img_y_cr_cb_eq, cv2.COLOR_YCR_CB2BGR)

    img_prep = img_bgr_eq



    # --- Convert BGR To RGB (Just On cv2) ---

    # b, g, r = cv2.split(img_bgr_eq)

    # img_prep = cv2.merge((r,g,b))

    

    # -------- Add Padding --------- #

    if add_padding:

        img_prep_h, img_prep_w = img_prep.shape[0], img_prep.shape[1]  # (height, width)

        ratio = float(IMG_SIZE) / max(img_prep_h, img_prep_w)

        shape_no_padding = (int(img_prep_h * ratio), int(img_prep_w * ratio))



        img_prep_no_padding = cv2.resize(img_prep, shape_no_padding[::-1])

        

        size_h = IMG_SIZE - shape_no_padding[0]

        size_w = IMG_SIZE - shape_no_padding[1]

        

        top, bottom = size_h // 2, size_h - (size_h // 2)

        left, right = size_w // 2, size_w - (size_w // 2)



        PADDING_COLOR = (0, 0, 0)  # black

        img_prep = cv2.copyMakeBorder(

            img_prep_no_padding,

            top,

            bottom,

            left,

            right,

            cv2.BORDER_CONSTANT,

            value=PADDING_COLOR

        )

    

    return img_prep
import cv2

IMG_SIZE = 224

# Save Preprocessed Train Images (Path: ../train_prep)

if not os.listdir(TRAIN_IMG_PREP_PATH):  # If PATH_IMG_TRAIN_PREP is empty

    for idx, row in df_train.iterrows():

        img_file_name = row['img_file']

        img_prep = img_he_pad(img_file_name, add_padding=True)

        cv2.imwrite(os.path.join(TRAIN_IMG_PREP_PATH, img_file_name), img_prep)



# Save Preprocessed Test Images (Path: ../test_prep)

if not os.listdir(TEST_IMG_PREP_PATH):

    for idx, row in df_test.iterrows():

        img_file_name = row['img_file']

        img_prep = img_he_pad(img_file_name, add_padding=True)

        cv2.imwrite(os.path.join(TEST_IMG_PREP_PATH, img_file_name), img_prep)
# Define Function For Test

def test_he_padding(img_file_name):

    # Show Cropped Image

    img_crop = PIL.Image.open(os.path.join(TRAIN_CROPPED_PATH, img_file_name))

    plt.figure(figsize=(12, 9))

    plt.subplot(1, 2, 1)

    plt.title(f'Cropped Image - {img_file_name}')

    plt.imshow(img_crop)

    plt.axis('off')



    # Show Preprocessed Image

    img_he_pad = PIL.Image.open(os.path.join(TRAIN_IMG_PREP_PATH, img_file_name))

    plt.subplot(1, 2, 2)

    plt.title(f'Historgram Equalized Cropped Image(Add Padding) - {img_file_name}')

    plt.imshow(img_he_pad)

    plt.axis('off')

    

    # Show Result

    plt.show()

    

# Test Histogram Equalization & Add Padding

test_he_padding(img_file_name=df_train['img_file'].iloc[114])
from sklearn.metrics import f1_score



def f1_metric(y_true, y_pred):



    def recall(y_true, y_pred):

        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))

        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))

        recall = (true_positives + K.epsilon()) / (possible_positives + K.epsilon())

        return recall



    def precision(y_true, y_pred):

        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))

        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))

        precision = (true_positives + K.epsilon()) / (predicted_positives + K.epsilon())

        return precision



    precision = precision(y_true, y_pred)

    recall = recall(y_true, y_pred)

    return 2*((precision*recall)/(precision+recall+K.epsilon()))
from keras_applications.resnext import ResNeXt101#, preprocess_input

# from keras.applications.resnet_v2. import ResNet50, preprocess_input

# from keras_applications.resnet import ResNet101, preprocess_input

from keras.preprocessing.image import ImageDataGenerator



# Parameter

img_size = (224, 224)

epochs = 70

batch_size =16



# Define Generator config

train_datagen = ImageDataGenerator(

    rotation_range=30,

    horizontal_flip = True, 

    vertical_flip = False,

    #zoom_range=0.30,

    #width_shift_range=0.2,

    #height_shift_range=0.2,

    #shear_range=0.5,

    brightness_range=[0.5, 1.5],

    fill_mode='nearest',

    rescale=1./255)

    #preprocessing_function=preprocess_input)



val_datagen = ImageDataGenerator(rescale=1./255)#preprocessing_function=preprocess_input)

test_datagen = ImageDataGenerator(rescale=1./255)#preprocessing_function=preprocess_input)
from keras.models import Sequential, Model

from keras.layers import Dense, Dropout, Flatten, Activation, Conv2D, GlobalAveragePooling2D

from keras import layers, models, optimizers, utils, backend,regularizers

# import keras

def get_model(model_name='ResNeXt101'):

    resNet_model = ResNeXt101(include_top= False, input_shape = (224,224,3)

                            , backend =backend, layers=layers, models = models,

                             utils = utils

                            )

    # resNet_model.summary()

    

    model = Sequential()

    model.add(resNet_model)

    model.add(GlobalAveragePooling2D())

    model.add(layers.Dropout(0.25))  # 과적합 줄여보기

    model.add(Dense(196, activation='softmax', kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(l=0.01)))

#     model.add(Dense(196, activation='softmax', kernel_initializer='he_normal'))

#     model.add(Dense(196, kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(l=0.01)))

#     model.add(LeakyReLU(alpha=0.01))    

    model.summary()

    adam = optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-8)

    model.compile(optimizer = adam,loss = 'categorical_crossentropy', metrics=['acc',f1_metric])

    # compile 할때 넣어줘야지 아래에서 early stopping 할때 사용 가능하다

    return model



def get_steps(num_samples,batch_size):

    if (num_samples % batch_size)>0:

        return (num_samples // batch_size) + 1

    else :

        return num_samples // batch_size
from sklearn.model_selection import StratifiedKFold

k_folds = 5

kfold =StratifiedKFold(n_splits = k_folds, random_state = 1990)

df_train['class'] = df_train['class'].astype('str')

df_train= df_train[['img_file','class']]

df_test = df_test[['img_file']]
%%time

file=[]

from keras.callbacks import ModelCheckpoint, EarlyStopping,ReduceLROnPlateau

for idx,(train_index, valid_index) in enumerate(kfold.split(

                                df_train['img_file'],df_train['class'])):

    if idx != 0 : continue # 여러번 나눠서 돌리기 위함

   # if idx == 1 : continue

    #if idx == 2 : continue

    #if idx == 3 : continue    

    traindf = df_train.iloc[train_index,:].reset_index()

#     validdf = df_train.iloc[valid_index,:].reset_index()

#     traindf.to_csv('%s_traindf'%idx,index=False)

#     validdf.to_csv('%s_validdf'%idx,index=False)

    

    nb_train_samples = len(traindf)

#     nb_validation_samples = len(validdf)

#     nb_test_samples = len(df_test)

    # Make Generator

    train_generator = train_datagen.flow_from_dataframe(

        dataframe=traindf, 

        directory=TRAIN_CROPPED_PATH,#'../input/train/',

        x_col = 'img_file',

        y_col = 'class',

        target_size = img_size,

        color_mode='rgb',

        class_mode='categorical',

        batch_size=batch_size,

        seed=42

    )









test_generator = test_datagen.flow_from_dataframe(

            dataframe=df_test,

            directory=TEST_IMG_PREP_PATH,#TEST_CROPPED_PATH,#'../input/test',

            x_col='img_file',

            y_col=None,

            target_size= img_size,

            color_mode='rgb',

            class_mode=None,

            batch_size=batch_size,

            shuffle=False

            )
path = '../input/models10/'

lst = os.listdir(path)

print(lst)


%%time

tta_steps = 5 #models5 기준 1회 0.950 ,5회 0.949 , 10회 0.9507

prediction = []

for i, name in enumerate(lst):

    preds =[]

    print(name)

    model = get_model()

    model.load_weights(os.path.join(path,name))

    for j in tqdm(range(tta_steps)):                

        test_generator.reset()

        nb_test_samples = len(df_test)

        pred = model.predict_generator(

            generator = test_generator,

            steps = get_steps(nb_test_samples, batch_size),

            verbose=1

            )

        preds.append(pred)

        

        gc.collect()

#         print(np.mean(preds,axis=0))    

    pd.DataFrame(np.mean(preds,axis=0)).to_csv('%s.csv'%i, index= False)

#     prediction.append(np.mean(preds,axis=0)) 

    del preds

    gc.collect()

for i, name in enumerate(lst):

    prediction.append(np.array(pd.read_csv('%s.csv'%i)))

# print(prediction)

y_pred = np.mean(prediction,axis=0)

print(y_pred)
path = '../input/2019-3rd-ml-month-with-kakr'

preds_class_indices = np.argmax(y_pred,axis =1)

labels = (train_generator.class_indices)

labels = dict((v,k) for k,v in labels.items())

final_pred = [labels[k] for k in preds_class_indices]



submission = pd.read_csv(os.path.join(path, 'sample_submission.csv'))

submission["class"] = final_pred

submission.to_csv("submission_rev03.csv", index=False)

submission.head()

              