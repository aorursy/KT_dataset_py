import sys

IN_COLAB = 'google.colab' in sys.modules
# model-specific constants:



# image w/h :

#     ResNet50 : (224, 224)

#     Xception : (299, 299)

IMAGE_WIDTH, IMAGE_HEIGHT = (299, 299)
# 전체적으로 문제없이 돌아가는지만 확인할 때 사용.

TESTFLIGHT = False



# train을 통해 weight를 만들려 할 때 사용

GENERATE_WEIGHTS = False



BATCH_SIZE = 32

EPOCHS = 40

K_FOLDS = 5

PATIENCE = 6



# 오류 없이 돌아가는지 확인하려 할 때 사용한다:

if TESTFLIGHT:

    EPOCHS = 3

    K_FOLDS = 3



ASSIGNED_FOLD_JOBS = [x for x in range(K_FOLDS)]
import pandas as pd

import os

from pathlib import Path

import shutil



DATA_PATH = "../input"

CROPPED_DATA_PATH = "../cropped"

MODEL_PATH = "../models"



if IN_COLAB:

    DATA_PATH = "./input"

    CROPPED_DATA_PATH = "./cropped"

    MODEL_PATH = "./models"

else:

    # path 목록

    DATA_PATH = "../input"            # input 데이터 경로

    CROPPED_DATA_PATH = "./cropped"  # crop한 이미지를 저장할 경로

    MODEL_PATH = "./"          # training 완료된 weight 파일이 저장될 경로

    

    if os.path.exists(MODEL_PATH) == False:

        Path(MODEL_PATH).mkdir(parents=True, exist_ok=True)

    

    # 직접 사용하지 않는다:

    PRE_MODEL_NAME = "kakr-3rd-xception"

    INPUT_DATA_NAME = "2019-3rd-ml-month-with-kakr"

    

    pre_models_path = os.path.join(DATA_PATH, PRE_MODEL_NAME)

    

    # weight를 생성하려는 목적이 아니면 weight파일을 미리 복사해 둔다.



    if os.path.exists(pre_models_path):

        for fname in os.listdir(pre_models_path):

            filepath = os.path.join(pre_models_path, fname)

            if os.path.isfile(filepath):

                if GENERATE_WEIGHTS == True:

                    if fname.find("h5") > 0:

                        continue

                destfilepath = os.path.join(MODEL_PATH, fname)

                print("copy file ", filepath, " to ", destfilepath)

                shutil.copy(filepath, destfilepath)



    if os.path.exists(os.path.join(DATA_PATH, INPUT_DATA_NAME)):

        DATA_PATH = os.path.join(DATA_PATH, INPUT_DATA_NAME)

        

print("Paths : ")

print("----------------------------------")

print("DATA_PATH         : ", DATA_PATH)

print("CROPPED_DATA_PATH : ", CROPPED_DATA_PATH)

print("MODEL_PATH        : ", MODEL_PATH)
# colab에서 구동을 위한 것이므로 코드 참고 시에는 무시하면 됩니다.

def tmp_copy_weights_to_model_path():

    for fname in os.listdir("./"):



        if fname.find("h5") > -1:

            dstpath = os.path.join(MODEL_PATH, fname)

            shutil.copy(fname, dstpath)



if IN_COLAB == True:

    tmp_copy_weights_to_model_path()
# csv 파일을 읽어서 DataFrame을 생성합니다.

df_train = pd.read_csv(os.path.join(DATA_PATH, "train.csv"))

df_test = pd.read_csv(os.path.join(DATA_PATH, "test.csv"))

df_class = pd.read_csv(os.path.join(DATA_PATH, "class.csv"))



classes = df_class['id'].values.astype('str').tolist()

num_classes_out = len(classes)



if os.path.exists(MODEL_PATH) == False:

    Path(MODEL_PATH).mkdir(parents=True, exist_ok=True)

    

    

import zipfile

z = zipfile.ZipFile(os.path.join(DATA_PATH, "train.zip"))

z.extractall("./train")

z.close()



import zipfile

z = zipfile.ZipFile(os.path.join(DATA_PATH, "test.zip"))

z.extractall("./test")

z.close()
df_train.head()
df_test.head()
df_class.head()
import seaborn as sns

import matplotlib.pyplot as plt



fig, ax = plt.subplots(2, 1, figsize=(24, 5))

def check_count_by_class():

    df_merged = df_train.append(df_test, ignore_index=True, sort=False)

    sns.countplot(x='class', data=df_merged, ax=ax[0])

    ax[1].set_xlim([0, 300])

    df_merged['class'].plot.kde(ax=ax[1])

    print("max count = ", df_merged['class'].value_counts().index[0], ":", df_merged['class'].value_counts()[df_merged['class'].value_counts().index[0]])

    

    a = df_merged['class'].value_counts()

    print("MAX count : ", a.idxmax().astype('int'), a[a.idxmax()])

    print("MIN count : ", a.idxmin().astype('int'), a[a.idxmin()])    



check_count_by_class()

from PIL import Image, ImageDraw

import matplotlib.pyplot as plt

import numpy as np

import os





widths = []

heights = []



for idx, row in df_train.iterrows():    

    im = Image.open(os.path.join("./train", row['img_file']))

    w, h = im.size

    widths.append(w)

    heights.append(h)

    

widths = np.array(widths)

heights = np.array(heights)

print("width  min/max/mean : ", np.min(widths), np.max(widths), np.mean(widths))

print("height min/max/mean : ", np.min(heights), np.max(heights), np.mean(heights))



bins = [x for x in range(-1, 6000, 200)]



import matplotlib.pyplot as plt

import seaborn as sns

fig, ax = plt.subplots(2, 2, figsize=(16, 10))



# draw as hist:

ax[0][0].hist(widths, bins=bins)

ax[0][0].set_title("width hist")

ax[0][1].hist(heights, bins=bins)

ax[0][1].set_title("height hist")



# draw as kde:

sns.kdeplot(np.array(widths), ax=ax[1][0])

ax[1][0].set_title("width kde")

sns.kdeplot(np.array(heights), ax=ax[1][1])

ax[1][1].set_title("height kde")

    

from PIL import Image, ImageDraw

import matplotlib.pyplot as plt

import matplotlib.patches as patches



n = 5



fix, ax = plt.subplots(n, 1, figsize = (10, 40))

axidx = 0



df_sample = df_train.sample(n)

for idx, row in df_sample.iterrows():

    im = Image.open(os.path.join("./train", row['img_file']))

    

    # 이미지 내에 박스를 그린다.

    draw = ImageDraw.Draw(im)

    #draw.rectangle((row['bbox_x1'], row['bbox_y1'], row['bbox_x2'], row['bbox_y2']), outline='blue', width=2)

    

    print("bbox:", row['bbox_x1'], row['bbox_y1'], row['bbox_x2'], row['bbox_y2'])

    width = row['bbox_x2'] - row['bbox_x1']

    height = row['bbox_y2'] - row['bbox_y1']

    rect = patches.Rectangle((row['bbox_x1'], row['bbox_y1']), width, height, linewidth=3, edgecolor='r',facecolor='none')

    ax[axidx].add_patch(rect)

    ax[axidx].imshow(im)

    

    axidx += 1



def checkratio(df):

    df['ratio'] = (df['bbox_x2'] - df['bbox_x1']) / (df['bbox_y2'] - df['bbox_y1'])

    return df



fig, ax = plt.subplots(1, 1, figsize=(12, 8))



df_merged = df_train.append(df_test, ignore_index=True, sort=False)

df_tmp = checkratio(df_merged)

df_tmp['ratio'].plot.kde()
image_width, image_height = IMAGE_WIDTH, IMAGE_HEIGHT



# crop해서 만들 이미지의 W/H 비율을 1:1로 한다.

# 모델의 이미지 입력은 width,height가 동일하므로 1:1만이 가능함.

default_ratio = 1.0



from pathlib import Path

    

def get_fixed_img(filename, area, ratio, output_size):

    debug = False

    im = Image.open(filename)

    cropIm = im.crop(area)

    

    if debug:

        print("crop w/h=", cropIm.width, cropIm.height)

    

    w, h = cropIm.width, cropIm.height

    # w : h = w/h ratio : 1

    fixedW, fixedH = w, h

    if w/h >= ratio:

        fixedH = w / ratio

    else:

        fixedW = h * ratio

    fixedW, fixedH = int(fixedW), int(fixedH)



    if debug:

        print("fixed w/h=", fixedW, fixedH, "ratio=", fixedW/fixedH)

    

    newIm = Image.new("RGB", (fixedW, fixedH))

    newIm.paste(cropIm, ((fixedW - w)//2, (fixedH - h)//2))

    

    #newIm = newIm.resize(output_size, resample=Image.NEAREST)

    return newIm



def img_from_row(row, path, ratio, w, h):

    filepath = os.path.join(path, row['img_file'])

    area = (row['bbox_x1'], row['bbox_y1'], row['bbox_x2'], row['bbox_y2'])

    return get_fixed_img(filepath, area, ratio, (w, h))    



def make_crop_img(df, dirname):

    # 저장될 디렉토리부터 생성

    dirpath_crop = os.path.join(CROPPED_DATA_PATH, dirname)

    if os.path.exists(dirpath_crop) == False:

        Path(dirpath_crop).mkdir(parents=True, exist_ok=True)



    for idx, row in df.iterrows():

        src_path = dirname

        target_img_path = os.path.join(dirpath_crop, row['img_file'])

        

        # 파일이 없거나, 사이즈가 0이면 새로 만든다.

        isvalid = os.path.exists(target_img_path)

        isvalid = (isvalid and (os.path.getsize(target_img_path) > 0))

        if isvalid == False:

            im = img_from_row(row, src_path, default_ratio, image_width, image_height)

            im.save(target_img_path)



def make_cropped_imgs():

    dirpath_crop = os.path.join(CROPPED_DATA_PATH, "train")

    print(dirpath_crop)

    make_crop_img(df_train, "train")

    

    dirpath_crop = os.path.join(CROPPED_DATA_PATH, "test")

    print(dirpath_crop)

    make_crop_img(df_test, "test")



make_cropped_imgs()
# show cropped image

def show_cropped_imgs():

    dirpath_crop = os.path.join(CROPPED_DATA_PATH, "train")    

    n = 4

    fix, ax = plt.subplots(n, 1, figsize = (10, 40))

    axidx = 0

    df_sample = df_train.sample(n)

    for idx, row in df_sample.iterrows():

        im = Image.open(os.path.join(dirpath_crop, row['img_file']))



        # 이미지 내에 박스를 그린다.

        draw = ImageDraw.Draw(im)

        ax[axidx].imshow(im)

        axidx += 1

    plt.show()

    

show_cropped_imgs()
epochs = EPOCHS

batch_size = BATCH_SIZE



def get_total_batch(num_samples, batch_size):    

    if (num_samples % batch_size) > 0 :

        return (num_samples // batch_size) + 1

    else :

        return num_samples // batch_size
from keras import backend as K

def recall(y_target, y_pred):

    # clip(t, clip_value_min, clip_value_max) : clip_value_min~clip_value_max 이외 가장자리를 깎아 낸다

    # round : 반올림한다

    y_target_yn = K.round(K.clip(y_target, 0, 1)) # 실제값을 0(Negative) 또는 1(Positive)로 설정한다

    y_pred_yn = K.round(K.clip(y_pred, 0, 1)) # 예측값을 0(Negative) 또는 1(Positive)로 설정한다



    # True Positive는 실제 값과 예측 값이 모두 1(Positive)인 경우이다

    count_true_positive = K.sum(y_target_yn * y_pred_yn) 



    # (True Positive + False Negative) = 실제 값이 1(Positive) 전체

    count_true_positive_false_negative = K.sum(y_target_yn)



    # Recall =  (True Positive) / (True Positive + False Negative)

    # K.epsilon()는 'divide by zero error' 예방차원에서 작은 수를 더한다

    recall = count_true_positive / (count_true_positive_false_negative + K.epsilon())



    # return a single tensor value

    return recall





def precision(y_target, y_pred):

    # clip(t, clip_value_min, clip_value_max) : clip_value_min~clip_value_max 이외 가장자리를 깎아 낸다

    # round : 반올림한다

    y_pred_yn = K.round(K.clip(y_pred, 0, 1)) # 예측값을 0(Negative) 또는 1(Positive)로 설정한다

    y_target_yn = K.round(K.clip(y_target, 0, 1)) # 실제값을 0(Negative) 또는 1(Positive)로 설정한다



    # True Positive는 실제 값과 예측 값이 모두 1(Positive)인 경우이다

    count_true_positive = K.sum(y_target_yn * y_pred_yn) 



    # (True Positive + False Positive) = 예측 값이 1(Positive) 전체

    count_true_positive_false_positive = K.sum(y_pred_yn)



    # Precision = (True Positive) / (True Positive + False Positive)

    # K.epsilon()는 'divide by zero error' 예방차원에서 작은 수를 더한다

    precision = count_true_positive / (count_true_positive_false_positive + K.epsilon())



    # return a single tensor value

    return precision





def f1score(y_target, y_pred):

    _recall = recall(y_target, y_pred)

    _precision = precision(y_target, y_pred)

    # K.epsilon()는 'divide by zero error' 예방차원에서 작은 수를 더한다

    _f1score = ( 2 * _recall * _precision) / (_recall + _precision+ K.epsilon())

    

    # return a single tensor value

    return _f1score
from sklearn.model_selection import StratifiedKFold, KFold

skfold = StratifiedKFold(n_splits=K_FOLDS, random_state=2019)
#ref: https://github.com/yu4u/cutout-random-erasing/blob/master/cifar10_resnet.py

def get_random_eraser(p=0.5, s_l=0.02, s_h=0.4, r_1=0.3, r_2=1/0.3, v_l=0, v_h=255, pixel_level=False):

    def eraser(input_img):

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



    return eraser
from keras.applications.resnet50 import ResNet50, preprocess_input

from keras.preprocessing.image import ImageDataGenerator



datagen_train = ImageDataGenerator(

    rescale=1./255,

    featurewise_center=False,  # set input mean to 0 over the dataset

    samplewise_center=False,  # set each sample mean to 0

    featurewise_std_normalization=False,  # divide inputs by std of the dataset

    samplewise_std_normalization=False,  # divide each input by its std

    zca_whitening=False,  # apply ZCA whitening

    rotation_range=20,  # randomly rotate images in the range (degrees, 0 to 180)

    zoom_range = 0.3, # Randomly zoom image 

    width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)

    height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)

    horizontal_flip=True,  # randomly flip images

    vertical_flip=False,  # randomly flip images

    preprocessing_function = get_random_eraser(v_l=0, v_h=255),

    )



datagen_val = ImageDataGenerator(

    rescale=1./255,

#     featurewise_center=False,  # set input mean to 0 over the dataset

#     samplewise_center=False,  # set each sample mean to 0

#     featurewise_std_normalization=False,  # divide inputs by std of the dataset

#     samplewise_std_normalization=False,  # divide each input by its std

#     zca_whitening=False,  # apply ZCA whitening

#     rotation_range=15,  # randomly rotate images in the range (degrees, 0 to 180)

#     zoom_range = 0.2, # Randomly zoom image 

#     width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)

#     height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)

#     horizontal_flip=True,  # randomly flip images

#     vertical_flip=False,  # randomly flip images

#     preprocessing_function = preprocess_input

)





# 아래 안해주면 에러남. categorical이어서 기준 col이 숫자값이면 안되는 것인듯.

df_train['class'] = df_train['class'].astype('str')
from keras.applications.resnet50 import ResNet50, preprocess_input

from keras.applications.xception import Xception

from keras import optimizers

from keras.models import Sequential, Model

from keras.layers import Dense, Dropout, Flatten, Activation, Conv2D, GlobalAveragePooling2D



__models = {}



def get_model(base_model, show_summary=False):

    base_model = base_model(weights='imagenet', include_top=False, input_shape = (IMAGE_WIDTH, IMAGE_HEIGHT,3))



    model = Sequential()

    model.add(base_model)

    model.add(GlobalAveragePooling2D())



    # 중간 layer 추가(dropout)

    model.add(Dense(2048, activation='relu', kernel_initializer='he_normal'))

    model.add(Dropout(0.15))



    model.add(Dense(196, activation='softmax', kernel_initializer='lecun_normal'))

    if show_summary:

        model.summary()



    optimizer = optimizers.Nadam(lr=0.0002)

    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['acc', recall, precision, f1score])

    

    return model

    

__models = {"Xception" : Xception}
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, LambdaCallback



def get_callbacks(model_save_filename, patience):

    es = EarlyStopping(monitor='val_f1score', min_delta=0, patience=patience, verbose=1, mode='max')



    rr = ReduceLROnPlateau(monitor = 'val_f1score', factor = 0.5, patience = patience / 2,

                           min_lr=0.000001,

                           verbose=1,

                           mode='max')



    mc = ModelCheckpoint(filepath=model_save_filename, monitor='val_f1score', verbose=1,

                           save_best_only=True, mode='max')



    return [es, rr, mc]

    

    
#aaa = skfold.split(X=df_train['img_file'], y=df_train['class'])

import ssl

from keras.models import model_from_json



ssl._create_default_https_context = ssl._create_unverified_context

history_list = {}

for _m in __models:

    print("Model : ", _m)

        

    # 미리 fold를 나누어 생생해 둔 dataframe 파일을 사용한다.

    for fold_index in ASSIGNED_FOLD_JOBS:



        # 모델 생성

        model = get_model(__models[_m], fold_index == 0)



        # weight를 저장할 파일명을 생성

        model_save_filename = ("%s_%d.h5" % (_m , fold_index))

        model_save_filepath = os.path.join(MODEL_PATH, model_save_filename)



        # 나눠진 dataframe을 load

        df_train_filename = ("fold_%d_train.csv" % fold_index)

        df_val_filename = ("fold_%d_val.csv" % fold_index)



        dataframe_train = pd.read_csv(os.path.join(MODEL_PATH, df_train_filename))

        dataframe_val = pd.read_csv(os.path.join(MODEL_PATH, df_val_filename))



        # 아래 안해주면 에러남. categorical이어서 기준 col이 숫자값이면 안되는 것인듯.

        dataframe_train['class'] = dataframe_train['class'].astype('str')

        dataframe_val['class'] = dataframe_val['class'].astype('str')



        # ImageDataGenerator 생성(train/val)

        datagen_train_flow = datagen_train.flow_from_dataframe(dataframe=dataframe_train,

                                                   directory=os.path.join(CROPPED_DATA_PATH, "train"),

                                                   x_col='img_file',

                                                   y_col="class",

                                                   classes = classes,

                                                   target_size=(IMAGE_WIDTH, IMAGE_HEIGHT),

                                                   color_mode='rgb',

                                                   class_mode='categorical',

                                                   batch_size=batch_size,

                                                   shuffle = True)



        datagen_val_flow = datagen_val.flow_from_dataframe(dataframe=dataframe_val,

                                                   directory=os.path.join(CROPPED_DATA_PATH, "train"),

                                                   x_col='img_file',

                                                   y_col="class",

                                                   classes = classes,

                                                   target_size=(IMAGE_WIDTH, IMAGE_HEIGHT),

                                                   color_mode='rgb',

                                                   class_mode='categorical',

                                                   batch_size=batch_size,

                                                   shuffle = True)

        

        if GENERATE_WEIGHTS == True:

            if os.path.exists(model_save_filepath) == True:

                os.remove(model_save_filepath)



        # 동일 이름의 weight 파일이 있으면 넘어간다.

        if os.path.exists(model_save_filepath) == True:

            print(">>>", model_save_filepath, " already trained... skip!")

            continue

        

        train_steps = get_total_batch(dataframe_train.shape[0], batch_size)

        val_steps = get_total_batch(dataframe_val.shape[0], batch_size)

        

        if TESTFLIGHT:

            train_steps = 10

            val_steps = 10

            

        history = model.fit_generator(datagen_train_flow,

            epochs=epochs,

            steps_per_epoch = train_steps,

            validation_data = datagen_val_flow,

            validation_steps = val_steps,

            callbacks = get_callbacks(model_save_filepath, PATIENCE),

            verbose=0)

        

        history_list[model_save_filename] = history

        

        model = None

        

    
if GENERATE_WEIGHTS == True:

    fig, ax = plt.subplots(1, 1, figsize=(12,8 * len(__models)))

    from cycler import cycler



    # set color cycle : 그래프 색깔을 알아서 cycling해준다.

    x = np.linspace(0, 1, 10)

    number = 5

    cmap = plt.get_cmap('gnuplot')

    colors = [cmap(i) for i in np.linspace(0, 1, number)]

    ax.set_prop_cycle(cycler('color', colors))



    for hname in history_list:

        history = history_list[hname]

        plot_label = "val_score : " + hname

        ax.plot(history.history['val_f1score'], label=plot_label)        

    ax.legend()        

    plt.show()
datagen_submit = ImageDataGenerator(

    rescale=1./255,

)



def load_sub_models():

    sub_models = []

    for _m in __models:

        print("Model ", _m, " : ")

        for _, _, filenames in os.walk(MODEL_PATH):

            for fname in filenames:

                if fname.find(_m) >= 0 and fname.find(".h5") >= 0:        

                    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> loading ", fname)

                    model = get_model(__models[_m])



                    # model에 데이터파일을 올린다.

                    # model 명이 있는 것들 중에서만 올린다.



                    fpath = os.path.join(MODEL_PATH, fname)

                    print("model weight fpath:", fpath)

                    model.load_weights(fpath)



                    sub_models.append(model)

    return sub_models



submodels = load_sub_models()
from numpy import dstack



def make_meta_learner_dataset(submodels, df, imgdirname, tta = False):

    datagen_submit = ImageDataGenerator(rescale=1./255)

    stackX = None

    for model in submodels:



		# make prediction :

        datagen_metalearner_flow = datagen_submit.flow_from_dataframe(

            dataframe=df,

            directory=os.path.join(CROPPED_DATA_PATH, imgdirname),

            x_col='img_file',

            y_col=None,

            target_size= (IMAGE_WIDTH, IMAGE_HEIGHT),

            color_mode='rgb',

            class_mode=None,

            batch_size=batch_size,

            shuffle=False)

        

        datagen_metalearner_augmented_flow = datagen_train.flow_from_dataframe(

            dataframe=df,

            directory=os.path.join(CROPPED_DATA_PATH, imgdirname),

            x_col='img_file',

            y_col=None,

            target_size=(IMAGE_WIDTH, IMAGE_HEIGHT),

            color_mode='rgb',

            class_mode=None,

            batch_size=batch_size,

            shuffle=False)

        

        datagen_flow = datagen_metalearner_flow

        if tta == True:

            print("select tta flow")

            datagen_flow = datagen_metalearner_augmented_flow



        datagen_flow.reset()

        pred = model.predict_generator(generator = datagen_flow,

                                       steps = get_total_batch(df.shape[0], batch_size),

                                       verbose=1)

        

        if stackX is None:

            stackX = pred

        else:

            stackX = dstack((stackX, pred))

   

    print("stackX.shape = ", stackX.shape)

        

    # flatten predictions to [rows, members x probabilities]

    stackX = stackX.reshape((stackX.shape[0], stackX.shape[1]*stackX.shape[2]))

    return stackX
import keras

from keras import layers, models
def make_meta_learner_model(input_shape, output_class_count, dropout):

    print(input_shape)

    print(output_class_count)

    print(dropout)

    print(input_shape[1] * 2)

    

    model = models.Sequential()



    model.add(layers.Dense(units=input_shape[1] * 2, activation='relu', kernel_initializer='he_normal'))

    model.add(layers.Dropout(dropout))

    

    model.add(layers.Dense(units=int(input_shape[1] / 2), activation='relu', kernel_initializer='he_normal'))

    model.add(layers.Dropout(dropout))



    model.add(layers.Dense(units=int(input_shape[1] / 4), activation='relu', kernel_initializer='he_normal'))

    model.add(layers.Dropout(dropout))

    

    model.add(layers.Dense(units=output_class_count, activation='softmax', kernel_initializer='he_normal'))

    

    model.compile(loss="categorical_crossentropy", optimizer='adam', metrics=['acc', recall, precision, f1score])

    

    return model





def get_callbacks_for_meta_learner(model_save_filename, patience):

    es = EarlyStopping(monitor='f1score', min_delta=0, patience=patience, verbose=1, mode='max')



    rr = ReduceLROnPlateau(monitor = 'f1score', factor = 0.5, patience = patience / 2,

                           min_lr=0.000001,

                           verbose=1, 

                           mode='max')



    mc = ModelCheckpoint(filepath=model_save_filename, monitor='f1score', verbose=1,

                           save_best_only=True, mode='max')



    return [es, rr, mc]
from keras.utils import to_categorical



print("Build dataset for meta-learner...")

meta_train_X = None



META_LEARNER_DATA_EPOCHS = 1

for epoch in range(META_LEARNER_DATA_EPOCHS):

    print("Epoch ", epoch)

    epochs_x = make_meta_learner_dataset(submodels, df_train, "train")

    if meta_train_X is None:

        meta_train_X = epochs_x

    else:

        meta_train_X = vstack([meta_train_X, epochs_x])

print("meta_train_X.shape=", meta_train_X.shape)

from numpy import vstack



# 모델이 훈련된 label값에 맞게 Y값을 만들어야 한다.

meta_train_Y = None

labels = (datagen_train_flow.class_indices)

y = df_train['class'].values

y = [labels[x] for x in y]

y = to_categorical(y)

for i in range(META_LEARNER_DATA_EPOCHS):

    if meta_train_Y is None:

        meta_train_Y = y

    else:

        meta_train_Y = vstack([meta_train_Y, y])

print("meta_train_Y.shape=", meta_train_Y.shape)
def train_meta_learner(X, Y):

    

    meta_learner_model_save_filename = ("meta_learner_model.h5")

    meta_learner_model_save_filepath = os.path.join(MODEL_PATH, meta_learner_model_save_filename)

    

    print("Training meta-learner model...")

    meta_learner_model = make_meta_learner_model(X.shape, len(classes), dropout=0.3)

    meta_learner_model.fit(X, Y, epochs=20, verbose=1, batch_size=128, callbacks=get_callbacks_for_meta_learner(meta_learner_model_save_filepath, 6))

    

    meta_learner_model.load_weights(meta_learner_model_save_filepath)

    

    return meta_learner_model



meta_learner_model = train_meta_learner(meta_train_X, meta_train_Y)



tta_len = 4



preds = []

for i in range(tta_len):

    meta_submit_X = make_meta_learner_dataset(submodels, df_test, "test", tta=True)

    pred = meta_learner_model.predict(meta_submit_X, batch_size=32)

    preds.append(pred)

    

pred = np.mean(np.array(preds), axis=0)



submit_Y = np.argmax(pred, axis=1)

labels = (datagen_train_flow.class_indices)

labels = dict((v,k) for k, v in labels.items())



predictions = [labels[k] for k in submit_Y]



submission = pd.DataFrame()

submission['img_file'] = df_test['img_file']

submission["class"] = predictions

submission.to_csv("submission.csv", index=False)