import os

import sys

sys.path.append(os.path.abspath('../input/efficientnet/efficientnet-master/efficientnet-master/'))



import pandas as pd

import os.path



# image directory path

#../input/aptos-train-dataset/aptos-train-images/aptos-train-images/



#original data : ../input/aptos2019-blindness-detection

TRAIN_DATA_PATH = "../input/aptos2019-blindness-detection"

TEST_DATA_PATH = "../input/aptos2019-blindness-detection"

PREPROCESSED_IMAGE_PATH = "./preprocessed"

MODEL_PATH = "./models"

MODEL_FILE_NAME = "effnet_b3_single.h5"



TRAIN_CSV_FILE_PATH = os.path.join(TRAIN_DATA_PATH, "train.csv")

TRAIN_IMAGE_FILE_PATH = "../input/aptos2019-blindness-detection/train_images"



TEST_CSV_FILE_PATH = "../input/aptos2019-blindness-detection/test.csv"

TEST_IMAGE_FILE_PATH = "../input/aptos2019-blindness-detection/test_images"



"""

전체 커널이 제대로 돌아기는지 확인할 때 사용한다.

submission이 정상적으로 진행되는지까지 학인

"""

CHECK_KERNEL_VALID = False



IMG_WIDTH = 456

IMG_HEIGHT = 456

IMG_CHANNELS = 3



BATCH_SIZE = 4



NUM_FOLDS = 6



EPOCHS = 30

if CHECK_KERNEL_VALID:

    EPOCHS = 1



GENERATE_WEIGHTS = True # weight를 새로 생성한다.



TRAIN_OVER_PRETRAINED = True # 기존의 weight에 추가 train한다.



RESCALE_DN = 128.0
from pathlib import Path

import shutil



if os.path.exists(MODEL_PATH) == False:

    Path(MODEL_PATH).mkdir(parents=True, exist_ok=True)

    



# weight를 생성하려는 목적이 아니면 weight파일을 미리 복사해 둔다.



pre_models_path = "../input/aptos-data-files"



pre_model_filepath = os.path.join(pre_models_path, "effnet_b3_single.h5")



if os.path.exists(pre_models_path):

    for fname in os.listdir(pre_models_path):

        filepath = os.path.join(pre_models_path, fname)        

        if os.path.isfile(filepath):

            if GENERATE_WEIGHTS == True and TRAIN_OVER_PRETRAINED == False:

                if fname.find("h5") > 0:

                    continue

            destfilepath = os.path.join(MODEL_PATH, fname)

            print("Copy File ", filepath, " >>> ", destfilepath)

            shutil.copy(filepath, destfilepath)
df_train = pd.read_csv(TRAIN_CSV_FILE_PATH)

df_train['id_code'] = df_train['id_code'] + ".png"

df_test = pd.read_csv(TEST_CSV_FILE_PATH)

df_train.head()
df_test.head()
import cv2

#from PIL import Image, ImageDraw

import matplotlib.pyplot as plt

import matplotlib.patches as patches



n = 3



fix, ax = plt.subplots(n, n, figsize = (16, 16))

axidx = 0



df_sample = df_train.sample(n * n)

for idx, row in df_sample.iterrows():

    imgpath = os.path.join(TRAIN_IMAGE_FILE_PATH, row['id_code'])

    

    im = cv2.imread(imgpath)

    # Note : In the case of color images, the decoded images will have the channels stored in B G R order.

    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB) # RGB로 바꿔주지 않으면 이미지가 파랗게 나온다.



    ax[int(axidx / n)][axidx % n].imshow(im)

    axidx += 1
import numpy as np



def crop_image_from_gray(img, tol=7):

    """

    Applies masks to the orignal image and 

    returns the a preprocessed image with 

    3 channels

    

    (img > tolerance) 로 mask를 생성하고 np.any()를 사용해서

    merge되는 값들 중 valid한 값이 있는 줄은 np.ix_()로 골라냄

        -> False인 줄은 모두 제거됨

    """

    # If for some reason we only have two channels

    if img.ndim == 2:

        mask = img > tol

        return img[np.ix_(mask.any(1),mask.any(0))]

    # If we have a normal RGB images

    elif img.ndim == 3:

        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        mask = gray_img > tol

        

        check_shape = img[:,:,0][np.ix_(mask.any(1),mask.any(0))].shape[0]

        if (check_shape == 0): # image is too dark so that we crop out everything,

            return img # return original image

        else:

            img1=img[:,:,0][np.ix_(mask.any(1),mask.any(0))]

            img2=img[:,:,1][np.ix_(mask.any(1),mask.any(0))]

            img3=img[:,:,2][np.ix_(mask.any(1),mask.any(0))]

            img = np.stack([img1,img2,img3],axis=-1)

        return img



def preprocess_image(image, sigmaX=10):

    """

    The whole preprocessing pipeline:

    1. Read in image

    2. Apply masks

    3. Resize image to desired size

    4. Add Gaussian noise to increase Robustness

    """

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) #RGB colorspace로 g변경

    image = crop_image_from_gray(image)

    image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))

    # cv2.addWeighted(src1, alpha, src2, beta, gamma[, dst[, dtype]]) 

    #image = cv2.addWeighted (image, 4, cv2.GaussianBlur(image, (0,0) ,sigmaX), -4, 128)

    

    '''

    cv2.addWeighted() : Calculates the weighted sum of two arrays.

    원본 이미지에 블러처리된 원본 이미지에 음수 가중치를 준 것을 더해서 윤곽을 강조

    (gamma로 contrast를 줌)

    '''

    image = cv2.addWeighted (image, 4, cv2.GaussianBlur(image, (0,0) ,sigmaX), -4, gamma=128)

    return image
fix, ax = plt.subplots(n, n, figsize = (20, 20))



axidx = 0    

for idx, row in df_sample.iterrows():

    filename = row['id_code']

    imgpath = os.path.join(TRAIN_IMAGE_FILE_PATH, filename)

    im = preprocess_image(cv2.imread(imgpath))

    ax[int(axidx / (n))][axidx % n].imshow(im)

    ax[int(axidx / (n))][axidx % n].set_title(row['id_code'])

    axidx += 1

    
def get_preds_and_labels(model, generator):

    """

    Get predictions and labels from the generator

    """

    preds = []

    labels = []

    for _ in range(int(np.ceil(generator.samples / BATCH_SIZE))):

        x, y = next(generator)

        preds.append(model.predict(x))

        labels.append(y)

    # Flatten list of numpy arrays

    return np.concatenate(preds).ravel(), np.concatenate(labels).ravel()



from sklearn.metrics import cohen_kappa_score



from keras.callbacks import Callback



class Metrics(Callback):

    """

    A custom Keras callback for saving the best model

    according to the Quadratic Weighted Kappa (QWK) metric

    """

    def __init__(self, model, val_generator, model_save_filepath):

        self.model = model

        self.val_generator = val_generator

        self.model_save_filepath = model_save_filepath

        

    def on_train_begin(self, logs={}):

        """

        Initialize list of QWK scores on validation data

        """

        self.val_kappas = []



    def on_epoch_end(self, epoch, logs={}):

        """

        Gets QWK score on the validation data

        """

        # Get predictions and convert to integers

        y_pred, labels = get_preds_and_labels(self.model, self.val_generator)

        y_pred = np.rint(y_pred).astype(np.uint8).clip(0, 4)

        # We can use sklearns implementation of QWK straight out of the box

        # as long as we specify weights as 'quadratic'

        _val_kappa = cohen_kappa_score(labels, y_pred, weights='quadratic') # QWK 방법을 사용

        self.val_kappas.append(_val_kappa)

        print(f"val_kappa: {round(_val_kappa, 4)}")

        if _val_kappa == max(self.val_kappas):

            print("Validation Kappa has improved. Saving model.")

            self.model.save(self.model_save_filepath)

        return
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint



def get_callbacks(model, val_generator, model_save_filepath):

    # Monitor MSE to avoid overfitting and save best model

    es = EarlyStopping(monitor='val_loss', mode='auto', verbose=1, patience=12)

    # factor : 변경 시 multiplier

    lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4,

                           verbose=1, mode='auto', min_delta=0.0001)

    km = Metrics(model, val_generator, model_save_filepath)

    return [es, lr, km]
def get_total_batch(num_samples, batch_size):    

    if (num_samples % batch_size) > 0 :

        return (num_samples // batch_size) + 1

    else :

        return num_samples // batch_size
from keras import backend as K

from keras.activations import elu

from keras.optimizers import Adam

from keras.models import Sequential

from keras.layers import Dense, Conv2D, Flatten, GlobalAveragePooling2D, Dropout



from efficientnet import EfficientNetB3





def build_model_effnet_b3(load_weights = True):

    """

    A custom implementation of EfficientNetB5

    for the APTOS 2019 competition

    (Regression)

    """

    

    # Load in EfficientNetB5

    effnet_b3 = EfficientNetB3(weights=None,

                        include_top=False,

                        input_shape=(IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS))

    if load_weights == True:

        effnet_b3.load_weights('../input/efficientnet-keras-weights-b0b5/efficientnet-b3_imagenet_1000_notop.h5')

    

    model = Sequential()

    model.add(effnet_b3)

    model.add(GlobalAveragePooling2D()) # 각각의 채널을 평균해서 변환함. 바로 FC로 변경된다.

    model.add(Dropout(0.5))

    model.add(Dense(5, activation=elu))

    model.add(Dense(1, activation="linear")) # 0~4까지 단계적 증감값이 출력이므로 activation은 선형으로 한다.

    model.compile(loss='mse',

                  optimizer=Adam(lr=0.00005), 

                  metrics=['mse', 'acc'])

    print(model.summary())

    return model





def get_model(load_weight):

    return build_model_effnet_b3(load_weight)

    

def get_pretrained_model(model_filepath):

    #model_filepath = os.path.join(MODEL_PATH, MODEL_FILE_NAME)

    model = get_model(load_weight = False)

    model.load_weights(model_filepath)

    return model

    
from keras.backend import clear_session

import gc



# Reset Keras Session

def clear_memory():

    clear_session()

    for i in range(20):

        gc.collect()  
import os

import gc

import psutil 



from keras import backend as K

from keras.preprocessing.image import ImageDataGenerator





def train_single():    

    model_save_filepath = os.path.join(MODEL_PATH, MODEL_FILE_NAME)

    

    model = None

    if TRAIN_OVER_PRETRAINED:

        print("train with existing weight :", pre_model_filepath)

        model = get_pretrained_model(pre_model_filepath) # 이전에 훈련한 weight로 초기화된 모델을 사용한다.

    else:

        model = get_model() # imageNet weight로 초기화된 모델을 사용한다.

    

    # Add Image augmentation to our generator

    train_datagen = ImageDataGenerator(rescale = 1./RESCALE_DN,

                                       preprocessing_function=preprocess_image, 

                                       rotation_range=360,

                                       horizontal_flip=True,

                                       validation_split=0.15,

                                       vertical_flip=True)

    



    # Use the dataframe to define train and validation generators

    train_generator = train_datagen.flow_from_dataframe(df_train,

                                                        x_col='id_code', 

                                                        y_col='diagnosis',

                                                        directory = TRAIN_IMAGE_FILE_PATH,

                                                        target_size=(IMG_WIDTH, IMG_HEIGHT),

                                                        batch_size=BATCH_SIZE,

                                                        class_mode='other',

                                                        subset='training')



    val_generator = train_datagen.flow_from_dataframe(df_train,

                                                      x_col='id_code',

                                                      y_col='diagnosis',

                                                      directory = TRAIN_IMAGE_FILE_PATH,

                                                      target_size=(IMG_WIDTH, IMG_HEIGHT),

                                                      batch_size=BATCH_SIZE,

                                                      class_mode='other',

                                                      subset='validation')

    

    if TRAIN_OVER_PRETRAINED == False:

        if GENERATE_WEIGHTS == True:

            if os.path.exists(model_save_filepath) == True:

                os.remove(model_save_filepath)



        # skip if weight file exists and not use pre-trained

        if os.path.exists(model_save_filepath) == True:

            print(">>>>>>>>>>", model_save_filepath, " already trained... skip!")

            return



    train_steps = get_total_batch(train_generator.samples, BATCH_SIZE)

    val_steps = get_total_batch(val_generator.samples, BATCH_SIZE)

    print("Steps : train=", train_steps, " validation=", val_steps)



    # make callbacks

    callbacks = get_callbacks(model=model, val_generator=val_generator, model_save_filepath=model_save_filepath)



    # First training phase (train top layer)

    model.fit_generator(train_generator,

                        steps_per_epoch = train_steps,

                        epochs = EPOCHS,

                        validation_data = val_generator,

                        validation_steps = val_steps,

                        callbacks = callbacks)

    

def train_models():

    clear_memory()



    train_single()



    # clear used memory

    clear_memory()

            





train_models()
import numpy as np

import scipy as sp

from functools import partial



class OptimizedRounder(object):

    """

    An optimizer for rounding thresholds

    to maximize Quadratic Weighted Kappa score

    

    regression 값으로 나온 실수값을 정수로 변경.

    label과 비교하여  기준값(coef)이 fit 된다.

    """

    def __init__(self):

        self.coef_ = 0



    def _kappa_loss(self, coef, X, y):

        """

        Get loss according to using current coefficients

        

        현재 coef로 rounding한 값을 label과의 cohen kappa score 계산한다.

        loss함수로 사용할 것이므로 음수로 리턴해준다.

        (값이 좋아질수록 음수가 커진다->loss가 줄어든다.)

        """

        X_p = np.copy(X)

        for i, pred in enumerate(X_p):

            if pred < coef[0]:

                X_p[i] = 0

            elif pred >= coef[0] and pred < coef[1]:

                X_p[i] = 1

            elif pred >= coef[1] and pred < coef[2]:

                X_p[i] = 2

            elif pred >= coef[2] and pred < coef[3]:

                X_p[i] = 3

            else:

                X_p[i] = 4



        ll = cohen_kappa_score(y, X_p, weights='quadratic')

        return -ll



    def fit(self, X, y):

        """

        Optimize rounding thresholds

        """

        loss_partial = partial(self._kappa_loss, X=X, y=y)

        initial_coef = [0.5, 1.5, 2.5, 3.5]

        # scipy를 이용 initial_coef값을 최적화한다.

        self.coef_ = sp.optimize.minimize(loss_partial, initial_coef, method='nelder-mead')



    def predict(self, X, coef):

        """

        Make predictions with specified thresholds

        """

        X_p = np.copy(X)

        for i, pred in enumerate(X_p):

            if pred < coef[0]:

                X_p[i] = 0

            elif pred >= coef[0] and pred < coef[1]:

                X_p[i] = 1

            elif pred >= coef[1] and pred < coef[2]:

                X_p[i] = 2

            elif pred >= coef[2] and pred < coef[3]:

                X_p[i] = 3

            else:

                X_p[i] = 4

        return X_p



    def coefficients(self):

        return self.coef_['x']
clear_memory()



"""

train 데이터로 최적화된 OptimizedRounder를 얻는다.

"""

df_train = pd.read_csv(TRAIN_CSV_FILE_PATH)

df_train['id_code'] = df_train['id_code'] + ".png"



val_datagen = ImageDataGenerator(rescale = 1./RESCALE_DN,

                                 preprocessing_function=preprocess_image)



val_generator = val_datagen.flow_from_dataframe(df_train,

                                                x_col='id_code',

                                                y_col='diagnosis',

                                                directory = TRAIN_IMAGE_FILE_PATH,

                                                target_size=(IMG_WIDTH, IMG_HEIGHT),

                                                batch_size=BATCH_SIZE,

                                                class_mode='other')





model_save_filepath = os.path.join(MODEL_PATH, MODEL_FILE_NAME)    



model = get_pretrained_model(model_save_filepath)        



y_val_preds, val_labels = get_preds_and_labels(model, val_generator)

optR = OptimizedRounder()

optR.fit(y_val_preds, val_labels)

coefficients = optR.coefficients()





clear_memory()





def make_submission():



    # Place holder for diagnosis column

    test_df = pd.read_csv(TEST_CSV_FILE_PATH)

    test_df['id_code'] = test_df['id_code'] + ".png" # 확장자 명이 없으므로 추가해야 한다.

    

    test_df['diagnosis'] = np.zeros(test_df.shape[0]) 

    # For preprocessing test images

    

    datagen = ImageDataGenerator(rescale = 1./RESCALE_DN,

                                 preprocessing_function=preprocess_image)

    

    test_generator = datagen.flow_from_dataframe(

                                            test_df, 

                                            x_col='id_code',

                                            y_col='diagnosis',

                                            directory=TEST_IMAGE_FILE_PATH,

                                            target_size=(IMG_WIDTH, IMG_HEIGHT),

                                            batch_size=BATCH_SIZE,

                                            class_mode='other',

                                            shuffle=False)



    model = get_pretrained_model(model_save_filepath)



    y_test, _ = get_preds_and_labels(model, test_generator)

    

    steps = get_total_batch(test_df.shape[0], BATCH_SIZE)

    y_test = model.predict_generator(generator = test_generator,

                                           steps = steps,

                                           verbose = 0)

    

    y_test = optR.predict(y_test, coefficients).astype(np.uint8)



    test_df['diagnosis'] = y_test

    # Remove .png from ids

    test_df['id_code'] = test_df['id_code'].str.replace(r'.png$', '')

    test_df.to_csv('submission.csv', index=False)



make_submission()

clear_memory()