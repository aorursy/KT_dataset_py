import os

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

plt.style.use('seaborn')

sns.set(font_scale=1.5) 



import cv2

from PIL import Image

from tqdm import tqdm_notebook as tqdm



import scipy



from sklearn.model_selection import train_test_split, StratifiedKFold, KFold

from sklearn.metrics import f1_score, accuracy_score



from IPython.display import clear_output, FileLink





import keras

from keras import backend as K



from keras.applications.resnet50 import ResNet50, preprocess_input

from keras.callbacks import Callback, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

from keras.preprocessing.image import ImageDataGenerator

from keras.models import Sequential

from keras.optimizers import Adam, RMSprop

from keras import layers





print(os.listdir('../input/'))

%matplotlib inline



#ignore warnings

import warnings

warnings.filterwarnings('ignore')



SIZE = 224

BATCH_SIZE = 64
train_df = pd.read_csv('../input/2019-3rd-ml-month-with-kakr/train.csv')

train_df.head()
train_df.describe()
class_count = train_df.groupby('class')['img_file'].count().values

class_count
print(f"Min count class: {np.argmin(class_count)+1}, Max count class: {np.argmax(class_count)+1}")
print(f"Min: {np.min(class_count)}, Max {np.max(class_count)}, Mean {np.mean(class_count):.2f}, STD {np.std(class_count):.2f}, Total num {sum(class_count)}")
class_info = pd.read_csv('../input/2019-3rd-ml-month-with-kakr/class.csv')

class_info.set_index('id', inplace=True)

class_info.head()
test_df = pd.read_csv('../input/2019-3rd-ml-month-with-kakr/test.csv')

test_df.head()
TRAIN_IMG_PATH = '../input/2019-3rd-ml-month-with-kakr/train'

TEST_IMG_PATH = '../input/2019-3rd-ml-month-with-kakr/test'
def crop_boxing_img(img_name, margin=16) :

    if img_name.split('_')[0] == "train" :

        PATH = TRAIN_IMG_PATH

        data = train_df

    elif img_name.split('_')[0] == "test" :

        PATH = TEST_IMG_PATH

        data = test_df

        

    img = Image.open(os.path.join(PATH, img_name))

    pos = data.loc[data["img_file"] == img_name, \

                   ['bbox_x1','bbox_y1', 'bbox_x2', 'bbox_y2']].values.reshape(-1)



    width, height = img.size

    x1 = max(0, pos[0] - margin)

    y1 = max(0, pos[1] - margin)

    x2 = min(pos[2] + margin, width)

    y2 = min(pos[3] + margin, height)

    

    if abs(pos[2] - pos[0]) > width or abs(pos[3] - pos[1]) > height:

        print(f'{img_name} is wrong bounding box, img size: {img.size},  bbox_x1: {pos[0]}, bbox_x2: {pos[2]}, bbox_y1: {pos[1]}, bbox_y2: {pos[3]}')

        return img



    return img.crop((x1,y1,x2,y2))
train_path = '../train_crop'

test_path = '../test_crop'
if (os.path.isdir(train_path) == False):

    os.mkdir(train_path)



if (os.path.isdir(test_path) == False):

    os.mkdir(test_path)

    

for i, image in enumerate(tqdm(train_df['img_file'])):

    cropped = crop_boxing_img(image)

    cropped.save(os.path.join(train_path, image))

    

for i, image in enumerate(tqdm(test_df['img_file'])):

    cropped = crop_boxing_img(image)

    cropped.save(os.path.join(test_path, image))
def display_samples(path, imgid, label, columns=4, rows=3):

    fig=plt.figure(figsize=(5*columns, 4*rows))



    for i in range(columns*rows): 

        img = cv2.imread(os.path.join(path, f'{imgid[i]}'))

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        fig.add_subplot(rows, columns, i+1)

        plt.title(label[i])

        plt.imshow(img)

    

    plt.tight_layout()



display_samples(train_path, train_df['img_file'], train_df['class'])
def train_datagen():

    return ImageDataGenerator(rescale=1./255,

                              rotation_range = 40,

                              width_shift_range=0.2,

                              height_shift_range=0.2,

                              shear_range=0.10,

                              zoom_range=0.20,

                              fill_mode='nearest',

                              horizontal_flip=True,  # randomly flip images

                              vertical_flip=False,  # randomly flip images

                              preprocessing_function=preprocess_input

                             )



def val_datagen():

    return ImageDataGenerator(rescale=1./255, preprocessing_function=preprocess_input)

train_df['class'] = train_df['class'].astype(str)
classes = list(map(str, set(train_df['class'].ravel())))

class_num = len(classes)


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
def build_model():

    resnet50 = ResNet50(weights='../input/resnet50/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5',

                        include_top=False, input_shape=(SIZE,SIZE,3))

    

    model = Sequential()

    model.add(resnet50)

    model.add(layers.GlobalAveragePooling2D())

    model.add(layers.Dropout(0.5))

    model.add(layers.Dense(class_num, activation='softmax'))

    

    model.compile(loss='categorical_crossentropy',

                  optimizer=Adam(lr=0.0001),

                  metrics=['acc', f1score]

                 )

    

    return model
from datetime import datetime

from pytz import timezone, utc

KST = timezone('Asia/Seoul')



def print2(string):  

    os.system(f'echo \"{string}\"')

    print(string)



# commit시 진행상황 모니터링을 위해 추가된 클래스

class EpochLogWrite(Callback):

    def __init__(self):

        self.fold = 0

    def on_train_begin(self, logs={}):

        self.fold += 1

        print2(f'<< Fold {self.fold} >>')

    def on_epoch_begin(self, epoch, logs={}):

        tmx = utc.localize(datetime.utcnow()).astimezone(KST).time()

        print2('Epoch #{} begins at {}'.format(epoch+1, tmx))

#     def on_epoch_end(self, epoch, logs={}):

#         tmx = utc.localize(datetime.utcnow()).astimezone(KST).time()

#         print2('Epoch #{} ends at {}  acc={} val_acc={} val_f1score={}'.format(epoch+1, tmx, round(logs['acc'],4), 

#                                                                           round(logs['val_acc'],4), round(logs['val_f1score'],4) ))
def get_callback(model_name, patient):

    # 마지막 최고 score로부터 patience epoch만큼 score가 상승되지 않으면 training stop

    ES = EarlyStopping(

        monitor='val_f1score', 

        patience=patient, 

        mode='max', 

        verbose=1)

    # 마지막 최고 score로부터 patience epoch만큼 score가 상승되지 않으면 현재learning rate*factor값으로 learning rate업데이트

    RR = ReduceLROnPlateau(

        monitor = 'val_f1score', 

        factor = 0.65, 

        patience = patient//2, 

        min_lr=0.000001, 

        verbose=1, 

        mode='max')

    # 최고 score를 얻은 모델 저장.

    MC = ModelCheckpoint(

        filepath=model_name, 

        monitor='val_f1score', 

        verbose=1, 

        save_best_only=True, 

        mode='max')



    return [ES, RR, MC]
k_folds = 5

skf = StratifiedKFold(n_splits=k_folds, random_state=2019)
# del model

# del train_gen

# del val_gen

# K.clear_session()
models, hilist = [], []

toprint = EpochLogWrite()



for i, (train_idx, valied_idx) in enumerate(skf.split(train_df['img_file'], train_df['class'])):

    

    X_train = train_df.iloc[train_idx, :].reset_index()

    X_val = train_df.iloc[valied_idx, :].reset_index()

    

    train_gen = train_datagen().flow_from_dataframe(dataframe=X_train,

                                                    directory=train_path,

                                                    x_col='img_file',

                                                    y_col='class',

                                                    target_size= (SIZE,SIZE),

                                                    color_mode='rgb',

                                                    class_mode='categorical',

    #                                                 classes = classes,

                                                    batch_size=BATCH_SIZE,

                                                    seed=2019,

                                                    shuffle=True)

    

    val_gen = val_datagen().flow_from_dataframe(dataframe=X_val,

                                                    directory=train_path,

                                                    x_col='img_file',

                                                    y_col='class',

                                                    target_size= (SIZE,SIZE),

                                                    color_mode='rgb',

                                                    class_mode='categorical',

    #                                                 classes = classes,

                                                    batch_size=BATCH_SIZE,

                                                    seed=2019,

                                                    shuffle=True)



    model_name = './f1_resnet50_{}.h5'.format(i+1)

    models.append(model_name)

    

    model = build_model()

    

    print("\nStart")

    history = model.fit_generator(train_gen,

                                  steps_per_epoch=len(X_train)/BATCH_SIZE,

                                  epochs=30,

                                  validation_data=val_gen, 

                                  validation_steps=len(X_val)/BATCH_SIZE,

                                  shuffle=False,

                                  callbacks=get_callback(model_name, 5) + [toprint]

                                 )

    hilist.append(history)



    del model


fig=plt.figure(figsize=(15, 25))



for i in range(5): 

    history_df = pd.DataFrame(hilist[i].history)

    fig.add_subplot(5, 2, 2*i+1)

    plt.plot(history_df[['acc', 'val_acc', 'val_f1score']]) 

    plt.title(models[i])

    plt.legend(['acc', 'val_acc', 'val_f1score'])

    

    fig.add_subplot(5, 2, 2*(i+1))

    plt.plot(history_df['lr']) 

    plt.title(models[i])

    plt.legend(['lr'])



plt.tight_layout()
test_gen = val_datagen().flow_from_dataframe(dataframe=test_df,

                                                directory=test_path,

                                                x_col='img_file',

                                                y_col=None,

                                                target_size= (SIZE,SIZE),

                                                color_mode='rgb',

                                                class_mode=None,

                                                batch_size=BATCH_SIZE,

                                                shuffle=False)
predictions = []



for i, name in enumerate(models):

    model = build_model()

    model.load_weights(name)

    test_gen.reset()

    y_test = model.predict_generator(generator=test_gen, steps = len(test_df)/BATCH_SIZE, verbose=1 )

    predictions.append(y_test)

    

    del model

    
y_pred = np.mean(predictions, axis=0)

y_pred = np.argmax(y_pred, axis=1)



# Generator class dictionary mapping

labels = (train_gen.class_indices)

labels = dict((v,k) for k,v in labels.items())

preds = [labels[k] for k in y_pred]
submit_df = pd.read_csv('../input/2019-3rd-ml-month-with-kakr/sample_submission.csv')

submit_df['class'] = preds



print(submit_df.shape)

print(submit_df.head())
submit_df.to_csv('./submission.csv', index=False)