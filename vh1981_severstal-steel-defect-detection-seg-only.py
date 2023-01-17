from IPython.core.display import display, HTML

display(HTML("<style>.container { width:80% !important; }</style>"))
import os

import json

import gc



import cv2

import keras

from keras import backend as K

from keras import layers

from keras.preprocessing.image import ImageDataGenerator

from keras.models import Model, load_model

from keras.layers import Input

from keras.layers.convolutional import Conv2D, Conv2DTranspose

from keras.layers.pooling import MaxPooling2D

from keras.layers.merge import concatenate

from keras.optimizers import Adam

from keras.callbacks import Callback, ModelCheckpoint

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

from tqdm import tqdm

from sklearn.model_selection import train_test_split
from pathlib import Path

import shutil



INPUT_PATH = "../input"



BOOTSTRAP = False



DF_TRAIN_PATH = os.path.join(INPUT_PATH, "severstal-steel-defect-detection/train.csv")

DF_TEST_PATH = os.path.join(INPUT_PATH, "severstal-steel-defect-detection/sample_submission.csv")



TRAIN_IMAGE_PATH = os.path.join(INPUT_PATH, "severstal-steel-defect-detection/train_images")

TEST_IMAGE_PATH = os.path.join(INPUT_PATH, "severstal-steel-defect-detection/test_images")

DATA_PATH = os.path.join(INPUT_PATH, "severstal-steel-defect-detection-data-files")



GENERATE_WEIGHTS = False

USE_CALLBACK = True



EPOCHS = 16

if BOOTSTRAP:

    EPOCHS = 1

CHANNELS = 3



MODEL_NAME = "model_single_segmentation.h5"



data_dir_path = "../input/severstal-steel-defect-detection-data-files"

if os.path.exists(data_dir_path):

    for fname in os.listdir(data_dir_path):

        filepath = os.path.join(data_dir_path, fname)

        print(filepath)

        if os.path.isfile(filepath):

            if GENERATE_WEIGHTS == True:

                if fname.find("h5") > 0:

                    continue

                if fname.find("json") > 0:

                    continue

            destfilepath = os.path.join("./", fname)

            print("copy file ", filepath, " to ", destfilepath)

            shutil.copy(filepath, destfilepath)

                

train_df = pd.read_csv(DF_TRAIN_PATH, index_col=False)

train_df.head()
# make ImageId / ClassId / hasMask

if "ImageId_ClassId" in list(train_df.columns):

    train_df['ImageId'] = train_df['ImageId_ClassId'].apply(lambda x: x.split('_')[0])

    train_df['ClassId'] = train_df['ImageId_ClassId'].apply(lambda x: x.split('_')[1])

train_df.head()
'''

add rows with empty 'EncodedPixels' (same as old train.csv file)

'''

rows = []

for imageId in train_df['ImageId'].unique():

    for i in range(4):        

        exists = ((train_df['ImageId'] == imageId) & (train_df['ClassId'] == i + 1)).any()

        if exists == False:

            rows.append({'ImageId':imageId, 'ClassId':i+1, 'EncodedPixels':np.NaN})

            

tmp_df = pd.DataFrame(rows)

train_df = pd.concat([train_df, tmp_df], ignore_index=True)



train_df['hasMask'] = ~ train_df['EncodedPixels'].isna()

train_df = train_df.sort_values(['ImageId', 'ClassId']).reset_index(drop=True)



train_df.head(20)
# 이미지 중에 hasMask가 하나라도 있는 것을 구분하기 위해 ImageId로 정렬하고

# sum을 적용한다. 숫자가 아닌 column은 적용시 사라진다.



mask_count_df = train_df.groupby('ImageId').agg(np.sum).reset_index()

mask_count_df.sort_values('hasMask', ascending=False, inplace=True)

mask_count_df = mask_count_df.drop('ClassId', axis=1)

mask_count_df.head()
non_missing_train_idx = mask_count_df[mask_count_df['hasMask'] > 0]

non_missing_train_idx.head()
def mask2rle(img):

    '''

    img: numpy array, 1 - mask, 0 - background

    Returns run length as string formated

    '''

    pixels= img.T.flatten()

    pixels = np.concatenate([[0], pixels, [0]])

    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1

    runs[1::2] -= runs[::2]

    return ' '.join(str(x) for x in runs)



def rle2mask(rle, input_shape):

    '''

    rle: run-length as string formated (start length)

    shape: (height, width) of array to return 

    Returns numpy array, 1 - mask, 0 - background

    '''

    height, width = input_shape[:2]

    

    mask= np.zeros(width * height).astype(np.uint8)

    

    """    

    RLE가 (시작점,길이)의 반복이므로, 짝수/홀수로 분리해서 시작점 배열과

    길이 배열을 만든다.

    s[1:] : 1부터 끝까지

    s[1:][::2] : s[1:]배열에 2씩 건너뛰며 추출한 값들의 배열을 얻는다.

    """

    array = np.asarray([int(x) for x in rle.split()])

    starts = array[0::2]

    lengths = array[1::2]

    

    for index, start in enumerate(starts):

        begin = int(start - 1)

        end = int(begin + lengths[index])        

        mask[begin : end] = 1

        

    """    

    img의 pixel 순서는 좌측 세로줄부터 위에서 아래쪽으로 이어지므로 순서에 맞게

    만들어서 넘겨야 한다.

    width/height는 행과 열에 맞게 [height, width, ...] 로 만들어야 한다.



    ex) width=4, height=3인 경우

    

    s = [1,2,3,4,5,6,7,8,9,10,11,12]

        => 1,2,3이 좌측 첫번쩨 세로줄, 4,5,6은 두번째 줄



    s.reshape(4,3) :

    [[ 1  2  3]

     [ 4  5  6]

     [ 7  8  9]

     [10 11 12]]



    s.reshape(4,3).T :

    [[ 1  4  7 10]

     [ 2  5  8 11]

     [ 3  6  9 12]]

    """

    return mask.reshape(width, height).T



# https://www.kaggle.com/titericz/building-and-visualizing-masks

def mask2contour(mask, width=3):

    # CONVERT MASK TO ITS CONTOUR

    w = mask.shape[1]

    h = mask.shape[0]

    mask2 = np.concatenate([mask[:,width:],np.zeros((h,width))],axis=1)

    mask2 = np.logical_xor(mask,mask2)

    mask3 = np.concatenate([mask[width:,:],np.zeros((width,w))],axis=0)

    mask3 = np.logical_xor(mask,mask3)

    return np.logical_or(mask2,mask3) 



def mask2pad(mask, pad=2):

    # ENLARGE MASK TO INCLUDE MORE SPACE AROUND DEFECT

    w = mask.shape[1]

    h = mask.shape[0]

    

    # MASK UP

    for k in range(1,pad,2):

        temp = np.concatenate([mask[k:,:],np.zeros((k,w))],axis=0)

        mask = np.logical_or(mask,temp)

    # MASK DOWN

    for k in range(1,pad,2):

        temp = np.concatenate([np.zeros((k,w)),mask[:-k,:]],axis=0)

        mask = np.logical_or(mask,temp)

    # MASK LEFT

    for k in range(1,pad,2):

        temp = np.concatenate([mask[:,k:],np.zeros((h,k))],axis=1)

        mask = np.logical_or(mask,temp)

    # MASK RIGHT

    for k in range(1,pad,2):

        temp = np.concatenate([np.zeros((h,k)),mask[:,:-k]],axis=1)

        mask = np.logical_or(mask,temp)

    

    return mask 



def build_masks(rles, input_shape):

    """transfer run-length-encoding strings to mask image array

    

    Args:

        rles : array of rle string

        input_shape : mask shape

        

    Returns:

        (h, w, 5) mask array

    """

    depth = 5

    masks = np.zeros((*input_shape, depth))



    assert len(rles) == 4    

    

    for i, rle in enumerate(rles):

        if type(rle) is str:

            masks[:, :, i] = rle2mask(rle, input_shape)



    #print("masks.shape ", masks.shape)

    m2 = np.sum(masks, axis=-1).astype('bool')

    m2 = np.logical_not(m2).astype('int')

    m2 = m2.reshape(input_shape[0], input_shape[1], 1)

    

    #print("masks.shape = ", masks.shape, "m2.shape",  m2.shape)

    masks[:, :, 4] = m2.reshape(input_shape) #masks.shape =  (256, 1600, 5) m2.shape (256, 1600, 1)

    

    return masks #(256, 1600, 4)



def build_rles(masks):

    width, height, depth = masks.shape

    

    rles = [mask2rle(masks[:, :, i])

            for i in range(depth)]

    

    return rles

columns = 1

rows = 8

fig = plt.figure(figsize=(10,5 * rows))

df = train_df

import math



grp = mask_count_df['ImageId'].values



ax_idx = 1

for filename in grp:

    if ax_idx > rows * columns * 2:

        break

    

    subdf = df[df['ImageId'] == filename].reset_index()

    row = ax_idx

    col = 0

    fig.add_subplot(rows * 2, columns, ax_idx).set_title(filename)



    # show defect mask

    img = cv2.imread(os.path.join(TRAIN_IMAGE_PATH, filename ))

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    for _, row in subdf.iterrows():        

        pixels = str(row['EncodedPixels'])

        if pixels != 'nan' and len(pixels) > 10:

            #print(row['EncodedPixels'], type(row['EncodedPixels']))

            mask = rle2mask(row['EncodedPixels'], (256,1600))

            mask = mask2pad(mask,pad=3)

            mask = mask2contour(mask,width=4)

            # print(img.shape, mask.shape)

            classId = int(row['ClassId'])

            img[mask == 1, classId % 3] = 255

    plt.imshow(img)

    ax_idx += 1



    # show non-defect mask

    fig.add_subplot(rows * 2, columns, ax_idx).set_title(filename + " - no mask")

    img = cv2.imread(os.path.join(TRAIN_IMAGE_PATH, filename ))

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)



    rles = subdf['EncodedPixels'].values # 1개의 이미지마다 4개씩 있음(대부분 비어있음)

    masks = build_masks(rles, input_shape=(256,1600)) #(256, 1600, 4)



    no_defect_mask = masks[:, :, 4] # set defect mask

    # print("img.shape:", img.shape)

    # print("no_defect_mask.shape:", no_defect_mask.shape)

    # print(masks.shape, no_defect_mask.shape)

    img[no_defect_mask == 1, 0] = 255

    plt.imshow(img)

    ax_idx += 1

    

plt.show()
class DataGenerator(keras.utils.Sequence):

    'Generates data for Keras'

    def __init__(self, list_IDs, df, target_df=None, mode='fit',

                 base_path=TRAIN_IMAGE_PATH,

                 batch_size=32, dim=(256, 1600), n_channels=CHANNELS,

                 n_classes=4, random_state=2019, shuffle=True):

        self.dim = dim

        self.batch_size = batch_size

        self.df = df

        self.mode = mode

        self.base_path = base_path

        self.target_df = target_df # ImageId로 label(4 masks)을 가져올 때 사용한다.

        self.list_IDs = list_IDs # df.index

        self.n_channels = n_channels

        self.n_classes = n_classes

        self.shuffle = shuffle

        self.random_state = random_state

        

        self.on_epoch_end()



    ##########################################################

    # DataGenerator Sub Methods:

    ##########################################################

    def __len__(self):

        'Denotes the number of batches per epoch'



        # 갯수가 빠질듯.

        ret = 0

        if (len(self.list_IDs) % self.batch_size) > 0:

            ret = int(len(self.list_IDs) / self.batch_size) + 1

        else:

            ret = int(np.floor(len(self.list_IDs) / self.batch_size))

        

        return ret



    def __getitem__(self, index):

        'Generate one batch of data'

        """

        batch 하나에 해당하는 데이터(train이면 X,y, predict면 X 만)를 만들어서 리턴한다.        

        """

        # Generate indexes of the batch

        start = index * self.batch_size

        end = min(len(self.list_IDs), (index + 1) * self.batch_size)

        #print("start/end = ", start, end)

        #indexes = self.indexes[index*self.batch_size : (index+1)*self.batch_size]

        indexes = self.indexes[start : end]





        # Find list of IDs

        list_IDs_batch = [self.list_IDs[k] for k in indexes]        

        #print("list_IDs_batch :", list_IDs_batch)

        X = self.__generate_X(list_IDs_batch)

        # X.shape : (16, 256, 1600, 1)

        

        if self.mode == 'fit':

            y = self.__generate_y(list_IDs_batch)

            return X, y

        

        elif self.mode == 'predict':

            return X



        else:

            raise AttributeError('The mode parameter should be set to "fit" or "predict".')



        

    def on_epoch_end(self):        

        'Updates indexes after each epoch'        

        self.indexes = np.arange(len(self.list_IDs)) # 그냥 0 ~ n까지 배열

        if self.shuffle == True:

            np.random.seed(self.random_state)

            np.random.shuffle(self.indexes)

    ##########################################################

    

    

    def __generate_X(self, list_IDs_batch):

        'Generates data containing batch_size samples'

        # Initialization

        X = np.empty((len(list_IDs_batch), *self.dim, self.n_channels)) #(?, h, w, 채널수(1))

        

        # Generate data

        for i, ID in enumerate(list_IDs_batch):            

            im_name = self.df['ImageId'].loc[ID]

            img_path = f"{self.base_path}/{im_name}"

            #img = self.__load_grayscale(img_path)

            img = self.__load_rgb(img_path)



            #print("im_name", im_name)

            

            # Store samples

            X[i,] = img



        return X

    

    def __generate_y(self, list_IDs_batch):

        y = np.empty((len(list_IDs_batch), *self.dim, self.n_classes), dtype=int)

        

        for i, ID in enumerate(list_IDs_batch):

            im_name = self.df['ImageId'].loc[ID]

            image_df = self.target_df[self.target_df['ImageId'] == im_name]

            

            # y값은 RLE를 읽고 mask로 만들어서 사용

            rles = image_df['EncodedPixels'].values # 1개의 이미지마다 4개씩 있음(대부분 비어있음)

            masks = build_masks(rles, input_shape=self.dim) #(256, 1600, 5)

            

            y[i, ] = masks

        

        return y #(batch_size, 256, 1600, 5)

    

    def __load_grayscale(self, img_path):

        """

        load image as grayscale

        """

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        img = img.astype(np.float32) / 255.        

        img = np.expand_dims(img, axis=-1) # [h, w] => [h, w, 1]



        return img

    

    def __load_rgb(self, img_path):

        img = cv2.imread(img_path)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = img.astype(np.float32) / 255.



        return img
TEST_INDEX = 16



# TEST :



def test_DataGenerator_src(index = 0):

    BATCH_SIZE = 4    

    rows = BATCH_SIZE

    columns = 1

    

    train_idx = non_missing_train_idx.index



    fig = plt.figure(figsize=(12, BATCH_SIZE * 6))    



    ax_idx = 1

    for i in range(BATCH_SIZE):

        if ax_idx > rows * columns * 2:

            break



        cur_row = mask_count_df.loc[train_idx[BATCH_SIZE * index + i]]

        filename = cur_row['ImageId']        

        image_df = train_df[train_df['ImageId'] == filename] # ImageId마다 4개씩 있음.

        image_df = image_df.fillna("")



        fig.add_subplot(rows * 2, columns, ax_idx).set_title(filename)

        img = cv2.imread(os.path.join(TRAIN_IMAGE_PATH, filename ))

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        

        for _, row in image_df.iterrows():

            #print("row['EncodedPixels']", type(row['EncodedPixels']), row['EncodedPixels'])

            if len(str(row['EncodedPixels'])) > 0 :

                mask = rle2mask(row['EncodedPixels'], (256,1600))

                mask = mask2pad(mask,pad=3)

                mask = mask2contour(mask,width=4)

                classId = int(row['ClassId'])

                img[mask == 1, (classId - 1) % 3] = 255                

        plt.imshow(img)

        ax_idx += 1



        fig.add_subplot(rows * 2, columns, ax_idx).set_title(filename + "- no defect")

        img = cv2.imread(os.path.join(TRAIN_IMAGE_PATH, filename ))

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        masks = build_masks(image_df['EncodedPixels'].values, input_shape=(256,1600))

        non_defect_mask = masks[:, :, 4]

        img[non_defect_mask > 0, 0] = 255

        plt.imshow(img)

        ax_idx += 1

        

    plt.show()

    

test_DataGenerator_src(TEST_INDEX)
# TEST :



def test_DataGenerator(index = 0):

    BATCH_SIZE = 4



    rows = BATCH_SIZE

    columns = 1    

    

    train_idx = non_missing_train_idx.index



    fig = plt.figure(figsize=(12, BATCH_SIZE * 6))



    gen = DataGenerator(

        train_idx,

        df=mask_count_df,

        target_df=train_df,

        batch_size=BATCH_SIZE, 

        n_classes=5,

        shuffle = False)



    X, y = gen.__getitem__(index)



    print("X.shape :", X.shape)

    print("y.shape :", y.shape)    



    ax_idx = 1

    for i in range(BATCH_SIZE):

        if ax_idx > rows * columns * 2:

            break

            

        img = X[i]

        img = img * 255

        img = img.astype(int)



        fig.add_subplot(rows * 2, columns, ax_idx).set_title(str(i) + "- source image")



        for mask_index in range(4):

            mask = y[i, :, :, mask_index]

            mask = mask2pad(mask,pad=3)

            mask = mask2contour(mask,width=4)

            y[i, :, :, mask_index] = mask

            k = y[i, :, :, mask_index]            

            img[k == 1, mask_index % 3] = 255        

        plt.imshow(img)

        ax_idx += 1



        fig.add_subplot(rows * 2, columns, ax_idx).set_title(str(i) + "- mask area")

        img = X[i]

        img = img * 255

        img = img.astype(int)

        no_defect_mask = y[i, :, :, 4]

        img[no_defect_mask > 0, 0] = 255

        plt.imshow(img)

        ax_idx += 1

        

    plt.show()



    

test_DataGenerator(TEST_INDEX)
# make image generators 



BATCH_SIZE = 4



train_idx, val_idx = train_test_split(

    mask_count_df.index, # 모든 파일을 입력으로 사용해야 하므로 non_missing_train_idx.index를 사용하지 않음

    random_state=2019,

    test_size=0.15

)



train_generator = DataGenerator(

    train_idx,

    df=mask_count_df,

    target_df=train_df,

    batch_size=BATCH_SIZE,

    n_classes=5)



val_generator = DataGenerator(

    val_idx,

    df=mask_count_df,

    target_df=train_df,

    batch_size=BATCH_SIZE,

    n_classes=5)
def dice_coef(y_true, y_pred, smooth=1):

    print(y_true.shape, y_pred.shape)

    y_true_f = K.flatten(y_true)

    y_pred_f = K.flatten(y_pred)

    intersection = K.sum(y_true_f * y_pred_f)

    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
# https://github.com/qubvel/segmentation_models



! pip install segmentation-models
import segmentation_models as sm

from segmentation_models import Unet

import keras



def build_model():



    # class는 5가 되어야 하고, activation -> softmax, loss : cross entropy

    preprocess = sm.get_preprocessing('resnet34')

    model = Unet(backbone_name='resnet34', input_shape=(256,1600, 3), classes=5, activation='softmax')

    

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])

    print("Model Input => Output : ", model.input_shape, " ======> ", model.output_shape)



    #model.summary()



    return model



model = build_model()



model.summary()
def get_pretrained_model():

    """

    get model with loaded weight & json model file

    """

    model_json_file_name = MODEL_NAME.split('.')[0] + ".json"

    json_file = open(model_json_file_name, "r")

    loaded_model_json = json_file.read()

    json_file.close()

    loaded_model = model_from_json(loaded_model_json)

    loaded_model.load_weights(MODEL_NAME)

    loaded_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])

    print("Loaded Model Input => Output : ", loaded_model.input_shape, " ======> ", loaded_model.output_shape)

    return loaded_model

    
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, LambdaCallback



es = EarlyStopping(monitor='val_acc', min_delta=0, patience = 3, verbose=1, mode='max')



rl = ReduceLROnPlateau(monitor = 'val_acc', factor = 0.5, patience = 2,

                       min_lr=0.0000001,

                       verbose=1, 

                       mode='max')



checkpoint = ModelCheckpoint(

    MODEL_NAME,

    monitor='val_acc',

    verbose=1,

    save_best_only=True,

    save_weights_only=False,

    mode='max')



if GENERATE_WEIGHTS:

    history = None

    

    history = model.fit_generator(

        train_generator,

        validation_data=val_generator,

        callbacks=[es, rl, checkpoint],

        use_multiprocessing=False,

        workers=1,

        epochs=EPOCHS)

    

    # save model as json file

    # weight is already stored by callback(checkpoint)

    model_json = model.to_json()

    model_json_file_name = MODEL_NAME.split('.')[0] + ".json"

    with open(model_json_file_name, "w") as json_file: 

        json_file.write(model_json)

    

    hdf = pd.DataFrame(history.history)

    hdf[['loss', 'val_loss']].plot()

    hdf[['acc', 'val_acc']].plot()

    #hdf[['dice_coef', 'val_dice_coef']].plot()

    

from keras.models import model_from_json



# model = build_model()

# model.load_weights(MODEL_NAME)

model = get_pretrained_model()



check_df = mask_count_df.sample(10)



gen = DataGenerator(

        check_df.index,

        df=mask_count_df,

        target_df=train_df,

        batch_size=BATCH_SIZE, 

        n_classes=5,

        shuffle=False)



predict = model.predict_generator(gen)
def get_one_hot(targets, nb_classes):    

    res = np.eye(nb_classes)[np.array(targets).reshape(-1)] # flatten pixels and ont-hot encode

    return res.reshape(list(targets.shape)+[nb_classes]) # restore shape(plus add ont-hot dimension)



def pred_to_onehot(pred, nb_classes=5):

    retval = np.argmax(pred, axis=-1)

    retval = get_one_hot(retval.reshape(-1), nb_classes)

    retval = retval.reshape(pred.shape)

    return retval
pred = pred_to_onehot(predict)



rows = 10

columns = 1



fig = plt.figure(figsize=(12, 6 * rows))



ax_idx=1

for index, bindex, in enumerate(check_df.index):

    if ax_idx > rows * columns * 2:

        break

            

    fname = check_df['ImageId'].loc[bindex]

    image_df = train_df[train_df['ImageId'] == fname]



    fig.add_subplot(rows * 2, columns, ax_idx).set_title(fname + " - source image")

    img = cv2.imread(os.path.join(TRAIN_IMAGE_PATH, fname ))

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    plt.imshow(img)

    ax_idx += 1



    fig.add_subplot(rows * 2, columns, ax_idx).set_title(fname + " - prediction")

    for mask_index in range(4):

        mask = pred[index, :, :, mask_index]        

        img[mask == 1, mask_index % 3] = 255

    plt.imshow(img)

    ax_idx += 1



plt.show()
def get_test_imgs_df():

    """build test image DataFrame from test image directory"""

    test_df = []

    for fname in os.listdir(TEST_IMAGE_PATH):

        test_df.append(fname)        



    test_df = pd.DataFrame({'ImageId' : test_df, 'EncodedPixels' : ''})    

    ret = test_df[['ImageId']].reset_index()

    return ret





def get_test_df():

    """build image mask DataFrame"""

    test_df = []

    for fname in os.listdir(TEST_IMAGE_PATH):

        filepath = os.path.join(TEST_IMAGE_PATH, fname)    

        if os.path.isfile(filepath):

            for i in range(4):

                img_cls = fname + "_" + str(i + 1)

                test_df.append(img_cls)

    

    test_df = pd.DataFrame({'ImageId_ClassId' : test_df, 'EncodedPixels' : ''})

    test_df['ImageId'] = test_df['ImageId_ClassId'].apply(lambda x: x.split('_')[0])

    test_df['ClassId'] = test_df['ImageId_ClassId'].apply(lambda x: x.split('_')[1])

    test_df['EncodedPixels'] = ""

    test_df.reset_index()

    

    return test_df
#######################################################################################

# FIXME : TEST CODE:

#######################################################################################



from keras.backend import clear_session

import gc



def show_test_prediction_head(n=60):



    # Reset Keras Session

    def clear_memory():

        clear_session()

        for i in range(20):

            gc.collect()



    clear_memory()

    TEST_BATCH_SIZE = n



    test_df = get_test_df()

    test_images_df = get_test_imgs_df()    



    batch_idx = list(range(0, min(test_images_df.shape[0], TEST_BATCH_SIZE)))

    print("running: ", 0, " - ", min(test_images_df.shape[0], TEST_BATCH_SIZE))



#     model = build_model()

#     model.load_weights(MODEL_NAME)

    model = get_pretrained_model()

    

    test_generator = DataGenerator(

        batch_idx,

        df=test_images_df,

        base_path = TEST_IMAGE_PATH,

        target_df=test_df, #label mask를 만들 때 사용하는 DataFrame. mode = 'predict' 인 경우 필요없음.

        mode = 'predict',

        batch_size=TEST_BATCH_SIZE,

        shuffle=False,

        n_classes=5)



    src_generator = DataGenerator(

        batch_idx,

        df=test_images_df,

        base_path = TEST_IMAGE_PATH,

        target_df=test_df,

        batch_size=TEST_BATCH_SIZE,

        shuffle=False,

        n_classes=5)



    X, _ = src_generator.__getitem__(0)

    

    # make prediction

    predict = model.predict_generator(test_generator)

    pred = pred_to_onehot(predict)



    columns = 1

    rows = TEST_BATCH_SIZE

    fig = plt.figure(figsize=(12, 3 * rows))



    ax_idx = 1

    for i in range(rows):

        if ax_idx > rows * columns:

            break

            

        # add plot

        fig.add_subplot(rows, columns, ax_idx).set_title(str(i))

        

        # source image

        img = X[i]

        img = img * 255

        img = img.astype(int)    



        # draw mask over image

        for mask_index in range(4):

            k = pred[i, :, :, mask_index]            

            img[k == 1, mask_index % 3] = 255

        plt.imshow(img)

        ax_idx += 1



    plt.show()



# show_test_prediction_head()



#######################################################################################
from keras.backend import clear_session

import gc



# Reset Keras Session

def clear_memory():

    clear_session()

    for i in range(20):

        gc.collect()



clear_memory()



TEST_BATCH_SIZE = 100

df_submit = []

MIN_MASK_PIXEL_THRESHOLD = 3500



test_df = get_test_df()

test_images_df = get_test_imgs_df()



# 하나의 이미지마다 동일 크기의 5개 mask 이미지가 생성되기 때문에

# 메모리 소비가 커서 나눠서 처리해야 한다.

for batch_start in range(0, test_images_df.shape[0], TEST_BATCH_SIZE):

    batch_idx = list(range(batch_start, min(test_images_df.shape[0], batch_start + TEST_BATCH_SIZE)))

    print("running: ", batch_start, " - ", min(test_images_df.shape[0], batch_start + TEST_BATCH_SIZE))



    model = get_pretrained_model()



    test_generator = DataGenerator(

        batch_idx,

        df=test_images_df,

        base_path = TEST_IMAGE_PATH,

        target_df=test_df,

        mode = 'predict',

        batch_size=BATCH_SIZE,

        shuffle=False,

        n_classes=5)

    

    predict = model.predict_generator(test_generator)

    pred = pred_to_onehot(predict)



    for index, bindex, in enumerate(batch_idx):

        fname = test_images_df['ImageId'].loc[bindex]

        image_df = test_df[test_df['ImageId'] == fname]



        pred_masks = pred[index, ]

        #print("pred_masks.shape :", pred_masks.shape)



        # threshold 이하 pixel 수는 모두 없앤다.

        for mask_index in range(4):

            pixelcnt = np.count_nonzero(pred_masks[:,:,mask_index])        

            if pixelcnt < MIN_MASK_PIXEL_THRESHOLD:

                pred_masks[:,:,mask_index] = 0



        pred_masks = pred_masks[:, :, :-1] # drop non-defect mask values

        pred_rles = build_rles(pred_masks)

        image_df['EncodedPixels'] = pred_rles        

        df_submit.append(image_df)

    

    clear_memory()



df_submit = pd.concat(df_submit)

print(df_submit.shape[0])

df_submit.head()
df_temp = df_submit



df_temp['maskPixelCount'] = df_temp['EncodedPixels'].map(str).apply(len)

df_temp = df_temp.sort_values(['maskPixelCount'], ascending=[False])

df_temp = df_temp.reset_index()



columns = 1

rows = 40

fig = plt.figure(figsize=(12, 6 * rows))



ax_idx = 1

for index, row in df_temp.iterrows():

    if ax_idx > rows * columns:

        break



    #print("index:", index, "imageid", row["ImageId"], "class", row["ClassId"])



    filename = row['ImageId']

    fig.add_subplot(rows, columns, ax_idx).set_title(filename)

    img = cv2.imread(os.path.join(TEST_IMAGE_PATH, filename))

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    

    mask = rle2mask(row['EncodedPixels'], (256,1600))

    img[mask == 1, 0] = 255

            

    plt.imshow(img)

    ax_idx += 1

        

plt.show()

df_submit.head(20)
df_submit[['ImageId_ClassId', 'EncodedPixels']].to_csv('submission.csv', index=False)