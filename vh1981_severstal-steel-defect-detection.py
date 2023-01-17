import sys

IN_COLAB = 'google.colab' in sys.modules

!nvidia-smi -L
# colab에서 구동하는 경우 서버의 구글 드라이브 파일을 다운받는다.



import os

#input.zip : https://drive.google.com/open?id=1Cb9kjJ40Sc7hs3TtDREGjdytH7479PKu

#model.h5 : https://drive.google.com/open?id=1CnF0Ailc2s8ob0JieXrhTK4u1YEr_HaD

#model_predict_missing_mask.h5 : https://drive.google.com/open?id=1Sr6D8utBeOEnQ3BUGEdCCwPkYiPfY_uM



def download_file_gd(file_id, fpathname, unzip=False):

    from google_drive_downloader import GoogleDriveDownloader as gdd

    if os.path.exists(fpathname) == False:

        gdd.download_file_from_google_drive(file_id=file_id, dest_path=fpathname, unzip=unzip, showsize=False)

    else:

        print(fpathname, ": already downloaded")



files = {

    "1Cb9kjJ40Sc7hs3TtDREGjdytH7479PKu" : "./input/severstal-steel-defect-detection/input.zip", 

    "1CnF0Ailc2s8ob0JieXrhTK4u1YEr_HaD" : "./model.h5", 

    "1Sr6D8utBeOEnQ3BUGEdCCwPkYiPfY_uM" : "./input/severstal-steel-defect-detection-data-files/model_predict_missing_mask.h5", 

}



if IN_COLAB:

    for f in files:

        print(f, files[f])

        download_file_gd(file_id=f, fpathname=files[f], unzip=(files[f].find(".zip") >= 0))

        

    # unzip train/test zip file

    import zipfile

    zipfile.ZipFile("./input/severstal-steel-defect-detection/train_images.zip").extractall("./input/severstal-steel-defect-detection/train_images")

    zipfile.ZipFile("./input/severstal-steel-defect-detection/test_images.zip").extractall("./input/severstal-steel-defect-detection/test_images")        



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



#sys.path.append(os.path.abspath('../input/efficientnet/efficientnet-master/efficientnet-master/'))



INPUT_PATH = "./input"

if IN_COLAB == False:

    INPUT_PATH = "../input"



DF_TRAIN_PATH = os.path.join(INPUT_PATH, "severstal-steel-defect-detection/train.csv")

DF_TEST_PATH = os.path.join(INPUT_PATH, "severstal-steel-defect-detection/sample_submission.csv")



TRAIN_IMAGE_PATH = os.path.join(INPUT_PATH, "severstal-steel-defect-detection/train_images")

TEST_IMAGE_PATH = os.path.join(INPUT_PATH, "severstal-steel-defect-detection/test_images")

DATA_PATH = os.path.join(INPUT_PATH, "severstal-steel-defect-detection-data-files")



GENERATE_WEIGHTS = True



EPOCHS = 12



USE_CALLBACK = True

    

if IN_COLAB == False:    

    data_dir_path = "../input/severstal-steel-defect-detection-data-files"

    if os.path.exists(data_dir_path):

        for fname in os.listdir(data_dir_path):

            filepath = os.path.join(data_dir_path, fname)

            print(filepath)

            if os.path.isfile(filepath):

                if GENERATE_WEIGHTS == True:

                    if fname.find("h5") > 0:

                        continue

                destfilepath = os.path.join("./", fname)

                print("copy file ", filepath, " to ", destfilepath)

                shutil.copy(filepath, destfilepath)

                

train_df = pd.read_csv(DF_TRAIN_PATH)

'''

image 파일명과 ClassId가 _로 연결되어 있어서 분리해서 별도 column으로 만든다.

'''

train_df['ImageId'] = train_df['ImageId_ClassId'].apply(lambda x: x.split('_')[0])

train_df['ClassId'] = train_df['ImageId_ClassId'].apply(lambda x: x.split('_')[1])

train_df['hasMask'] = ~ train_df['EncodedPixels'].isna()

train_df.head()
# 이미지 중에 hasMask가 하나라도 있는 것을 구분하기 위해 ImageId로 정렬하고

# sum을 적용한다. 숫자가 아닌 column은 적용시 사라진다.



mask_count_df = train_df.groupby('ImageId').agg(np.sum).reset_index()

mask_count_df.sort_values('hasMask', ascending=False, inplace=True)

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



def build_masks(rles, input_shape):

    depth = len(rles)

    masks = np.zeros((*input_shape, depth))

    

    for i, rle in enumerate(rles):

        if type(rle) is str:

            masks[:, :, i] = rle2mask(rle, input_shape)    

    

    return masks #(256, 1600, 4)



def build_rles(masks):

    width, height, depth = masks.shape

    

    rles = [mask2rle(masks[:, :, i])

            for i in range(depth)]

    

    return rles

columns = 1

rows = 10

fig = plt.figure(figsize=(20,80))

df = train_df[train_df['hasMask']] #mask가 있는 것만 추린다.



df_maskCnt = pd.DataFrame({'maskCount' : df.groupby('ImageId').size()})

df = pd.merge(df, df_maskCnt, on="ImageId")

df = df[df['maskCount'] > 1]

df = df.sort_values(by='maskCount', ascending=False) # 최대한 valid한 mask가 많은 것을 보여주도록



grp = df.groupby('ImageId')



ax_idx = 1

for filename, g in grp:

    if ax_idx > rows * columns * 2:

        break

    

    subdf = df[df['ImageId'] == filename].reset_index()

    row = ax_idx

    col = 0



    fig.add_subplot(rows * 2, columns, ax_idx).set_title(filename)

    img = cv2.imread(os.path.join(TRAIN_IMAGE_PATH, filename ))

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)    

    

    plt.imshow(img)

    

    ax_idx += 1

    fig.add_subplot(rows * 2, columns, ax_idx).set_title(filename)



    for _, row in subdf.iterrows():

        mask = rle2mask(row['EncodedPixels'], (256,1600))

        classId = int(row['ClassId'])                

        img[mask == 1, classId % 3] = 255        

            

    plt.imshow(img)

    ax_idx += 1

        

plt.show()
class DataGenerator(keras.utils.Sequence):

    'Generates data for Keras'

    def __init__(self, list_IDs, df, target_df=None, mode='fit',

                 base_path=TRAIN_IMAGE_PATH,

                 batch_size=32, dim=(256, 1600), n_channels=1,

                 n_classes=4, random_state=2019, shuffle=True):

        self.dim = dim

        self.batch_size = batch_size

        self.df = df

        self.mode = mode

        self.base_path = base_path

        self.target_df = target_df

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



        #print("ret=", ret)

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

            img = self.__load_grayscale(img_path)           



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

            masks = build_masks(rles, input_shape=self.dim) #(256, 1600, 4)

            

            y[i, ] = masks

        

        return y #(batch_size, 256, 1600, 4)

    

    def __load_grayscale(self, img_path):

        """

        이미지를 gray-scale로 읽어서 돌려준다.

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
non_missing_train_idx.index

mask_count_df.head()
TEST_INDEX = 18





# TEST :



def test_DataGenerator_src(index = 0):

    BATCH_SIZE = 16

    

    train_idx = non_missing_train_idx.index



    fig = plt.figure(figsize=(20,80))



    columns = 1

    rows = BATCH_SIZE



    ax_idx = 1

    for i in range(BATCH_SIZE):

        if ax_idx > rows * columns:

            break



        cur_row = mask_count_df.loc[train_idx[BATCH_SIZE * index + i]]

        filename = cur_row['ImageId']

        print(train_idx[BATCH_SIZE * index + i], filename, cur_row['hasMask'])

        image_df = train_df[train_df['ImageId'] == filename] # ImageId마다 4개씩 있음.

        image_df = image_df.fillna("")



        fig.add_subplot(rows, columns, ax_idx).set_title(filename)

        img = cv2.imread(os.path.join(TRAIN_IMAGE_PATH, filename ))

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        

        for _, row in image_df.iterrows():

            #print("row['EncodedPixels']", type(row['EncodedPixels']), row['EncodedPixels'])

            if len(str(row['EncodedPixels'])) > 0 :

                mask = rle2mask(row['EncodedPixels'], (256,1600))

                classId = int(row['ClassId'])

                img[mask == 1, (classId - 1) % 3] = 255

                

        plt.imshow(img)

        ax_idx += 1

        

    plt.show()

    

test_DataGenerator_src(TEST_INDEX)
# TEST :



def test_DataGenerator(index = 0):

    BATCH_SIZE = 16

    

    train_idx = non_missing_train_idx.index    



    fig = plt.figure(figsize=(20,80))



    gen = DataGenerator(

        train_idx,

        df=mask_count_df,

        target_df=train_df,

        batch_size=BATCH_SIZE, 

        n_classes=4,

        shuffle=False)



    X, y = gen.__getitem__(index)



    print("X.shape :", X.shape)

    print("y.shape :", y.shape)



    columns = 1

    rows = BATCH_SIZE



    ax_idx = 1

    for i in range(BATCH_SIZE):

        if ax_idx > rows * columns:

            break

            

        img = X[i].reshape(X[i].shape[0], X[i].shape[1])

        img = img * 255

        img = img.astype(int)

        img = np.stack((img, img, img), axis=2)    



        fig.add_subplot(rows, columns, ax_idx).set_title(str(i))



        for mask in range(4):

            k = y[i, :, :, mask]

            #print("k.shape", k.shape)

            img[k == 1, mask % 3] = 255

                

        plt.imshow(img)

        ax_idx += 1

        

    plt.show()



    

test_DataGenerator(TEST_INDEX)
BATCH_SIZE = 16



train_idx, val_idx = train_test_split(

    non_missing_train_idx.index,  # NOTICE DIFFERENCE

    random_state=2019,

    test_size=0.15

)



train_generator = DataGenerator(

    train_idx, 

    df=mask_count_df,

    target_df=train_df,

    batch_size=BATCH_SIZE, 

    n_classes=4

)



val_generator = DataGenerator(

    val_idx,

    df=mask_count_df,

    target_df=train_df,

    batch_size=BATCH_SIZE,

    n_classes=4

)
def dice_coef(y_true, y_pred, smooth=1):

    print(y_true.shape, y_pred.shape)

    y_true_f = K.flatten(y_true)

    y_pred_f = K.flatten(y_pred)

    intersection = K.sum(y_true_f * y_pred_f)

    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
def build_model(input_shape):

    """

    input : [batch_size, h, w, 1] 그레이스케일 이미지

    output : [batch_size, h, w, 4] 4 defect mask (class 1~4)

    """

    inputs = Input(input_shape)



    c1 = Conv2D(8, (3, 3), activation='relu', padding='same') (inputs)

    c1 = Conv2D(8, (3, 3), activation='relu', padding='same') (c1)

    p1 = MaxPooling2D((2, 2)) (c1)



    c2 = Conv2D(16, (3, 3), activation='relu', padding='same') (p1)

    c2 = Conv2D(16, (3, 3), activation='relu', padding='same') (c2)

    p2 = MaxPooling2D((2, 2)) (c2)



    c3 = Conv2D(32, (3, 3), activation='relu', padding='same') (p2)

    c3 = Conv2D(32, (3, 3), activation='relu', padding='same') (c3)

    p3 = MaxPooling2D((2, 2)) (c3)



    c4 = Conv2D(64, (3, 3), activation='relu', padding='same') (p3)

    c4 = Conv2D(64, (3, 3), activation='relu', padding='same') (c4)

    p4 = MaxPooling2D(pool_size=(2, 2)) (c4)



    c5 = Conv2D(64, (3, 3), activation='relu', padding='same') (p4)

    c5 = Conv2D(64, (3, 3), activation='relu', padding='same') (c5)

    p5 = MaxPooling2D(pool_size=(2, 2)) (c5)



    c55 = Conv2D(128, (3, 3), activation='relu', padding='same') (p5)

    c55 = Conv2D(128, (3, 3), activation='relu', padding='same') (c55)

    

    u6 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same') (c55)

    print("u6:", u6.shape, "c5:", c5.shape)

        

    # u6 : (?, ?, ?, 64)  c5 : (?, 16, 100, 64)

    u6 = concatenate([u6, c5]) # axis가 지정되어 있지 않으면 마지막 dim에 붙는다.

    # u6 : (?, 16, 100, 128)



    print("u6 after:", u6.shape, "c5:", c5.shape)

    c6 = Conv2D(64, (3, 3), activation='relu', padding='same') (u6)

    c6 = Conv2D(64, (3, 3), activation='relu', padding='same') (c6)



    u71 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (c6)

    u71 = concatenate([u71, c4]) # channel이 64가 된다.

    c71 = Conv2D(32, (3, 3), activation='relu', padding='same') (u71)

    c61 = Conv2D(32, (3, 3), activation='relu', padding='same') (c71)



    u7 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (c61)

    u7 = concatenate([u7, c3]) # channel=64

    c7 = Conv2D(32, (3, 3), activation='relu', padding='same') (u7)

    c7 = Conv2D(32, (3, 3), activation='relu', padding='same') (c7)



    u8 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same') (c7)

    u8 = concatenate([u8, c2]) # channel=32

    c8 = Conv2D(16, (3, 3), activation='relu', padding='same') (u8)

    c8 = Conv2D(16, (3, 3), activation='relu', padding='same') (c8)



    u9 = Conv2DTranspose(8, (2, 2), strides=(2, 2), padding='same') (c8)

    u9 = concatenate([u9, c1], axis=3) # channel=16

    c9 = Conv2D(8, (3, 3), activation='relu', padding='same') (u9)

    c9 = Conv2D(8, (3, 3), activation='relu', padding='same') (c9)



    # y의 channel이 4이므로 출력 채널도 4로 맞춘다. 크기는 입력과 동일하게 한다.

    outputs = Conv2D(4, (1, 1), activation='sigmoid') (c9)



    model = Model(inputs=[inputs], outputs=[outputs])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[dice_coef])

    

    return model
model = build_model((256, 1600, 1))

model.summary()
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, LambdaCallback



es = EarlyStopping(monitor='val_dice_coef', min_delta=0, patience = 3, verbose=1, mode='max')



rl = ReduceLROnPlateau(monitor = 'val_dice_coef', factor = 0.5, patience = 2,

                       min_lr=0.0000001,

                       verbose=1, 

                       mode='max')



checkpoint = ModelCheckpoint(

    './model.h5', 

    monitor='val_dice_coef',

    verbose=1,

    save_best_only=True,

    save_weights_only=False,

    mode='max'

)



if GENERATE_WEIGHTS:

    history = None

    if USE_CALLBACK:

        history = model.fit_generator(

            train_generator,

            validation_data=val_generator,

            callbacks=[es, rl, checkpoint],

            use_multiprocessing=False,

            workers=1,

            epochs=EPOCHS)

    else :

        history = model.fit_generator(

            train_generator,

            validation_data=val_generator,

            callbacks=[checkpoint],

            use_multiprocessing=False,

            workers=1,

            epochs=10)

    

    hdf = pd.DataFrame(history.history)

    hdf[['loss', 'val_loss']].plot()

    hdf[['dice_coef', 'val_dice_coef']].plot()

def get_test_imgs_df():

    '''

    Test 이미지 디렉토리의 이미지 파일명을 사용해서 ImageId가 

    파일명을 가지고 있는 DataFrame을 생성한다.

    '''

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



    test_image_df = test_df.groupby('ImageId').agg(np.sum).reset_index()

    test_image_df = test_image_df[['ImageId']]

    test_image_df.reset_index()

    test_image_df.head(20)

    return test_image_df





def get_test_df():

    '''

    Test 이미지 디렉토리의 이미지들로 ImageId_ClassId를 가진 DataFrame을 만든다.

    '''

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
BATCH_SIZE = 64

test_imgs = get_test_imgs_df()

print(TEST_IMAGE_PATH)

def create_test_gen():

    return ImageDataGenerator(rescale=1/255.).flow_from_dataframe(

        test_imgs,

        #directory='../input/severstal-steel-defect-detection/test_images',

        directory= TEST_IMAGE_PATH,

        x_col='ImageId',

        class_mode=None,

        target_size=(256, 256),

        batch_size=BATCH_SIZE,

        shuffle=False)



test_gen = create_test_gen()



classify_model = load_model(os.path.join(DATA_PATH, "model_predict_missing_mask.h5"))

classify_model.summary()

test_missing_pred = classify_model.predict_generator(

    test_gen,

    steps=len(test_gen),

    verbose=1

)



# print(test_imgs.shape)

# print(len(test_missing_pred))



test_imgs['allMissing'] = test_missing_pred

test_imgs.head()



plt.figure(figsize=(10, 5))

test_imgs2 = test_imgs.sort_values(by=['allMissing'])

plt.plot(test_imgs2['allMissing'].values)

plt.xlabel("count")

plt.ylabel("allMissing")

plt.show()
filtered_test_imgs = test_imgs[test_imgs['allMissing'] < 0.5]

print(filtered_test_imgs.shape)

filtered_test_imgs.head()
sub_df = get_test_df()



filtered_mask = sub_df['ImageId'].isin(filtered_test_imgs["ImageId"].values)

filtered_sub_df = sub_df[filtered_mask].copy()

null_sub_df = sub_df[~filtered_mask].copy()

#null_sub_df['EncodedPixels'] = null_sub_df['EncodedPixels'].apply(lambda x: ' ')



filtered_sub_df['EncodedPixels'] = ""

null_sub_df['EncodedPixels'] = ""



filtered_sub_df.reset_index(drop=True, inplace=True)

filtered_test_imgs.reset_index(drop=True, inplace=True)



print(filtered_sub_df.shape)

print(null_sub_df.shape)



print("filtered images: ")

print(filtered_test_imgs.head())



print("filtered df: ")

print(filtered_sub_df.head())



print("null df: ")

print(null_sub_df.head())
from keras.backend import clear_session

import gc



# Reset Keras Session

def clear_memory():

    clear_session()

    for i in range(20):

        gc.collect()  



clear_memory()



test_df = get_test_df() #test image DataFrame



TEST_BATCH_SIZE = 100

df_submit = []

MIN_MASK_PIXEL_THRESHOLD = 3500



test_image_df = filtered_test_imgs



# 하나의 이미지마다 동일 크기의 4개 mask 이미지가 생성되기 때문에

# 메모리 소비가 커서 나눠서 처리해야 한다.

for batch_start in range(0, test_image_df.shape[0], TEST_BATCH_SIZE):

    batch_idx = list(range(batch_start, min(test_image_df.shape[0], batch_start + TEST_BATCH_SIZE)))

    print("running: ", batch_start, " - ", min(test_image_df.shape[0], batch_start + TEST_BATCH_SIZE))



    model = build_model((256, 1600, 1))

    model.load_weights('model.h5')



    test_generator = DataGenerator(

        batch_idx,

        df=test_image_df,

        base_path = TEST_IMAGE_PATH,

        target_df=test_df,

        mode = 'predict',

        batch_size=BATCH_SIZE,

        n_classes=4)

    

    predict = model.predict_generator(test_generator)



    for index, bindex, in enumerate(batch_idx):

        fname = test_image_df['ImageId'].iloc[bindex]

        image_df = test_df[test_df['ImageId'] == fname]



        pred_masks = predict[index, ].round().astype(int)        

        #print("pred_masks.shape :", pred_masks.shape)



        for mask_index in range(4):

            pixelcnt = np.count_nonzero(pred_masks[:,:,mask_index])

            #print(index, mask_index, pixelcnt)

            if pixelcnt < MIN_MASK_PIXEL_THRESHOLD:

                pred_masks[:,:,mask_index] = 0



        pred_rles = build_rles(pred_masks)



        #print(len(pred_rles))



        image_df['EncodedPixels'] = pred_rles        

        df_submit.append(image_df)

    

    clear_memory()



df_submit = pd.concat(df_submit)

df_submit = pd.concat([df_submit, null_sub_df])



print(df_submit.shape[0])

df_submit.head()
df_temp = df_submit



df_temp['maskPixelCount'] = df_temp['EncodedPixels'].map(str).apply(len)

df_temp = df_temp.sort_values(['maskPixelCount'], ascending=[False])

df_temp = df_temp.reset_index()

#df_temp.head(80)



columns = 1

rows = 20

fig = plt.figure(figsize=(20,80))



ax_idx = 1

for index, row in df_temp.iterrows():

    if ax_idx > rows * columns:

        break



    print("index:", index, "imageid", row["ImageId"], "class", row["ClassId"])



    filename = row['ImageId']

    fig.add_subplot(rows, columns, ax_idx).set_title(filename)

    img = cv2.imread(os.path.join(TEST_IMAGE_PATH, filename))

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    

    mask = rle2mask(row['EncodedPixels'], (256,1600))

    img[mask == 1, 0] = 255

            

    plt.imshow(img)

    ax_idx += 1

        

plt.show()

df_submit[['ImageId_ClassId', 'EncodedPixels']].to_csv('submission.csv', index=False)
#df_submit[['ImageId_ClassId']].head(20)

df_submit[['ImageId_ClassId']].shape[0]
df2 = pd.read_csv(DF_TEST_PATH)

df2.shape[0]
from IPython.display import FileLinks

FileLinks('.') # input argument is specified folder