import time

s_time = time.time()

import os



import sys

sys.path.append(os.path.abspath('../input/efficientnet/efficientnet-master/efficientnet-master/'))

from efficientnet import EfficientNetB3
import numpy as np

import pandas as pd

import scipy as sp





from keras.layers import Input, Dense, Dropout, GlobalAveragePooling2D, GlobalMaxPooling2D, Concatenate, BatchNormalization,PReLU

from keras import backend as K

from keras.models import Model, load_model

from keras.utils import Sequence

from keras.regularizers import l2

from keras.optimizers import Adam

from keras.callbacks import ReduceLROnPlateau, EarlyStopping



from tqdm import tqdm

import glob

from functools import partial

from multiprocessing import Pool

from matplotlib import pyplot as plt

from efficientnet import EfficientNetB3 , preprocess_input

from sklearn.metrics import cohen_kappa_score, mean_squared_error

from albumentations import HorizontalFlip, Compose, RandomRotate90, RandomBrightness, Resize,OneOf, VerticalFlip,Rotate, RandomBrightnessContrast



from sklearn.model_selection import StratifiedKFold

import cv2



import os

IMAGE_SIZE = 299

batch_size = 32

TEST_DIR = 'test'
data=pd.read_csv('../input/aptos2019-blindness-detection/test.csv')

#data['id_code'] = data['id_code'].apply(lambda x : os.path.join('test', x)+'.png')

subm = pd.read_csv('../input/aptos2019-blindness-detection/sample_submission.csv')

data.head()
## First, we calculate how we split our inference to several batches



if len(data) % 32 :

    split_count = len(data)//32 + 1

else :

    split_count = len(data) // 32

    

split_batch = np.array_split(np.arange(split_count) , 4)

batch_end =[min(len(data), (split_batch[i].max()+1)*batch_size) for i in range(len(split_batch))]

batch_start = [0] + batch_end[:-1]

batch_start, batch_end
def load_cropped(path , image_size = (224,224)):

    image = cv2.imread(path)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image = crop_image_from_gray(image)

    image = cv2.resize(image, image_size, interpolation = cv2.INTER_LANCZOS4)

    return image



def save_image(path , directory = 'test' , image_size = (224,224)) :

    filename = path.split('/')[-1]

    image = load_cropped(path , image_size)

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    new_path = os.path.join(directory,filename)

    cv2.imwrite(new_path, image)



    

##This function will do preprocessing and save file in a directory

def prepare_files(test_img_list, image_size = (224,224)) :



    save_test = partial(save_image , directory = TEST_DIR, image_size = image_size)

    if not(os.path.isdir(TEST_DIR)) :

        os.mkdir(TEST_DIR)

    with Pool(os.cpu_count()) as p :

        list(tqdm(p.imap(save_test, test_img_list), total=len(test_img_list)))



def crop_image_from_gray(img,tol=7):

    if img.ndim ==2:

        mask = img>tol

        return img[np.ix_(mask.any(1),mask.any(0))]

    elif img.ndim==3:

        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        mask = gray_img>tol

        

        check_shape = img[:,:,0][np.ix_(mask.any(1),mask.any(0))].shape[0]

        if (check_shape == 0): # image is too dark so that we crop out everything,

            return img # return original image

        else:

            img1=img[:,:,0][np.ix_(mask.any(1),mask.any(0))]

            img2=img[:,:,1][np.ix_(mask.any(1),mask.any(0))]

            img3=img[:,:,2][np.ix_(mask.any(1),mask.any(0))]

    #         print(img1.shape,img2.shape,img3.shape)

            img = np.stack([img1,img2,img3],axis=-1)

    #         print(img.shape)

        return img

    

def get_head(linear_size = [512,256,128] , probs_dropout = 0.5) :

    inp = Input(shape = (None,None,base_model.output_shape[-1]))

    x_avg = GlobalAveragePooling2D()(inp)

    x_max = GlobalMaxPooling2D()(inp)

    x = Concatenate()([x_avg,x_max])

    x = BatchNormalization()(x)

    x = Dropout(rate = probs_dropout)(x)

    x = Dense(linear_size[0], activation = 'tanh', kernel_regularizer = l2(1e-5))(x)

    #x = PReLU()(x)



    for n in linear_size[1:] :

        x = BatchNormalization()(x)

        x = Dropout(rate = probs_dropout)(x)

        x = Dense(n, kernel_regularizer = l2(1e-4))(x)

        x = PReLU()(x)

    x = Dense(1 , kernel_regularizer = l2(1e-4))(x)

    return Model(inp,x)
train_aug = Compose([

    #Resize(IMAGE_SIZE,IMAGE_SIZE),

    Rotate(360,border_mode = cv2.BORDER_CONSTANT, value = 0),

    OneOf([

        HorizontalFlip(),

        VerticalFlip()

    ]),

    RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3,p=1)

])
class ImageLoader(Sequence) :

    def __init__(self , image_list, 

                        image_cat = None, 

                        batch_size = 32, 

                        shuffle = True, 

                        include_last = True, 

                        transforms = None,

                        use_ben = True,

                        sigma_ben = 15, #Default value sigma is not randomized

                        randomized_sigma_ben_range = None, ##Apply randomized sigma for ben's preprocessing

                        crop = True,

                        resized = True) :



        self.image_list = np.array(image_list)

        self.crop = crop

        self.use_ben  = use_ben

        self.have_transform = False

        if self.use_ben :

            if not(randomized_sigma_ben_range is None) :

                self.sigma_ben_range = randomized_sigma_ben_range

                self.random_ben = True

            else :

                self.random_ben = False

                self.sigma_ben = sigma_ben

            

        if transforms is None :

            self.have_transform = False

            if not(resized) :

                self.have_transform = True

                self.transforms = Resize(IMAGE_SIZE,IMAGE_SIZE)

        else :

            self.have_transform = True 

            self.transforms = transforms

        

        if len(self.image_list) == 0 :

            print('List is empty please recheck')

        else :

            print('List contains {} images'.format(len(self.image_list)))

        #print(self.image_list)

        if image_cat is None :

            self.image_cat = None

        else :

            self.image_cat = np.array(image_cat)

            assert len(self.image_list) == len(self.image_cat) , 'Image List and Cat Mismatch'

            

        self.batch_size = batch_size 

        self.shuffle = shuffle

        self.index = np.arange(len(self.image_list), dtype = int)

        #print(type(image_list))

        if shuffle :

            self.shuffle_index()

        self.include_last = include_last

        

    def __len__(self) :

        if self.include_last :

            if len(self.image_list) % self.batch_size :

                return len(self.image_list) // self.batch_size + 1

            else :

                return len(self.image_list) // self.batch_size

        else :

            return len(self.image_list) // self.batch_size 

    

    def __getitem__(self , index) :

        

        if self.include_last and (index == (len(self.image_list) // self.batch_size)) :

            batch_count = len(self.image_list) % self.batch_size

        else :

            batch_count = self.batch_size 

        

        X = np.empty((batch_count,IMAGE_SIZE,IMAGE_SIZE,3))

        

        idxs = np.arange(index*self.batch_size, (index*self.batch_size)+batch_count )

        if self.use_ben :

            if self.random_ben :

                sigma = np.random.randint(self.sigma_ben_range[0], self.sigma_ben_range[1] , size = batch_count)

            else :

                sigma = [self.sigma_ben]*batch_count



        for i , idx in enumerate(idxs) :

            img = cv2.imread(self.image_list[idx])

            img = cv2.cvtColor(img , cv2.COLOR_BGR2RGB)

            if self.crop :

                img = crop_image_from_gray(img)

            if self.have_transform :

                img = self.transforms(image=img)['image']

            if self. use_ben :

                img = cv2.addWeighted (img,4, cv2.GaussianBlur(img, (0,0) , sigma[i]) ,-4 ,128)

            X[i] = img

            

        X = preprocess_input(X)

        if not(self.image_cat is None) :

            return X, self.image_cat[idxs]

        else :

            return X

    

    def shuffle_index(self) :

        np.random.shuffle(self.index)

            

        self.image_list = self.image_list[self.index]

            

        if not(self.image_cat is None) :

            self.image_cat = self.image_cat[self.index]

            

    def on_epoch_end(self) :

        if self.shuffle :

            self.shuffle_index()
models = []

K.clear_session()

for i in range(4) :

    base_model = EfficientNetB3(weights = None , include_top = False , input_shape = (IMAGE_SIZE,IMAGE_SIZE,3))

    head_model = get_head()

    model = Model(base_model.input, head_model(base_model.output))

    model.load_weights('../input/finetune-effnet/model_{}.h5'.format(i))

    models.append(model)
##Get the correct file path, from id

def get_path(path) :

    def corrector(x) :

        return os.path.join(path, x) + '.png'

    return corrector



##Function to clear directory where images are stored,

#rmdir is not necessary but for sanity check since if there are still any remains it will throw an error.

def clean_images() :

    if os.path.isdir('test') :

        for f in glob.glob('test/*.png') :

            os.remove(f)

        os.rmdir('test')
TTA_count = 8

preds = np.zeros([len(data),4]) ##Array storing prediction



for i, (batch_s, batch_e) in enumerate(zip(batch_start,batch_end)) :

    

    test_img_list = data.iloc[batch_s:batch_e,0].apply(get_path('../input/aptos2019-blindness-detection/test_images/')).tolist() 

    prepare_files(test_img_list, image_size = (IMAGE_SIZE,IMAGE_SIZE))

    

    test_img_list = data.iloc[batch_s:batch_e,0].apply(get_path('test')).tolist()

    

    loader = ImageLoader(test_img_list,

                         shuffle = False,

                         crop = False,

                         transforms=train_aug,

                         resized = True)

    for j in range(4) :

        for k in range(TTA_count) :

            preds[slice(batch_s,batch_e),j:j+1] += models[j].predict_generator(loader , 

                                                                              use_multiprocessing = True, 

                                                                              workers = os.cpu_count())/TTA_count

    ##Clean image after each batch inference job VERY IMPORTANT!

    clean_images()

    

    ##Sanity check that directory is cleaned

    try :

        print(len(os.listdir('test')))

    except :

        print('Batch {} Cleared'.format(i+1)) 

        print()
## Copied from predict method of optimized rounder

def get_labels(X , coef = [0.5,1.5,2.5,3.5]) :

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



## Predetermined coefficient, you can try optimize this value or use the default one

coef = [0.5353928773986922, 1.5749586701298883, 2.448495965240568, 3.393477738062221]
### +2 Since I trained my network to output values centered at 0

subm['diagnosis'] = get_labels(preds.mean(axis = 1) + 2, coef).astype(int)

subm.to_csv('submission.csv', index = False)
print('Kernel Runtime : {:.3f} minute'.format((time.time() - s_time) / 60 ))