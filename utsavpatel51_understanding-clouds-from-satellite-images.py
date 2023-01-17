import numpy as np

import pandas as pd

import cv2

import matplotlib.pyplot as plt

from keras.preprocessing.image import load_img,img_to_array,array_to_img

import keras

from keras.applications.xception import Xception

from keras import layers

from keras.models import Model

from keras.optimizers import Adam

from sklearn.model_selection import train_test_split

from keras.callbacks import ModelCheckpoint

import scipy

from tqdm import tqdm_notebook
def plot_curve(history):

    fig,axs = plt.subplots(1,2,figsize=(15,5))

    axs[0].plot(history.history['accuracy'])

    axs[0].plot(history.history['val_accuracy'])

    axs[0].set_title('model accuracy')

    axs[0].set_ylabel('accuracy')

    axs[0].set_xlabel('epoch')

    axs[0].legend(['train', 'test'], loc='upper left')

    # summarize history for loss

    axs[1].plot(history.history['loss'])

    axs[1].plot(history.history['val_loss'])

    axs[1].set_title('model loss')

    axs[1].set_ylabel('loss')

    axs[1].set_xlabel('epoch')

    axs[1].legend(['train', 'test'], loc='upper left')

    plt.show()
train_df = pd.read_csv('../input/understanding_cloud_organization/train.csv')

train_df.head()
train_df['Image'] = train_df['Image_Label'].apply(lambda x:x.split('_')[0])

train_df['label'] = train_df['Image_Label'].apply(lambda x:x.split('_')[1])

train = pd.DataFrame({'Image':train_df['Image'][::4]})

train['e1'] = train_df['EncodedPixels'][::4].values

train['e2'] = train_df['EncodedPixels'][1::4].values

train['e3'] = train_df['EncodedPixels'][2::4].values

train['e4'] = train_df['EncodedPixels'][3::4].values



train.set_index('Image',inplace=True,drop=True)



train.fillna('',inplace=True)



categoty = ['c1','c2','c3','c4']

train[categoty] = (train[['e1','e2','e3','e4']]!='').astype('int8')
train.head()
def rle2maskX(mask_rle, shape=(2100,1400), shrink=1):

    # Converts rle to mask size shape then downsamples by shrink

    s = mask_rle.split()

    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]

    starts -= 1

    ends = starts + lengths

    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)

    for lo, hi in zip(starts, ends):

        img[lo:hi] = 1

    return img.reshape(shape).T[::shrink,::shrink]

def mask2rle(img, shape=(525,350)):    

    pixels= img.T.flatten()

    pixels = np.concatenate([[0], pixels, [0]])

    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1

    runs[1::2] -= runs[::2]

    return ' '.join(str(x) for x in runs)



class DataGenerator(keras.utils.Sequence):

    'Generates data for Keras'

    def __init__(self, list_IDs, batch_size=8, shuffle=False, width=512, height=352, scale=1/128., sub=1., mode='train',

                 path='../input/understanding_cloud_organization/train_images/', flips=False):

        'Initialization'

        self.list_IDs = list_IDs

        self.shuffle = shuffle

        self.batch_size = batch_size

        self.path = path

        self.scale = scale

        self.sub = sub

        self.path = path

        self.width = width

        self.height = height

        self.mode = mode

        self.flips = flips

        self.on_epoch_end()

        

    def __len__(self):

        'Denotes the number of batches per epoch'

        ct = int(np.floor( len(self.list_IDs) / self.batch_size))

        if len(self.list_IDs)>ct*self.batch_size: ct += 1

        return int(ct)



    def __getitem__(self, index):

        'Generate one batch of data'

        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        X, y = self.__data_generation(indexes)

        if (self.mode=='train')|(self.mode=='validate'): return X, y

        else: return X



    def on_epoch_end(self):

        'Updates indexes after each epoch'

        self.indexes = np.arange(int( len(self.list_IDs) ))

        if self.shuffle: np.random.shuffle(self.indexes)



    def __data_generation(self, indexes):

        'Generates data containing batch_size samples' 

        # Initialization

        lnn = len(indexes)

        X = np.empty((lnn,self.height,self.width,3),dtype=np.float32)

        y = np.zeros((lnn,4),dtype=np.int8)

        

        # Generate data

        for k in range(lnn):

            img = cv2.imread(self.path + self.list_IDs[indexes[k]])

            img = cv2.resize(img,(self.width,self.height),interpolation = cv2.INTER_AREA)

            # AUGMENTATION FLIPS

            hflip = False; vflip = False

            if (self.flips):

                if np.random.uniform(0,1)>0.5: hflip=True

                if np.random.uniform(0,1)>0.5: vflip=True

            if vflip: img = cv2.flip(img,0) # vertical

            if hflip: img = cv2.flip(img,1) # horizontal

            # NORMALIZE IMAGES

            X[k,] = img*self.scale - self.sub      

            # LABELS

            if (self.mode=='train')|(self.mode=='validate'):

                y[k,] = train.loc[self.list_IDs[indexes[k]],['c1','c2','c3','c4']].values

        return X, y
id_train,id_val = train_test_split(train.index,random_state=21,test_size=0.2)

train_gen = DataGenerator(id_train,shuffle=True,flips=True)

val_gen = DataGenerator(id_val,mode='validate')
def plotImages(images_arr):

    fig, axes = plt.subplots(1, 5, figsize=(20,20))

    axes = axes.flatten()

    for img, ax in zip( images_arr, axes):

        ax.imshow(img)

    plt.tight_layout()

    plt.show()

augmented_images = [train_gen[0][0][0] for i in range(5)]

plotImages(augmented_images)
'''xception_model = Xception(weights='imagenet',include_top=False,input_shape=(None,None,3))

for layer in xception_model.layers:

    if not isinstance(layer,layers.BatchNormalization):

        layer.trainable = False

x = xception_model.output

x = layers.GlobalAveragePooling2D()(x)

x = layers.Dense(4,activation='sigmoid')(x)



model = Model(inputs=xception_model.input,outputs=x)



model.compile(optimizer=Adam(learning_rate=0.001),loss='binary_crossentropy',metrics=['accuracy'])

model.summary()'''
'''checkpoint = ModelCheckpoint('best_xception_model.h5', monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')

callbacks_list = [checkpoint]'''
'''history1 = model.fit_generator(train_gen,epochs=5,verbose=1,validation_data=val_gen,callbacks=callbacks_list)'''
'''plot_curve(history1)'''
from keras.models import load_model

model = load_model('../input/xception-model/best_xception_model.h5')

model.summary()
train1 = train.loc[train.index.isin(id_val)].copy()

test_local_gen = DataGenerator(train1.index.values, mode='predict')

pred= model.predict_generator(test_local_gen, verbose=2)
val_gen = DataGenerator(id_val,mode='validate')
test_df = pd.read_csv('../input/understanding_cloud_organization/sample_submission.csv')
test_df.head()

test_df['Image'] = test_df['Image_Label'].apply(lambda x:x.split('_')[0])

test_df['label'] = test_df['Image_Label'].apply(lambda x:x.split('_')[1])

test_df.head()
test = pd.DataFrame({'Image':test_df['Image'][::4]})

test.set_index('Image',inplace=True,drop=True)

test.head()
all_layer_weights = model.layers[-1].get_weights()[0]

cam_model = Model(inputs=model.input, 

        outputs=(model.layers[-3].output, model.layers[-1].output))
cam_model.output
for i in range(1,5): test['p'+str(i)] = ''

for i in range(1,5): test['pp'+str(i)] = 0
PATH='../input/understanding_cloud_organization/test_images/'

for i,f in tqdm_notebook(enumerate(test.index.values),total=len(test)):

    img = cv2.resize( cv2.imread(PATH+f), (512, 352))

    x = np.expand_dims(img, axis=0)/128. -1.

    last_conv_output, pred_vec = cam_model.predict(x) 

    last_conv_output = np.squeeze(last_conv_output)

    

    for pred in [0,1,2,3]:

        # CREATE FOUR MASKS FROM ACTIVATION MAPS

        layer_weights = all_layer_weights[:, pred]  

        final_output = np.dot(last_conv_output.reshape((16*11, 2048)), layer_weights).reshape(11,16) 

        final_output = scipy.ndimage.zoom(final_output, (32, 32), order=1)

        mx = np.round( np.max(final_output),1 )

        mn = np.round( np.min(final_output),1 )

        final_output = (final_output-mn)/(mx-mn)

        final_output = cv2.resize(final_output,(525,350))

        test.loc[f,'p'+str(pred+1)] = mask2rle( (final_output>0.3).astype(int) )

        test.loc[f,'pp'+str(pred+1)] = pred_vec[0,pred]
test
df  = pd.DataFrame(columns=['Image_Label','EncodedPixels'])
im=[]

pi=[]
for id_,row in enumerate(test.itertuples()):

        #-----------------

        label = 'Fish'

        im.append(row.Index+'_'+label)

        if row.pp1>0.75:

            pi.append(row.p1)

        else:

            pi.append(np.nan)

        #-----------------

        label = 'Flower'

        im.append(row.Index+'_'+label)

        if row.pp2>0.75:

            pi.append(row.p2)

        else:

            pi.append(np.nan)

        #-----------------

        label = 'Gravel'

        im.append(row.Index+'_'+label)

        if row.pp3>0.75:

            pi.append(row.p3)

        else:

            pi.append(np.nan)

        #-----------------

        label = 'Sugar'

        im.append(row.Index+'_'+label)

        if row.pp4>0.75:

            pi.append(row.p4)

        else:

            pi.append(np.nan)
df['Image_Label'] = im

df['EncodedPixels'] = pi
df.head()
from IPython.display import HTML

import pandas as pd

import numpy as np

import base64



# function that takes in a dataframe and creates a text link to  

# download it (will only work for files < 2MB or so)

def create_download_link(df, title = "Download CSV file", filename = "submit.csv"):  

    csv = df.to_csv(index=None)

    b64 = base64.b64encode(csv.encode())

    payload = b64.decode()

    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'

    html = html.format(payload=payload,title=title,filename=filename)

    return HTML(html)
create_download_link(df)
df.to_csv(r'submit.csv')
from IPython.display import FileLink

FileLink(r'submit.csv')