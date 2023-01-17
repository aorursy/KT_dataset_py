# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import zipfile

import h5py

from keras.optimizers import Adam

import cv2

from keras.utils import to_categorical

import glob, os

from matplotlib import pyplot as plt

import h5py

from sklearn.metrics import accuracy_score

import numpy as np

from tqdm import tqdm

import time

import gc

from keras.applications import *

from keras.layers import *

from keras import backend as K

from keras.models import Model
lbl=[]

img=np.zeros((3064,224,224))

for i in range(1,3065):

    try:

        path='/kaggle/input/brain-tumour/brainTumorDataPublic_1766/'

        with h5py.File(path+str(i)+'.mat') as f:

          images = f['cjdata']

          resized = cv2.resize(images['image'][:,:], (224,224), interpolation = cv2.INTER_CUBIC )

          x=np.asarray(resized)

          x=(x-np.min(x))/(np.max(x)-np.min(x))

          x=x.reshape((1,224,224))

          img[i-1]=x

          lbl.append(int(images['label'][0]))

    except:

        try:

          path='/kaggle/input/brain-tumour/brainTumorDataPublic_22993064/'

          with h5py.File(path+str(i)+'.mat') as f:

              images = f['cjdata']

              resized = cv2.resize(images['image'][:,:], (224,224), interpolation = cv2.INTER_CUBIC )

              x=np.asarray(resized)

              x=(x-np.min(x))/(np.max(x)-np.min(x))

              x=x.reshape((1,224,224))

              img[i-1]=x

              lbl.append(int(images['label'][0]))

        except:

            try:

              path='/kaggle/input/brain-tumour/brainTumorDataPublic_15332298/'

              with h5py.File(path+str(i)+'.mat') as f:

                  images = f['cjdata']

                  resized = cv2.resize(images['image'][:,:], (224,224), interpolation = cv2.INTER_CUBIC )

                  x=np.asarray(resized)

                  x=(x-np.min(x))/(np.max(x)-np.min(x))

                  x=x.reshape((1,224,224))

                  img[i-1]=x

                  lbl.append(int(images['label'][0]))

            except:

              path='/kaggle/input/brain-tumour/brainTumorDataPublic_7671532/'

              with h5py.File(path+str(i)+'.mat') as f:

                  images = f['cjdata']

                  resized = cv2.resize(images['image'][:,:], (224,224), interpolation = cv2.INTER_CUBIC )

                  x=np.asarray(resized)

                  x=(x-np.min(x))/(np.max(x)-np.min(x))

                  x=x.reshape((1,224,224))

                  img[i-1]=x

                  lbl.append(int(images['label'][0]))



path='/kaggle/input/braintumour/cvind (2).mat'



with h5py.File(path) as f:

      data=f['cvind']

      idx=data[0]

dk={}

dk['images']=img

dk['fold']=idx

dk['label']=lbl

np.save('final.npy',dk)

df=np.load('final.npy',allow_pickle=True)

df=df.item()
del([dk])

gc.collect()
del([images,img,lbl,x,resized])

gc.collect()
def unison_shuffled_copies(a, b):

    assert len(a) == len(b)

    p = np.random.permutation(len(a))

    return a[p], b[p]









#get train and test splits

def get_trn_tst(df,tst_fold):

  idx=np.asarray(df['fold'])

  y=np.asarray(df['label'])

  y-=1

  img=np.asarray(df['images'])

  gc.collect()

  trn_y=np.asarray(y[(idx!=tst_fold)])

  trn_img=np.asarray(img[(idx!=tst_fold)])

  tst_y=np.asarray(y[(idx==tst_fold)])

  tst_img=img[idx==tst_fold]

  trn_img=np.repeat(trn_img.reshape((trn_img.shape[0],224,224,1)),3,axis=3)

  tst_img=np.repeat(tst_img.reshape((tst_img.shape[0],224,224,1)),3,axis=3)

  return (trn_img.copy(),trn_y.copy()),(tst_img.copy(),tst_y.copy())

mod=densenet.DenseNet121(include_top=True, weights='imagenet')

mod.summary()
def load_model(last=True):   

  K.clear_session()

  mod=densenet.DenseNet121(include_top=True, weights='imagenet')

  out_1=mod.layers[-2]

  out=Dense(3,activation='softmax')(out_1.output)

  model=Model(inputs=mod.input,outputs=out)

  if last:

    for i in range(len(model.layers)):

        model.layers[i].trainable = False

  model.layers[-1].trainable=True

  return model
best_accuracy_last={}

final_accuracy_last={}

history_last={}

answers_last={}

times_last={}
def upd(dk,data):

    if dk==0:

        dk=data

    else:

        for ky in data.keys():

            dk[ky].extend(data[ky])

    return dk

for index in tqdm(range(1,6)):

  epoch=50

  pre_acc=0

  best=0

  fold='fold_'+str(index)

  trn,tst=get_trn_tst(df,index)

  history_last[fold]=0

  

  plt.imshow(trn[0][0])

  plt.show()

  plt.imshow(tst[0][0])

  plt.show()







  trn_x,trn_y=unison_shuffled_copies(trn[0],trn[1])

  tst_x,tst_y=unison_shuffled_copies(tst[0],tst[1])

  





  model=load_model()





  

  #compiling the model

  model.compile(optimizer=Adam(3e-4), 

                     loss='categorical_crossentropy', 

                     metrics=['accuracy'])

  

  

  #fitting the model

  #timing

  start=time.time()

  for i in range(epoch):

      hist=model.fit(trn_x,to_categorical(trn_y),batch_size=32,epochs=1,validation_data=[tst_x,to_categorical(tst_y)])

      pre=model.predict(tst_x)

      pre=np.argmax(pre,1)

      new_acc=accuracy_score(pre,tst_y)

      if new_acc>best:

            best_accuracy_last[fold]=new_acc

            best=new_acc

      

      #storing the result

      history_last[fold]=upd(history_last[fold],hist.history)

  end=time.time()

  times_last[fold]=end-start





  #getting the prediction 

  pre=model.predict(tst_x)

  







  #select the maximum position

  pre=np.argmax(pre,1)



  

  

  

  #getting the accuracy

  new_acc=accuracy_score(pre,tst_y)



  





  #storing the predictions

  final_accuracy_last[fold]=new_acc



  





















  #storing the answers

  answers_last[fold]=tst_y

    

    

    

    

  #freeing memory

  del([trn,tst,trn_x,trn_y,tst_x,tst_y,model])

  gc.collect()
np.save('best_accuracy_last.npy',best_accuracy_last)

np.save('final_accuracy_last.npy',final_accuracy_last)

np.save('fistory_last.npy',history_last)

np.save('answers_last.npy',answers_last)

np.save('times_last.npy',times_last)