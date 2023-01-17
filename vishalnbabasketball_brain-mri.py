import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
import tensorflow.compat.v1 as tf
#import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from albumentations import (
    PadIfNeeded,
    HorizontalFlip,
    VerticalFlip,    
    CenterCrop,    
    Crop,
    Compose,
    Transpose,
    RandomRotate90,
    ElasticTransform,
    GridDistortion, 
    OpticalDistortion,
    RandomSizedCrop,
    OneOf,
    CLAHE,
    RandomBrightnessContrast,    
    RandomGamma    
)
import random

#Get path to all the image and masks
image_path,mask_path=[],[]
main_path='../input/lgg-mri-segmentation/lgg-mri-segmentation/kaggle_3m/'
folders=os.listdir(main_path)
folders.remove('data.csv')
folders.remove('README.md')
for folder in tqdm(folders):
    tmp_path=os.path.join(main_path,folder)
    for file in os.listdir(tmp_path):
        if 'mask' in file.split('.')[0].split('_'):
            mask_path.append(os.path.join(tmp_path,file))
        else:
            image_path.append(os.path.join(tmp_path,file))
#Now lets check if we ther's any image without mask
img_wm=[]
for img_p in tqdm(image_path):
    img_p=img_p.split('.')
    img_p[2]=img_p[2]+'_mask'
    img_p='.'.join(img_p)
    if img_p not in mask_path:
        img_wm.append(img_p)
if len(img_wm)==0:
    print('All the images have masks')

del mask_path
mask_path=[]
for img_p in tqdm(image_path):
    img_p=img_p.split('.')
    img_p[2]=img_p[2]+'_mask'
    img_p='.'.join(img_p)
    mask_path.append(img_p)
    
#Lets plot some samples
rows,cols=3,3
fig=plt.figure(figsize=(10,10))
for i in range(1,rows*cols+1):
    fig.add_subplot(rows,cols,i)
    img_path=image_path[i]
    msk_path=mask_path[i]
    img=cv2.imread(img_path)
    img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    msk=cv2.imread(msk_path)
    plt.imshow(img)
    plt.imshow(msk,alpha=0.4)
plt.show()
tf.disable_eager_execution()
def lrelu(x,threshold=0.1):
    return tf.maximum(x,x*threshold)

def conv_layer(x,n_filters,k_size,stride,padding='SAME'):
    x=tf.layers.conv2d(x,filters=n_filters,kernel_size=k_size,strides=stride,padding=padding)
    x=tf.nn.relu(x)
    return x

def max_pool(x,pool_size):
    x=tf.layers.max_pooling2d(x,pool_size=pool_size)
    return x

def conv_transpose(x,n_filters,k_size,stride,padding='SAME'):
    x=tf.layers.conv2d_transpose(x,filters=n_filters,kernel_size=k_size,strides=stride,padding=padding)
    x=tf.nn.relu(x)
    return x


#Placeholders
image=tf.placeholder(tf.float32,[None,256,256,3],name='Input_image')
mask=tf.placeholder(tf.float32,[None,256,256,3],name='Image_mask')


#########################Beta Network

#Branch-0
layer_1=conv_layer(image,n_filters=64,k_size=4,stride=1)
mp_1=tf.layers.max_pooling2d(layer_1,pool_size=2,strides=2)

layer_2=conv_layer(mp_1,n_filters=128,k_size=4,stride=1)
mp_2=tf.layers.max_pooling2d(layer_2,pool_size=2,strides=2)

layer_3=conv_layer(mp_2,n_filters=256,k_size=4,stride=1)
mp_3=tf.layers.max_pooling2d(layer_3,pool_size=2,strides=2)

layer_4=conv_layer(mp_3,n_filters=512,k_size=4,stride=1)
mp_4=tf.layers.max_pooling2d(layer_4,pool_size=2,strides=2)

layer_5=conv_layer(mp_4,n_filters=1024,k_size=4,stride=1)
mp_5=tf.layers.max_pooling2d(layer_5,pool_size=2,strides=2)


#Branch_1
layer_b1=conv_layer(image,n_filters=128,k_size=4,stride=1)
mp_b1=tf.layers.max_pooling2d(layer_b1,pool_size=2,strides=2)

beta_1=tf.keras.layers.add([layer_2,mp_b1])

layer_b2=conv_layer(beta_1,n_filters=256,k_size=4,stride=1)
mp_b2=tf.layers.max_pooling2d(layer_b2,pool_size=2,strides=2)

beta_2=tf.keras.layers.add([layer_3,mp_b2])

layer_b3=conv_layer(beta_2,n_filters=512,k_size=4,stride=1)
mp_b3=tf.layers.max_pooling2d(layer_b3,pool_size=2,strides=2)

beta_3=tf.keras.layers.add([mp_b3,layer_4])

layer_b4=conv_layer(beta_3,n_filters=1024,k_size=4,stride=1)
mp_b4=tf.layers.max_pooling2d(layer_b4,pool_size=2,strides=2)

beta_4=tf.keras.layers.add([mp_b4,layer_5])

beta_0=layer_1
########################################################




#Downsample
#64
x_layer_1=conv_layer(image,n_filters=64,k_size=5,stride=1)
x_layer_1=conv_layer(x_layer_1,n_filters=64,k_size=4,stride=1)
x_layer_1=conv_layer(x_layer_1,n_filters=64,k_size=4,stride=2)
x_batch_1=tf.layers.batch_normalization(x_layer_1)#128x128x64

#128
x_layer_2=conv_layer(x_batch_1,n_filters=128,k_size=5,stride=1)
x_layer_2=conv_layer(x_layer_2,n_filters=128,k_size=4,stride=1)
x_layer_2=conv_layer(x_layer_2,n_filters=128,k_size=4,stride=2)
x_batch_2=tf.layers.batch_normalization(x_layer_2)#64x64x128

#256
x_layer_3=conv_layer(x_batch_2,n_filters=256,k_size=5,stride=1)
x_layer_3=conv_layer(x_layer_3,n_filters=256,k_size=4,stride=1)
x_layer_3=conv_layer(x_layer_3,n_filters=256,k_size=4,stride=2)
x_batch_3=tf.layers.batch_normalization(x_layer_3)#32x32x256

#512
x_layer_4=conv_layer(x_batch_3,n_filters=512,k_size=5,stride=1)
x_layer_4=conv_layer(x_layer_4,n_filters=512,k_size=4,stride=1)
x_layer_4=conv_layer(x_layer_4,n_filters=512,k_size=4,stride=2)
x_batch_4=tf.layers.batch_normalization(x_layer_4)#16x16x512

#1024
x_layer_5=conv_layer(x_batch_4,n_filters=1024,k_size=4,stride=1)
x_layer_5=conv_layer(x_layer_5,n_filters=1024,k_size=4,stride=8)
x_batch_5=tf.layers.batch_normalization(x_layer_5)#8x8x1024


#Upsample
#1024
y_layer_1=conv_transpose(x_batch_5,n_filters=1024,k_size=4,stride=8)
y_layer_1=tf.keras.layers.add([y_layer_1,beta_4])
y_layer_1=conv_layer(y_layer_1,n_filters=1024,k_size=4,stride=1)
y_batch_1=tf.layers.batch_normalization(y_layer_1)


#512
y_layer_2=conv_transpose(y_batch_1,n_filters=512,k_size=5,stride=2)
y_layer_2=tf.keras.layers.add([y_layer_2,beta_3])
y_layer_2=conv_layer(y_layer_2,n_filters=512,k_size=4,stride=1)
y_layer_2=conv_layer(y_layer_2,n_filters=512,k_size=4,stride=1)
y_batch_2=tf.layers.batch_normalization(y_layer_2)

#256
y_layer_3=conv_transpose(y_batch_2,n_filters=256,k_size=5,stride=2)
y_layer_3=tf.keras.layers.add([y_layer_3,beta_2])
y_layer_3=conv_layer(y_layer_3,n_filters=256,k_size=4,stride=1)
y_layer_3=conv_layer(y_layer_3,n_filters=256,k_size=4,stride=1)
y_batch_3=tf.layers.batch_normalization(y_layer_3)


#128
y_layer_4=conv_transpose(y_batch_3,n_filters=128,k_size=3,stride=2)
y_layer_4=tf.keras.layers.add([y_layer_4,beta_1])
y_layer_4=conv_layer(y_layer_4,n_filters=128,k_size=2,stride=1)
y_layer_4=conv_layer(y_layer_4,n_filters=128,k_size=2,stride=1)
y_batch_4=tf.layers.batch_normalization(y_layer_4)

#64
y_layer_5=conv_transpose(y_batch_4,n_filters=64,k_size=2,stride=2)
y_layer_5=tf.keras.layers.add([y_layer_5,beta_0])
y_layer_5=conv_layer(y_layer_5,n_filters=64,k_size=1,stride=1)
y_layer_5=conv_layer(y_layer_5,n_filters=64,k_size=1,stride=1)
y_batch_5=tf.layers.batch_normalization(y_layer_5)

#Output
logits=tf.layers.conv2d(y_batch_5,activation=None,filters=3,kernel_size=1,strides=1,padding='SAME')
out=tf.nn.sigmoid(logits)
#Loss and Optimizer
loss=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=mask,logits=logits))
train_opt=tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)
train_images,val_images,train_masks,val_masks=train_test_split(image_path,mask_path,test_size=0.2)
train_loss_list=[]
val_loss_list=[]
EPOCHS=50
BATCH_SIZE=25
train_batches=int(len(train_images)//BATCH_SIZE)
val_batches=int(len(val_images)//BATCH_SIZE)
saver=tf.train.Saver()
def read_images(image_path,mask_path,is_train):
    tmp_images=[]
    tmp_masks=[]
    aug = Compose([VerticalFlip(p=0.5), 
                   RandomRotate90(p=0.5),
                   GridDistortion(p=1),
                   Transpose(p=1)])
    
    for img,msk in zip(image_path,mask_path):
        img=cv2.imread(img)
        msk=cv2.imread(msk)
        
        #Apply Augmentation
        if is_train:
            augmented = aug(image=img, mask=msk)
            img = augmented['image']
            msk = augmented['mask']
        tmp_images.append(img.astype('float')/255.)
        tmp_masks.append(msk.astype('float')/255.)  
        
    tmp_images=np.reshape(tmp_images,(BATCH_SIZE,256,256,3))
    tmp_masks=np.reshape(tmp_masks,(BATCH_SIZE,256,256,3))
    return tmp_images,tmp_masks
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    tmp_v_loss=0
    for epoch in range(EPOCHS):
        print('EPOCH: {}/{}'.format(epoch+1,EPOCHS))
        #Training
        for batch_ix in tqdm(range(train_batches)):
            img_ix=train_images[batch_ix*BATCH_SIZE:(batch_ix+1)*BATCH_SIZE]
            msk_ix=train_masks[batch_ix*BATCH_SIZE:(batch_ix+1)*BATCH_SIZE]
            X,y=read_images(img_ix,msk_ix,is_train=True)
            t_loss,_=sess.run([loss,train_opt],feed_dict={image:X,mask:y})
        
        #Validation
        for batch_ix in tqdm(range(val_batches)):
            img_ix=val_images[batch_ix*BATCH_SIZE:(batch_ix+1)*BATCH_SIZE]
            msk_ix=val_masks[batch_ix*BATCH_SIZE:(batch_ix+1)*BATCH_SIZE]
            X,y=read_images(img_ix,msk_ix,is_train=False)
            v_loss=sess.run(loss,feed_dict={image:X,mask:y})
        
        print('Training Loss: {}'.format(t_loss))
        print('Validation Loss: {}'.format(v_loss))
        train_loss_list.append(t_loss)
        val_loss_list.append(v_loss)
        
        #Save model 
        if epoch==0:
            print('Loss improved from {} to {}'.format(0,v_loss))
            saver.save(sess,'seg_model.ckpt')
            tmp_v_loss=v_loss
            
        elif v_loss<tmp_v_loss:
            print('Loss improved from {} to {}'.format(tmp_v_loss,v_loss))
            saver.save(sess,'seg_model.ckpt')
            tmp_v_loss=v_loss
        print('==============================================')
        
    #Print Samples
    preds=sess.run(out,feed_dict={image:X})
    rows,cols=3,3
    fig=plt.figure(figsize=(10,10))
    for i in range(1,rows*cols+1):
        fig.add_subplot(rows,cols,i)
        plt.imshow(preds[i])
    plt.show()    
epochs=range(1,len(train_loss_list)+1)
plt.plot(epochs,train_loss_list,'b',color='red',label='Training Loss')
plt.plot(epochs,val_loss_list,'b',color='blue',label='Validation Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()
#Randomaly select 10 ima from validation dataset
val_indexes=np.arange(len(val_images))
val_image_indexes=random.choices(val_indexes,k=10)
tmp_val=np.zeros((10,256,256,3))
for ix,ix_ in enumerate(val_image_indexes):
    img=cv2.imread(val_images[ix_])
    img=img.astype('float')/255.
    tmp_val[ix]=img

with tf.Session() as sess:
    saver.restore(sess,'seg_model.ckpt')
    print('Model restored')
    preds=sess.run(out,feed_dict={image:tmp_val})


#Plot true and predicted side by side
rows,cols=4,2
fig=plt.figure(figsize=(10,10))
for i in range(1,rows*cols+1):
    fig.add_subplot(rows,cols,i)
    if (i-1)%2==0:
        img=cv2.imread(val_masks[val_indexes[i-1]])
    else:
        img=preds[i-1]
    plt.imshow(img)
plt.show()