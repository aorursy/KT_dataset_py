# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import cv2

% matplotlib inline

import gc

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os

print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
from keras.applications import VGG16

from keras import backend as K



# Load the pre-trained VGG16 model

model = VGG16(weights='imagenet', include_top=False)

print(model.summary())
def deprocess_image(x):

    # normalize tensor: center on 0., ensure std is 0.1

    x -= x.mean()

    x /= (x.std() + 1e-5)

    x *= 0.2



    # clip to [0, 1]

    x += 0.5

    x = np.clip(x, 0, 1)



    # convert to RGB array

    x *= 255

    x = np.clip(x, 0, 255).astype('uint8')

    return x

def rescale(img,rescale_factor):

    """

    Rescale image given rescale factor

    """

    rescale_size = (int(img.shape[1]*rescale_factor),int(img.shape[2]*rescale_factor))

    rescaled_img = np.empty((1,rescale_size[0],rescale_size[1],3))

    rescaled_img[0,:,:,0] = cv2.resize(img[0,:,:,0], rescale_size, cv2.INTER_AREA)

    rescaled_img[0,:,:,1] = cv2.resize(img[0,:,:,1], rescale_size, cv2.INTER_AREA)

    rescaled_img[0,:,:,2] = cv2.resize(img[0,:,:,2], rescale_size, cv2.INTER_AREA)

    

    return rescaled_img

def shift(img):

    """

    Randomly Shift image 0.1-0.3*width

    """

    shifted_img = np.empty(img.shape)

    shift_pixel = np.random.randint(int(0.1*img.shape[1]),int(0.3*img.shape[1]))

    direction = [1 if np.random.rand()>0.5 else -1]

    M = np.float32([[1,0,direction[0]*shift_pixel],[0,1,0]])

    shifted_img[0,:,:,0] = cv2.warpAffine(img[0,:,:,0],M,img.shape[1:3])

    shifted_img[0,:,:,1] = cv2.warpAffine(img[0,:,:,1],M,img.shape[1:3])

    shifted_img[0,:,:,2] = cv2.warpAffine(img[0,:,:,2],M,img.shape[1:3])

    return shifted_img

def recover_details(original_img, shrunk_img, shape, dreamed_img):

    """

    Use the difference between orignal image to the scaled image to restore details

    """

    eta = 0.9995

    diff = np.empty((1,shape[0],shape[1],3))

    diff[0,:,:,0] = cv2.resize(original_img[0,:,:,0],shape,cv2.INTER_AREA) - cv2.resize(shrunk_img[0,:,:,0],shape,cv2.INTER_AREA)

    diff[0,:,:,1] = cv2.resize(original_img[0,:,:,1],shape,cv2.INTER_AREA) - cv2.resize(shrunk_img[0,:,:,1],shape,cv2.INTER_AREA)

    diff[0,:,:,2] = cv2.resize(original_img[0,:,:,2],shape,cv2.INTER_AREA) - cv2.resize(shrunk_img[0,:,:,2],shape,cv2.INTER_AREA)

    

    return eta*dreamed_img +(1-eta)*diff



def DreamProcess(layers_name,

                 image = np.random.random((1, 300, 300, 3)) * 20 + 128., 

                 size=(300,300),

                 noise=False):

    """

    Get layers outputs and loss and iteration function

    """

    loss = K.variable(0.)

    # Define the loss and grads implementation

    for layer in layers_name.keys(): 

        weight = layers_name[layer]

        layer_output = model.get_layer(layer).output

        scaling = K.prod(K.cast(K.shape(layer_output),'float32'))

        loss += weight*K.sum(K.square(layer_output[:, :, :, :]) )/scaling

    grads = K.gradients(loss, model.input)[0]

    #L-1 gradient normalization

    grads /= K.maximum(K.mean(K.abs(grads)), 1e-5)   # L-1 gradient normalization

    # Retrieve loss and gradients given input image

    iterate = K.function([model.input], [loss, grads])

    print('.zZ')

    """

    Run gradient ascent

    """

    def gradientAscent(img, iters=10, step = 0.01):

        for i in range(iters):

            # Dream patterns

            loss_value, grads_value = iterate([img])

            img += grads_value * step

        return img

    """

    Dream process with rescaling/shifting

    """

    original_img = image.copy()

    random_repeat = 6

    fig,ax = plt.subplots(1,random_repeat,figsize=(18,6))

    plt.subplots_adjust(wspace=0, hspace=0)

    # Shrink image as the begining image

    shrunk_img  = rescale(original_img,0.1)

    for i in range(random_repeat):

        print('Repeat at {0} size: {1}'.format(i,shrunk_img.shape[1:3]))

        # scaling

        resized_img = rescale(shrunk_img,1.4)

        # gradient ascent

        dreamed_img = gradientAscent(resized_img)

        # recover details

        shrunk_img = recover_details(original_img, shrunk_img, resized_img.shape[1:3], dreamed_img)

        

        ax[i].imshow(cv2.cvtColor(deprocess_image(shrunk_img[0]), cv2.COLOR_BGR2RGB))

        ax[i].axis('off')

        ax[i].set_aspect('equal')

        ax[i].set_title('Repetition {0}'.format(i+1),fontsize=16)

        

    """

    Some visualizations

    """

    fig,ax = plt.subplots(1,2, figsize=(18,12))

    ax[0].imshow(cv2.cvtColor(deprocess_image(original_img[0]), cv2.COLOR_BGR2RGB))

    ax[1].imshow(cv2.cvtColor(deprocess_image(shrunk_img[0]), cv2.COLOR_BGR2RGB))

    ax[0].set_title('Original',fontsize=20)

    ax[1].set_title('Dream',fontsize=20)

    for a in ax:

        a.axis('off')

        a.set_aspect('equal')

    return deprocess_image(shrunk_img[0])
layers_name = {'block5_conv2':2,'block2_conv2':0.1}
%%time

image= cv2.resize(cv2.imread('../input/sky-image/sky.JPG'),(640,640), cv2.INTER_AREA)

img = np.empty((1,image.shape[0],image.shape[1],3))

img[0,:,:,:] = image

final_img = DreamProcess(layers_name,

             image = img, 

             size=(img.shape[1],img.shape[2]))
%%time

image= cv2.resize(cv2.imread('../input/myportrait/Li_Zheng-brown.jpg'),(640,640), cv2.INTER_AREA)

img[0,:,:,:] = image

final_img = DreamProcess(layers_name,

             image = img, 

             size=(img.shape[1],img.shape[2]))
%%time

image= cv2.resize(cv2.imread('../input/sky-image/sky.JPG'),(640,640), cv2.INTER_AREA)

image = randomNoise(image)

img = np.empty((1,image.shape[0],image.shape[1],3))

img[0,:,:,:] = image

final_img = DreamProcess(layers_name,

             image = img, 

             size=(img.shape[1],img.shape[2]))
%%time

image= cv2.resize(cv2.imread('../input/myportrait/Li_Zheng-brown.jpg'),(640,640), cv2.INTER_AREA)

image = randomNoise(image)

img[0,:,:,:] = image

final_img = DreamProcess(layers_name,

             image = img, 

             size=(img.shape[1],img.shape[2]))