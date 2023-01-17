# Listing all the imports

# ! pip install imutils

! mkdir /kaggle/working/trained_images/

! rm -rf /kaggle/working/trained_images/

! mkdir /kaggle/working/trained_images/

!pip install git+https://github.com/qubvel/efficientnet

import cv2

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import os

import time

# import imutils

import math
! ls /kaggle/working

! ls /kaggle/working/trained_images

!ls -l /kaggle/working/trained_images/ | egrep -c '^-' 
# image height and image width ----> GLOBAL

img_ht = 256

img_wd = 256



def displayImage(display_name, image):

    cv2.namedWindow(display_name,cv2.WINDOW_AUTOSIZE)

    cv2.imshow(display_name, image)



def findContourEye(thresh_image):

    cnts = cv2.findContours(thresh_image.copy(), cv2.RETR_EXTERNAL,

	cv2.CHAIN_APPROX_SIMPLE)

#     cnts = imutils.grab_contours(cnts)

    cnts = max(cnts[0], key=cv2.contourArea)

    return cnts



def findContourEyeExtreme(cnts):

    # Locating extreme points on all 4 sides

    leftmost = tuple(cnts[cnts[:,:,0].argmin()][0])

    rightmost = tuple(cnts[cnts[:,:,0].argmax()][0])

    topmost = tuple(cnts[cnts[:,:,1].argmin()][0])

    bottommost = tuple(cnts[cnts[:,:,1].argmax()][0])

    # Locating the top left and bottom right corner

    x1 = leftmost[0]

    y1 = topmost[1]

    x2 = rightmost[0]

    y2 = bottommost[1]

    return x1,y1,x2,y2 



def findRadiusAndCentreOfContourEye(cnts):

    M = cv2.moments(cnts)

    if( M["m00"]==0):

        cX, cY = 0, 0

    else:

        cX = int(M["m10"] / M["m00"])

        cY = int(M["m01"] / M["m00"])

    if(cX < cY):

        r = cX

    else:

        r = cY

    return cX,cY,r



def drawCentreOnContourEye(image,cnts,cX,cY):

    cv2.drawContours(image, [cnts], -1, (0, 255, 0), 2)

    cv2.circle(image, (cX, cY), 7, (255, 255, 255), -1)

    cv2.putText(image, "center", (cX - 20, cY - 20),

    cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2)

    return image

    

def Radius_Reduction(img,cX,cY,r):

    h,w,c=img.shape

    Frame=np.zeros((h,w,c),dtype=np.uint8)

    cv2.circle(Frame,(int(cX),int(cY)),int(r), (255,255,255), -1)

    Frame1=cv2.cvtColor(Frame, cv2.COLOR_BGR2GRAY)

    img1 =cv2.bitwise_and(img,img,mask=Frame1)

    return img1



def imageResize(image, ht, wd):

    # resized_image = imutils.resize(image, height = ht, width = wd)

    resized_image = cv2.resize(image,(wd,ht))

    return resized_image



def crop_black(image):

    org = image.copy()

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    thresh = cv2.threshold(blurred, 10, 255, cv2.THRESH_BINARY)[1]

    # displayImage('thresh',thresh)

    cnts = findContourEye(thresh)

    x1,y1,x2,y2 = findContourEyeExtreme(cnts)

    # print(x1,y1,x2,y2)

    crop = org[y1:y2, x1:x2]

    crop = imageResize(crop, img_ht, img_wd)

    # displayImage("cr1",crop)

    return crop



def imageAugmentation(image):

    x_flip = cv2.flip( image, 0 )

    y_flip = cv2.flip( image, 1 )

    xy_flip = cv2.flip(x_flip,1)

    return x_flip, y_flip, xy_flip



def imageHistEqualization(image):

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))

    cl = clahe.apply(l)

    limg = cv2.merge((cl,a,b))

    final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    return final



def subtract_median_bg_image(im):

    k = np.max(im.shape)//20*2+1

    bg = cv2.medianBlur(im, k)

    sub_med = cv2.addWeighted (im, 4, bg, -4, 100)

    return sub_med



def colorEnhancement(image1,image2):

    image_final = cv2.bitwise_and(image1,image2)

    return image_final



def imageAugSave(path,img1,img2,img3,img4,img_ht,img_wd):

    count = len(os.listdir(path))



    img1 = imageResize(img1, img_ht, img_wd)

    img2 = imageResize(img2, img_ht, img_wd)

    img3 = imageResize(img3, img_ht, img_wd)

    img4 = imageResize(img4, img_ht, img_wd)



    cv2.imwrite(os.path.join(path , '%d.png'%(count+1)), img1)

    cv2.imwrite(os.path.join(path , '%d.png'%(count+2)), img2)

    cv2.imwrite(os.path.join(path , '%d.png'%(count+3)), img3)

    cv2.imwrite(os.path.join(path , '%d.png'%(count+4)), img4)

    return count+1,count+2,count+3,count+4



def processed_test_save(path,img,img_ht,img_wd):

    count = len(os.listdir(path))

    img = imageResize(img,img_ht,img_wd)

    cv2.imwrite(os.path.join(path , '%d.png'%(count+1)), img)

    return count+1
img_ht = 256

img_wd = 256

path_toCollect =  '/kaggle/input/aptos2019-blindness-detection/train_images'

path_toSave = '/kaggle/working/trained_images'

train_data = pd.read_csv('/kaggle/input/aptos2019-blindness-detection/train.csv')

newDataframe_cols = ['id_code','diagnosis'] 

trained_data = pd.DataFrame(columns=newDataframe_cols)
def feedToPipeline(image_name,diagnosis_type):

    global path_toCollect

    global path_toCollect

    global img_ht,img_wd

    global trained_data, train_data



    try:

        image_name = str(image_name) + '.png'

        image = cv2.imread(os.path.join(path_toCollect,image_name))

        image = imageResize(image, img_ht, img_wd)

        org_copy = image.copy()

        image_crop = crop_black(image)

        image_clahe = imageHistEqualization(image_crop)

        sub_med = subtract_median_bg_image(image_clahe)

        image_final = colorEnhancement(sub_med, image_clahe)

        aug1, aug2, aug3 = imageAugmentation(image_final)

        count1,count2,count3,count4 = imageAugSave(path_toSave,image_final, aug1, aug2, aug3,img_ht,img_wd)

        count1 = str(count1) + '.png'

        count2 = str(count2) + '.png'

        count3 = str(count3) + '.png'

        count4 = str(count4) + '.png'

        len_trained_data = len(trained_data)

        trained_data.loc[len_trained_data]   = [count1,diagnosis_type] 

        trained_data.loc[len_trained_data+1] = [count2,diagnosis_type] 

        trained_data.loc[len_trained_data+2] = [count3,diagnosis_type] 

        trained_data.loc[len_trained_data+3] = [count4,diagnosis_type]

#         print("Processed")

    except:

        print("+========================+")

        pass
def preprocess(img):

    global path_toCollect

    global path_toCollect

    global img_ht,img_wd

    global trained_data, train_data



    try:

        image_name = str(image_name) + '.png'

        image = cv2.imread(os.path.join(path_toCollect,image_name))

        image = imageResize(image, img_ht, img_wd)

        org_copy = image.copy()

        image_crop = crop_black(image)

        image_clahe = imageHistEqualization(image_crop)

        sub_med = subtract_median_bg_image(image_clahe)

        image_final = colorEnhancement(sub_med, image_clahe)

#         print("Processed")

    except:

        print("+========================+")

        pass
start = time.time()



# # Vectorize approach took 846 seconds and the for loop took 905 seconds to process more than 3 thousand images

# # 

# # np.vectorize(feedToPipeline)(train_data['id_code'],train_data['diagnosis'])

# # 

from tqdm.notebook import tqdm

for i in tqdm(range(len(train_data))): 

# for i in tqdm(range(1)): 

#     print(i)

    feedToPipeline(train_data['id_code'][i],train_data['diagnosis'][i])

# # 

trained_data.to_csv('/kaggle/working/final_trained.csv',index = False)

!ls /kaggle/working/

# !ls /kaggle/working/trained_images

!ls -l /kaggle/working/trained_images/ | egrep -c '^-' 
import gc

gc.collect()

gc.collect()
import os

import cv2

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import tensorflow as tf

from keras.models import Sequential

from keras.layers import Dense, Dropout, Conv2D, BatchNormalization, LeakyReLU, Flatten, Activation, MaxPool2D, GlobalAveragePooling2D

from keras.preprocessing.image import ImageDataGenerator

from keras.optimizers import Adam, RMSprop

from keras.callbacks import EarlyStopping,ReduceLROnPlateau,LearningRateScheduler
train_df = pd.read_csv("/kaggle/working/final_trained.csv")

train_df["id_code"]=train_df["id_code"]

train_df['diagnosis'] = train_df['diagnosis'].astype(str)

train_df.head()
# Example of images 

img_names = train_df['id_code'][:10]

plt.figure(figsize=[15,15])

i = 1

for img_name in img_names:

    img = cv2.imread("/kaggle/working/trained_images/%s" % img_name)[...,[2, 1, 0]]

    ht,wd,ch = img.shape

    print(ht,wd,ch)

    plt.subplot(6, 5, i)

    plt.imshow(img)

    i += 1

plt.show()

gc.collect()
nb_classes = 5

lbls = list(map(str, range(nb_classes)))

batch_size = 16

img_size = 256

nb_epochs = 5
%%time



train_datagen=ImageDataGenerator(

    rescale=1./255, 

    validation_split=0.25,

    horizontal_flip = True, 

    vertical_flip = True,

    rotation_range = 90,

    zoom_range = 0.3,

    width_shift_range = 0.3,

    height_shift_range=0.3

    )



train_generator=train_datagen.flow_from_dataframe(

    dataframe=train_df,

    directory="/kaggle/working/trained_images",

    x_col="id_code",

    y_col="diagnosis",

    batch_size=batch_size,

    shuffle=True,

    class_mode="categorical",

    classes=lbls,

    target_size=(img_size,img_size),

    subset='training')



valid_generator=train_datagen.flow_from_dataframe(

    dataframe=train_df,

    directory="/kaggle/working/trained_images",

    x_col="id_code",

    y_col="diagnosis",

    batch_size=batch_size,

    shuffle=True,

    class_mode="categorical", 

    classes=lbls,

    target_size=(img_size,img_size),

    subset='validation')
# def build_model():

    

#     model = Sequential()

    

#     model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(256,256,3)))

#     model.add(BatchNormalization())

#     model.add(Conv2D(64, kernel_size=3, activation='relu'))

#     model.add(BatchNormalization())

#     model.add(MaxPool2D(pool_size=(2,2)))

    

#     model.add(Conv2D(128, kernel_size=3, activation='relu'))

#     model.add(BatchNormalization())

#     model.add(Conv2D(128, kernel_size=3, activation='relu'))

#     model.add(BatchNormalization())

#     model.add(MaxPool2D(pool_size=(2,2)))

    

#     model.add(Conv2D(256, kernel_size=3, activation='relu'))

#     model.add(BatchNormalization())

#     model.add(Conv2D(256, kernel_size=3, activation='relu'))

#     model.add(BatchNormalization())

#     model.add(MaxPool2D(pool_size=(2,2)))



#     model.add(Flatten())

#     model.add(Dense(1024, activation='relu'))

#     model.add(Dropout(0.5))

#     model.add(Dense(1024, activation='relu'))

#     model.add(Dropout(0.5))

#     model.add(Dense(5, activation='softmax'))



    

#     return model
from efficientnet.keras import EfficientNetB3

effnet = EfficientNetB3(weights='imagenet', include_top=False, input_shape=(256,256,3))
for i, layer in enumerate(effnet.layers):

    if "batch_normalization" in layer.name:

        effnet.layers[i] = GroupNormalization(groups=32, axis=-1, epsilon=0.00001)
def build_model():

    

    model = Sequential()

    model.add(effnet)

    model.add(GlobalAveragePooling2D())

    model.add(Dropout(0.5))

    

    model.add(Dense(256, activation='relu'))

    model.add(Dropout(0.5))

    model.add(Dense(256, activation='relu'))

    model.add(Dropout(0.5))

    model.add(Dense(5, activation='softmax'))



    print(model.summary())

    return model
es = EarlyStopping(monitor='val_loss',

                                      mode='auto',

                                      verbose=1,

                                      patience=10)



learning_rate_reduction = ReduceLROnPlateau(monitor='val_loss',

                                            patience=3,

                                            verbose=1,

                                            mode = 'auto',

                                            factor=0.25,

                                            min_lr=0.000001)



optimizer = Adam(learning_rate=0.5e-4, beta_1=0.9, beta_2=0.999, amsgrad=False)

model = build_model()

model.compile(

    loss='binary_crossentropy',

    optimizer=optimizer,

    metrics=['accuracy']

)
model.fit_generator(

    generator=train_generator, 

#     steps_per_epoch  = 128, 

    validation_data  = valid_generator,

#     validation_steps = 128,

#     epochs = 11, 

    epochs = 2, 

    use_multiprocessing=True,

    verbose = 1,

    callbacks = [es, learning_rate_reduction]

)

model.save("effnet_b3.h5")
! rm -rf /kaggle/working/trained_images/

! ls /kaggle/working
file_src = '/kaggle/input/aptos2019-blindness-detection/test_images'

def predict(img):

    img = cv2.imread(os.path.join(file_src,img))

    img = cv2.resize(img,(256,256))

    image_crop = crop_black(img)

    image_clahe = imageHistEqualization(image_crop)

    sub_med = subtract_median_bg_image(image_clahe)

    image_final = colorEnhancement(sub_med, image_clahe)

    img = image_final/255

    img = np.expand_dims(img,axis = 0)

    ans = model.predict(img)

    ans = np.argmax(ans)

    return int(ans)

test_df = pd.read_csv('/kaggle/input/aptos2019-blindness-detection/sample_submission.csv')

test_df.head()
for i in tqdm(range(len(test_df))):

    img_name = str(test_df['id_code'][i]) + '.png'

    ans = predict(img_name)

    test_df['diagnosis'][i] = ans

    print(i,ans)
test_df.to_csv('submission.csv',index = False)
for i in range(1000):

    print(test_df['diagnosis'][i])
test_df['diagnosis'][1]