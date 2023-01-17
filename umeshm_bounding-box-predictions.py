import tensorflow  as tf
from keras.preprocessing import image
from keras.applications import inception_v3
import numpy as np
import matplotlib.pyplot as plt
from PIL import  Image
# Importing Inception_V3 preprocessing input functions
from keras.applications.inception_v3 import preprocess_input
from keras.models import Model
from keras.applications.inception_v3 import decode_predictions
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.inception_v3 import preprocess_input
from keras.utils.data_utils import GeneratorEnqueuer
data = np.load('../input/object_localization.npy',encoding='latin1')
import keras
from keras.layers import Input,GlobalAveragePooling2D,Dense
from sklearn.model_selection import train_test_split
X = data[:,0]
Y = data[:,1:3]
X_train, X_test, y_train,y_test = train_test_split(X,Y,test_size=0.2,random_state=42)
#  Formatting Data into proper input shape
x=[]
for i in y_train:
    x.append(i[1])
y_train_coord = np.array(x)

x=[]
for i in y_test:
    x.append(i[1])
y_test_coord = np.array(x)

x=[]
for i in X_train:
    x.append(i)
X_train = np.array(x)

x=[]
for i in X_test:
    x.append(i)
X_test = np.array(x)
X_train = preprocess_input(X_train)
X_test = preprocess_input(X_test)

Input_Tensor = Input(shape=(227,227,3))
baseModel = inception_v3.InceptionV3(weights='imagenet',include_top=False,input_tensor=Input_Tensor,pooling='max')
x = baseModel.output
x = Dense(1024,activation='relu')(x)
x = Dense(128,activation='relu')(x)
x = Dense(64,activation='relu')(x)
predictions = Dense(4,activation='linear')(x)

model = Model(inputs=baseModel.input,outputs=predictions)
for layer in baseModel.layers:
    layer.trainable = False
model.compile(optimizer=keras.optimizers.adam(),loss='mean_squared_error')
history = model.fit(X_train,y_train_coord,epochs=100,verbose=1,validation_split=0.2)
import cv2 as cv
def draw_rect(im, cords, color = None):
    """Draw the rectangle on the image
    
    Parameters
    ----------
    
    im : numpy.ndarray
        numpy image 
    
    cords: numpy.ndarray
        Numpy array containing bounding boxes of shape `N X 4` where N is the 
        number of bounding boxes and the bounding boxes are represented in the
        format `x1 y1 x2 y2`
        
    Returns
    -------
    
    numpy.ndarray
        numpy image with bounding boxes drawn on it
        
    """
    
    im = im.copy()
    
    cords = cords[:,:4]
    cords = cords.reshape(-1,4)
    if not color:
        color = [255,0,0]
    for cord in cords:
        
        pt1, pt2 = (cord[0], cord[1]) , (cord[2], cord[3])
                
        pt1 = int(pt1[0]), int(pt1[1])
        pt2 = int(pt2[0]), int(pt2[1])
    
        im = cv.rectangle(im.copy(), pt1, pt2, color, int(max(im.shape[:2])/200),)
    return im
model.evaluate(X_test,y_test_coord)
model.save('bounding_box.h5')
#  Prediction on test data set
def prepare_test_data(x):
    test = np.reshape(X_test[x],(1,227,227,3))
    coord = model.predict(test)   
    img_arr= np.reshape(test,(227,227,3))
    y=np.reshape(y_train[x][1],(1,4))
    im1 = draw_rect(img_arr,coord)
    im2 = draw_rect(img_arr,y)
    return im1,im2
    
for i in range(0,10):    
    im1,im2 = prepare_test_data(89)
    plt.imshow(im1)
    plt.imshow(im2)

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()