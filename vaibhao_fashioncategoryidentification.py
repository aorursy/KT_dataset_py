# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input/fashiondatacolor-images/fashiondata/FashionData/"))

# Any results you write to the current directory are saved as output.
import keras
import tensorflow as tf
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import cv2
from keras import regularizers
from keras.applications import InceptionV3,InceptionResNetV2,ResNet50
from keras.models import Sequential
from keras.layers import  Activation,Dense,Flatten, GlobalAveragePooling2D
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
import shutil
from keras.optimizers import Adam, SGD, RMSprop
import time,os
from keras.utils import plot_model
import os
import pickle
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau
import matplotlib.pyplot as plt
from keras.models import Sequential, save_model,load_model
from keras import backend as K
import tensorflow as tf
'''with K.tf.device('/gpu:0'):
    config = tf.ConfigProto(intra_op_parallelism_threads=4,\
           inter_op_parallelism_threads=4, allow_soft_placement=True,\
           device_count = {'CPU' : 1, 'GPU' : 1})
    session = tf.Session(config=config)
    K.set_session(session)'''

os.path.dirname(os.path.abspath('__file__')) + '/'
#class for training and preprocessing
class FashionTraining():
    #Intiailise the object with Data
    def __init__(self, TrainingDataPath, valDataPath, numClassses, lr = 0.001, 
                 epochs=50, l2_labmda = 0.0001, batchSize = 64):
        self.batchSize = batchSize
        self.path = os.path.dirname(os.path.abspath('__file__')) + '/'
        #Path of Training Data
        self.trainPath = TrainingDataPath
        #Path of Validation Data
        self.valDataPath = valDataPath
        #creating base model for multiGPU 
        self.baseModel = None
        #Resnet weight path 
        self.weightPath = '../input/keras-pretrained-models/inception_resnet_v2_weights_tf_dim_ordering_tf_kernels_notop.h5'
        #self.weightPath = '../input/keras-pretrained-models/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'
        #defining the ImageGenerator Object
        self.trainDataGen = ImageDataGenerator(#rescale = 1./255,
                                              preprocessing_function = self.preprocessing) 
                                   #'''shear_range = 0.15, 
                                   #zoom_range = 0.05,
                                   #rotation_range=0.2,
                                   #width_shift_range=0.1,
                                   #height_shift_range=0.1
                                   #horizontal_flip = True'''
                                   #)
        #self.testDataGen = ImageDataGenerator()
        self.valDataGen = ImageDataGenerator(#rescale = 1./255
                                            preprocessing_function = self.preprocessing
                                            )
        
        self.reduceLR= ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1,
                                         mode='auto',min_delta=1e-4, cooldown=0, min_lr=0)

        self.EarlyCheckPt = EarlyStopping(monitor='val_loss', min_delta=0, 
                                          patience=5, verbose=1, mode='auto')

        
        self.ModelCkPt = ModelCheckpoint('fashionModel_v1.h5', monitor='val_loss', 
                                         verbose=1, save_best_only=True, save_weights_only=False, 
                                         mode='auto', period=1)
        
        self.train_generator = self.trainDataGen.flow_from_directory(self.trainPath + '/',
                                          target_size = (224,224),
                                          class_mode = 'categorical',
                                          batch_size = self.batchSize,
                                          color_mode='rgb',
                                          shuffle=True,
                                          seed=88
                                          )
        
        self.validation_generator = self.valDataGen.flow_from_directory(self.valDataPath + '/',
                                          target_size = (224,224),
                                          class_mode = 'categorical',
                                          batch_size = self.batchSize,
                                          color_mode='rgb',
                                          shuffle=True,
                                          seed=88
                                          )
        
        self.LR = lr,
        self.l2Lambda = l2_labmda
        self.epochs = epochs
        self.optimizer = RMSprop(self.LR,decay=1e-5)
        self.numCategories = numClassses
        self.model = None
        self.hist = None
        with open('FasshionCategories.pkl','wb') as f:
            pickle.dump(self.train_generator.class_indices,f)
#-------------------------------------------------------------------            
    def preprocessing(self,image):
        #img = cv2.imread('/Users/vk250027/Documents/InStoreVideo/FashionData/Validation_Directory/abstract/66137.jpg')
        #print('image batch has shape : ', images.shape)
        #for img in images:
        gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY).astype('uint8')        
        blur = cv2.GaussianBlur(gray,(3,3),0)  
        #print('Blur image has shape : ', blur.shape)
        #creating a mask for the image
        (t, maskLayer) = cv2.threshold(blur.copy(), 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)        
        # make a mask suitable for color images
        #mask = cv2.merge([maskLayer, maskLayer, maskLayer]) (not required if using mask argument in bitwise and)
        #Cropping the image (i.e. identifying the person first)
        person = cv2.bitwise_and(image,image, mask = maskLayer)
        '''
        Part 2 : cropting the ROI (shirt area)
        a.. iodentifying the skin color and substracting from parent one = result will be ROI
        '''
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        #defining lower and upper bound for skin color
        lower = np.array([0, 48, 80], dtype = "uint8")
        upper = np.array([20, 255, 255], dtype = "uint8")
        converted = cv2.cvtColor(person, cv2.COLOR_BGR2HSV)
        skinMask = cv2.inRange(converted, lower, upper)
        skinMask = cv2.erode(skinMask, kernel, iterations = 2)
        skinMask = cv2.dilate(skinMask, kernel, iterations = 2)
        # blur the mask to help remove noise, then apply the mask to the frame
        skinMask = cv2.GaussianBlur(skinMask, (3, 3), 0)
        # this will identify the skin portion of a person
        skin = cv2.bitwise_and(person, person, mask = skinMask)
        #substracting the skin frtom orifinal person image
        diff = person - skin
        #return the substracted image for processing 
        return diff
        
        '''
        _, contours, hierarchy = cv2.findContours(maskLayer, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        #t = cv2.cvtColor(maskLayer.copy(),cv2.COLOR_GRAY2BGR)
        
        #t = cv2.drawContours(t,contours,-1,(0,255,255),2)
        
        #imgArea = img.shape[0] * img.shape[1]
        # getting the area for each contour and identifying the max area
        index = np.argmax([cv2.contourArea(item) for item in contours ])
        #filtering the contours based on max area 
        #index = np.argmax(areas)
         
        #indexes = np.where(np.array(areas) > 0.005 * imgArea)
        
        #getting the coordinate of ROI
        x,y,w,h = cv2.boundingRect(contours[index])
        img = np.array(images[y:y+h,x:x+w],dtype=np.uint8)
        img = cv2.resize(img,(224,224), cv2.INTER_AREA)
        return img
        '''  
 #-------------------------------------------------------------------           
    def getModel(self, numClases):
        optim = self.optimizer
        
        self.model = Sequential()
        self.model.add(InceptionResNetV2(include_top=False,weights=self.weightPath,input_shape=(224,224,3),
                                    pooling='avg'))
        '''self.model.add(ResNet50(include_top=False,weights=self.weightPath,input_shape=(224,224,3),
                                    pooling='avg'))'''
        self.model.add(Dense(numClases,activation='softmax', kernel_initializer='he_uniform',
                                      kernel_regularizer=regularizers.l2(self.l2Lambda)))
        
        self.model.layers[0].trainable = False
        
        self.model.compile(optimizer=optim,loss='categorical_crossentropy',metrics=['accuracy'])
        
        print(self.model.summary())
        plot_model(self.model, to_file= 'modelStructure.png')
        
        print('Model configured..!')
#-------------------------------------------------------------------
    def training(self):
        stepsPerEpochs = self.train_generator.samples//self.batchSize
        #stepsPerEpochs = self.validation_generator.samples//self.batchSize
        validationSteps = self.validation_generator.samples//self.batchSize        
        #with tf.device('/gpu:1'):
            
        # =============================================================================
        #         config = tf.ConfigProto( device_count = {'GPU': 1 , 'CPU': 56} )
        #         sess = tf.Session(config=config) 
        #         keras.backend.set_session(sess)
        # =============================================================================
        self.hist = self.model.fit_generator(self.train_generator,
                            steps_per_epoch =stepsPerEpochs,
                            epochs=self.epochs,
                            validation_data = self.validation_generator,
                            validation_steps = validationSteps,
                            verbose = 1,
                            callbacks = [self.reduceLR, 
                                        self.EarlyCheckPt,
                                        self.ModelCkPt]
                                                #AltModelCheckpoint(self.path+'Models/OCR_Epochs.h5',self.baseModel)] 
                                        )
                #except Exception as e:
                    #print("Got issues : ", e)
                   
        #saving the model after final Epoch
        save_model(self.model,'finalModel_v1.h5')
        self.model.set_weights(self.model.get_weights())
        self.model.save(filepath='finalModel_weights_v1.h5')        
        N = np.arange(0, self.epochs)
        plt.style.use("ggplot")
        plt.figure()
        plt.plot(N, self.hist.history["loss"], label="train_loss")
        plt.plot(N, self.hist.history["val_loss"], label="val_loss")
        plt.plot(N, self.hist.history["acc"], label="train_acc")
        plt.plot(N, self.hist.history["val_acc"], label="val_acc")
        plt.title("Training Loss and Accuracy (Simple NN)")
        plt.xlabel("Epoch #")
        plt.ylabel("Loss/Accuracy")
        plt.legend()
        plt.savefig('OutputStructure.png')
#-------------------------------------------------------------------
if __name__ == '__main__' :
    path = os.path.dirname(os.path.abspath('__file__')) + '/'
    trainPath = '../input/fashiondatacolor-images/fashiondata/FashionData/Train_Directory/'
    valPath = '../input/fashiondatacolor-images/fashiondata/FashionData/Validation_Directory/'
    numClasses = len([item for item in os.listdir(trainPath) if '.DS_Store' not in item])
    obj = FashionTraining(TrainingDataPath=trainPath, valDataPath= valPath, numClassses= numClasses,
                          epochs=20)
    obj.getModel(numClasses)
    obj.training()
    print('Trainig completed...!!')
