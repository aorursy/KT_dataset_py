# Data used can be found at:

# https://www.kaggle.com/c/facial-keypoints-detection/data



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from PIL import Image

import seaborn as sns

import keras as K # importing this just in case



from keras import backend



#from VGG16 import VGG16_Obj # contains the keras models I made, for local testing

#from TrainImageObj import TrainImage # contains some viewing and V&V data 



import tensorflow as tf

import os 

import sys

from datetime import datetime
# process the image in the TF/Keras pretrained format

def formatImage(dataInput):

    masterTemp = []

    for i in dataInput:

        tp = np.reshape(i.split(' '),(96,96))

        imagePIL = Image.fromarray(np.array(tp, dtype='int32'))

        imageResized = imagePIL.resize((224,224), Image.BILINEAR) # nearest neighbor resolution increase

        imageResized = np.array(imageResized)

        temp = []

        for n in range(3):

            temp.append(imageResized)

        masterTemp.append(temp)

    dataInput = np.array(masterTemp)

    dataInput = np.swapaxes(dataInput,1,2)

    dataInput = np.swapaxes(dataInput,2,3)

    return dataInput



# use class to make the system more modular 

class VGG16_Obj:

    def __init__ (self, projectDirectory = os.getcwd()):

        # home computer has weights that can be leveraged for this, but right now jupyter can handle it

        # interesting experiment: running this model with pretrained weights from the imagenet challenge and then again here without 

        pass

        #self.projectDirectory = projectDirectory

        #self.modelFC   = self.VGG16()

        #self.modelNoFC = self.VGG16(FC_Include = False) # add additional method to call this out

        #self.modelReducedFC = self.AddFCtoVGG16FeatureExtractor()



    def VGG16(self,ClassicVGG16=True, 

                  FC_Include = True,

                  l2_weight = 5e-04,

                  like_its_hot = 0.4, # drop regulation

                  FeatureExtractorTraining = False, 

                  FCTraining = True,

                  weights= 'imagenet', 

                  input_tensor=None): 

        ''' 

        # Inputs

        FC_Include = using the network as a feature extractor based on the convolutional layers

        FullyConnected = if training is needed for the fully connected layer

        classificationNumber = number of components 

        FeatureExtractorTraining = if you need to train the middle layers

        weights = 'imagenet' means using the weights from a network pretrained by imagenet challenge

        

        Rules:

        CANNOT have MORE than 1000 outputs (can't really see a case where they're would be >1k but eat your heart out.

        Use the excess and keep training on the FC layers true

        

        # Returns

            VGG16 Network

        '''

        if weights not in {'imagenet', None}:

            raise ValueError('The `weights` argument should be either '

                             '`None` (random initialization) or `imagenet` '

                             '(pre-training on ImageNet).')

        # Determine proper input shape

        if ClassicVGG16:

            inputShape = (224, 224, 3)

        else:

            inputShape = (None, None, 3)

        

        img_input = K.Input(inputShape)

        

        # Block 1     

        b1_1 = K.layers.Conv2D(64, (3, 3), 

                      activation='relu', 

                      padding='same', # border_mode is now padding

                      name='block1_conv1')

        b1_1.trainable = FeatureExtractorTraining

        x = b1_1(img_input)

        

        b1_2 = K.layers.Conv2D(64, (3, 3), 

                      activation='relu', 

                      padding='same', 

                      name='block1_conv2')

        b1_2.trainable = FeatureExtractorTraining

        x = b1_2(x)#_normalized)

        

        x = K.layers.MaxPooling2D((2,2), strides=(2,2), name='block1_pool')(x) 

        

        # Block 2

        b2_1 = K.layers.Conv2D(128, 

                      (3, 3), 

                      activation='relu', 

                      padding='same', 

                      name='block2_conv1')

        b2_1.trainable = FeatureExtractorTraining

        x = b2_1(x)

        

        b2_2 = K.layers.Conv2D(128, (3, 3), 

                               activation='relu', 

                               padding='same', 

                               name='block2_conv2')

        b2_2.trainable = FeatureExtractorTraining

        x = b2_2(x)#_normalized)

        

        x = K.layers.MaxPooling2D((2,2), strides=(2,2) , name='block2_pool')(x)#_normalized) # decrease the amout of data points with no rounding loss

        

        # Block 3

        # convolution block

        

        b3_1 = K.layers.Conv2D(256, (3, 3), 

                      activation='relu', 

                      padding='same', 

                      name='block3_conv1')

        b3_1.trainable = FeatureExtractorTraining

        x = b3_1(x)

        

        b3_2 = K.layers.Conv2D(256, (3, 3), 

                      activation='relu', 

                      padding='same', 

                      name='block3_conv2')

        b3_2.trainable = FeatureExtractorTraining

        x = b3_2(x)#_normalized)



        b3_3 = K.layers.Conv2D(256, (3, 3), 

                      activation='relu', 

                      padding='same', 

                      name='block3_conv3')

        b3_3.trainable = FeatureExtractorTraining

        x = b3_3(x)#_normalized)

        

        x = K.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

        

        # Block 4 identity doc

        

        b4_1 = K.layers.Conv2D(512, (3, 3), 

                      activation='relu', 

                      padding='same', 

                      name='block4_conv1')

        b4_1.trainable = FeatureExtractorTraining

        x = b4_1(x)

        

        b4_2 = K.layers.Conv2D(512, (3, 3), 

                      activation='relu', 

                      padding='same', 

                      name='block4_conv2')

        b4_2.trainable = FeatureExtractorTraining

        x = b4_2(x)#_normalized)



        b4_3 = K.layers.Conv2D(512, (3, 3), 

                      activation='relu', 

                      padding='same', 

                      name='block4_conv3')

        b4_3.trainable = FeatureExtractorTraining

        x = b4_3(x)#_normalized)

        

        x = K.layers.MaxPooling2D((2,2), strides=(2,2), name='block4_pool')(x)

        

        #Block 5

        b5_1 = K.layers.Conv2D(512, (3, 3), 

                      activation='relu', 

                      padding='same', 

                      name='block5_conv1')

        b5_1.trainable = FeatureExtractorTraining

        x = b5_1(x)

        

        b5_2 = K.layers.Conv2D(512, (3, 3), 

                      activation='relu', 

                      padding='same', 

                      name='block5_conv2')

        b5_2.trainable = FeatureExtractorTraining

        x = b5_2(x)

        

        b5_3 = K.layers.Conv2D(512, (3, 3), 

                      activation='relu', 

                      padding='same', 

                      name='block5_conv3')

        b5_3.trainable = FeatureExtractorTraining

        x = b5_3(x)

        

        x = K.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)



        x = K.layers.Flatten(name='flatten')(x)



        if FC_Include:

            # Classification block

            #x = K.layers.Flatten(name='flatten')(x) # moved this inside the lines

            

            #x = Dropout(like_its_hot, name = 'regulator_0')(x)

            fc1 = K.layers.Dense(4096, activation='relu',

                        kernel_regularizer=K.regularizers.l2(l2_weight),

                        name='fc1')

            fc1.trainable = FCTraining

            x = fc1(x)

            

            x = K.layers.Dropout(like_its_hot, name = 'regulator_1')(x)

            

            fc2 = K.layers.Dense(4096, 

                        activation='relu', 

                        kernel_regularizer=K.regularizers.l2(l2_weight),

                        name='fc2')

            fc2.trainable = FCTraining

            x = fc2(x)

            

            x = K.layers.Dropout(like_its_hot, name = 'regulator_2')(x)

            

            pred  = K.layers.Dense(1000, 

                          activation='softmax', 

                          kernel_regularizer=K.regularizers.l2(l2_weight),

                          name='pred')(x)

            

            model = K.Model(img_input, pred)

            

        else: ########################################################################################

            #print ("You got no legs Lieutenant Dan!!!")

            model = K.Model(img_input,x)

            

        # load weights

        if weights == 'imagenet':

            currentCwd = os.getcwd()

            os.chdir(self.projectDirectory) # hard coded for my directory

            if FC_Include == False:

                modelWeights = model.load_weights('vgg16Weights_noFC.h5')            

            elif FC_Include == True: # only include the top if has 

                modelWeights = model.load_weights('vgg16Weights_FCincluded.h5')

            os.chdir(currentCwd)

        return model





    def AddFCtoVGG16FeatureExtractor(self, fc1Variable = 500, 

                                    fc2Variable = 200, predVariable = 30, 

                                    l2_weight = 1e-03, like_its_hot = 0.4):

        '''

        Purpose: utalize Transfer Learning and slam a vgg16 network together with different FC layers

        Output: model with just that (if you do a model.summary it misses the new model but data is there)

        '''

        

        #TK modified here 11/03 

        model1 = self.VGG16(ClassicVGG16=True, # use the 224x224x3 base to make it the most apples to apples comparison 

                    FC_Include = False,# remove the additional layers

                    l2_weight = 5e-04,# Ridge Regression weights

                    like_its_hot = 0.25, # drop regulation

                    FeatureExtractorTraining = True,# go ahead and train 

                    FCTraining = True,

                    weights= None,# not using transfer learnign

                    input_tensor=None) # this is where I import the Transfer Learning Model



        #model1 = self.modelNoFC # get the pretrained network with no FC layer

        # end modifications here 11/03

        

        #make the second model we slam together

        model2 = K.models.Sequential(name="FC_Layers_Model") # MUST use Keras API to add layers together

        model2.add(K.layers.Dense(fc1Variable, #define amount of variables in function header

                      activation='relu',

                      kernel_regularizer=K.regularizers.l2(l2_weight),

                      name  ="fc1"))

        

        model2.add(K.layers.Dropout(like_its_hot, name = 'regulator_1')) # add regulation

        model2.add(K.layers.Dense(fc2Variable, #define amount of variables in function header

                     activation='relu',

                     kernel_regularizer=K.regularizers.l2(l2_weight),

                     name ="fc2"))

        model2.add(K.layers.Dropout(like_its_hot, name = 'regulator_2')) #add a little more regulation

        model2.add(K.layers.Dense(predVariable, #define amount of variables in function header

                      #activation='softmax', # don't want an activator function here

                      kernel_regularizer=K.regularizers.l2(l2_weight),

                      name  ="pred"))



        linkingOutput = model2(model1.output) #link the two models here

        finalModel = K.Model(model1.input, linkingOutput)

        return finalModel

print("Model Created")
# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



# Any results you write to the current directory are saved as output.



####################################### definitions for data import#######################################

cwd = os.getcwd()

test = False # just incase there is something we need to test in this

start = datetime.now()

masterStart = datetime.now()



projectDirectory = "/home/ted/Python_Projects/FacialKeypointsDetection"#"C:\\Users\\Ted\\Projects\\Python\\FacialKeypointsDetection"

dataDir = "/media/ted/Elements/TK_PracticeDatabases/facial-keypoints-detection"#"E:\\TK_PracticeDatabases\\facial-keypoints-detection"



dirname = projectDirectory



for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))





os.chdir("/kaggle/input/facial-keypoints-detection/test")

testData = pd.read_csv("test.csv")



os.chdir("/kaggle/input/facial-keypoints-detection/training")

trainingData = pd.read_csv("training.csv")



os.chdir("/kaggle/input/facial-keypoints-detection")

idLookupData = pd.read_csv("IdLookupTable.csv")

sampleSubData = pd.read_csv("SampleSubmission.csv")

#os.chdir(projectDirectory)



# borrowed these line from Karan Jakhar's post

b4NanFill = trainingData.isnull().any().value_counts() # only present for the print statement

trainingData.fillna(0, inplace=True) # replace nans with '0' 

afterNanFill = trainingData.isnull().any().value_counts() # only present for the print statement

#"Big thank you to Karan Jakhar's post for these lines \n\nB4Nan Handling :  \n"

print(str(b4NanFill) + " \n\nHandling After: \n" + str(afterNanFill) +"\n")

del b4NanFill

del afterNanFill # don't need to delete these to free up memory but doesn't hurt





################################################################################################################
dataFromFiles = trainingData['Image']

dataInput = formatImage(dataFromFiles)



validationData = trainingData.drop('Image', axis=1)



os.chdir(cwd)



if test: # in the begining take a look at the training data

    print(trainingData)



importComplete = datetime.now()

print("Import Finished: " + str(dataInput.shape)+" "+ str(importComplete - start))
########################################## Generate the model ##########################################

start = datetime.now()



#os.chdir(projectDirectory)

vgg16 = VGG16_Obj()



model = vgg16.AddFCtoVGG16FeatureExtractor(fc1Variable = 1000, # use the VGG to then swap out the fully connected levels

                                           fc2Variable = 200, # gradual is typically good

                                           predVariable = 30, 

                                           l2_weight = 1e-03, 

                                           like_its_hot = 0.4)



model.summary() # this will give us the VGG16 without the fully connected layers and label the new FCs as sequentional_1

# we are using the VGG16 as a feature extractor and not changing them!!!



# check if tensorflow sees my GPU

print("\nDoes the computer recognize a GPU: " + str(len(backend.tensorflow_backend._get_available_gpus())))



importComplete = datetime.now()

print("\nModified Vgg16 Generated: " + str(importComplete - start))

plt.close() # close the figure for non-jupyter notebook applications
########################################## fit to model ##########################################

start = datetime.now()

##log all the loss data

#csvLogging = K.callbacks.CSVLogger("FacialPointExtractor_Loss.log", append=True)



model.compile(optimizer = 'adam', # adam seems to be the best optimizer I've used (mostly for classification tho) 

              loss = 'mse', # mse because we want a continuous number!!!

              metrics = ['mae', 'accuracy']) # print/log some extra info

#'''

model.fit(dataInput, validationData,  #need data to insert 

          batch_size=128, # make this a power of 2 for best performance

          epochs=20, # more epochs better the prediction until gain is saturated, start small and go bigger if needed

          shuffle=True, # shuffling is never a bad idea...

          verbose=True, # show me progress per epoch

          #callbacks=[csv_logger] # log the data?

          validation_split=0.2) # take a 1/5 of the data for validation purposes '''





doneFitting = datetime.now()

print("\nFitting Complete: " + str(doneFitting-start))
# Test data Prep, same as before, should make this a function...

dataInput = testData['Image']

masterTemp = []

for i in dataInput:

    tp = np.reshape(i.split(' '),(96,96))

    imagePIL = Image.fromarray(np.array(tp, dtype='int32'))

    imageResized = imagePIL.resize((224,224), Image.BILINEAR) # nearest neighbor resolution increase

    imageResized = np.array(imageResized)

    temp = []

    for n in range(3):

        temp.append(imageResized)

    masterTemp.append(temp)

dataInput = np.array(masterTemp)

dataInput = np.swapaxes(dataInput,1,2)

dataInput = np.swapaxes(dataInput,2,3)



pred = model.predict(dataInput)

print("Pred Complete")
train_data = trainingData

test_data = testData

lookid_data = idLookupData

sampleSubData



lookid_list = list(lookid_data['FeatureName'])

imageID = list(lookid_data['ImageId']-1)

pre_list = list(pred)



rowid = lookid_data['RowId']

rowid=list(rowid)



feature = []

for f in list(lookid_data['FeatureName']):

    feature.append(lookid_list.index(f))

preded = []

for x,y in zip(imageID,feature):

    preded.append(pre_list[x][y])



rowid = pd.Series(rowid,name = 'RowId')

loc = pd.Series(preded,name = 'Location')

submission = pd.concat([rowid,loc],axis = 1)

submission.to_csv('face_key_detection_submission.csv',index = False)

os.getcwd()