import numpy as np

import matplotlib.pyplot as plt

from PIL import Image

import seaborn as sns

import os

import shutil



from tensorflow.keras import Model

from tensorflow.keras.callbacks import Callback, TensorBoard, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Add, Dropout, ReLU, BatchNormalization, Input, Lambda

from tensorflow.keras import backend as K

from tensorflow.keras.metrics import Recall, Precision, AUC 

from tensorflow.keras.preprocessing.image import ImageDataGenerator

import pandas as pd

sns.set()

import tensorflow as tf

PATH = '/kaggle/input/breast-histopathology-images'

folders = os.listdir('/kaggle/input/breast-histopathology-images')

folders.sort()

folders.remove('IDC_regular_ps50_idx5')
def get_x_y(string):

    _, _, x, y, c = string.split('_')

    return [int(x[1:]), int(y[1:]), int(c[-5])]



def getImg(Pid):

    base = os.path.join(PATH, Pid)

    stringList = []

    stringList.extend(base + '/0/' + s for s in os.listdir(base + '/0'))

    stringList.extend(base + '/1/' + s for s in os.listdir(base + '/1'))

    length = len(stringList)

    

    index = np.zeros((length, 3), dtype = np.int32)   

    smallImg = 255*np.ones((length, 50, 50, 3), dtype = np.int8)

    

    for i in range(length):

        index[i] = get_x_y(stringList[i])

        tempImg = Image.open(stringList[i])

        

        if tempImg.size != (50,50):

            smallImg[i, :tempImg.size[1], :tempImg.size[0]] = np.array(tempImg)

        else : 

            smallImg[i] = np.array(tempImg)

            

    maxVal = np.max(index, axis = 0)

    fullImg = 255 * np.ones((maxVal[1] + 50, maxVal[0] + 50, 4), dtype = np.int8)

    for i in range(length):

        x, y, c = index[i]

        fullImg[y : y+50, x : x+50, :3] = smallImg[i]

        if c == 0:

            fullImg[y : y+50, x : x+50, 3] = 150

    return fullImg



def viewImg():

    idx = np.random.randint(0,high = len(folders))

    basePath = os.path.join(PATH, folders[idx])

    imgs0 = os.listdir(os.path.join(basePath, '0'))

    imgs1 = os.listdir(os.path.join(basePath, '1'))

    base0 = os.path.join(basePath,'0')

    base1 = os.path.join(basePath,'1')

    idx0 = np.random.choice(np.arange(len(imgs0)), size = 25)

    idx1 = np.random.choice(np.arange(len(imgs1)), size = 25)

    

    print('Few negative image of Pacitent ID ' + str(folders[idx]))

    plt.figure(figsize = [9,9])

    for i in range(25):

        plt.subplot(5,5,i+1)

        plt.xticks([])

        plt.yticks([])

        plt.imshow(Image.open(os.path.join(base0, imgs0[idx0[i]])))

    plt.show()

    

    print('Few positive image of Pacitent ID ' + str(folders[idx]))

    plt.figure(figsize = [9,9])

    for i in range(25):

        plt.subplot(5,5,i+1)

        plt.xticks([])

        plt.yticks([])

        plt.imshow(Image.open(os.path.join(base1, imgs1[idx1[i]])))

    plt.show()

viewImg()

    

def viewRandom9():

    idx = np.random.choice(np.arange(len(folders)) ,size = 9)

    plt.figure(figsize = [16,16])

    for i in range(9):        

        plt.subplot(3,3, i+1)

        plt.xticks([])

        plt.yticks([])

        plt.title(str(folders[idx[i]]))

        plt.imshow(getImg(folders[idx[i]]))



viewRandom9()
def splitData(folders, trainingData, testData, valData):

    """

    folders : list of folders

    train/test/valDataFraction : fraction of train/test/valDataFraction

    """

    PATH = '/kaggle/input/breast-histopathology-images'

    length = len(folders)

    """os.mkdir('/kaggle/temp')

    OutputPath = '/kaggle/temp'

    

    os.mkdir(os.path.join(OutputPath, 'train'))

    os.mkdir(os.path.join(OutputPath, 'test')) 

    os.mkdir(os.path.join(OutputPath, 'val'))

    

    trainDir = os.path.join(OutputPath, 'train')

    testDir = os.path.join(OutputPath, 'test')

    valDir = os.path.join(OutputPath, 'val')

    

    os.mkdir(os.path.join(trainDir, '0'))

    os.mkdir(os.path.join(trainDir, '1'))

    trainDir0 = os.path.join(trainDir, '0')

    trainDir1 = os.path.join(trainDir, '1')

    

    os.mkdir(os.path.join(testDir, '0'))

    os.mkdir(os.path.join(testDir, '1'))

    testDir0 = os.path.join(testDir, '0')

    testDir1 = os.path.join(testDir, '1')

    

    os.mkdir(os.path.join(valDir, '0'))

    os.mkdir(os.path.join(valDir, '1'))

    valDir0 = os.path.join(valDir, '0')

    valDir1 = os.path.join(valDir, '1')

"""

    trainDF = valDF = testDF = pd.DataFrame(columns = ['img', 'lable'])

    index = np.arange(length)

    np.random.shuffle(index)

    

    for i in range(int(trainingData * length)):

        folder = os.path.join(PATH, folders[index[i]])

        src0 = os.path.join(folder, '0')

        src1 = os.path.join(folder, '1')

        

        lst0 = [os.path.join(src0, f) for f in os.listdir(src0)]

        lst1 = [os.path.join(src1, f) for f in os.listdir(src1)]

        df0 = pd.DataFrame(list(zip(lst0, np.zeros(len(lst0), dtype = np.int8))), columns = trainDF.columns)

        df1 = pd.DataFrame(list(zip(lst1, np.ones(len(lst1), dtype = np.int8))), columns = trainDF.columns)

        trainDF = trainDF.append(df0, ignore_index = True)

        trainDF = trainDF.append(df1, ignore_index = True)

        

        

    for i in range(int(trainingData * length), int((testData + trainingData)* length)):

        folder = os.path.join(PATH, folders[index[i]])

        src0 = os.path.join(folder, '0')

        src1 = os.path.join(folder, '1')

        

        

        lst0 = [os.path.join(src0, f) for f in os.listdir(src0)]

        lst1 = [os.path.join(src1, f) for f in os.listdir(src1)]

        df0 = pd.DataFrame(list(zip(lst0, np.zeros(len(lst0), dtype = np.int8))), columns = trainDF.columns)

        df1 = pd.DataFrame(list(zip(lst1, np.ones(len(lst1), dtype = np.int8))), columns = trainDF.columns)

        valDF = valDF.append(df0, ignore_index = True)

        valDF = valDF.append(df1, ignore_index = True)

    

    for i in range(int((testData + trainingData)* length), length):

        folder = os.path.join(PATH, folders[index[i]])

        src0 = os.path.join(folder, '0')

        src1 = os.path.join(folder, '1')

        

        lst0 = [os.path.join(src0, f) for f in os.listdir(src0)]

        lst1 = [os.path.join(src1, f) for f in os.listdir(src1)]

        df0 = pd.DataFrame(list(zip(lst0, np.zeros(len(lst0), dtype = np.int8))), columns = trainDF.columns)

        df1 = pd.DataFrame(list(zip(lst1, np.ones(len(lst1), dtype = np.int8))), columns = trainDF.columns)

        testDF = testDF.append(df0, ignore_index = True)

        testDF = testDF.append(df1, ignore_index = True)

        

    trainDF['lable'] = trainDF['lable'].astype('str')

    valDF['lable'] = valDF['lable'].astype('str')

    testDF['lable'] = testDF['lable'].astype('str')

    

    print(trainDF.info())

    print(valDF.info())

    print(testDF.info())

    return trainDF, valDF, testDF

        



trainDF, valDF, testDF = splitData(folders, 0.8, 0.1, 0.1)

logdir = 'logs'



class My_callback(Callback):

    def on_epoch_end(self, epoch, logs = {}):

        if (logs['acc'] > 0.93):

            self.model.stop_training = True    



esCallback = EarlyStopping(monitor = 'acc', patience = 3)

mcCallback = ModelCheckpoint(filepath = 'modelOn.hdf5', monitor = 'val_loss', save_best_only = True)

lrCallback = ReduceLROnPlateau(monitor = 'val_loss', factor = 0.1, patience = 3)

tbCallback = TensorBoard(log_dir = logdir)

csCallback = My_callback()


trainDataGen = ImageDataGenerator(rescale = 1./255,

                                 horizontal_flip = True,

                                 vertical_flip = True)

valDataGen = ImageDataGenerator(rescale = 1./255)



trainFlow = trainDataGen.flow_from_dataframe(trainDF, x_col = 'img', y_col = 'lable', 

                                             class_mode = 'binary', 

                                             target_size = (50,50), 

                                             validate_filenames = False)



valFlow = valDataGen.flow_from_dataframe(valDF, x_col = 'img', y_col = 'lable', 

                                         class_mode = 'binary', 

                                         target_size = (50,50), 

                                         validate_filenames = False)



testFlow = valDataGen.flow_from_dataframe(testDF, x_col = 'img', y_col = 'lable', 

                                          class_mode = 'binary', 

                                          target_size = (50,50), 

                                          validate_filenames = False)
def resNet(layer, filterIn, filterOut, conv, block, isFirst):

    

    if isFirst: 

        x = Conv2D(filterIn, (1,1), (2,2), name = conv + '_' + block + '_1')(layer)

        x = BatchNormalization(name = conv + '_' + block + '_1_' + 'Norm', trainable = True)(x)

        x = ReLU(name = conv + '_' + block + '_1_' + 'ReLU')(x)

        

        x = Conv2D(filterIn, (3,3), padding='same', name = conv + '_' + block + '_2')(x)

        x = BatchNormalization(name = conv + '_' + block + '_2_' + 'Norm', trainable = True)(x)

        x = ReLU(name = conv + '_' + block + '_2_' + 'ReLU')(x)

        

        x = Conv2D(filterOut, (1,1), name = conv + '_' + block + '_3')(x)

        x = BatchNormalization(name = conv + '_' + block + '_3_' + 'Norm', trainable = True)(x)

        

        y = Conv2D(filterOut,(1,1), (2,2), name = conv + '_' + block + '_0')(layer)

        y = BatchNormalization(name = conv + '_' + block + '_0_' + 'Norm', trainable = True)(y)

        

    else:

        x = Conv2D(filterIn, (1,1), name = conv + '_'+ block + '_1')(layer)

        x = BatchNormalization(name = conv + '_' + block + '_1_' + 'Norm', trainable = True)(x)

        x = ReLU(name = conv + '_' + block + '_1_' + 'ReLU')(x)

        

        x = Conv2D(filterIn, (3,3), name = conv + '_'+ block + '_2', padding='same')(x)

        x = BatchNormalization(name = conv + '_' + block + '_2_' + 'Norm', trainable = True)(x)

        x = ReLU(name = conv + '_' + block + '_2_' + 'ReLU')(x)

        

        x = Conv2D(filterOut, (1,1), name = conv + '_'+ block + '_3')(x)

        x = BatchNormalization(name = conv + '_' + block + '_3_' + 'Norm', trainable = True)(x)

        

        y = layer

        

    out = Add()([x,y])

    out = ReLU(name = conv + '_' + block + '_final_' + 'ReLU')(out)

    return out



def resNetAll(blockFilter, nosOfblock, layer_):

    x = layer_

    for i in range(len(blockFilter)-1):

        for j in range(nosOfblock[i]):

            if j == 0:

                x = resNet(x, blockFilter[i], blockFilter[i+1], 'conv' + str(i+1), 'block' + str(j+1), True)

            else:

                x = resNet(x, blockFilter[i], blockFilter[i+1], 'conv' + str(i+1), 'block' + str(j+1), False)

    

    return x 

            



inputs = (None, None, 3)

inp = Input(inputs)

x = Conv2D(16, (3,3), padding= 'same', name = 'InitialInput')(inp)

x = BatchNormalization(name = 'InitialBatch',trainable = True)(x)

x = ReLU(name = 'InitialReLU')(x)

x = resNetAll([32, 32, 64, 64], [3,5,5], x)

x = Conv2D(512, (7,7))(x)

x = BatchNormalization(trainable = True)(x)

x = ReLU()(x)

x = Conv2D(16, (1,1))(x)

x = BatchNormalization(trainable = True)(x)

x = ReLU()(x)

x = Conv2D(1, (1,1), activation = 'sigmoid')(x)

x = Lambda(lambda x: K.reshape(x,shape =  [-1, 1]))(x)

model = Model(inp, x)

model.compile('adam','binary_crossentropy' ,['acc','mse', Precision(), Recall(), AUC(),])
totalOnes = np.sum(trainDF['lable'].astype('int'))

totalZeros = len(trainDF) - totalOnes

zeroClass, oneClass = totalOnes/ (totalOnes + totalZeros), totalZeros / (totalOnes + totalZeros)

weightDist = {0 : zeroClass, 1 : oneClass}


history = model.fit(trainFlow,

                   epochs = 30,

                   callbacks = [tbCallback,csCallback, lrCallback, esCallback, mcCallback],

                   class_weight = weightDist,

                   steps_per_epoch = 5000,

                   validation_data = valFlow,

                )

model.save('/kaggle/working/model.h5')   
plt.figure()

plt.plot(history.history['acc'])

plt.plot(history.history['val_acc'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'val'], loc='upper left')

plt.show()
plt.figure()

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'val'], loc='upper left')

plt.show()
plt.figure()

plt.plot(history.history['mse'])

plt.plot(history.history['val_mse'])

plt.title('model MSE')

plt.ylabel('MSE')

plt.xlabel('epoch')

plt.legend(['train', 'val'], loc='upper left')

plt.show()
plt.figure()

plt.plot(history.history['precision'])

plt.plot(history.history['val_precision'])

plt.title('model precision')

plt.ylabel('precision')

plt.xlabel('epoch')

plt.legend(['train', 'val'], loc='upper left')

plt.show()
plt.figure()

plt.plot(history.history['recall'])

plt.plot(history.history['val_recall'])

plt.title('model recall')

plt.ylabel('recall')

plt.xlabel('epoch')

plt.legend(['train', 'val'], loc='upper left')

plt.show()
plt.figure()

plt.plot(history.history['auc'])

plt.plot(history.history['val_auc'])

plt.title('model auc')

plt.ylabel('auc')

plt.xlabel('epoch')

plt.legend(['train', 'val'], loc='upper left')

plt.show()
res = model.evaluate(testFlow)

print("Model loss      on testDataset " + str(res[0]))

print("Model acc       on testDataset " + str(res[1]))

print("Model mse       on testDataset " + str(res[2]))

print("Model precision on testDataset " + str(res[3]))

print("Model recall    on testDataset " + str(res[4]))

print("Model AUC       on testDataset " + str(res[5]))