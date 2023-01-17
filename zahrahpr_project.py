

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from scipy import stats

from keras.models import Sequential

from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout



from keras import optimizers

random_seed = 611

np.random.seed(random_seed)
plt.style.use('ggplot')

def readData(filePath):

    columnNames = ['user_id','activity','timestamp','x-axis','y-axis','z-axis']

    data = pd.read_csv(filePath,header = None, names=columnNames,na_values=';')

    return data



def featureNormalize(dataset):

    u = np.mean(dataset,axis=0)

    sigma = np.std(dataset,axis=0)

    return (dataset-u)/sigma
def plotAxis(axis,x,y,title):

    axis.plot(x,y)

    axis.set_title(title)

    axis.xaxis.set_visible(False)

    axis.set_ylim([min(y)-np.std(y),max(y)+np.std(y)])

    axis.set_xlim([min(x),max(x)])

    axis.grid(True)
def plotActivity(activity,data):

    fig,(ax0,ax1,ax2) = plt.subplots(nrows=3, figsize=(15,10),sharex=True)

    plotAxis(ax0,data['timestamp'],data['x-axis'],'x-axis')

    plotAxis(ax1,data['timestamp'],data['y-axis'],'y-axis')

    plotAxis(ax2,data['timestamp'],data['z-axis'],'z-axis')

    plt.subplots_adjust(hspace=0.2)

    fig.suptitle(activity)

    plt.subplots_adjust(top=0.9)

    plt.show()
def windows(data,size):

    start = 0

    while start< data.count():

        yield int(start), int(start + size)

        start+= (size/2)
# segmenting the time series

def segment_signal(data, window_size = 90):

    segments = np.empty((0,window_size,3))

    labels= np.empty((0))

    for (start, end) in windows(data['timestamp'],window_size):

        x = data['x-axis'][start:end]

        y = data['y-axis'][start:end]

        z = data['z-axis'][start:end]

        if(len(data['timestamp'][start:end])==window_size):

            segments = np.vstack([segments,np.dstack([x,y,z])])

            labels = np.append(labels,stats.mode(data['activity'][start:end])[0][0])

    return segments, labels
''' Main Code '''

# # # # # # # # #   reading the data   # # # # # # # # # # 

# Path of file #

dataset = readData('../input/activity-dataset.txt')

# plotting a subset of the data to visualize

for activity in np.unique(dataset['activity']):

    subset = dataset[dataset['activity']==activity][:180]

    plotActivity(activity,subset)

# segmenting the signal in overlapping windows of 90 samples with 50% overlap

segments, labels = segment_signal(dataset) 

#categorically defining the classes of the activities

labels = np.asarray(pd.get_dummies(labels),dtype = np.int8)

# defining parameters for the input and network layers



numOfRows = segments.shape[1]

numOfColumns = segments.shape[2]

numChannels = 1

numFilters = 128 

# kernal size of the Conv2D layer

kernalSize1 = 2

# max pooling window size

poolingWindowSz = 2

# number of filters in fully connected layers

numNueronsFCL1 = 128

numNueronsFCL2 = 128

numNueronsFCL3 = 128



# split ratio for test and validation

trainSplitRatio = 0.8

# number of epochs

Epochs = 20

# batchsize

batchSize = 10

# number of total clases

numClasses = labels.shape[1]

# dropout ratio for dropout layer

dropOutRatio = 0.2

# reshaping the data for network input

reshapedSegments = segments.reshape(segments.shape[0], numOfRows, numOfColumns,1)

# splitting in training and testing data

trainSplit = np.random.rand(len(reshapedSegments)) < trainSplitRatio

trainX = reshapedSegments[trainSplit]

testX = reshapedSegments[~trainSplit]

trainX = np.nan_to_num(trainX)

testX = np.nan_to_num(testX)

trainY = labels[trainSplit]

testY = labels[~trainSplit]



def cnnModel():

    model = Sequential()

    # adding the first convolutionial layer with 32 filters and 5 by 5 kernal size, using the rectifier as the activation function

    model.add(Conv2D(numFilters, (kernalSize1,kernalSize1),input_shape=(numOfRows, numOfColumns,1),activation='relu'))

    # adding a maxpooling layer

    model.add(MaxPooling2D(pool_size=(poolingWindowSz,poolingWindowSz),padding='valid'))

    # adding a dropout layer for the regularization and avoiding over fitting

    model.add(Dropout(dropOutRatio))

    # flattening the output in order to apply the fully connected layer

    model.add(Flatten())

    # adding first fully connected layer with 256 outputs

    model.add(Dense(numNueronsFCL1, activation='relu'))

    #adding second fully connected layer 128 outputs

    model.add(Dense(numNueronsFCL2, activation='relu'))

    #adding second fully connected layer 128 outputs

    model.add(Dense(numNueronsFCL3, activation='relu'))



    # adding softmax layer for the classification

    model.add(Dense(numClasses, activation='softmax'))

    # Compiling the model to generate a model

    adam = optimizers.Adam(lr = 0.0005, decay=1e-6)

    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

    return model

model = cnnModel()

for layer in model.layers:

    print(layer.name)

model.fit(trainX,trainY, validation_split=1-trainSplitRatio,epochs=20,batch_size=batchSize,verbose=2)

score = model.evaluate(testX,testY,verbose=2)

print('Baseline Error: %.2f%%' %(100-score[1]*100))

from keras.models import load_model

import numpy as np

from sklearn import metrics

import matplotlib.pyplot as plt

import os



os.environ['QT_PLUGIN_PATH'] = ''

def plot_cm(cM, labels,title):

    # normalizing the confusionMatrix for showing the probabilities

    cmNormalized = np.around((cM/cM.sum(axis=1)[:,None])*100,2)

    # creating a figure object

    fig = plt.figure()

    # plotting the confusion matrix

    plt.imshow(cmNormalized,interpolation=None,cmap = plt.cm.Blues)

    # creating a color bar and setting the limits

    plt.colorbar()

    plt.clim(0,100)

    # assiging the title, x and y labels

    plt.xlabel('Predicted Values')

    plt.ylabel('Ground Truth')

    plt.title(title + '\n%age confidence')

    # defining the ticks for the x and y axis

    plt.xticks(range(len(labels)),labels,rotation = 60)

    plt.yticks(range(len(labels)),labels)

    # number of occurences in the boxes

    width, height = cM.shape 

    print('Accuracy for each class is given below.')

    for predicted in range(width):

        for real in range(height):

            color = 'black'

            if(predicted == real):

                color = 'white'

                print(labels[predicted].ljust(12)+ ':', cmNormalized[predicted,real], '%')

            plt.gca().annotate(

                    '{:d}'.format(int(cmNormalized[predicted,real])),xy=(real, predicted),

                    horizontalalignment = 'center',verticalalignment = 'center',color = color)

    # making sure that the figure is not clipped

    plt.tight_layout()

    # saving the figure

# loading the pretrained model#loading the testData and groundTruth data

test_x = testX

groundTruth = testY

# evaluating the model

score = model.evaluate(test_x,groundTruth,verbose=2)

print('Baseline Error: %.2f%%' %(100-score[1]*100))

'''

 Creating and plotting a confusion matrix



'''

# defining the class labels

labels = ['Downstairs','Jogging','Sitting','Standing','Upstairs','Walking']

# predicting the classes

predictions = model.predict(test_x,verbose=2)

# getting the class predicted and class in ground truth for creation of confusion matrix

predictedClass = np.zeros((predictions.shape[0]))

groundTruthClass = np.zeros((groundTruth.shape[0]))

for instance in range (groundTruth.shape[0]):

    predictedClass[instance] = np.argmax(predictions[instance,:])

    groundTruthClass[instance] = np.argmax(groundTruth[instance,:])

# obtaining a confusion matrix  

cm = metrics.confusion_matrix(groundTruthClass,predictedClass)

# plotting the confusion matrix

plot_cm(cm, labels,'Confusion Matrix')


from keras.models import load_model

import numpy as np

from sklearn import metrics

import matplotlib.pyplot as plt

import os



os.environ['QT_PLUGIN_PATH'] = ''

def plot_cm(cM, labels,title):

    # normalizing the confusionMatrix for showing the probabilities

    cmNormalized = np.around((cM/cM.sum(axis=1)[:,None])*100,2)

    # creating a figure object

    fig = plt.figure()

    # plotting the confusion matrix

    plt.imshow(cmNormalized,interpolation=None,cmap = plt.cm.Blues)

    # creating a color bar and setting the limits

    plt.colorbar()

    plt.clim(0,100)

    # assiging the title, x and y labels

    plt.xlabel('Predicted Values')

    plt.ylabel('Ground Truth')

    plt.title(title + '\n%age confidence')

    # defining the ticks for the x and y axis

    plt.xticks(range(len(labels)),labels,rotation = 60)

    plt.yticks(range(len(labels)),labels)

    # number of occurences in the boxes

    width, height = cM.shape 

    print('Accuracy for each class is given below.')

    for predicted in range(width):

        for real in range(height):

            color = 'black'

            if(predicted == real):

                color = 'white'

                print(labels[predicted].ljust(12)+ ':', cmNormalized[predicted,real], '%')

            plt.gca().annotate(

                    '{:d}'.format(int(cmNormalized[predicted,real])),xy=(real, predicted),

                    horizontalalignment = 'center',verticalalignment = 'center',color = color)

    # making sure that the figure is not clipped

    plt.tight_layout()

    # saving the figure

# loading the pretrained model

model = load_model('../input/model.h5')

#loading the testData and groundTruth data

test_x = np.load('../input/testData.npy')

groundTruth = np.load('../input/groundTruth.npy')

# evaluating the model

score = model.evaluate(test_x,groundTruth,verbose=2)

print('Baseline Error: %.2f%%' %(100-score[1]*100))

'''

 Creating and plotting a confusion matrix



'''

# defining the class labels

labels = ['Downstairs','Jogging','Sitting','Standing','Upstairs','Walking']

# predicting the classes

predictions = model.predict(test_x,verbose=2)

# getting the class predicted and class in ground truth for creation of confusion matrix

predictedClass = np.zeros((predictions.shape[0]))

groundTruthClass = np.zeros((groundTruth.shape[0]))

for instance in range (groundTruth.shape[0]):

    predictedClass[instance] = np.argmax(predictions[instance,:])

    groundTruthClass[instance] = np.argmax(groundTruth[instance,:])

# obtaining a confusion matrix  

cm = metrics.confusion_matrix(groundTruthClass,predictedClass)

# plotting the confusion matrix

plot_cm(cm, labels,'Confusion Matrix')






