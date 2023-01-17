import pandas as pd # used for reading and manipulating the data
import seaborn as sns # used for plotting
import matplotlib.pyplot as plt
from keras.utils.np_utils import to_categorical # to_categorical converts vectors to "1-hot" vectors
from sklearn.model_selection import train_test_split
from sklearn import svm
import numpy as np
from keras.optimizers import Adam, RMSprop
from keras.losses import categorical_crossentropy
from keras.models import Sequential # used to define the ML model
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Lambda, Flatten, Dense, Conv2D, MaxPool2D, Dropout
from keras.callbacks import ReduceLROnPlateau, LearningRateScheduler
from sklearn.metrics import confusion_matrix


dataToUse = 5000 # how much data out of the dataset do we want to use? theres 42,000 in total.
totalData = 42000
imageDimensions = [28,28]
# store the paths to the train and test datasets. Using the kaggle notebook these are available 
# via the "Data" tab at the top right corner of the interface.
datasetTrainPath = "../input/digit-recognizer/train.csv"
datasetTestPath = "../input/digit-recognizer/test.csv"
#datasetTrain = pd.read_csv(datasetTrainPath, nrows = dataToUse)
#datasetTest = pd.read_csv(datasetTestPath, nrows = dataToUse)
datasetTrain = pd.read_csv(datasetTrainPath)
datasetTest = pd.read_csv(datasetTestPath)
datasetTrain.head(5) ## look at the first few items of each dataset
datasetTest.head(5)
# split the train data into x and y, in this case digit image and digit label, respectiely.
datasetTrainY = datasetTrain["label"]
datasetTrainX = datasetTrain.drop(labels = ["label"] , axis = 1)
# the test data can't be split as it is unlabeled and contains "only x".
# check data for uniformity
labelDistributionPlot = sns.countplot(datasetTrainY)
datasetTrainX.isnull() # converts all non-null values to "False" and all null values to "True"
datasetTrainX.isnull().any() # returns True or False in case there are / there aren't any null values for every column
datasetTrainX.isnull().any().describe() # describes the results in a more human-readable manner
datasetTrainY.head(5)
datasetTrainY.isnull().any() # for y we can't use "describe" because "any" already returns just one result because
# there is just one single column
datasetTest.isnull().any().describe()
# values converts the first ([0]) datasetTrainX data into a numpy.ndarray
# reshape reshapes trainImageExample into a 28x28 grayscale image. the 1 is for grayscale (for RGB we would have 3)
# and the -1 is for the number of images, which is set to -1 in order to make the reshape function
# pick the correct number automatically based on the other dimensions chosen.
trainImageExample = datasetTrainX.values.reshape(-1,*imageDimensions,1)
# plot the whole width and height (":,:") of the third ([2]) train image ("0" is for the only (grayscale) channel)
datasetTrainPlot = plt.imshow(trainImageExample[2][:,:,0])
plt.figure(figsize=(12,10))
numToPlot = 30
for i in range(30):
    plt.subplot(3,10,i+1)
    plt.imshow(datasetTrainX.values[i].reshape(*imageDimensions), interpolation = "nearest")
plt.show()
plt.figure(figsize=(12,10))
numToPlot = 30
for i in range(30):
    plt.subplot(3,10,i+1)
    plt.imshow(datasetTest.values[i].reshape(*imageDimensions), interpolation = "nearest")
plt.show()
# split 20% of the train data set into a validation data set in a random fashion.
# if the data was not evenly distributed we would have possibly needed
# to have used stratify = True in order to avoid unequally splitting the data
randomSeed = 0
datasetTrainX , datasetValX , datasetTrainY , datasetValY = train_test_split(datasetTrainX, datasetTrainY,
                                                                             test_size = 0.2, random_state = randomSeed)
# it is probably enough to either have .astype("float32") or the ".0" in order to convert the expression to a float
datasetTrainX = datasetTrainX.astype("float32") / 255.0
datasetValX = datasetValX.astype("float32") / 255.0
datasetTrainY = to_categorical(datasetTrainY, num_classes = 10)
datasetValY = to_categorical(datasetValY, num_classes = 10)
# reshape the training data
# values converts datasetTrainX into a numpy.ndarray
# reshape datasetTrainX into a 28x28 grayscale image. the 1 is for grayscale (for RGB we would have 3)
# and the -1 is for the number of images, which is set to -1 in order to make the reshape function
# pick the correct number automatically based on the other dimensions chosen.
# instead of -1 we could have used datasetTrainX.shape[0] and datasetValX.shape[0]
datasetTrainX = datasetTrainX.values.reshape(-1,*imageDimensions,1)

# reshape the validation data
datasetValX = datasetValX.values.reshape(-1,*imageDimensions,1)
#batch_size = int(64*(dataToUse/totalData))
# explanation on batch_size: with 42000 examples and 80% 20% split we have 42000 * 0.8 = 33600 train examples
# bath_size 64 means we will have 33600/64 = 525 batches.
# steps_per_epoch should in that case not be 33600 but 525 because each step is a batch.
# this is unlike whats done in https://www.kaggle.com/poonaml/deep-neural-network-keras-way/notebook
# where the number of steps is simply the number of train data examples. I don't understand why his notebook compiles.
# similarly validation_steps should not be genFlowVal.n but genFlowVal.n/batch_size
batch_size = 64

gen = ImageDataGenerator() # used for data augmentation, which we don't do for this model.
genFlowTrain = gen.flow(datasetTrainX, datasetTrainY, batch_size = batch_size)
genFlowVal = gen.flow(datasetValX, datasetValY, batch_size = batch_size)
datasetTrainXMean = datasetTrainX.mean().astype(np.float32)
datasetTrainXstd = datasetTrainX.std().astype(np.float32)

def standardize(x):
    return (x-datasetTrainXMean)/datasetTrainXstd

model = Sequential([
    Lambda(standardize, input_shape=(*imageDimensions,1)),
    Flatten(),
#    Dense(512, activation="relu"),
    Dense(10, activation="softmax")]) # 10 is the output size, softmax gives probability weights for each output

model.compile(optimizer = RMSprop(lr=0.001), loss = "categorical_crossentropy", metrics = ["accuracy"])
#model.compile(optimizer = Adam(lr=0.001), loss = "categorical_crossentropy", metrics = ["accuracy"])


print("the model's input shape is " + str(model.input_shape) + " and its output shape is " + str(model.output_shape))
epochs = 3
#steps_per_epoch = genFlowTrain.n # total number of examples in batchesTrain
steps_per_epoch = genFlowTrain.n // batch_size
validation_steps = genFlowVal.n // batch_size
#history = model.fit(datasetTrainX, datasetTrainY, batch_size = batch_size, epochs = epochs,
#                   validation_data = (datasetValX, datasetValY), verbose = 2)
history = model.fit_generator(generator = genFlowTrain, steps_per_epoch = steps_per_epoch,epochs = epochs,
                             validation_data = genFlowVal , validation_steps = validation_steps)
plt.plot(history.history["loss"], color = "r")
plt.plot(history.history["val_loss"], color = "g")
plt.title("train (red) and validation (green) losses")
plt.plot(history.history["accuracy"], color = "r")
plt.plot(history.history["val_accuracy"], color = "g")
plt.title("train (red) and validation (green) accuracy")
optimizer = RMSprop(lr = 0.001, rho = 0.9, epsilon = 1e-8, decay = 0.0)
model = Sequential()
# note the input_shape = (28,28,1) for the first Conv2D layer to account for the size of the input
model.add(Conv2D(filters = 32, kernel_size = (3,3), activation = "relu", input_shape = (*imageDimensions,1)))
model.add(Conv2D(filters = 32, kernel_size = (5,5), padding = "Same", activation = "relu"))
model.add(MaxPool2D(pool_size = (2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(filters = 64, kernel_size = (3,3), padding = "Same", activation = "relu"))
model.add(Conv2D(filters = 64, kernel_size = (3,3), padding = "Same", activation = "relu"))
model.add(MaxPool2D(pool_size = (2,2), strides = (2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256, activation = "relu"))
model.add(Dropout(0.5))
model.add(Dense(10, activation = "softmax")) # note "10" because thats the output size (number of digits)
# also note "softmax" which returns probabilities (weights) for each output (digit)
# note that the metrics used is accuracy, which is only used at the evaluation stage, not in the training stage.
model.compile(optimizer = RMSprop(), loss = categorical_crossentropy, metrics = ["accuracy"])
epochs = 10
batch_size = 64

gen = ImageDataGenerator() # used for data augmentation, which we don't do for this model.
#gen.fit(datasetTrainX)
genFlowTrain = gen.flow(datasetTrainX, datasetTrainY, batch_size = batch_size)
genFlowVal = gen.flow(datasetValX, datasetValY, batch_size = batch_size)
steps_per_epoch = genFlowTrain.n // batch_size # // performs rounded-down division to int
validation_steps = genFlowVal.n // batch_size 
learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 
                                            patience=3, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.0001)

history = model.fit_generator(generator = genFlowTrain, steps_per_epoch = steps_per_epoch, epochs = epochs,
                             validation_data = genFlowVal, validation_steps = validation_steps,
                             )
plt.plot(history.history["loss"], color = "r")
plt.plot(history.history["val_loss"], color = "g")
plt.title("train (red) and validation (green) losses")
plt.plot()
plt.plot(history.history["accuracy"], color = "r")
plt.plot(history.history["val_accuracy"], color = "g")
plt.title("train (red) and validation (green) accuracy")
plt.plot()
gen = ImageDataGenerator(zoom_range = 0.15, height_shift_range = 0.15, width_shift_range = 0.15, rotation_range = 15)
genFlowTrain = gen.flow(datasetTrainX, datasetTrainY, batch_size = batch_size)
genFlowVal = gen.flow(datasetValX, datasetValY, batch_size = batch_size)
annealer = LearningRateScheduler(lambda x: 1e-3 * 0.9 ** x)

epochs = 30
# use just a subset of the validation data in order to speed up the calculation. We will use the rest of it later.
validationDataSample = (datasetValX[:500,:], datasetValY[:500,:])
history2 = model.fit_generator(generator = genFlowTrain, steps_per_epoch = steps_per_epoch, epochs = epochs,
                             validation_data = validationDataSample, validation_steps = validation_steps,
                               callbacks = [annealer]
                             )
final_val_loss, final_val_acc = model.evaluate(datasetValX, datasetValY, verbose=0)
print("Final loss: {0:.4f}, final accuracy: {1:.4f}".format(final_val_loss, final_val_acc))
plt.plot(history2.history["loss"], color = "r")
plt.plot(history2.history["val_loss"], color = "g")
plt.title("train (red) and validation (green) losses")
plt.plot()
plt.plot(history2.history["accuracy"], color = "r")
plt.plot(history2.history["val_accuracy"], color = "g")
plt.title("train (red) and validation (green) accuracy")
plt.plot()
predictionsProbabilities = model.predict(datasetValX)
predictions = np.argmax(predictionsProbabilities, axis = 1)
actualLabels = np.argmax(datasetValY, axis = 1)
cm = confusion_matrix(actualLabels,predictions)
print(cm)

datasetTestArray = np.array(datasetTest)
datasetTestArray = datasetTestArray.reshape(-1,28,28,1)
predictions = model.predict(datasetTestArray)
print(predictions)
predictionsTest = []
for prediction in predictions:
    predictionsTest.append(np.argmax(prediction))
# create a new data structure with our predicted image labels
submission = pd.DataFrame({
    "ImageId" : datasetTest.index+1,
    "Label": predictionsTest
})
# convert it to a csv and save
submission.to_csv('submission.csv', index = False)