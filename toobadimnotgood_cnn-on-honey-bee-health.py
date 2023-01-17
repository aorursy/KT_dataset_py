#Vta92

#from https://github.com/vta92/DS_projects/blob/honey_bees_cnn/honey_bees/honey_bees.ipynb

#updated with more architectures. Best accuracy ~ 96%



import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



#to import images

from keras.preprocessing.image import ImageDataGenerator

from keras.preprocessing import image



from sklearn.utils import shuffle

from sklearn.metrics import accuracy_score





import matplotlib.pyplot as plt

%matplotlib inline
directory = "../input/bee_imgs/bee_imgs/"

#64x64 for faster training.

picture3 = image.load_img(directory+"041_073.png", target_size=(64,64))

picture3
df = pd.read_csv("../input/bee_data.csv")

#create a list to hold the 4d image tensors data

X_pics = [image.load_img(directory+img_name,target_size=(64,64)) for img_name in df["file"]]



#a list of np tensors

X = [np.array(image.img_to_array(i)) for i in X_pics]

#rescale for training, using minmax scaling

X = [i/255.0 for i in X]
#verified to be in order. Should be identical to the picture above

X_pics[2] #third picture
#summary of the target/labels

print(df.health.value_counts())

target_ids = []

for i in df.health:

    if i not in target_ids:

        target_ids.append(i)
#doing a label assignment by using a sparse matrix

#we can also utilize the keras util library for this task



y_keys = {"healthy":np.array([1,0,0,0,0,0]),

         "few varrao, hive beetles":np.array([0,1,0,0,0,0]),

         "Varroa, Small Hive Beetles":np.array([0,0,1,0,0,0]),

         "ant problems":np.array([0,0,0,1,0,0]),

         "hive being robbed":np.array([0,0,0,0,1,0]),

         "missing queen":np.array([0,0,0,0,0,1])}

y = [y_keys[i] for i in df.health]
#helper function

#input as 1 type of target only, return some random indices for image showing

def random_imgs(df,num_images,X_pics):

    index_lst = df["file"].sample(n=num_images,random_state=1).index

    image_lst = []

    for i in index_lst:

        image_lst.append(X_pics[i])

    return image_lst
healthy = random_imgs(df[df["health"]=="healthy"],4,X_pics)

hive_beetles = random_imgs(df[df["health"] == "few varrao, hive beetles"],4,X_pics)

ant_probs = random_imgs(df[df["health"] == "ant problems"],4,X_pics)

hive_robbed = random_imgs(df[df["health"] == "hive being robbed"],4,X_pics)

varroa = random_imgs(df[df["health"] == "Varroa, Small Hive Beetles"],4,X_pics)


#only plot 2x2 images. Helper function. One can always generalize the function if neccessary

def plot_bees(img_lst,title):

    fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(8,8))

    ax[0].imshow(img_lst[0])

    ax[0].set_title(title)

    ax[1].imshow(img_lst[1])

    #ax[1].set_title(title)

    ax[2].imshow(img_lst[2])

    #ax[2].set_title(title)

    ax[3].imshow(img_lst[3])

    #ax[3].set_title(title)

    

    plt.show()

    

#plot_bees(healthy,"healthy")
plot_bees(healthy,"healthy")

plot_bees(hive_beetles,"few varrao, hive beetles")

plot_bees(ant_probs,"ant problems")

plot_bees(hive_robbed,"hive being robbed")

plot_bees(varroa,"Varroa, Small Hive Beetles")
#Keras CNN

from keras.models import Sequential

from keras.layers import Convolution2D

from keras.layers import MaxPooling2D

from keras.layers import Flatten

from keras.layers import Dense

from keras.layers import Dropout

from keras.layers import Activation

from keras.layers import BatchNormalization

from keras import optimizers

from keras.preprocessing.image import ImageDataGenerator

from keras import callbacks





from sklearn.model_selection import train_test_split

history = callbacks.History() #need to be defined first
#LeNet's conv->pool->conv patterns

def train_cnn():

    #to combat overfitting, better optimization for CNN, we'll be using Batch normalization PRIOR to activation.

    #There has been a debate on where to use it, but the consensus has been to use it prior/after non-linearity (activation)

    model = Sequential()



    #3x3 matrix with 11 feature maps in total, conventional. 3d array for colored img, RGB. 255 in term of intensity max/min

    model.add(Convolution2D(11,3,3, input_shape=(64,64,3)))

    model.add(BatchNormalization())

    model.add(Activation("relu"))

    model.add(MaxPooling2D(pool_size=(2,2),padding="SAME"))

    



    model.add(Convolution2D(21,3,3, activation="relu"))

    model.add(BatchNormalization())

    model.add(Activation("relu"))

    model.add(MaxPooling2D(pool_size=(2,2),padding="SAME"))



    #third convo layer with more feature filter size, 41 for better detection.

    model.add(Convolution2D(41,3,3, activation="relu"))

    model.add(BatchNormalization())

    model.add(Activation("relu"))

    model.add(MaxPooling2D(pool_size=(2,2),padding="SAME"))



    #Flattening to input the fully connected layers

    model.add(Flatten())



    #dense layer section with after flattening

    #hidden layer, 200

    model.add(Dense(200, activation="relu"))

    model.add(Dropout(0.2))

    model.add(Dense(6, activation="softmax"))

    

    #smaller learning rate to optimize better. Default has periodic dips

    model.compile(optimizer=optimizers.rmsprop(lr=0.0001), loss="categorical_crossentropy",metrics=["accuracy"])



    return model

#splitting into train,test, val datasets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,stratify=y,random_state=1)

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2,stratify=y_train, random_state=1)
#uncomment for training

model1 = train_cnn()

history1 = model1.fit(np.array(X_train),np.array(y_train),validation_data=(np.array(X_val),np.array(y_val)),

                      verbose=True,shuffle=True,epochs=50)
def model_plot(history,epochs,title,y_range=[0.5,1.0],save=0 ):

    train_losses = history.history["loss"]

    val_losses = history.history["val_loss"]

    plt.plot([i for i in range(0,epochs)],train_losses,val_losses)

    plt.legend(["Train Loss","Val Loss"])

    plt.title(title)

    

    if save == 1:

        plt.savefig(title+"_Losses.jpg")

    plt.show()

    

    

    train_losses = history.history["acc"]

    val_losses = history.history["val_acc"]

    plt.plot([i for i in range(0,epochs)],train_losses,val_losses)

    plt.legend(["Train_acc","Val_acc"])

    plt.title(title)

    plt.ylim(y_range)

    

    if save == 1:

        plt.savefig(title+"_Accuracy.jpg")

    plt.show()
#uncomment for plotting

model_plot(history1,epochs=len(history1.epoch),title="baseline_cnn")
def datasets_split(X,y):

    print("original \n",pd.Series(y).value_counts(normalize=True))

    #split out test set and train set. To make things easier, let's make the y into a pandas Series for stratifying

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,stratify=y,random_state=1)

    #now from the train set, we split out the train set and the validation set. Remember, we can't validate with

    #newly generated data. it won't do our model any good!

    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2,stratify=y_train, random_state=1)

    

    return X_train, X_test, X_val, y_train, y_test, y_val



#example of stratified result

X_train, X_test, X_val, y_train, y_test, y_val = datasets_split(X,y)

print(pd.Series(y_val).value_counts(normalize=True))
#Data Augmentation. Generating additional data for testing

#refer to Keras for extra documentation as well as

#https://machinelearningmastery.com/image-augmentation-deep-learning-keras/ for a brief introduction

def data_aug(X_train):

    datagen = ImageDataGenerator(

        rotation_range=20,

        shear_range=0.2,

        zoom_range=0.2,

        horizontal_flip=True, 

        vertical_flip=True,

        width_shift_range=0.3,

        height_shift_range=0.3)

    

    datagen.fit(X_train)

    # fits the model on batches with real-time data augmentation:

    return datagen



datagen = data_aug(X_train)


#uncomment for training

model2 = train_cnn()

history2 = model2.fit_generator(datagen.flow(np.array(X_train),np.array(y_train),batch_size=50),

                              validation_data= (np.array(X_val),np.array(y_val)),

                              steps_per_epoch=len(X_train) / 50,epochs=50)
#uncomment for plotting after training

model_plot(history2,epochs=50,title="basic_CNN_with_image_augmentation")
df.head()
#The point of this excercise is to see whether oversampling helps at all, or we are just better off using the image augmentation technique

#we'll upsample until our each of our target has as many datapoints as the highest target value (healthy)

#X_pics were defined from the very beginning to contain all our images, and y contains all the targets.

def max_oversampling(df,X_pics,y):

    

    #we need to resplit the original dataset in a form of dataframe.

    #this is a total df with additional pixels and target columns

    

    df["pixels"] = [np.array(image.img_to_array(i)) for i in X_pics]

    df["pixels"] = df["pixels"]/255.0 #we re-imported, so have to rescale (otherwise the cnn won't learn)

    df["target"] = y

    

    #so we'll input the whole df, with pixels and target to help create a true"oversample" split solely with

    #the training data, not contaminating the val or test. We'll ignore the test/val sets

    #the df itself (X_train), should contain both the target and pixels.

    #we only use df_train (including both target and features as added in the above lines)

    df_train, temp1, temp2, temp3, temp4, temp5 = datasets_split(df,y)

    #print(df_train.pixels[0:5])

    #print(df_train.head())

    max_size = df_train["health"].value_counts().max()

    lst = [df_train]

    for classification, group in df_train.groupby('health'):

        lst.append(group.sample(max_size-len(group), replace=True))

    df_new = pd.concat(lst)

    return df_new
#re-split again for no reason, then just to keep the reference the same prior to any partition.

#X_train, X_test, X_val, y_train, y_test, y_val = datasets_split(X,y)
#df_new is the new training set only, NOT val or test set

df_new = max_oversampling(df,X_pics,y)
#quite important to shuffle our training set, since our function to upsample only concat the pd's together at the end

#for each categories. Thus, there is no randomization in our current df based on labels/target

df_new = shuffle(df_new)

print(df_new.health.value_counts())

df_new.pixels.head()
#now we need to extract the right y's from the new training dataframe using the previously defined keys (y_keys dict)

X_upsampled = df_new.pixels.tolist()

y_upsampled = [y_keys[i] for i in df_new["health"]]
#uncomment for training

model3 = train_cnn()

history3 = model3.fit(np.array(X_upsampled),np.array(y_upsampled),validation_data=(np.array(X_val),np.array(y_val)), verbose=True,shuffle=True,epochs=50)
#uncomment for plotting

model_plot(history3,epochs=50,title="max_oversampling")
print(df_new.health.value_counts())

print(df.health.value_counts()) 

#the whole dataset only has 40 missing queens data. So less than 80% of that will be in training set.

#oversampling these data to 2000+ points won't help it learn from the 30 data points. The same can be applied to other

#labels/targets
#let's create a proportional upsampling, where we don't upsampe the "healthy" label, and for every other label, we would

#upsample them by at most 4x, with the majority label having no multiplier.



#proportional_dict will contain the label (key), and mulplier of upsampling

#after playing around with the ratio, this is one of the more optimal result we've got

multiplier_dict = {"healthy":1,

         "few varrao, hive beetles":2, 

         "Varroa, Small Hive Beetles":2,

         "ant problems":2,

         "hive being robbed":3,

         "missing queen":4}

def proportional_oversampling(df,X_pics,y,multiplier_dict):

    

    #we need to resplit the original dataset in a form of dataframe.

    #this is a total df with additional pixels and target columns

    

    df["pixels"] = [np.array(image.img_to_array(i)) for i in X_pics]

    df["pixels"] = df["pixels"]/255.0 #normalize due to fresh import of dataframe

    df["target"] = y

    #again, we only care about the training oversampling, the val and the test should still be the same!

    df_train, temp1, temp2, temp3, temp4, temp5 = datasets_split(df,y)

    

    label_ids = df_train.health.value_counts().index.tolist() #list

    label_size = df_train.health.value_counts().values #list

    print("initial train set:\n",df_train.health.value_counts())

    result = [] #a list to hold all the sampled df's

    for i in range(len(label_ids)):

        #this function will 1: filter our the label/target of each "health" column,

        #multiply the number of sample size by the multiplier factor in the dictionary by randomly sampling with replacement

        #and finally append to the list for concat back to a single training dataframe.

        df_sampled = df_train[df_train["health"] == label_ids[i]].sample(n=(multiplier_dict[label_ids[i]]*label_size[i]),

        replace=True,random_state=101)

        result.append(df_sampled)

    df_new = pd.concat(result)

    #Shuffle so our data will come out in batches representing a generalized siutation. Otherwise, there will only

    #be 1 label for each batch.

    df_new = shuffle(df_new,random_state=202)

    return df_new
#since the last df_new was useless, we'll be re-using this variable name to establish our new baseline

df_new = proportional_oversampling(df,X_pics,y,multiplier_dict)
df_new.health.value_counts()
#example of stratified result

X_train, X_test, X_val, y_train, y_test, y_val = datasets_split(X,y)
X_upsampled2 = df_new.pixels.tolist()

y_upsampled2 = [y_keys[i] for i in df_new["health"]]

datagen = data_aug(X_upsampled2)



#uncomment for training



model4 = train_cnn()

history4 = model4.fit_generator(datagen.flow(np.array(X_upsampled2),np.array(y_upsampled2),batch_size=50),

                              validation_data=(np.array(X_val),np.array(y_val)),

                              steps_per_epoch=len(X_train)/50,epochs=50)
#uncomment for plotting

model_plot(history4,epochs=50,title="proportionally oversampled & img augmentation")
from keras import layers, Input, Model

#Since both Alex's net and VGG make use of the multiple conv blocks prior to pooling, let's just uniquely

#modify our structure to take advantage of these new concepts (in a way to fit our 64x64x3 features)

#and see if we can get any sort of improvement on our previous baseline cnn with augmentation. One could import

#the architecture from keras and tweak them; but let's build them from scratch!



#AlexNet's and VGG's conv->pool->conv->pool->[[conv->conv->conv->pool]] blocks ->fc

#pool sizes of 2x2 because we only have 64x64 features for each channel. We'll only be using 1 unit "block". Each block and subsequent blocks have the same hyperparameters.

#needed to reduce down learning rate by 2/3, we're seeing optimization issues

#Adam in favor of sgd



#to do: refactor the code with a helper function

def conv_block(input_tensor,avg_pool=False):

    convb = input_tensor

    for _ in range(3):

        convb = layers.Conv2D(filters=121,kernel_size=(3,3),padding="SAME")(convb)

        convb = layers.BatchNormalization()(convb)

        convb = layers.Activation("relu")(convb)

    if avg_pool == True:

        return layers.AveragePooling2D(pool_size=(2,2),padding="SAME")(convb)

    return layers.MaxPool2D(pool_size=(2,2),padding="SAME")(convb)

    

####



def train_alex_vgg():

    input_tensor = Input(shape=(64,64,3,))

    conv1 = layers.Conv2D(filters=31,kernel_size=(7,7),strides=(1,1),padding="SAME")(input_tensor) 

    conv1 = layers.BatchNormalization()(conv1)

    conv1 = layers.Activation("relu")(conv1)

    pool1 = layers.MaxPool2D(pool_size=(2,2),padding="SAME")(conv1)

    

    conv2 = layers.Conv2D(filters=71,kernel_size=(5,5),padding="SAME")(pool1)

    conv2 = layers.BatchNormalization()(conv2)

    conv2 = layers.Activation("relu")(conv2)

    pool2 = layers.MaxPool2D(pool_size=(2,2),padding="SAME")(conv2)

    

    #Block1

    convb1 = conv_block(pool2,avg_pool=False)    

    #Block2

    convb2 = conv_block(convb1,avg_pool=False)    

    #Block3

    convb3 = conv_block(convb2,avg_pool=False)

    #Block4

    convb4 = conv_block(convb3,avg_pool=True) #1x1 at this point, deeper architecture won't help.

    

    flat = layers.Flatten()(convb4)

    dense1 = layers.Dense(units=1000,activation="relu")(flat)

    dense1 = layers.Dropout(rate=0.2)(dense1)

    dense2 = layers.Dense(units=500)(dense1)

    dense2 = layers.Dropout(rate=0.2)(dense2)

    dense3 = layers.Dense(units=500)(dense2)

    dense3 = layers.Dropout(rate=0.2)(dense3)

    

    output_tensor = layers.Dense(units=6,activation="softmax")(dense3)



    model = Model(inputs=input_tensor, outputs=output_tensor)

    model.compile(optimizer=optimizers.Adam(lr=0.00003), loss="categorical_crossentropy",metrics=["accuracy"]) #lr needs to be this low, otherwise cannot converge with 10x mag.

    print(model.summary())

    

    return model
model5 = train_alex_vgg()

history5 = model5.fit_generator(datagen.flow(np.array(X_train),np.array(y_train),batch_size=50),

                              validation_data= (np.array(X_val),np.array(y_val)),

                              steps_per_epoch=len(X_train)/50,epochs=100) 
#uncomment for plotting

model_plot(history5,epochs=len(history5.epoch),title="Alexnet_vgg_cnn",save=1)
#inception-like architecture

#another popular architecture with multiple branches to learn the spatials features differently.

#also, we'll be using conv1x1 to reduce the number of channels down to 1 prior to looking.

#Since deep networks tend to overfit, we'll see how each conv layer of different kernel sizes will eval the data

#going wide instead of deeper.

#known to be better than alex net and vgg



#hard-coded like the original block, with dimensional reduction

def inception_block(input_tensor):

    branch1 = layers.Conv2D(filters=5,kernel_size=(1,1),padding="SAME")(input_tensor) 

    branch1 = layers.Conv2D(filters=5,kernel_size=(3,3),padding="SAME")(branch1)

    

    branch2 = layers.Conv2D(filters=5,kernel_size=(1,1),padding="SAME")(input_tensor)

    branch2 = layers.Conv2D(filters=5,kernel_size=(5,5),padding="SAME")(branch2)

    

    branch3 = layers.MaxPooling2D(pool_size=(3,3),strides=(1,1),padding="SAME")(input_tensor)

    branch3 = layers.Conv2D(filters=5,kernel_size=(1,1),padding="SAME")(branch3)

    #print(branch1.shape,branch2.shape,branch3.shape)

    output_tensor = layers.concatenate([branch1,branch2,branch3],axis=3)

    return output_tensor



def train_inception():

    input_tensor = Input(shape=(64,64,3,))

    #we won't go deep into the first stack of convolution due to the input sizes

    conv1 = layers.Conv2D(filters=64,kernel_size=(3,3),strides=(2,2),padding="SAME")(input_tensor) 

    conv1 = layers.BatchNormalization()(conv1)

    conv1 = layers.Activation("relu")(conv1)   

    conv2 = layers.Conv2D(filters=128,kernel_size=(3,3),padding="SAME")(conv1)

    conv2 = layers.BatchNormalization()(conv2)

    conv2 = layers.Activation("relu")(conv2)

    conv3 = layers.Conv2D(filters=128,kernel_size=(3,3),padding="SAME")(conv2)

    conv3 = layers.BatchNormalization()(conv3)

    conv3 = layers.Activation("relu")(conv3)

    pool1 = layers.MaxPooling2D(pool_size=(2,2),padding="SAME")(conv3)

    

    inception1 = inception_block(pool1)

    inception2 = inception_block(inception1)

    inception3 = inception_block(inception2)

    inception4 = inception_block(inception3)

    inception5 = inception_block(inception4)

    inception6 = inception_block(inception5)

    inception7 = inception_block(inception6)

    pool_final = layers.AveragePooling2D(pool_size=(2,2),padding="SAME")(inception6)

    

    flat = layers.Flatten()(pool_final)

    dense1 = layers.Dense(units=500,activation="relu")(flat)

    dense1 = layers.Dropout(rate=0.2)(dense1)

    

    output_tensor = layers.Dense(units=6,activation="softmax")(dense1)



    model = Model(inputs=input_tensor, outputs=output_tensor)

    model.compile(optimizer=optimizers.Adam(lr=0.0001), loss="categorical_crossentropy",metrics=["accuracy"]) #lr needs to be this low, otherwise cannot converge with 10x mag.

    print(model.summary())

    return model
model6 = train_inception()

history6 = model6.fit_generator(datagen.flow(np.array(X_train),np.array(y_train),batch_size=100),

                              validation_data= (np.array(X_val),np.array(y_val)),

                              steps_per_epoch=len(X_train)/100,epochs=100) 
#uncomment for plotting

model_plot(history6,epochs=len(history6.epoch),title="inception_cnn",save=1)
#Let's put them all together

model_plot(history1,epochs=len(history1.epoch),title="baseline_cnn",y_range=[0.5,1],save=1)

model_plot(history2,epochs=len(history2.epoch),title="img_augmentation",y_range=[0.5,1],save=1)

model_plot(history3,epochs=len(history3.epoch),title="max_oversampling",y_range=[0.5,1],save=1)

model_plot(history4,epochs=len(history4.epoch),title="proportional_oversampling_and_img_aug",y_range=[0.5,1],save=1)

model_plot(history5,epochs=len(history5.epoch),title="alex_vgg_block",y_range=[0.5,1],save=1)

model_plot(history6,epochs=len(history6.epoch),title="inception_block",y_range=[0.5,1],save=1)


#to save all the weights for the future

'''

model1.save_weights("model1.h5",overwrite=False)

model2.save_weights("model2.h5",overwrite=False)

model3.save_weights("model3.h5",overwrite=False)

model4.save_weights("model4.h5",overwrite=False)

model5.save_weights("model5.h5",overwrite=False)

model6.save_weights("model6.h5",overwrite=False)

'''
#to load the saved weights, remember to run all the cnn_defined stuff

'''

model1 = train_cnn()

model2 = train_cnn()

model3 = train_cnn()

model4 = train_cnn()

model5 = train_alex_vgg()

model6 = train_inception()



model1.load_weights("model1.h5")

model2.load_weights("model2.h5")

model3.load_weights("model3.h5")

model4.load_weights("model4.h5")

model5.load_weights("model5.h5")

model6.load_weights("model6.h5")

'''
#input is the list of models

def multi_pred(models,X_test,y_test):

    preds = [] #store all the predictions

    for model in models:

        pred_ = model.predict(np.array(X_test))

        preds.append([np.argmax(i) for i in pred_]) #return the index with maximum probability

    

    #transpose due to the fact that dataframe takes in the sample number as columns, and the 3 models as rows

    preds = pd.DataFrame(data=np.array(preds).T,columns=["model2","model3","model4","model5","model6"])

    preds["target"] = y_test

    preds["target"] = preds["target"].apply(np.argmax) #vectorization, no parameter input

    return preds
preds = multi_pred([model2,model3,model4,model5,model6],X_test,y_test)

preds.head(10)
accuracy_2 = accuracy_score(preds["model2"],preds["target"])

accuracy_3 = accuracy_score(preds["model3"],preds["target"])

accuracy_4 = accuracy_score(preds["model4"],preds["target"])

accuracy_5 = accuracy_score(preds["model5"],preds["target"])

accuracy_6 = accuracy_score(preds["model6"],preds["target"])
print(accuracy_2,accuracy_3,accuracy_4,accuracy_5,accuracy_6)
#input input is a test_prediction dataframe as defined by "preds" above

def accuracy_table(df):

    models_lst = df.columns.tolist() #putting model names to a list

    models_lst.remove("target")

    health_cols = ["healthy","few varrao, hive beetles","Varroa, Small Hive Beetles","ant problem","hive being robbed", "missing queen"]

    df_acc = []

    

    for model in models_lst:

        acc_lst = []

        for i in range(6): #i is health from 0 -> 5 as defined previously

            size = (df["target"] == i).sum()

            true = ((df[model] == i) &((df["target"] == i))).sum()

            acc_lst.append(true/size)

        #print(acc_lst)

        df_acc.append(acc_lst) #append each model accuracy into our df

    df_acc = pd.DataFrame(df_acc,columns=health_cols)

    df_acc.index = models_lst

    return df_acc
df_acc = accuracy_table(preds)

df_acc
from keras.models import load_model

model2.summary()


#now let's grab a sample from previous import. Let's use a "ant_problem" label that we had pulled out randomly prev

#this is crucial because ant problems are generally well-classified as seen above. We'll see what makes our filters

#to recognize that behavior

vis_sample = ant_probs[0]

vis_sample = np.expand_dims(vis_sample,axis=0) #need the extra dimension for processing

print(np.shape(vis_sample))

vis_sample = vis_sample/255.
plt.imshow(vis_sample[0])
from keras import models

layer_outputs = [layer.output for layer in model2.layers[:12]] #top 10 layer, look at the summary above (top 10 lines) 

activation_model = models.Model(inputs=model2.input, outputs=layer_outputs)
layer_outputs
activations = activation_model.predict(vis_sample)
#first layer activation

first = activations[0]

#4th channel of the first layer on the ant-problem picture. There are 11 filters total in the first conv layer

for i in range(0,11):

    plt.matshow(first[0,:,:,i],cmap='viridis')
#visualizing every channel in all those conv layers

#helper function courtesy of Francois Challet in Deep Learning With Python book

np.seterr(invalid='warn')

layer_names = []

for layer in model2.layers[:12]: #up to 11 layers stack

    layer_names.append(layer.name)

    #print(layer_names)

    

img_per_row = 10

for layer_name, layer_activation in zip(layer_names, activations):

    n_features = layer_activation.shape[-1]

    size = layer_activation.shape[1]

    

    n_cols = n_features//img_per_row

    display_grid = np.zeros((size*n_cols,img_per_row*size))

    

    for col in range(n_cols):

        for row in range(img_per_row):

            channel_img = layer_activation[0,:,:,col*img_per_row+row]

            channel_img -= channel_img.mean()

            channel_img /= channel_img.std()

            channel_img *64

            channel_img += 128

            #channel_img = np.clip(channel_img,0,255).astype("uint8")

            display_grid[col*size : (col + 1) * size, row*size : (row + 1)*size] = channel_img



    scale = 1. / size

    plt.figure(figsize=(scale * display_grid.shape[1], scale * display_grid.shape[0]))

    plt.title(layer_name)

    plt.grid(False)

    plt.imshow(display_grid, aspect='auto', cmap='viridis')