import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os # To check filepath
print(os.listdir("../input")) #to check the name of the directory inside which we have our files
#a particular folder

from glob import glob# To find all the pathnames matching a specified pattern 

files = glob('../input/breast-histopathology-images/**/*', recursive=True) 

files[0]
extention=list() #will store end 3 letters of all the file names (extentions)

for image in files:

    ext=image[-3:]

    if ext not in extention:

        extention.append(ext)

alpha_ext=list()

for ex in extention: #any valid image will have extention in alphabets 

    if ex.isalpha() == True: #this line checks for such alphabet extentions

        alpha_ext.append(ex)

print(alpha_ext)
Data = glob('../input/breast-histopathology-images/**/*.png', recursive=True)  #we extract only png files
print(Data[0]) 
len(Data)
from os import listdir

base_path = "../input/breast-histopathology-images/IDC_regular_ps50_idx5/"

folder = listdir(base_path)

len(folder)
from PIL import Image #adds support for opening, manipulating, and saving many different image file formats

from tqdm import tqdm #adds progress bar for the loops

dimentions=list()

x=1

for images in (Data):

    dim = Image.open(images)

    size= dim.size

    if size not in dimentions:

        dimentions.append(size)

        x+=1

    if(x>8): #going through all the images will take up lot of memory, so therefore we will check until we get three different dimentions.

        break

print(dimentions)
import cv2 #used for computer vision tasks such as reading image from file, changing color channels etc

import matplotlib.pyplot as plt #for plotting various graph, images etc.

def view_images(image): #function to view an image

    image_cv = cv2.imread(image) #reads an image

    plt.imshow(cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)); #displays an image

view_images(Data[12596])
def plot_images(photos) : #to plot multiple image

    x=0

    for image in photos:

        image_cv = cv2.imread(image)

        plt.subplot(5, 5, x+1)

        plt.imshow(cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB));

        plt.axis('off');

        x+=1

plot_images(Data[:25])
total_images= 277524

data = pd.DataFrame(index=np.arange(0, total_images), columns=["patient_id", "path", "target"])



k = 0

for n in range(len(folder)):

    patient_id = folder[n]

    patient_path = base_path + patient_id 

    for c in [0,1]:

        class_path = patient_path + "/" + str(c) + "/"

        subfiles = listdir(class_path)

        for m in range(len(subfiles)):

            image_path = subfiles[m]

            data.iloc[k]["path"] = class_path + image_path

            data.iloc[k]["target"] = c

            data.iloc[k]["patient_id"] = patient_id

            k += 1  
data.head()
data.shape
data.info()
from tqdm import tqdm

import csv #to open and write csv files

Data_output=list()

Data_output.append(["target"])

for file_name in tqdm(Data):

    Data_output.append([file_name[-10:-4]])

with open("output.csv", "w") as f:

    writer = csv.writer(f)

    for val in Data_output:

        writer.writerows([val])
import pandas as pd # Allows the use of display() for DataFrames

data_output = pd.read_csv("output.csv")

#print(data_output.head(),"\n","\n","\n",data_output.tail())

def class_output(images,x,i):  #to display image along with their labels

    fig = plt.figure()

    ax = plt.subplot(2, 2,i)

    ax.set_title(data_output.loc[x].item())

    view_images(images)

    i+=1

    return

k=0 #we have to show only one image of class0 therefore this variable is to check that

l=0 #we have to show only one image of class1 therefore this variable to check that

i=0 #for subplot position

for x in range(1,len(Data)):

    if(data_output.loc[x].item()=="class0" and k!=1):

        k+=1

        i+=1

        class_output(Data[x],x,i)

    elif(data_output.loc[x].item()=="class1" and l!=1):

        l+=1

        i+=1

        class_output(Data[x],x,i)

    elif(k==0 or l==0):

        continue

    else:

        break
def get_cancer_dataframe(patient_id, cancer_id):

    path = base_path + patient_id + "/" + cancer_id

    files = listdir(path)

    dataframe = pd.DataFrame(files, columns=["filename"])

    path_names = path + "/" + dataframe.filename.values

    dataframe = dataframe.filename.str.rsplit("_", n=4, expand=True)

    dataframe.loc[:, "target"] = np.int(cancer_id)

    dataframe.loc[:, "path"] = path_names

    dataframe = dataframe.drop([0, 1, 4], axis=1)

    dataframe = dataframe.rename({2: "x", 3: "y"}, axis=1)

    dataframe.loc[:, "x"] = dataframe.loc[:,"x"].str.replace("x", "", case=False).astype(np.int)

    dataframe.loc[:, "y"] = dataframe.loc[:,"y"].str.replace("y", "", case=False).astype(np.int)

    return dataframe



def get_patient_dataframe(patient_id):

    df_0 = get_cancer_dataframe(patient_id, "0")

    df_1 = get_cancer_dataframe(patient_id, "1")

    patient_df = df_0.append(df_1)

    return patient_df
example = get_patient_dataframe(data.patient_id.values[0])

example.head()
fig, ax = plt.subplots(6,3,figsize=(20, 25))



patient_ids = data.patient_id.unique()



for n in range(6):

    for m in range(3):

        patient_id = patient_ids[m + 3*n]

        example = get_patient_dataframe(patient_id)

        ax[n,m].scatter(example.x.values, example.y.values, c=example.target.values, cmap="coolwarm", s=20);

        ax[n,m].set_title("patient " + patient_id)
class1 = data_output[(data_output["target"]=="class1" )].shape[0]

class0 = data_output[(data_output["target"]=="class0" )].shape[0]

objects=["class1","class0"]

y_pos = np.arange(len(objects))

count=[class1,class0]

plt.bar(y_pos, count, align='center', alpha=0.5)

plt.xticks(y_pos, objects)

plt.ylabel('Number of images')

plt.title('Target distribution')

 

plt.show()
percent_class1=class1/len(Data)

percent_class0=class0/len(Data)

print("Total Class1 images :",class1)

print("Total Class0 images :",class0)

print("Percent of class 0 images : ", percent_class0*100)

print("Percent of class 1 images : ", percent_class1*100)
from sklearn.utils import shuffle #to shuffle the data

Data,data_output= shuffle(Data,data_output)
from tqdm import tqdm

data=list()

for img in tqdm(Data):

    image_ar = cv2.imread(img)

    data.append(cv2.resize(image_ar,(50,50),interpolation=cv2.INTER_CUBIC))
data_output=data_output.replace(to_replace="class0",value=0)

data_output=data_output.replace(to_replace="class1",value=1)
from keras.utils import to_categorical #to hot encode the output labels

data_output_encoded =to_categorical(data_output, num_classes=2)

print(data_output_encoded.shape)
from sklearn.model_selection import train_test_split

data=np.array(data)

X_train, X_test, Y_train, Y_test = train_test_split(data, data_output_encoded, test_size=0.3)

print("Number of train files",len(X_train))

print("Number of test files",len(X_test))

print("Number of train_target files",len(Y_train))

print("Number of  test_target  files",len(Y_test))
X_train=X_train[0:70000]

Y_train=Y_train[0:70000]

X_test=X_test[0:30000]

Y_test=Y_test[0:30000]
from keras.utils import to_categorical #to hot encode the data

from imblearn.under_sampling import RandomUnderSampler #For performing undersampling



X_train_shape = X_train.shape[1]*X_train.shape[2]*X_train.shape[3]

X_test_shape = X_test.shape[1]*X_test.shape[2]*X_test.shape[3]

X_train_Flat = X_train.reshape(X_train.shape[0], X_train_shape)

X_test_Flat = X_test.reshape(X_test.shape[0], X_test_shape)



random_US = RandomUnderSampler(ratio='auto') #Constructor of the class to perform undersampling

X_train_RUS, Y_train_RUS = random_US.fit_sample(X_train_Flat, Y_train) #resamples the dataset

X_test_RUS, Y_test_RUS = random_US.fit_sample(X_test_Flat, Y_test) #resamples the dataset

del(X_train_Flat,X_test_Flat)



class1=1

class0=0



for i in range(0,len(Y_train_RUS)): 

    if(Y_train_RUS[i]==1):

        class1+=1

for i in range(0,len(Y_train_RUS)): 

    if(Y_train_RUS[i]==0):

        class0+=1

#For Plotting the distribution of classes

classes=["class1","class0"]

y_pos = np.arange(len(classes))

count=[class1,class0]

plt.bar(y_pos, count, align='center', alpha=0.5)

plt.xticks(y_pos, objects)

plt.ylabel('Number of images')

plt.title('Class distribution')

 

plt.show()





#hot encoding them

Y_train_encoded = to_categorical(Y_train_RUS, num_classes = 2)

Y_test_encoded = to_categorical(Y_test_RUS, num_classes = 2)



del(Y_train_RUS,Y_test_RUS)



for i in range(len(X_train_RUS)):

    X_train_RUS_Reshaped = X_train_RUS.reshape(len(X_train_RUS),50,50,3)

del(X_train_RUS)



for i in range(len(X_test_RUS)):

    X_test_RUS_Reshaped = X_test_RUS.reshape(len(X_test_RUS),50,50,3)

del(X_test_RUS)

X_test, X_valid, Y_test, Y_valid = train_test_split(X_test_RUS_Reshaped, Y_test_encoded, test_size=0.2,shuffle=True)
print("Number of train files",len(X_train_RUS_Reshaped))

print("Number of valid files",len(X_valid))

print("Number of train_target files",len(Y_train_encoded))

print("Number of  valid_target  files",len(Y_valid))

print("Number of test files",len(X_test))

print("Number of  test_target  files",len(Y_test))
from sklearn.utils import shuffle

X_train,Y_train= shuffle(X_train_RUS_Reshaped,Y_train_encoded)
print(Y_train_encoded.shape)

print(Y_test.shape)

print(Y_valid.shape)
print("Training Data Shape:", X_train.shape)

print("Validation Data Shape:", X_valid.shape)

print("Testing Data Shape:", X_test.shape)

print("Training Label Data Shape:", Y_train.shape)

print("Validation Label Data Shape:", Y_valid.shape)

print("Testing Label Data Shape:", Y_test.shape)
import itertools #create iterators for effective looping

#Plotting the confusion matrix for checking the accuracy of the model

def plot_confusion_matrix(cm, classes,

                          title='Confusion matrix',

                          cmap=plt.cm.Blues):

    """

    This function prints and plots the confusion matrix.

    Normalization can be applied by setting `normalize=True`.

    """

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title)

    plt.colorbar()

    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, rotation=45)

    plt.yticks(tick_marks, classes)



    fmt = '.2f'

    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        plt.text(j, i, format(cm[i, j], fmt),

                 horizontalalignment="center",

                 color="white" if cm[i, j] > thresh else "black")



    plt.ylabel('True label')

    plt.xlabel('Predicted label')

    plt.tight_layout()
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D #Import layers for the model

from keras.layers import Dropout, Flatten, Dense 

from keras.models import Sequential #Our model will be Sequential



model = Sequential()

model.add(Conv2D(filters=32,kernel_size=(3,3),strides=2,padding='same',activation='relu',input_shape=(50,50,3)))

model.add(Flatten()) #Flattens the matrix into a vector

model.add(Dense(2, activation='softmax')) 

model.summary()
model.compile(optimizer= 'adam', loss='categorical_crossentropy', metrics=['accuracy']) #Compiling the model
from keras.callbacks import ModelCheckpoint  #Checkpoint to save the best weights of the model.

checkpointer = ModelCheckpoint(filepath='weights.best.cnn.hdf5', 

                               verbose=1, save_best_only=True) 

model.fit(X_train, Y_train, 

          validation_data=(X_valid, Y_valid),

          epochs=6, batch_size=256, callbacks=[checkpointer], verbose=2,shuffle=True)
model.load_weights('weights.best.cnn.hdf5') #Load the saved weights from file.
predictions = [np.argmax(model.predict(np.expand_dims(feature, axis=0))) for feature in tqdm(X_test)]
from sklearn.metrics import confusion_matrix #to plot confusion matrix

class_names=['IDC(-)','IDC(+)']

cnf_matrix_bench=confusion_matrix(np.argmax(Y_test, axis=1), np.array(predictions))

plot_confusion_matrix(cnf_matrix_bench, classes=class_names,

                      title='Confusion matrix')
from keras.preprocessing.image import ImageDataGenerator  #For Image argumentaton

datagen = ImageDataGenerator(

        shear_range=0.2,

        rotation_range=40,

        width_shift_range=0.2,

        height_shift_range=0.2,

        zoom_range=0.2,

        rescale=1/255.0,

        horizontal_flip=True,

        vertical_flip=True)
X_valid_e=X_valid/255.0 #rescaling X_valid

X_test_e=X_test/255.0 #rescaling X_Test
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D

from keras.layers import Dropout, Flatten, Dense

from keras.models import Sequential



argum_model = Sequential()

argum_model.add(Conv2D(filters=32,kernel_size=(3,3),strides=2,padding='same',activation='relu',input_shape=X_train.shape[1:]))

argum_model.add(Dropout(0.15))

argum_model.add(MaxPooling2D(pool_size=2,strides=2))

argum_model.add(Conv2D(filters=64,kernel_size=(3,3),strides=2,padding='same',activation='relu'))

argum_model.add(Dropout(0.25))

argum_model.add(Conv2D(filters=128,kernel_size=(3,3),strides=2,padding='same',activation='relu'))

argum_model.add(Dropout(0.35))

argum_model.add(Conv2D(filters=512,kernel_size=(3,3),strides=2,padding='same',activation='relu'))

argum_model.add(Dropout(0.45))

argum_model.add(Flatten())

argum_model.add(Dense(2, activation='softmax'))

argum_model.summary()
argum_model.compile(loss='categorical_crossentropy', optimizer='AdaDelta', metrics=['accuracy'])
from keras.callbacks import ModelCheckpoint

checkpointer = ModelCheckpoint(filepath='weights.bestarg.hdf5', verbose=1, save_best_only=True)
batch_size=32

epochs=13

argum_model.fit_generator(datagen.flow(X_train, Y_train, batch_size), 

          validation_data=(X_valid_e, Y_valid), steps_per_epoch=500,

          epochs=epochs,callbacks=[checkpointer], verbose=0)
argum_model.load_weights('weights.bestarg.hdf5')
predictions_arg = [np.argmax(argum_model.predict(np.expand_dims(feature, axis=0))) for feature in tqdm(X_test_e)]
from sklearn.metrics import confusion_matrix

class_names=['IDC(-)','IDC(+)']

cnf_matrix_Arg=confusion_matrix(np.argmax(Y_test, axis=1), np.array(predictions_arg))

plot_confusion_matrix(cnf_matrix_Arg, classes=class_names,

                      title='Confusion matrix')
from keras.applications.vgg16 import VGG16 #downloading model for transfer learning

arg_model = VGG16(include_top=False,weights='imagenet', input_tensor=None, input_shape=None, pooling=None,)
from keras.applications.vgg19 import preprocess_input #preprocessing the input so that it could work with the downloaded model

bottleneck_train=arg_model.predict(preprocess_input(X_train),batch_size=50,verbose=1) #calculating bottleneck features, this inshure that we hold the weights of bottom layers
from keras.applications.vgg19 import preprocess_input

bottleneck_valid=arg_model.predict(preprocess_input(X_valid),batch_size=50,verbose=1)
from keras.applications.vgg19 import preprocess_input

bottleneck_test=arg_model.predict(preprocess_input(X_test),batch_size=50,verbose=1)


from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D

from keras.layers import Dropout, Flatten, Dense

from keras.models import Sequential



model_transfer = Sequential()

model_transfer.add(GlobalAveragePooling2D(input_shape=bottleneck_train.shape[1:]))

model_transfer.add(Dense(32,activation='relu'))

model_transfer.add(Dropout(0.15))

model_transfer.add(Dense(64,activation='relu'))

model_transfer.add(Dropout(0.20))

model_transfer.add(Dense(128,activation='relu'))

model_transfer.add(Dropout(0.25))

model_transfer.add(Dense(256,activation='relu'))

model_transfer.add(Dropout(0.35))

model_transfer.add(Dense(512,activation='relu'))

model_transfer.add(Dropout(0.45))



model_transfer.add(Dense(2, activation='softmax'))



model_transfer.summary()
model_transfer.compile(loss='categorical_crossentropy', optimizer='AdaDelta', metrics=['accuracy'])
from keras.callbacks import ModelCheckpoint

checkpointer = ModelCheckpoint(filepath='weights.bestarg.tranfer.hdf5', verbose=1, save_best_only=True)
batch_size=64

epochs=20

model_transfer.fit(bottleneck_train, Y_train, batch_size,

          validation_data=(bottleneck_valid, Y_valid),

          epochs=epochs,callbacks=[checkpointer], verbose=1)
model_transfer.load_weights('weights.bestarg.tranfer.hdf5')
predictions_transfer = [np.argmax(model_transfer.predict(np.expand_dims(feature, axis=0))) for feature in bottleneck_test]
from sklearn.metrics import confusion_matrix

class_names=['IDC(-)','IDC(+)']

cnf_matrix_transfer_vgg16=confusion_matrix(np.argmax(Y_test, axis=1), np.array(predictions_transfer))

plot_confusion_matrix(cnf_matrix_transfer_vgg16, classes=class_names,

                      title='Confusion matrix')
from keras.applications.vgg19 import VGG19 #downloading model for transfer learning

arg_model = VGG19(include_top=False,weights='imagenet', input_tensor=None, input_shape=None, pooling=None,)
from keras.applications.vgg19 import preprocess_input #preprocessing the input so that it could work with the downloaded model

bottleneck_train=arg_model.predict(preprocess_input(X_train),batch_size=50,verbose=1) #calculating bottleneck features, this inshure that we hold the weights of bottom layers
from keras.applications.vgg19 import preprocess_input

bottleneck_valid=arg_model.predict(preprocess_input(X_valid),batch_size=50,verbose=1)
from keras.applications.vgg19 import preprocess_input

bottleneck_test=arg_model.predict(preprocess_input(X_test),batch_size=50,verbose=1)


from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D

from keras.layers import Dropout, Flatten, Dense

from keras.models import Sequential



model_transfer_vgg19 = Sequential()

model_transfer_vgg19.add(GlobalAveragePooling2D(input_shape=bottleneck_train.shape[1:]))

model_transfer_vgg19.add(Dense(32,activation='relu'))

model_transfer_vgg19.add(Dropout(0.15))

model_transfer_vgg19.add(Dense(64,activation='relu'))

model_transfer_vgg19.add(Dropout(0.20))

model_transfer_vgg19.add(Dense(128,activation='relu'))

model_transfer_vgg19.add(Dropout(0.25))

model_transfer_vgg19.add(Dense(256,activation='relu'))

model_transfer_vgg19.add(Dropout(0.35))

model_transfer_vgg19.add(Dense(512,activation='relu'))

model_transfer_vgg19.add(Dropout(0.45))



model_transfer_vgg19.add(Dense(2, activation='softmax'))



model_transfer_vgg19.summary()
model_transfer_vgg19.compile(loss='categorical_crossentropy', optimizer='AdaDelta', metrics=['accuracy'])
from keras.callbacks import ModelCheckpoint

checkpointer = ModelCheckpoint(filepath='weights.bestarg.tranfer.hdf5', verbose=1, save_best_only=True)
batch_size=64

epochs=20

model_transfer_vgg19.fit(bottleneck_train, Y_train, batch_size,

          validation_data=(bottleneck_valid, Y_valid),

          epochs=epochs,callbacks=[checkpointer], verbose=1)
model_transfer_vgg19.load_weights('weights.bestarg.tranfer.hdf5')
predictions_transfer = [np.argmax(model_transfer.predict(np.expand_dims(feature, axis=0))) for feature in bottleneck_test]
from sklearn.metrics import confusion_matrix

class_names=['IDC(-)','IDC(+)']

cnf_matrix_transfer_vgg19=confusion_matrix(np.argmax(Y_test, axis=1), np.array(predictions_transfer))

plot_confusion_matrix(cnf_matrix_transfer_vgg19, classes=class_names,

                      title='Confusion matrix')
#Bar chart to compare different models

tp=0

for i in range(0,len(Y_test)): #Number of positive cases

    if(np.argmax(Y_test[i])==1):

        tp+=1

#Senstivity of models

confusion_bench_s=cnf_matrix_bench[1][1]/tp *100 

confusion_Arg_s=cnf_matrix_Arg[1][1]/tp *100

confusion_transfer_s_vgg16=cnf_matrix_transfer_vgg16[1][1]/tp *100

confusion_transfer_s_vgg19=cnf_matrix_transfer_vgg19[1][1]/tp *100



classes=["benchmark","data argum","TL Vgg16","TL Vgg19"]

objects=["benchmark","data argum","TL Vgg16","TL Vgg19"]

y_pos = np.arange(len(classes))

count=[confusion_bench_s,confusion_Arg_s,confusion_transfer_s_vgg16,confusion_transfer_s_vgg19]

plt.bar(y_pos, count, align='center', alpha=0.5)

plt.xticks(y_pos, objects)

plt.ylabel('Percentage')

plt.title('Sensitivity')



plt.show()
tp=0

tn=0

for i in range(0,len(Y_test)):  #Number of postive cases

    if(np.argmax(Y_test[i])==1): 

        tp+=1

for i in range(0,len(Y_test)): #number of negative cases

    if(np.argmax(Y_test[i])==0):

        tn+=1

confusion_bench=cnf_matrix_bench[0][0]/tn *100

confusion_Arg=cnf_matrix_Arg[0][0]/tn *100

confusion_transfer_vgg16=cnf_matrix_transfer_vgg16[0][0]/tn *100

confusion_transfer_vgg19=cnf_matrix_transfer_vgg19[0][0]/tn *100

classes=["benchmark","data argum","TL Vgg16","TL Vgg19"]

objects=["benchmark","data argum","TL Vgg16","TL Vgg19"]

y_pos = np.arange(len(classes))

count=[confusion_bench_s,confusion_Arg_s,confusion_transfer_s_vgg16,confusion_transfer_s_vgg19]

plt.bar(y_pos, count, align='center', alpha=0.5)

plt.xticks(y_pos, objects)

plt.ylabel('Percentage')

plt.title('Specificity')



plt.show()
col=['Models','Senstivity','Specificity']

results=pd.DataFrame(columns=col) #dataframe to store the results

results.loc[0]=['Bench',confusion_bench_s,confusion_bench]

results.loc[1]=['Image Arg model',confusion_Arg_s,confusion_Arg]

results.loc[2]=['TL Vgg16',confusion_transfer_s_vgg16,confusion_transfer_vgg16]

results.loc[3]=['TL Vgg16',confusion_transfer_s_vgg19,confusion_transfer_vgg19]
display(results)