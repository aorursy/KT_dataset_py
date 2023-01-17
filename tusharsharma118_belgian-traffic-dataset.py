# Install tf 2.0 preview GPU version

!pip install tensorflow-gpu==2.0.0-beta1 
# Common Imports

import os                                   # For os commands (dir cwd etc)

import zipfile                              # for extracting data set files

                                 

import skimage                              # for scikit-learn image operations

from matplotlib import pyplot as plt        # for visualizing data

import numpy as np                          # for numerical python

import random                               # for random sampling in range(),number_of_values



import tensorflow as tf

print(tf.test.gpu_device_name())

print(tf.version)
print(os.getcwd())

print(os.listdir('../'))
DATASET_PATH = "../input/belgiumts-dataset/"

os.chdir(DATASET_PATH)

print(os.getcwd())

print(os.listdir())
def load_images(data_directory):

  

  # lists to store Images and labels

  images = []

  labels = []

  log_index = 0

  # get list of all directories present in the data_directory path

  directories = [dir for dir in os.listdir(data_directory)

                 if os.path.isdir(os.path.join(data_directory,dir))] # to make sure that we include only directories and not any files present in the folder

  print(len(directories))

  for dir in directories:

    current_directory = os.path.join(data_directory,dir)

    # Gather all fileNames in the given directory to load images into images array using sklearn

    file_names = [os.path.join(current_directory,file) 

                  for file in os.listdir(current_directory)

                  if file.endswith('.ppm')

                 ]

    

    # Load all given Images into the Images array

    for file in file_names:

      images.append(skimage.data.imread(file))

      labels.append(int(dir))

      log_index+=1

      # print('Loading File: {0}'.format(log_index))

  print('Successfully Loadded  {0} images!'.format(len(images)))

  return np.array(images),np.array(labels)







# LOAD IMAGES 

  

ROOT_PATH = os.getcwd()  

TRAININ_DATA_PATH = ROOT_PATH + '/BelgiumTSC_Training/Training'  

TEST_DATA_PATH = ROOT_PATH + '/BelgiumTSC_Testing/Testing'

  

training_images, training_labels = load_images(TRAININ_DATA_PATH)

print('Training Data Sucessfully Loaded!!')

testing_images,testing_labels = load_images(TEST_DATA_PATH)

print('Test data sucessfully loaded!!')

index = 1                          # Replace the index to check out the shape of input images

print('Dimension of Image at index ' + str(index) + ':', training_images[index].shape)  

print('Number of training Images :' , training_images.size)

print('Number of Dimensions of Images array : ',training_images.ndim)                 # ndims - number of dimensions for np array images



print('Dimensions for labels :', training_labels.shape)

print('Label for Image at index ' + str(index) +': ',training_labels[index])

print('Number of Classes : ',len(set(training_labels)))



print('Some additional tidbits about the memory requirements of data ')

print('Size of an individual image: ' ,training_images.itemsize)
def transform_images(images,height,width):

  transformed_images = [skimage.transform.resize(image,(height,width)) for image in images]

  return  np.array(transformed_images)

from sklearn.model_selection import train_test_split
print(training_images[0].shape)

print(training_images.size)

full_data = np.append(training_images,testing_images)

full_labels = np.append(training_labels,testing_labels)

print('Collated Image Data size :',full_data.shape)

print('Collated Image Labels size :',full_labels.shape)

## Remove Data for classes with unknown labels

##idx = [index for index in range(len(full_labels)) if full_labels[index] != 42  if full_labels[index] != 43  if full_labels[index] != 55 ]
##print('Filtered Data Lenght:',len(idx))

##print('Full Data Length :',len(full_data))
"""

full_data = full_data[idx]

full_labels = full_labels[idx]

#full_labels = [label-2 for label in full_labels if  55 > label > 43]

#full_labels = [label-1 for label in full_labels if   label > 55]

temp = []

for label in full_labels:

    if label < 55 and label > 43 :

        temp.append(label - 2)

    elif label > 55 :

        temp.append(label - 1)

    else :

        temp.append(label)



full_labels = np.array(temp)

print(full_data.size)

print(full_labels.size)

"""
train_images_split,test_images_split,train_labels_split,test_labels_split = train_test_split(full_data,full_labels,stratify = full_labels,test_size = 0.2)

print('Train Image Split dims: ',train_images_split.size)

print('Train Labels Split dims: ',train_labels_split.size)

print('Test Image Split dims: ',test_images_split.size)

print('Test Labels Split dims: ',test_labels_split.size)
train_images_split,val_images_split,train_labels_split,val_labels_split = train_test_split(train_images_split,train_labels_split,stratify = train_labels_split,test_size = 0.2)

print('Train Image Split dims: ',train_images_split.size)

print('Train Labels Split dims: ',train_labels_split.size)

print('Test Image Split dims: ',val_images_split.size)

print('Test Labels Split dims: ',val_labels_split.size)
train_split_images = transform_images(train_images_split,128,128)

test_split_images = transform_images(test_images_split,128,128)

val_split_images = transform_images(val_images_split,128,128)



print('Finished Trasforming Images for Train,Test and Validation Sets')
def show_images_of_all_classes(train_images,training_labels):

  num_cols =  8

  classes = len(set(training_labels))

  if classes % num_cols == 0:

      num_rows =  classes / num_cols

  else:

      num_rows = int(classes / num_cols) + 1

  plt.figure(figsize=(15,15))

  i = 1

  for class_number in set(training_labels):

    

    indices = np.where(training_labels == class_number)

    plt.subplot(num_rows,num_cols, i)

    i += 1

    plt.axis('off')

    plt.imshow(train_images[indices[0][1]])

    plt.title('Class :{0} [{1}] '.format(class_number,len(indices[0])))   #np.count_nonzero(labels == labels[index])))

  plt.show()

print(set(full_labels))
show_images_of_all_classes(full_data,full_labels)
# Create a dictionary with Class Names

classnames = {

              0 : 'Warning for a bad road surface',

              1 : 'Warning for a speed bump',

              2 : 'Warning for a slippery road surface',

              3 : 'Warning for a curve to the left',

              4 : 'Warning for a curve to the right',

              5 : 'Warning for a double curve, first left then right',                                                    # Merge Classes 5 & 6 later

              6 : 'Warning for a double curve, first left then right',

              7 : 'Watch out for children ahead',

              8 : 'Watch out for  cyclists',

              9 : 'Watch out for cattle on the road',

              10: 'Watch out for roadwork ahead',

              11: 'Traffic light ahead',

              12: 'Watch out for railroad crossing with barriers ahead',

              13: 'Watch out ahead for unknown danger',

              14: 'Warning for a road narrowing',

              15: 'Warning for a road narrowing on the left',

              16: 'Warning for a road narrowing on the right',

              17: 'Warning for side road on the right',

              18: 'Warning for an uncontrolled crossroad',

              19: 'Give way to all drivers',

              20: 'Road narrowing, give way to oncoming drivers',

              21: 'Stop and give way to all drivers',

              22: 'Entry prohibited (road with one-way traffic)',

              23: 'Cyclists prohibited',

              24: 'Vehicles heavier than indicated prohibited',

              25: 'Trucks prohibited',

              26: 'Vehicles wider than indicated prohibited',

              27: 'Vehicles higher than indicated prohibited',

              28: 'Entry prohibited',

              29: 'Turning left prohibited',

              30: 'Turning right prohibited',

              31: 'Overtaking prohibited',

              32: 'Driving faster than indicated prohibited (speed limit)',

              33: 'Mandatory shared path for pedestrians and cyclists',

              34: 'Driving straight ahead mandatory',

              35: 'Mandatory left',

              36: 'Driving straight ahead or turning right mandatory',

              37: 'Mandatory direction of the roundabout',

              38: 'Mandatory path for cyclists',

              39: 'Mandatory divided path for pedestrians and cyclists',

              40: 'Parking prohibited',

              41: 'Parking and stopping prohibited',

              42: '',

              43: '',

              44: 'Road narrowing, oncoming drivers have to give way',

              45: 'Parking is allowed',

              46: 'parking for handicapped',

              47: 'Parking for motor cars',

              48: 'Parking for goods vehicles',

              49: 'Parking for buses',

              50: 'Parking only allowed on the sidewalk',

              51: 'Begin of a residential area',

              52: 'End of the residential area',

              53: 'Road with one-way traffic',

              54: 'Dead end street',

              55: '', 

              56: 'Crossing for pedestrians',

              57: 'Crossing for cyclists',

              58: 'Parking exit',

              59: 'Information Sign : Speed bump',

              60: 'End of the priority road',

              61: 'Begin of a priority road'

    }
# Training Data Generator



training_datagen = tf.keras.preprocessing.image.ImageDataGenerator(zoom_range=0.2,width_shift_range=0.3,height_shift_range=0.2,shear_range=0.25,fill_mode='nearest')



validation_datagen = tf.keras.preprocessing.image.ImageDataGenerator()
# Ready our generators for passing into models 

train_generator = training_datagen.flow(train_split_images,train_labels_split,batch_size=32)



validation_generator = validation_datagen.flow(val_split_images,val_labels_split,batch_size=32)

# A little test

# try and visualize your images



nyan_generator = training_datagen.flow(train_split_images[1:2], train_labels_split[1:2],batch_size=1)



sign = [next(nyan_generator) for i in range(0,10)]

fig, ax = plt.subplots(1,10, figsize=(16, 6))

print('Labels:', [item[1][0] for item in sign])

l = [ax[i].imshow(sign[i][0][0]) for i in range(0,10)]
def conv_net(train_images_dims,num_of_classes,filter_size = 2,num_convolutions=64,num_strides=2):

  # pre process image dimensions

  if (len(train_images_dims) == 3):    # Channel Last

    train_images_dims = (train_images_dims[1],train_images_dims[2])   

  elif (len(train_images_dims) == 4):

    train_images_dims = (train_images_dims[1],train_images_dims[2],train_images_dims[3])

  

  model  = tf.keras.Sequential()

  

  #Conv1

  model.add(tf.keras.layers.Conv2D(int(num_convolutions),(filter_size,filter_size),activation='relu',input_shape= train_images_dims))

  model.add(tf.keras.layers.MaxPool2D(pool_size=(2,2),strides=num_strides))

    

  #Conv2

  model.add(tf.keras.layers.Conv2D(int(num_convolutions),(filter_size,filter_size),activation='relu'))

  model.add(tf.keras.layers.MaxPool2D(pool_size=(2,2),strides=num_strides))



  #Conv3

  model.add(tf.keras.layers.Conv2D(int(num_convolutions),(filter_size,filter_size),activation='relu'))

  model.add(tf.keras.layers.MaxPool2D(pool_size=(2,2),strides=num_strides))

    

  #Conv4

  model.add(tf.keras.layers.Conv2D(int(num_convolutions),(filter_size,filter_size),activation='relu'))

  model.add(tf.keras.layers.MaxPool2D(pool_size=(2,2),strides=num_strides))



  #Conv5

  model.add(tf.keras.layers.Conv2D(int(num_convolutions) ,(filter_size,filter_size),activation='relu'))

  model.add(tf.keras.layers.MaxPool2D(pool_size=(2,2),strides=num_strides))

  

  #Flatten and add Dense Layer

  model.add(tf.keras.layers.Flatten())

  #Dense 1

  model.add(tf.keras.layers.Dense(512,activation='relu'))

  model.add(tf.keras.layers.Dropout(0.5))

  #Dense 2

  model.add(tf.keras.layers.Dense(512,activation='relu'))

  model.add(tf.keras.layers.Dropout(0.5))

  

  #Output Layer

  model.add(tf.keras.layers.Dense(num_of_classes,activation = 'softmax'))

  return model
monitor = tf.keras.callbacks.EarlyStopping(monitor = 'val_loss',patience = 8,restore_best_weights = True, min_delta = 0.01)
model_regularized = conv_net(train_split_images.shape,len(set(train_labels_split)),filter_size=2,num_convolutions=512)
model_regularized.compile(optimizer=tf.keras.optimizers.Adam(), loss='sparse_categorical_crossentropy',metrics = ['accuracy'])

model_regularized.summary()
history = model_regularized.fit_generator(train_generator, validation_data=validation_generator,steps_per_epoch=(len(train_split_images) / 32),epochs = 52,verbose=1,callbacks=[monitor])  # 32 = batch size
# Get training and test loss histories

training_loss = history.history['loss']

validation_loss = history.history['val_loss']



# Create count of the number of epochs

epoch_count = range(1, len(training_loss) + 1)



# Visualize loss history

plt.plot(epoch_count, training_loss, 'r--')

plt.plot(epoch_count, validation_loss, 'b-')

plt.legend(['Training Loss', 'Validation Loss'])

plt.xlabel('Epoch')

plt.ylabel('Loss')

plt.show();
#history = model_regularized.fit_generator(train_generator, validation_data=validation_generator,steps_per_epoch=(len(training_images128) / 32),epochs = 30,verbose=1)  # 32 = batch size
# Get training and test loss histories

training_loss = history.history['loss']

test_loss = history.history['val_loss']



# Create count of the number of epochs

epoch_count = range(1, len(training_loss) + 1)



# Visualize loss history

plt.plot(epoch_count, training_loss, 'r--')

plt.plot(epoch_count, test_loss, 'b-')

plt.legend(['Training Loss', 'Validation Loss'])

plt.xlabel('Epoch')

plt.ylabel('Loss')

plt.show();
indexes = []

for label in set(test_labels_split):

        indexes.append(np.where(test_labels_split == label)[0])

   

#indexes = [ind[0][0] for ind in indexes]

print(len(indexes))

id  = []

for i in indexes:

    id.append(i.flat[0])



class_wise_test_img = test_split_images[id]

class_wise_test_labels = test_labels_split[id]

print(class_wise_test_img.shape)
model_regularized.evaluate(test_split_images,test_labels_split)
# Class wise image test

model_regularized.evaluate(class_wise_test_img,class_wise_test_labels)
for i,img in enumerate(class_wise_test_img):

     show_img = img

     img_arr = np.expand_dims(img,axis=0) # to add new dimension to meet required input dims



     classes = model_regularized.predict(img_arr)



     plt.imshow(img)

     #plt.axis('off')

     plt.show()

     predicted_class = np.argmax(classes)

     print('PREDICTION : The Image belongs to class : {}, with description : {}'.format(predicted_class,classnames[predicted_class]))

     print('ACTUAL : The Image belongs to class : {}, with description : {}'.format(class_wise_test_labels[i],classnames[class_wise_test_labels[i]]))
def predict_image(model,directory):

    for file_name in os.listdir(directory):

        if '.jpg' in file_name or '.png' in file_name:

            img = tf.keras.preprocessing.image.load_img(os.path.join(directory,file_name),target_size = (128,128))  # Defaults to rgb mode and returns a PIL Instance

            

            img_arr = tf.keras.preprocessing.image.img_to_array(img) # returns 3d numpy array

            show_img = img_arr

            img_arr = np.expand_dims(img_arr,axis=0) # to add new dimension to meet required input dims

            

            classes = model.predict(img_arr)

            

            plt.imshow(img)

            #plt.axis('off')

            plt.show()

            predicted_class = np.argmax(classes)

            print('The Image belongs to class : {}, with description : {}'.format(predicted_class,classnames[predicted_class]))

    
#print(os.listdir('../'))
predict_image(model_regularized,'../belgium-new-test-images')
os.chdir(r'../../working')
print(os.listdir())
# SAVE THE MODEL AS H5 File

model_regularized.save('final_model.h5')
# Try out tensorflow saved model to save the model

tf.saved_model.save(model_regularized,'model_regularized_tf2')
!pip install tensorflowjs
print(os.listdir('../../working'))
import tensorflowjs as tfjs

tfjs.converters.save_keras_model(model_regularized, "model_js")
from IPython.display import FileLink

FileLink(r'final_model.h5')