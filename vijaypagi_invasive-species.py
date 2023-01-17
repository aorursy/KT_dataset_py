from sklearn.datasets import load_files       
from keras.utils import np_utils
import numpy as np
import pandas as pd
from glob import glob
import random
import cv2                
import matplotlib.pyplot as plt                        
%matplotlib inline                               
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image                  
from tqdm import tqdm
from keras.applications.resnet50 import preprocess_input, decode_predictions
from PIL import ImageFile                            
ImageFile.LOAD_TRUNCATED_IMAGES = True 
import os
train_labels = pd.read_csv("/invasive-species/train_labels.csv")
sample_submission = pd.read_csv("/invasive-species/sample_submission.csv")

train_labels.dropna
train_labels.tail()

sample_submission.dropna
sample_submission.tail()

print('There are %d total training images.' % len(train_labels))
print('There are %d total testing images.' % len(sample_submission))
import matplotlib.image as mpatImg
def species_images(img_path):
    imgPrnt = mpatImg.imread(img_path)
    plt.figure(figsize=(10,10))
    plt.imshow(imgPrnt)
species_images('/invasive-species/train/1.jpg')
species_images('/invasive-species/train/29.jpg')
species_images('/invasive-species/train/298.jpg')
species_images('/invasive-species/train/1008.jpg')
species_images('/invasive-species/train/1007.jpg')
species_images('/invasive-species/train/2287.jpg')
species_images('/invasive-species/test/76.jpg')
species_images('/invasive-species/test/987.jpg')
species_images('/invasive-species/test/585.jpg')
species_images('/invasive-species/test/1212.jpg')
species_images('/invasive-species/test/1007.jpg')
species_images('/invasive-species/test/1431.jpg')
def smpl_visual(path, smpl, dim_y):
    
    smpl_pic = glob(smpl)
    fig = plt.figure(figsize=(20, 14))
    
    for i in range(len(smpl_pic)):
        ax = fig.add_subplot(round(len(smpl_pic)/dim_y), dim_y, i+1)
        plt.title("{}: Height {} Width {} Dim {}".format(smpl_pic[i].strip(path),
                                                         plt.imread(smpl_pic[i]).shape[0],
                                                         plt.imread(smpl_pic[i]).shape[1],
                                                         plt.imread(smpl_pic[i]).shape[2]
                                                        )
                 )
        plt.imshow(plt.imread(smpl_pic[i]))
        
    return smpl_pic

smpl_pic = smpl_visual('/invasive-species/train\\', '/invasive-species/train/112*.jpg', 4)
def visual_with_transformation (pic):

    for idx in list(range(0, len(pic), 1)):
        ori_smpl = cv2.imread(pic[idx])
        smpl_1_rgb = cv2.cvtColor(cv2.imread(pic[idx]), cv2.COLOR_BGR2RGB)
        smpl_1_lab = cv2.cvtColor(cv2.imread(pic[idx]), cv2.COLOR_BGR2LAB)
        smpl_1_gray =  cv2.cvtColor(cv2.imread(pic[idx]), cv2.COLOR_BGR2GRAY) 

        f, ax = plt.subplots(1, 4,figsize=(30,20))
        (ax1, ax2, ax3, ax4) = ax.flatten()
        train_idx = int(pic[idx].strip("/invasive-species/train\\").strip(".jpg"))
        print("The Image name: {} Is Invasive?: {}".format(pic[idx].strip("train\\"), 
                                                           train_labels.loc[train_labels.name.values == train_idx].invasive.values)
             )
        ax1.set_title("Original - BGR")
        ax1.imshow(ori_smpl)
        ax2.set_title("Transformed - RGB")
        ax2.imshow(smpl_1_rgb)
        ax3.set_title("Transformed - LAB")
        ax3.imshow(smpl_1_lab)
        ax4.set_title("Transformed - GRAY")
        ax4.imshow(smpl_1_gray)
        plt.show()

visual_with_transformation(smpl_pic)
img_path = "/invasive-species/train/"

print(img_path)

y = []
file_paths = []
for i in range(len(train_labels)):
    file_paths.append( img_path + str(train_labels.iloc[i][0]) +'.jpg' )
    y.append(train_labels.iloc[i][1])
y = np.array(y)
def centering_image(img):
    size = [256,256]
    
    img_size = img.shape[:2]
    
    # centering
    row = (size[1] - img_size[0]) // 2
    col = (size[0] - img_size[1]) // 2
    resized = np.zeros(list(size) + [img.shape[2]], dtype=np.uint8)
    resized[row:(row + img.shape[0]), col:(col + img.shape[1])] = img

    return resized
x = []
for i, file_path in enumerate(file_paths):
    #read image
    img = cv2.imread(file_path)
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    #resize
    if(img.shape[0] > img.shape[1]):
        tile_size = (int(img.shape[1]*256/img.shape[0]),256)
    else:
        tile_size = (256, int(img.shape[0]*256/img.shape[1]))

    #centering
    img = centering_image(cv2.resize(img, dsize=tile_size))
    
    #out put 224*224px 
    img = img[16:240, 16:240]
    x.append(img)

x = np.array(x)
img_path = "/invasive-species/test/"

test_names = []
file_paths = []

for i in range(len(sample_submission)):
    test_names.append(sample_submission.ix[i][0])
    file_paths.append( img_path + str(int(sample_submission.ix[i][0])) +'.jpg' )
    
test_names = np.array(test_names)
test_images = []
for file_path in file_paths:
    #read image
    img = cv2.imread(file_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    #resize
    if(img.shape[0] > img.shape[1]):
        tile_size = (int(img.shape[1]*256/img.shape[0]),256)
    else:
        tile_size = (256, int(img.shape[0]*256/img.shape[1]))

    #centering
    img = centering_image(cv2.resize(img, dsize=tile_size))
    
    #out put 224*224px 
    img = img[16:240, 16:240]
    test_images.append(img)
    
    path, ext = os.path.splitext( os.path.basename(file_paths[0]) )

test_images = np.array(test_images)
data_num = len(y)
random_index = np.random.permutation(data_num)

x_shuffle = []
y_shuffle = []
for i in range(data_num):
    x_shuffle.append(x[random_index[i]])
    y_shuffle.append(y[random_index[i]])
    
x = np.array(x_shuffle) 
y = np.array(y_shuffle)
val_split_num = int(round(0.2*len(y)))
x_train = x[val_split_num:]
y_train = y[val_split_num:]
x_test = x[:val_split_num]
y_test = y[:val_split_num]

print('x_train', x_train.shape)
print('y_train', y_train.shape)
print('x_test', x_test.shape)
print('y_test', y_test.shape)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
def invasiveSpeciesCapture(model_Capture, speciesPic):
    species_batch = np.expand_dims(speciesPic,axis=0)
    conv_species = model_Capture.predict(species_batch)
    
    conv_species = np.squeeze(conv_species, axis=0)
    print(conv_species.shape)
    conv_species = conv_species.reshape(conv_species.shape[:2])
    
    print(conv_species.shape)
    plt.imshow(conv_species)
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D,BatchNormalization,Convolution2D
from keras.layers import Dropout, Flatten, Dense
from keras.models import Sequential, Model, load_model
from keras import applications
from keras import optimizers
from keras.layers import Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam

model = Sequential()
print('Training set is ',x_train.shape[0])
print('Validation set is ',x_test.shape[1])


model.add(BatchNormalization(input_shape=(224, 224, 3)))

model.add(Conv2D(filters = 256,kernel_size=2,padding='same',activation = 'relu'))
model.add(MaxPooling2D(pool_size=2))

model.add(Dropout(0.3))
model.add(Conv2D(filters = 64,kernel_size=2,padding='same',activation = 'relu'))
model.add(MaxPooling2D(pool_size=2))

model.add(Conv2D(filters = 128,kernel_size=2,padding='same',activation = 'relu'))
model.add(MaxPooling2D(pool_size=2))

model.add(GlobalAveragePooling2D())
model.add(Dense(1,activation = 'sigmoid'))

model.summary()
model.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])
from keras.callbacks import ModelCheckpoint 

epochs = 60

checkpointer = ModelCheckpoint(filepath='weights.best.from_scratch.hdf5', 
                               verbose=1, save_best_only=True)

model_trained = model.fit(x_train, y_train, 
          validation_data=(x_test, y_test),
          epochs=epochs, batch_size=30, callbacks=[checkpointer], verbose=1)
model.load_weights('weights.best.from_scratch.hdf5')
plt.plot(model_trained.history['acc'])
plt.plot(model_trained.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
plt.plot(model_trained.history['loss'])
plt.plot(model_trained.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
print("Training loss: {:.2f} / Validation loss: {:.2f}".\
      format(model_trained.history['loss'][-1], model_trained.history['val_loss'][-1]))
print("Training accuracy: {:.2f}% / Validation accuracy: {:.2f}%".\
      format(100*model_trained.history['acc'][-1], 100*model_trained.history['val_acc'][-1]))
img_rows, img_cols, img_channel = 224, 224, 3

base_model = applications.VGG16(weights='imagenet', include_top=False, input_shape=(img_rows, img_cols, img_channel))
add_model = Sequential()
add_model.add(Flatten(input_shape=base_model.output_shape[1:]))

add_model.add(Dense(256, activation='relu'))

add_model.add(Dense(128, activation='relu'))

add_model.add(Dense(64, activation='relu'))

add_model.add(Dense(1, activation='sigmoid'))

vgg16_model = Model(inputs=base_model.input, outputs=add_model(base_model.output))
vgg16_model.compile(loss='binary_crossentropy', optimizer=Adam(lr=1E-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=1E-4),
          metrics=['accuracy'])
vgg16_model.summary()
from keras.callbacks import ModelCheckpoint 

batch_size = 32
epochs = 20

vgg16_train_datagen = ImageDataGenerator(
        rotation_range=31, 
        width_shift_range=0.1,
        height_shift_range=0.1, 
        horizontal_flip=True)
vgg16_train_datagen.fit(x_train)

vgg16_checkpointer = ModelCheckpoint(filepath='weights.best.vgg16.hdf5', 
                               verbose=1, save_best_only=True)


vgg16_history = vgg16_model.fit_generator(
    vgg16_train_datagen.flow(x_train, y_train, batch_size=batch_size),
    steps_per_epoch=x_train.shape[0] // batch_size,
    epochs=epochs,callbacks=[vgg16_checkpointer],
    validation_data=(x_test, y_test),
)
vgg16_model.load_weights('weights.best.vgg16.hdf5')
plt.plot(vgg16_history.history['acc'])
plt.plot(vgg16_history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
plt.plot(vgg16_history.history['loss'])
plt.plot(vgg16_history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
test_images = test_images.astype('float32')
test_images /= 255
print("Training loss: {:.2f} / Validation loss: {:.2f}".\
      format(vgg16_history.history['loss'][-1], vgg16_history.history['val_loss'][-1]))
print("Training accuracy: {:.2f}% / Validation accuracy: {:.2f}%".\
      format(100*vgg16_history.history['acc'][-1], 100*vgg16_history.history['val_acc'][-1]))
predictions = vgg16_model.predict(test_images)
for i, name in enumerate(test_names):
    sample_submission.loc[sample_submission['name'] == name, 'invasive'] = predictions[i]

sample_submission.to_csv("vgg16submit.csv", index=False)
add_model = Sequential()
add_model.add(Flatten(input_shape=base_model.output_shape[1:]))

add_model.add(Dense(256, activation='relu'))

add_model.add(Dense(128, activation='relu'))

add_model.add(Dense(64, activation='relu'))

add_model.add(Dense(1, activation='sigmoid'))

vgg16_model = Model(inputs=base_model.input, outputs=add_model(base_model.output))
vgg16_model.compile(loss='binary_crossentropy', optimizer=Adam(lr=1E-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=1E-4),
          metrics=['accuracy'])
vgg16_model.summary()