import glob, pylab, pandas as pd
import pydicom, numpy as np
!ls ../input
'''
with open('../input/GCP Credits Request Link - RSNA.txt')as f:
    content=f.readlines()
    print(content)
'''
df = pd.read_csv('../input/rsna-pneumonia-detection-challenge/stage_1_train_labels.csv')
print(df.iloc[0])
print(df.info())
print(df.iloc[4])
'''
patientId = df['patientId'][0]
dcm_file = '../input/stage_1_train_images/%s.dcm' % patientId
dcm_data = pydicom.read_file(dcm_file)

print(dcm_data)
'''
'''
im = dcm_data.pixel_array
print(type(im))
print(im.dtype)
print(im.shape)
'''
#pylab.imshow(im, cmap=pylab.cm.gist_gray)
#pylab.axis('off')
def parse_data(df):
    """
    Method to read a CSV file (Pandas dataframe) and parse the 
    data into the following nested dictionary:

      parsed = {
        
        'patientId-00': {
            'dicom': path/to/dicom/file,
            'label': either 0 or 1 for normal or pnuemonia, 
            'boxes': list of box(es)
        },
        'patientId-01': {
            'dicom': path/to/dicom/file,
            'label': either 0 or 1 for normal or pnuemonia, 
            'boxes': list of box(es)
        }, ...

      }

    """
    # --- Define lambda to extract coords in list [y, x, height, width]
    extract_box = lambda row: [row['y'], row['x'], row['height'], row['width']]

    parsed = {}
    for n, row in df.iterrows():
        # --- Initialize patient entry into parsed 
        pid = row['patientId']
        if pid not in parsed:
            parsed[pid] = {
                'dicom': '../input/stage_1_train_images/%s.dcm' % pid,
                'label': row['Target'],
                'boxes': []}

        # --- Add box if opacity is present
        if parsed[pid]['label'] == 1:
            parsed[pid]['boxes'].append(extract_box(row))

    return parsed
#parsed = parse_data(df)


#print(parsed[ 'c542f0f4-1903-4fee-ba0f-186203d35226'])
'''
def draw(data):
    """
    Method to draw single patient with bounding box(es) if present 

    """
    # --- Open DICOM file
    d = pydicom.read_file(data['dicom'])
    im = d.pixel_array

    # --- Convert from single-channel grayscale to 3-channel RGB
    im = np.stack([im] * 3, axis=2)

    # --- Add boxes with random color if present
    for box in data['boxes']:
        rgb = np.floor(np.random.rand(3) * 256).astype('int')
        im = overlay_box(im=im, box=box, rgb=rgb, stroke=6)

    pylab.imshow(im, cmap=pylab.cm.gist_gray)
    pylab.axis('off')

def overlay_box(im, box, rgb, stroke=1):
    """
    Method to overlay single box on image

    """
    # --- Convert coordinates to integers
    box = [int(b) for b in box]
    
    # --- Extract coordinates
    y1, x1, height, width = box
    y2 = y1 + height
    x2 = x1 + width

    im[y1:y1 + stroke, x1:x2] = rgb
    im[y2:y2 + stroke, x1:x2] = rgb
    im[y1:y2, x1:x1 + stroke] = rgb
    im[y1:y2, x2:x2 + stroke] = rgb

    return im
    '''
#draw(parsed['00436515-870c-4b36-a041-de91049b9ab4'])
#df_detailed = pd.read_csv('../input/stage_1_detailed_class_info.csv')
#print(df_detailed.iloc[0])
#patientId = df_detailed['patientId'][0]
#draw(parsed[patientId])
'''
summary = {}
for n, row in df_detailed.iterrows():
    if row['class'] not in summary:
        summary[row['class']] = 0
    summary[row['class']] += 1
    
print(summary)
'''
import glob, pylab, pandas as pd
import pydicom, numpy as np

import os
import csv
import random
from skimage import measure
from skimage.transform import resize

import tensorflow as tf
from tensorflow import keras
import skimage.exposure

from matplotlib import pyplot as plt






df_detailed = pd.read_csv('../input/rsna-pneumonia-detection-challenge/stage_1_detailed_class_info.csv')
print(df_detailed.iloc[6])
print(df_detailed.iloc[80])
# empty dictionary
nodule_locations = {}
# load table
with open(os.path.join('../input/rsna-pneumonia-detection-challenge/stage_1_train_labels.csv'), mode='r') as infile:
    reader = csv.reader(infile)
    # skip header
    next(reader, None)

    for rows in reader:
        filename = rows[0]
        location = rows[1:5]
        nodule = rows[5]
        # if row contains a nodule add label to dictionary
        # which contains a list of nodule locations per filename
        if nodule == '1':
            # convert string to float to int
            location = [int(float(i)) for i in location]
            # save nodule location in dictionary
            if filename in nodule_locations:
                nodule_locations[filename].append(location)
            else:
                nodule_locations[filename] = [location]

folder = '../input/rsna-pneumonia-detection-challenge/stage_1_train_images'
filenames = os.listdir(folder)
random.shuffle(filenames)
# split into train and validation filenames
n_valid_samples = 2000
train_filenames = filenames[n_valid_samples:]
valid_filenames = filenames[:n_valid_samples]
print('n train samples', len(train_filenames))
print('n valid samples', len(valid_filenames))
n_train_samples = len(filenames) - n_valid_samples
class generator(keras.utils.Sequence):
    
    def __init__(self, folder, filenames, nodule_locations=None, batch_size=32, image_size=128, shuffle=True, predict=False):
        self.folder = folder
        self.filenames = filenames
        self.nodule_locations = nodule_locations
        self.batch_size = batch_size
        self.image_size = image_size
        self.shuffle = shuffle
        self.predict = predict
        self.on_epoch_end()
        
    def __load__(self, filename):
        # load dicom file as numpy array
        img = pydicom.dcmread(os.path.join(self.folder, filename)).pixel_array
        # create empty mask
        #msk = np.zeros(img.shape)
        # get filename without extension
        filename = filename.split('.')[0]
        # if image contains nodules
        if filename in nodule_locations:
            # loop through nodules
            pneumonia=1
        else:
            pneumonia=0
                
        # resize both image and mask
        img = resize(img, (self.image_size, self.image_size), mode='reflect')
        #msk = resize(msk, (self.image_size, self.image_size), mode='reflect') > 0.5
        # add trailing channel dimension
        img = np.expand_dims(img, -1)
        #msk = np.expand_dims(msk, -1)
        return img,pneumonia # ,msk
    
    def __loadpredict__(self, filename):
        # load dicom file as numpy array
        img = pydicom.dcmread(os.path.join(self.folder, filename)).pixel_array
        # resize image
        img = resize(img, (self.image_size, self.image_size), mode='reflect')
        # add trailing channel dimension
        img = np.expand_dims(img, -1)
        return img
        
    def __getitem__(self, index):
        # select batch
        filenames = self.filenames[index*self.batch_size:(index+1)*self.batch_size]
        # predict mode: return images and filenames
        if self.predict:
            # load files
            imgs = [self.__loadpredict__(filename) for filename in filenames]
            # create numpy batch
            imgs = np.array(imgs)
            return imgs, filenames
        # train mode: return images and masks
        else:
            # load files
            items = [self.__load__(filename) for filename in filenames]
            # unzip images and masks
            imgs, pneumonia = zip(*items)
            
            # create numpy batch
            imgs = np.array(imgs)
            #imgs= [skimage.transform.resize(imgs, (128,128,1))]   
            pneumonia = np.array(pneumonia)
            #pneumonia=pneumonia.reshape(16,1,1,1)
            return imgs,pneumonia #, msks
        
    def on_epoch_end(self):
        if self.shuffle:
            random.shuffle(self.filenames)
        
    def __len__(self):
        if self.predict:
            # return everything
            return int(np.ceil(len(self.filenames) / self.batch_size))
        else:
            # return full batches only
            return int(len(self.filenames) / self.batch_size)
# create train and validation generators
folder = '../input/rsna-pneumonia-detection-challenge/stage_1_train_images'
train_gen = generator(folder, train_filenames, nodule_locations, batch_size=16, image_size=128, shuffle=True, predict=False)
valid_gen = generator(folder, valid_filenames, nodule_locations, batch_size=16, image_size=128, shuffle=False, predict=False)
#x_val_new, y_val=valid_gen
#validation_generator = test_datagen.flow(valid_gen,batch_size=16)

def identity_block(inputs,kernel_size,filters):
    filters1, filters2, filters3 = filters
    
    x = keras.layers.Conv2D(filters1, (1, 1)) (inputs)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)

    x = keras.layers.Conv2D(filters2, kernel_size,
               padding='same')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)

    x =keras.layers.Conv2D(filters3, (1, 1))(x)
    x = keras.layers.BatchNormalization()(x)

    x = keras.layers.add([x, inputs])
    x = keras.layers.ReLU()(x)
    return x

def webnet(input_size):
    inputs= keras.Input(shape=(input_size, input_size, 1))
    conv1 = keras.layers.Conv2D(64, (3, 3), activation='relu',padding='same')(inputs)
    #conv1 = Dropout(0.5)(conv1)
    conv1 = keras.layers.BatchNormalization()(conv1)
    conv1 = keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(conv1)
    #conv1 = Dropout(0.5)(conv1)
    conv1 = keras.layers.BatchNormalization()(conv1)
    pool1 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv1)

    
    conv2=identity_block(pool1,3,[64,64,64])
    #conv2 = Dropout(0.5)(conv2)
    conv2 = identity_block(conv2,3,[64,64,64])
    pool2 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = identity_block(pool2,3,[32,32,64])
    #conv3 = Dropout(0.5)(conv3)
    conv3 = identity_block(conv3,3,[32,32,64])
    pool3 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = identity_block(pool3,3,[32,32,64])
    #conv4 = Dropout(0.5)(conv4)
    conv4 = identity_block(conv4,3,[32,32,64])
    pool4 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = identity_block(pool4,3,[32,32,64])
    #conv5 = Dropout(0.5)(conv5)
    conv5 = identity_block(conv5,3,[32,32,64])
    conv5 = identity_block(conv5,3,[32,32,64])

    up6 = keras.layers.Concatenate(axis=-1)([keras.layers.UpSampling2D(size=(2, 2))(conv5), conv4])
    conv6 = keras.layers.Conv2D(64, (3, 3), activation='relu',padding='same')(up6)
    conv6 = identity_block(conv6,3,[32,32,64])
    #conv6 = Dropout(0.5)(conv6)
    conv6 = identity_block(conv6,3,[32,32,64])

    up7 = keras.layers.Concatenate(axis=-1)([keras.layers.UpSampling2D(size=(2, 2))(conv6), conv3])
    conv7 = keras.layers.Conv2D(64, (3, 3), activation='relu',padding='same')(up7)
    conv7 = identity_block(conv7,3,[32,32,64])
    #conv7 = Dropout(0.5)(conv7)
    conv7 = identity_block(conv7,3,[32,32,64])
    pool7 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv7)

    concat8 = keras.layers.Concatenate(axis=-1)([pool7, conv6])
    conv8 = keras.layers.Conv2D(64, (3, 3), activation='relu',padding='same')(concat8)
    conv8 = identity_block(conv8,3,[32,32,64])
    #conv8 = Dropout(0.5)(conv8)
    conv8 = identity_block(conv8,3,[32,32,64])
    pool8 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv8)

    concat9 =keras.layers.Concatenate()([pool8, conv5])
    conv9 = keras.layers.Conv2D(64, (3, 3), activation='relu',padding='same')(concat9)
    conv9 = identity_block(conv9,3,[32,32,64])
    #conv9 = Dropout(0.5)(conv9)
    conv9 = identity_block(conv9,3,[32,32,64])
    #conv9 = Dropout(0.5)(conv9)
       
    up10 = keras.layers.Concatenate(axis=-1)([keras.layers.UpSampling2D(size=(2, 2))(conv9), conv8])
    conv10 = keras.layers.Conv2D(64, (3, 3), activation='relu',padding='same')(up10)
    conv10 = identity_block(conv10,3,[32,32,64])
    #conv10 = Dropout(0.5)(conv6)
    conv10 = identity_block(conv10,3,[32,32,64])
    
    up11 = keras.layers.Concatenate(axis=-1)([keras.layers.UpSampling2D(size=(2, 2))(conv10), conv7])
    conv11 = keras.layers.Conv2D(64, (3, 3), activation='relu',padding='same')(up11)
    conv11 =identity_block(conv11,3,[32,32,64])
    #conv11 = Dropout(0.5)(conv11)
    conv11 = identity_block(conv11,3,[32,32,64])
    pool11 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv11)
    
    concat12 = keras.layers.Concatenate()([pool11, conv10])
    conv12 = keras.layers.Conv2D(64, (3, 3), activation='relu',padding='same')(concat12)
    conv12 = identity_block(conv12,3,[32,32,64])
    #conv12 = Dropout(0.5)(conv12)
    conv12 = identity_block(conv12,3,[32,32,64])
    pool12 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv12)
    
    concat13 = keras.layers.Concatenate(axis=-1)([pool12, conv9])
    conv13 = keras.layers.Conv2D(64, (3, 3), activation='relu',padding='same')(concat13)
    conv13 = identity_block(conv13,3,[32,32,64])
    #conv13 = Dropout(0.5)(conv13)
    conv13 = identity_block(conv13,3,[32,32,64])
     
    up14 = keras.layers.Concatenate(axis=-1)([keras.layers.UpSampling2D(size=(2, 2))(conv13), conv12])
    conv14 = keras.layers.Conv2D(64, (3, 3), activation='relu',padding='same')(up14)
    conv14 = identity_block(conv14,3,[32,32,64])
    conv14 = identity_block(conv14,3,[32,32,64])
    
    up15 = keras.layers.Concatenate(axis=-1)([keras.layers.UpSampling2D(size=(2, 2))(conv14), conv11])
    conv15 = keras.layers.Conv2D(64, (3, 3), activation='relu',padding='same')(up15)
    conv15 = identity_block(conv15,3,[32,32,64])
    #conv15 = Dropout(0.5)(conv15)
    conv15 = identity_block(conv15,3,[32,32,64])
    
    up16 = keras.layers.Concatenate(axis=-1)([keras.layers.UpSampling2D(size=(2, 2))(conv15), conv2])
    conv16 = keras.layers.Conv2D(64, (3, 3), activation='relu',padding='same')(up16)
    conv16 = identity_block(conv16,3,[32,32,64])
    #conv16 = Dropout(0.5)(conv16)
    conv16 = identity_block(conv16,3,[32,32,64])
    
    up17 = keras.layers.Concatenate(axis=-1)([keras.layers.UpSampling2D(size=(2, 2))(conv16), conv1])
    conv17 = keras.layers.Conv2D(64, (3, 3), activation='relu',padding='same')(up17)
    conv17 = identity_block(conv17,3,[32,32,64])
    #conv17 = Dropout(0.5)(conv17)
    #conv17 = AveragePooling2D((7, 7))(conv17)
    conv17 = keras.layers.Conv2D(1, (1, 1), activation='sigmoid', padding='same')(conv17)
    flat1=keras.layers.Flatten()(conv17)
    dense1= keras.layers.Dense(1, activation='sigmoid')(flat1)
    model = keras.Model(inputs=inputs, outputs=dense1)

    return model
model = webnet(input_size=128)
model.compile(optimizer='SGD',
              loss='binary_crossentropy',
              metrics=['accuracy'])
model.summary()
def cosine_annealing(x):
    lr = 0.001
    epochs = 25
    return lr*(np.cos(np.pi*x/epochs)+1.)/2
learning_rate = tf.keras.callbacks.LearningRateScheduler(cosine_annealing)

model_info=model.fit_generator(
 train_gen,
    steps_per_epoch=200,
    epochs=100,
    validation_data=valid_gen,
    validation_steps=50,
     callbacks=[tf.keras.callbacks.CSVLogger(os.path.join('training_log.csv'), append=True),
                                         #ReduceLROnPlateau(monitor='val_loss', mode='min', factor=0.5, verbose=1, patience=50),
                                         tf.keras.callbacks.ModelCheckpoint(os.path.join(
                                               #'weights.ep-{epoch:02d}-val_mean_IOU-{val_mean_IOU_gpu:.2f}_val_loss_{val_loss:.2f}.hdf5',
                                               'last_checkpoint.hdf5'),
                                               monitor='val_loss', mode='min', save_best_only=True, 
                                               save_weights_only=False, verbose=0)])
def plot_model_history(model_history):
    fig, axs = plt.subplots(1,2,figsize=(15,5))
    # summarize history for accuracy
    axs[0].plot(range(1,len(model_history.history['acc'])+1),model_history.history['acc'])
    axs[0].plot(range(1,len(model_history.history['val_acc'])+1),model_history.history['val_acc'])
    axs[0].set_title('Model Accuracy')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_xlabel('Epoch')
    axs[0].set_xticks(np.arange(1,len(model_history.history['acc'])+1),len(model_history.history['acc'])/10)
    axs[0].legend(['train', 'val'], loc='best')
    # summarize history for loss
    axs[1].plot(range(1,len(model_history.history['loss'])+1),model_history.history['loss'])
    axs[1].plot(range(1,len(model_history.history['val_loss'])+1),model_history.history['val_loss'])
    axs[1].set_title('Model Loss')
    axs[1].set_ylabel('Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].set_xticks(np.arange(1,len(model_history.history['loss'])+1),len(model_history.history['loss'])/10)
    axs[1].legend(['train', 'val'], loc='best')
    plt.show()
#plot_model_history(model_info)
model_path = os.path.join('../input/webnet1-rsna/last_checkpoint.hdf5')
model.load_weights(model_path)

folder = '../input/rsna-pneumonia-detection-challenge/stage_1_train_images'
filenames = os.listdir(folder)

# split into train and validation filenames
n_valid_samples = 2000

valid_filenames2 = filenames[:n_valid_samples]
valid_gen_ = generator(folder, valid_filenames2, nodule_locations, batch_size=16, image_size=128, shuffle=False, predict=False)
pred = model.predict_generator(valid_gen_)
print (pred.shape)
print ((pred[2]))
#import math


correct=0
for i in range(2000):
  
    if (pred[i]   >= 0.1 and df['Target'][i] == 1):
        correct +=1
    if (pred[i]  < 0.1 and df['Target'][i] == 0):
        correct +=1
        
    
        

accuracy= correct/2000

print(accuracy)   

#print(accuracy)
#print (df['Target'][:2000])

















        


