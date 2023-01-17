!pip install git+https://github.com/mjkvaak/ImageDataAugmentor



import numpy as np 



import pandas as pd



import matplotlib.pyplot as plt

import seaborn as sns



import cv2



import keras

from keras import losses

from keras.layers.normalization import BatchNormalization

from keras.layers.core import Activation,Flatten, Dropout, Dense

from keras.layers.convolutional import MaxPooling2D

from keras import Model

from keras.layers import GlobalAveragePooling2D

from keras.callbacks import EarlyStopping,ModelCheckpoint

from keras.optimizers import Nadam,Adam



from ImageDataAugmentor.image_data_augmentor import *

import albumentations



import warnings

warnings.filterwarnings('ignore')



from sklearn.model_selection import train_test_split

from sklearn.metrics import roc_curve, auc, roc_auc_score



import glob







df = pd.read_csv('../input/histopathologic-cancer-detection/train_labels.csv')

print(df.head(10))
column_names = list(df.columns)

print(len(df))



print(len(df[column_names[0]].unique()))

print(df[column_names[1]].unique())

# No duplciate rows
df['id']=df['id'].apply(lambda x: x+'.tif')

print(df['id'])
hist = df['label'].hist(bins=5)
fig, axs = plt.subplots(4,4,figsize=(10, 10), dpi=150)





images = []

for i in range(4):

    for j in range(4):

        

        No = np.random.randint(0,2000)

                

        image = cv2.imread('../input/histopathologic-cancer-detection/train/'+ df.iloc[No]['id'])

        images.append(axs[i, j].imshow(image))

        

        if df.iloc[No]['label'] == 1:

            axs[i,j].set_title('Cancerous')

        else:

            axs[i,j].set_title('Non Cancerous')

            

        axs[i,j].set_xticks([])

        axs[i,j].set_yticks([])

        



    

plt.show()

# This will be helpful for image augmentations, like we can do do channel shuffling in order for the model to extract features well or

# contrast enhancement to make features distinguishable.

del images
malignant_data = df[(df.label==1)]

malignant_image = malignant_data.iloc[1000]['id']



img = cv2.imread('../input/histopathologic-cancer-detection/train/'+malignant_image)

plt.imshow(img)

plt.title("Cancerous Image")

plt.show()

plt.hist(img[:, :,  0].ravel(), bins = 256, color = 'red')

plt.hist(img[:, :, 1].ravel(), bins = 256, color = 'Green')

plt.hist(img[:, :, 2].ravel(), bins = 256, color = 'Blue')

plt.xlabel('Intensity Value')

plt.ylabel('Count')

plt.legend(['Red_Channel', 'Green_Channel', 'Blue_Channel'])

plt.title("Cancerous Frequency plot")

plt.show()



benign_data = df[(df.label==0)]

benign_image = benign_data.iloc[1]['id']



img = cv2.imread('../input/histopathologic-cancer-detection/train/'+benign_image)

plt.imshow(img)

plt.title("Non Cancerous Image")

plt.show()

plt.hist(img[:, :, 0].ravel(), bins = 256, color = 'red')

plt.hist(img[:, :, 1].ravel(), bins = 256, color = 'Green')

plt.hist(img[:, :, 2].ravel(), bins = 256, color = 'Blue')

plt.xlabel('Intensity Value')

plt.ylabel('Count')

plt.legend(['Red_Channel', 'Green_Channel', 'Blue_Channel'])

plt.title("Non Cancerous Frequency plot")

plt.show()



del img,benign_data,malignant_data
AUGMENTATIONS = albumentations.Compose([

    

    albumentations.Flip(p=0.5),

    albumentations.Rotate(p=0.5),    

    albumentations.CLAHE(p=0.3), 

    albumentations.RandomContrast(p=0.3) # (Default varies from -0.2 to 0.2)

    

]) 



train_datagen = ImageDataAugmentor(

        rescale=1./255,

        augment = AUGMENTATIONS

      )
df['label'] = df['label'].astype('str')



train, test = train_test_split(df, test_size=0.2,random_state=42)

train, valid = train_test_split(train, test_size=0.1,random_state=42)



train_path = '../input/histopathologic-cancer-detection/train/'

valid_path = '../input/histopathologic-cancer-detection/train/'



train_generator = train_datagen.flow_from_dataframe(

                dataframe=train,

                directory=train_path,

                x_col = 'id',

                y_col = 'label',

                shuffle=True,

                subset='training',

                target_size=(94, 94),

                batch_size=64,

                class_mode="binary"

                )



valid_datagen = ImageDataAugmentor(

        rescale=1./255, )



validation_generator = valid_datagen.flow_from_dataframe(

                dataframe=valid,

                directory=valid_path,

                x_col = 'id',

                y_col = 'label',

                subset=None, 

                target_size=(94, 94),

                batch_size=64,

                shuffle=True,

                class_mode="binary"

                )
plt.imshow(train_generator[0][0][1])
plt.imshow(train_generator[0][0][2])
image_shape = (94,94, 3)



# Define base_model

TLModel = keras.applications.VGG19(weights='imagenet',

                  include_top=False,

                  input_shape=(image_shape))



# Make the botton 8 layers trainable

for layer in TLModel.layers[:-8]:

    layer.trainable = False





x = TLModel.output

x = GlobalAveragePooling2D()(x)



# Helps to learn new features

x = Dense(1000,activation='relu')(x)

x = Dense(500,activation='relu')(x)

x = Dense(500,activation='relu')(x)

x = BatchNormalization()(x)



output = Dense(1, activation='sigmoid')(x)



model = Model(inputs=TLModel.input, outputs=output)

    

model.summary()



opt = Nadam()

model.compile(optimizer= opt, loss=losses.binary_crossentropy, metrics=['accuracy'])
earlyStop = EarlyStopping(monitor='val_accuracy', mode='max',patience= 4)

Checkpoint = ModelCheckpoint(filepath='best_model.h5', monitor='val_accuracy', save_best_only=True,mode='max',verbose=1)



StepSizeTrain= int(train_generator.n/train_generator.batch_size)

StepSizeValid= int(validation_generator.n/validation_generator.batch_size)



hist = model.fit_generator(

                train_generator,

                steps_per_epoch=StepSizeTrain,

                validation_steps=StepSizeValid,

                epochs=25,

                validation_data=validation_generator

                ,callbacks=[earlyStop,Checkpoint],verbose=1)



# Summarize Accuracy

plt.plot(hist.history['accuracy'])

plt.plot(hist.history['val_accuracy'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()
# Summarize Loss 

plt.plot(hist.history["loss"])

plt.plot(hist.history['val_loss'])

plt.title("model loss")

plt.ylabel("loss")

plt.xlabel("Epoch")

plt.legend(["Loss","Validation Loss"])

plt.show()
test_datagen = ImageDataAugmentor(rescale=1./255)



test_path = '../input/histopathologic-cancer-detection/train/'

test_gen = test_datagen.flow_from_dataframe(

                dataframe=test,

                directory=test_path,

                x_col = 'id',

                y_col = 'label',      

                target_size=(94, 94),

                batch_size=1,

                shuffle=False,

               class_mode="binary"

                ) 



model.load_weights("best_model.h5")



# make predictions

predictions = model.predict_generator(test_gen, steps=len(test_gen), verbose=1)

False_Positive_rate, True_Positive_rate, Thresholds = roc_curve(test_gen.classes, predictions)

AUC = auc(False_Positive_rate, True_Positive_rate)
plt.figure(figsize=(20,20))

plt.plot([0, 1], [0, 1], 'k--')

plt.plot(False_Positive_rate, True_Positive_rate, label='area = {:.3f}'.format(AUC))

plt.xlabel('False positive rate')

plt.ylabel('True positive rate')

plt.title('ROC curve')

plt.legend(loc='best')

plt.show()
del test_gen, train_generator, validation_generator
test  = pd.DataFrame()

test_fileNames = [file for file in glob.glob("../input/histopathologic-cancer-detection/test/*.tif")]



test_fileNames.sort()



FileNames = []

for name in test_fileNames:

    Name = name.replace("../input/histopathologic-cancer-detection/test/", "")

    FileNames.append(Name)

    

test['id'] = FileNames
test_datagen = ImageDataAugmentor(rescale=1./255)



test_path = '../input/histopathologic-cancer-detection/test/'

test_gen = test_datagen.flow_from_dataframe(

                dataframe=test,

                directory=test_path,

                x_col = 'id',

                y_col = None,

                class_mode=None,

                target_size=(94, 94),

                batch_size=1,

                shuffle=False,   

                ) 
predictions = model.predict(test_gen, steps=len(test_gen), verbose=1)
output = pd.DataFrame()



output['id'] = test_gen.filenames

output['id'] = output['id'].str.replace('.tif','')

output['label'] = predictions





output.to_csv("submission.csv",index=False)

print(output)