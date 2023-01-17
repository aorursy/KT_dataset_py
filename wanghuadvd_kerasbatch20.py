import os
print(os.listdir("../input/dogs-vs-cats"))
print(os.listdir("/kaggle/working/"))
os.chdir("/kaggle/working/")
import zipfile
train_data = "../input/dogs-vs-cats/train.zip"
test_data = "../input/dogs-vs-cats/test1.zip"

with zipfile.ZipFile(train_data,"r") as z:
    z.extractall(".")
with zipfile.ZipFile(test_data,"r") as z:
    z.extractall(".")
    
print(os.listdir("/kaggle/working/train")[:5])
print(os.listdir("/kaggle/working/test1")[:5])
import numpy as np
import pandas as pd 
from keras.preprocessing.image import ImageDataGenerator, load_img
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import random
# define image related constants
FAST_RUN = False
IMAGE_WIDTH=128
IMAGE_HEIGHT=128
IMAGE_SIZE=(IMAGE_WIDTH, IMAGE_HEIGHT)
IMAGE_CHANNELS=3 #TGB channels
#filenames = os.listdir("C:/temp/DogsandCats/train") #local PC
filenames = os.listdir("/kaggle/working/train")
categories = []
for filename in filenames:
    category = filename.split('.')[0]
    if category == 'dog':
        categories.append(1)
    else:
        categories.append(0)

df = pd.DataFrame({
    'filename': filenames,
    'category': categories
})
print(df.head())
# see total in counts
df['category'].value_counts().plot.bar()
# check a sample image
sample = random.choice(filenames)
#image = load_img("C:/temp/DogsandCats/train/"+sample) # local PC
image = load_img("/kaggle/working/train/"+sample)
plt.imshow(image)
#build model with keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Dropout, Flatten, Dense, Activation, BatchNormalization
model = Sequential()

model.add(Conv2D(32, (3 , 3), activation='relu', input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS)))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2,2))) # default is (2,2)
model.add(Dropout(rate=0.25))

model.add(Conv2D(64, (3,3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2,2))) # default is (2,2)
model.add(Dropout(rate=0.25))

model.add(Conv2D(128, (3,3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2,2))) # default is (2,2)
model.add(Dropout(rate=0.25))

model.add(Conv2D(256, (3,3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2,2))) # default is (2,2)
model.add(Dropout(rate=0.25))

model.add(Flatten())
model.add(Dense(units=512, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(rate=0.5))
model.add(Dense(units=2, activation='sigmoid'))

model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

model.summary()
# callbacks - early stop & learning rate reduction
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
earlystop = EarlyStopping(patience=10)
learn_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', patience=2, verbose=1, factor=0.5, min_lr=0.0001)
callbacks = [earlystop, learn_rate_reduction]
# data preparation
df["category"] = df["category"].replace({0:'cat', 1:'dog'}) # dog is target
df_train, df_test = train_test_split(df, test_size=0.1, random_state = 42)
df_train = df_train.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)
df_train["category"].value_counts().plot.bar()
df_test["category"].value_counts().plot.bar()
total_train = df_train.shape[0]
total_test = df_test.shape[0]
batch_size = 20
# image data argumentation
# training generator
train_datagen = ImageDataGenerator(
    rotation_range= 15,
    rescale=1./255,
    shear_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1
)
train_generator = train_datagen.flow_from_dataframe(
    df_train,
    directory="/kaggle/working/train/",
    x_col='filename',
    y_col='category',
    target_size=IMAGE_SIZE,
    class_mode='categorical',
    batch_size=batch_size
)
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_dataframe(
    df_test,
    directory="/kaggle/working/train/",
    x_col='filename',
    y_col='category',
    target_size=IMAGE_SIZE,
    class_mode='categorical',
    batch_size=batch_size
)
# check performance of data argumentation
df_example = df_train.sample(n=1).reset_index(drop=True)
example_generator = train_datagen.flow_from_dataframe(
    df_example,
    directory="/kaggle/working/train/",
    x_col='filename',
    y_col='category',
    target_size=IMAGE_SIZE,
    class_mode='categorical'
)
plt.figure(figsize=(12,12))
for i in range(0,15):
    plt.subplot(5,3,i+1)
    for X_batch, Y_batch in example_generator:
        image = X_batch[0]
        plt.imshow(image)
        break
plt.tight_layout()
plt.show()
# fit model
epochs = 3 if FAST_RUN else 50 # 1 epoch meaning "one pass over the entire database" 
# with GPU enabled, about 150s per epoch and 135ms per step
history = model.fit(
    train_generator,
    epochs=epochs,
    validation_data=test_generator,
    validation_steps=total_test//batch_size,
    steps_per_epoch=total_train//batch_size,
    callbacks=callbacks
)
# created new branch of 15
# save trained model weights
model.save_weights("modelAug19.h5")
# virtualize training
fig, (ax1, ax2) = plt.subplots(2, 1, figsize = (12, 12))
ax1.plot(history.history['loss'], color='b',label="Training loss")
ax1.plot(history.history['val_loss'], color='r',label="Testing loss")
ax1.set_xticks(np.arange(1, epochs, 1))
ax1.set_yticks(np.arange(0, 1, 0.1))

ax2.plot(history.history['accuracy'], color='b',label="Training accuracy")
ax2.plot(history.history['val_accuracy'], color='r',label="Testing accuracy")
ax2.set_xticks(np.arange(1, epochs, 1))
#ax2.set_yticks(np.arrange(0, 1, 0.1)) # accuracy should be between 0 to 1 but not always true

legend = plt.legend(loc='best', shadow=True)
plt.tight_layout()
plt.show()
# prepare independent valdiation data
filenames_val = os.listdir("/kaggle/working/test1")
df_val = pd.DataFrame({
    'filename': filenames_val
})
n_samples = df_val.shape[0]
# testing generator
val_gen = ImageDataGenerator(rescale=1./255)
val_generator = val_gen.flow_from_dataframe(
    df_val,
    directory="/kaggle/working/test1",
    x_col='filename',
    y_col=None,
    class_mode=None,
    target_size=IMAGE_SIZE,
    batch_size=batch_size,
    shuffle=False
)
# prediction on independent validation sample
predict = model.predict_generator(val_generator, steps=np.ceil(n_samples/batch_size))
df_val['category'] = np.argmax(predict, axis=-1)
label_map = dict((v,k) for k,v in train_generator.class_indices.items())
print(label_map)
df_val['category'] = df_val['category'].replace(label_map)
df_val['category'] = df_val['category'].replace({'cat':0, 'dog':1})
# visualize prediction
df_val['category'].value_counts().plot.bar()
sample_val = df_val.head(18)
plt.figure(figsize=(12,24))
for index, row in sample_val.iterrows():
    filename = row['filename']
    catefory = row['category']
    img = load_img("/kaggle/working/test1/"+filename, target_size=IMAGE_SIZE)
    plt.subplot(6,3,index+1)
    plt.imshow(img)
    plt.xlabel(filename+'('+"{}".format(category)+')')
plt.tight_layout()
plt.show
# output to csv/excel file for checking
df_output = df_val.copy()
df_output['id'] = df_output['filename'].str.split('.').str[0]
df_output['label'] = df_output['category']
df_output.drop(['filename','category'],axis=1,inplace=True)
df_output.to_csv('predicted_output.csv',index=False)