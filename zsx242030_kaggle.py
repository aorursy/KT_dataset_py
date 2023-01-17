import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import skimage.io
import skimage.transform
import keras
from keras.layers import Conv2D,Dropout,Flatten,Dense,MaxPool2D
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.utils import plot_model
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from keras.applications.vgg16 import VGG16
trn_path = "../input/dataset/dataset_updated/training_set/"
test_path = "../input/dataset/dataset_updated/validation_set/"

cats = ['drawings', 'engraving', 'iconography', 'painting', 'sculpture']
n_cats = len(cats)
category_embeddings = {
    'drawings': 0,
    'engraving': 1,
    'iconography': 2,
    'painting': 3,
    'sculpture': 4
}

width,height,channels = 128,128,3
batchsize = 16
# training dataset metadata
n_imgs = []
for cat in cats:
    files = os.listdir(os.path.join(trn_path, cat))
    n_imgs += [len(files)]
    
plt.bar([_ for _ in range(n_cats)], n_imgs, tick_label=cats)
plt.show()
#lets visualize some of the images
fig,axes = plt.subplots(nrows=1,ncols=n_cats,figsize=(15,3))

cat_cpt=0
for cat in cats:
    category_path = os.path.join(trn_path,cat)
    img_name=os.listdir(category_path)[0]
    img = skimage.io.imread(os.path.join(category_path,img_name))
    img = skimage.transform.resize(img,(width,height,channels),mode='reflect')
    axes[cat_cpt].imshow(img,resample=True)
    axes[cat_cpt].set_title(cat,fontsize=8)
    cat_cpt += 1

plt.show()
#create the training dataset which will be tuples
#will be used to read images batch by batch
trn_data = []
for cat in cats:
    files = os.listdir(os.path.join(trn_path,cat))
    for file in files:
        trn_data += [(os.path.join(cat,file),cat)]
        
        
test_data = []
for cat in cats:
    files = os.listdir(os.path.join(test_path,cat))
    for file in files:
        test_data += [(os.path.join(cat,file),cat)]
def load_dataset(tuples_list, dataset_path):
    indexes = np.arange(len(tuples_list))
    np.random.shuffle(indexes)
    
    X = []
    y = []
    n_samples = len(indexes)
    cpt = 0
    for i in range(n_samples):
        t = tuples_list[indexes[i]]
        try:
            img = skimage.io.imread(os.path.join(dataset_path, t[0]))
            img = skimage.transform.resize(img, (width, height,channels), mode='reflect')
            X += [img]
            y_tmp = [0 for _ in range(n_cats)]
            y_tmp[category_embeddings[t[1]]] = 1
            y += [y_tmp]
        except OSError:
            pass
        
        cpt += 1
        
        if cpt % 1000 == 0:
            print("Processed {} images".format(cpt))
    return X, y
x_train, y_train = load_dataset(trn_data, trn_path)
x_val, y_val = load_dataset(test_data, test_path)
print(len(x_train))
print(len(y_train))
print(len(x_val))
print(len(y_val))
x_train = np.array(x_train)
y_train = np.array(y_train)
x_val=np.array(x_val)
y_val=np.array(y_val)
x_train.shape
y_train.shape
x_val.shape
y_val.shape
#data augmentation
trn_augs = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)
trn_augs.fit(x_train)
tmodel = Sequential()
tmodel.add(Conv2D(32, kernel_size=5, input_shape=(width, height,channels), activation='relu'))
tmodel.add(MaxPool2D(pool_size=(2, 2)))
tmodel.add(Conv2D(48, kernel_size=3, activation='relu'))
tmodel.add(MaxPool2D(pool_size=(2, 2)))
tmodel.add(Dropout(0.35))
tmodel.add(Flatten())
tmodel.add(Dense(512, activation='relu'))
tmodel.add(Dropout(0.25))
tmodel.add(Dense(256,activation='relu'))
tmodel.add(Dropout(0.10))
tmodel.add(Dense(n_cats, activation='softmax'))

tmodel.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

tmodel.summary()
trn_gen = trn_augs.flow(x_train,y_train,batch_size=batchsize)

history = tmodel.fit_generator(trn_gen,
                              validation_data=(x_val,y_val),
                              epochs=30,
                              verbose=1,
                              steps_per_epoch=200)
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 3))

axes[0].plot(history.history['loss'], label="Loss")
axes[0].plot(history.history['val_loss'], label="Validation loss")
axes[0].set_title('Loss')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].legend()


axes[1].plot(history.history['acc'], label="Accuracy")
axes[1].plot(history.history['val_acc'], label="Validation accuracy")
axes[1].set_title('Accuracy')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy')
axes[1].legend()
plt.tight_layout()

plt.show()


X_test = []
y_test = []
for t in test_data:
    try:
        img = skimage.io.imread(os.path.join(test_path, t[0]))
        img = skimage.transform.resize(img, (width, height, channels), mode='reflect')
        X_test += [img]
        y_test += [category_embeddings[t[1]]]
    except OSError:
        pass

X_test = np.array(X_test)
y_test = np.array(y_test)
pred = tmodel.predict(X_test, verbose=1)

y_pred = np.argmax(pred, axis=1)
print(classification_report(y_test, y_pred))

cmatrix = confusion_matrix(y_test, y_pred)
plt.imshow(cmatrix, cmap=plt.cm.Blues)
plt.title("Confusion matrix")
plt.colorbar()
plt.show()
print(cmatrix)
plot_model(tmodel,to_file='model.png')
