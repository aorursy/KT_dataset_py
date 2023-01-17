import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os



from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelBinarizer



from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, GlobalAveragePooling2D

from tensorflow.keras.models import Model, Sequential

from tensorflow.keras.applications import VGG16

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau



import matplotlib.pyplot as plt



import cv2
BASEPATH = "../input/images/Images/"



LABELS = set()



paths = []

    

for d in os.listdir(BASEPATH):

    LABELS.add(d)

    paths.append((BASEPATH+d, d))
# resizing and converting to RGB

def load_and_preprocess_image(path):

    image = cv2.imread(path)

    image = cv2.resize(image, (224,224))

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return image
X = []

y = []



for path, label in paths:

    for image_path in os.listdir(path):

        image = load_and_preprocess_image(path+"/"+image_path)

        

        X.append(image)

        y.append(label)
encoder = LabelBinarizer()



X = np.array(X)

y = encoder.fit_transform(np.array(y))



print(y[0])
print(X.shape)

print(y.shape)

plt.imshow(X[0])
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.1)
model = Sequential()



model.add(Conv2D(64,(3,3),activation="relu", padding="same"))

model.add(MaxPooling2D())

model.add(Dropout(0.2))



model.add(Conv2D(64,(3,3),activation="relu", padding="same"))

model.add(MaxPooling2D())

model.add(Dropout(0.2))



model.add(Conv2D(64,(3,3),activation="relu", padding="same"))

model.add(MaxPooling2D())

model.add(Dropout(0.2))



model.add(Conv2D(128,(3,3),activation="relu", padding="same"))

model.add(MaxPooling2D())



model.add(Conv2D(128,(3,3),activation="relu", padding="same"))

model.add(MaxPooling2D())



model.add(Flatten())



model.add(Dense(1024,activation="relu"))

model.add(Dropout(0.5))



model.add(Dense(256,activation="relu"))

model.add(Dropout(0.2))



model.add(Dense(len(LABELS),activation="softmax"))
base_model=VGG16(weights='imagenet',include_top=False)



x=base_model.output

x=GlobalAveragePooling2D()(x)

x=Dense(1024,activation='relu')(x)

x=Dense(1024,activation='relu')(x)

x=Dropout(0.5)(x)

x=Dense(512,activation='relu')(x)

preds=Dense(len(LABELS),activation='softmax')(x)



model=Model(inputs=base_model.input,outputs=preds)



for layer in model.layers[:-5]:

    layer.trainable=False

for layer in model.layers[-5:]:

    layer.trainable=True

    

model.compile("adam",loss="categorical_crossentropy",metrics=["accuracy"])



print(model.summary())
early_stopping = EarlyStopping(patience=5, verbose=1,restore_best_weights=True)

reduce_lr = ReduceLROnPlateau(factor=0.1, patience=3,verbose=1)
model.fit(X_train,y_train,batch_size=64,epochs=50,validation_data=(X_test,y_test), callbacks=[early_stopping, reduce_lr])
loss, acc = model.evaluate(X_test,y_test,verbose=0)

print(f"loss on the test set is {loss:.2f}")

print(f"accuracy on the test set is {acc:.3f}")
predictions = model.predict(X_test)
label_predictions = encoder.inverse_transform(predictions)
rows, cols = 5, 3

size = 25



fig,ax=plt.subplots(rows,cols)

fig.set_size_inches(size,size)

for i in range(rows):

    for j in range (cols):

        index = np.random.randint(0,len(X_test))

        ax[i,j].imshow(X_test[index])

        ax[i,j].set_title(f'Predicted: {label_predictions[index]}\n Actually: {encoder.inverse_transform(y_test)[index]}')

        

plt.tight_layout()