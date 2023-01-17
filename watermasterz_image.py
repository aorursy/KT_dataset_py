import tensorflow as tf

from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout

from tensorflow.keras.models import Model, Sequential

from tensorflow.keras.preprocessing.image import ImageDataGenerator

import matplotlib.pyplot as plt

dir = "../input/natural-images/natural_images"

img_size = (128, 128)
datagen = ImageDataGenerator(

    rescale=1./255, 

    horizontal_flip=True, 

    rotation_range=15, 

    validation_split=0.2

)





train_data = datagen.flow_from_directory(

    dir,

    class_mode='categorical',

    target_size=img_size,

    shuffle=True,

    subset='training',

    batch_size=32

)



test_data = datagen.flow_from_directory(

    dir,

    class_mode='categorical',

    target_size=img_size,

    shuffle=True,

    subset='validation',

    batch_size=32

)

labels = ['airplane', 'car', 'cat', 'dog', 'flower', 'fruit', 'motorbike', 'person']

def get_labels(preds):

    return labels[tf.math.argmax(preds).numpy()]
x,y = next(iter(train_data))

plt.imshow(x[5])

print(get_labels(y[5]))
seq = Sequential([

    Conv2D(32, 3, activation='relu', input_shape=(128, 128, 3)),

    Dropout(0.2),

    Conv2D(64, 5, activation='relu'),

    MaxPooling2D(2),

    Dropout(0.2),

    

    Conv2D(128, 5, activation='relu'),

    MaxPooling2D(4),

    Dropout(0.2),

    

    Flatten(),

    

    Dense(64, activation='relu'),

    

    Dense(8, activation='softmax')

])

seq.summary()

seq.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

history = seq.fit(train_data, epochs=25, validation_data=test_data, )
plt.figure(figsize=(11,5))

plt.xlabel("Epochs")

plt.ylabel("Accuracy")

plt.title("Accuracy vs Validation Accuracy")

plt.plot(history.history['acc'], label='Train')

plt.plot(history.history['val_acc'], label='Validation')

plt.legend()
plt.figure(figsize=(11,5))

plt.title('Loss vs Validation Loss')

plt.xlabel("Epochs")

plt.ylabel("Loss")

plt.plot(history.history['loss'], label='Train')

plt.plot(history.history['val_loss'], label='Validation')

plt.legend()