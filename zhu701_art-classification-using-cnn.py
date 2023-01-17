import numpy as np 

import pandas as pd

from keras.models import Sequential

from keras.layers import Conv2D, MaxPooling2D,Flatten,Dense, Dropout



classifier = Sequential()



# Step 1 - Convolution

classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))



# Step 2 - Pooling

classifier.add(MaxPooling2D(pool_size = (2, 2)))

classifier.add(Dropout(0.2))



# Adding a second layer

classifier.add(Conv2D(32, (3, 3), activation = 'relu'))

classifier.add(MaxPooling2D(pool_size = (2, 2)))

classifier.add(Dropout(0.2))



# Adding a third layer

classifier.add(Conv2D(64, (3, 3), activation = 'relu'))

classifier.add(MaxPooling2D(pool_size = (2, 2)))

classifier.add(Dropout(0.2))



# Step 3 - Flattening

classifier.add(Flatten())



# Step 4 - Full connection

classifier.add(Dense(units = 128, activation = 'relu'))

classifier.add(Dropout(0.2))

classifier.add(Dense(units = 5, activation = 'softmax'))



# Compiling 

classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

classifier.summary()
from keras.preprocessing.image import ImageDataGenerator



train_datagen = ImageDataGenerator(rescale = 1./255,

                                   shear_range = 0.2,

                                   zoom_range = 0.2,

                                   horizontal_flip = True)



test_datagen = ImageDataGenerator(rescale = 1./255)



training_set = train_datagen.flow_from_directory(directory="../input/art-images-drawings-painting-sculpture-engraving/dataset/dataset_updated/training_set/",

                                                 target_size = (64, 64),

                                                 color_mode="rgb",

                                                 batch_size = 32,

                                                 class_mode="categorical")



test_set = test_datagen.flow_from_directory(directory="../input/art-images-drawings-painting-sculpture-engraving/dataset/dataset_updated/validation_set/",

                                            target_size = (64, 64),

                                            batch_size = 32,

                                            class_mode = 'categorical')
def skip(pic):

    while True:

        try:

            data, labels = next(pic)

            yield data, labels

        except:

            pass
history = classifier.fit_generator(skip(training_set),

                         steps_per_epoch = 200,

                         epochs = 15,

                         validation_data = skip(test_set),

                         validation_steps = 50)
import matplotlib.pyplot as plt



plt.plot(history.history['accuracy'], label="Accuracy")

plt.plot(history.history['val_accuracy'], label="Validation accuracy")

plt.title('Accuracy')

plt.xlabel('Epoch')

plt.ylabel('Accuracy')

plt.legend()

plt.tight_layout()

plt.show()
# Checking the indices 

training_set.class_indices
# Define a prediction function 

from keras.preprocessing import image

def predict(img):

    test_image = image.load_img(img, target_size = (64, 64))

    test_image = image.img_to_array(test_image)

    test_image = np.expand_dims(test_image, axis = 0)

    result = classifier.predict(test_image)

    if result[0][0] == 1:

        prediction = 'drawing'

    elif result[0][1] == 1:

        prediction = 'engraving'

    elif result[0][2] == 1:

        prediction = 'iconography'

    elif result[0][3] == 1:

        prediction = 'painting'

    else:

        prediction='sculpture'

    print("The predited art type is",prediction)
predict('../input/art-pred/drawing.png')
predict('../input/art-pred/elephant engraving.jpg')
predict('../input/art-pred/buddha icon.jpg')