import numpy as np

import matplotlib.pyplot as plt

from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, array_to_img

from keras.layers import Conv2D, Flatten, MaxPooling2D, Dense

from keras.models import Sequential



import glob, os, random
base_path = './dataset-resized'

img_list = glob.glob(os.path.join(base_path, '*/*.jpg'))

print(len(img_list))
for i, img_path in enumerate(random.sample(img_list, 6)):

    img = load_img(img_path)

    img = img_to_array(img, dtype=np.uint8)

    

    plt.subplot(2, 3, i+1)

    plt.imshow(img.squeeze())
train_datagen = ImageDataGenerator(

    rescale=1./225, shear_range=0.1, zoom_range=0.1,

    width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True,

    vertical_flip=True, validation_split=0.1)



test_datagen = ImageDataGenerator(

    rescale=1./255, validation_split=0.1)

    

train_generator = train_datagen.flow_from_directory(

    base_path, target_size=(300, 300), batch_size=16,

    class_mode='categorical', subset='training', seed=0)



validation_generator = test_datagen.flow_from_directory(

    base_path, target_size=(300, 300), batch_size=16,

    class_mode='categorical', subset='validation', seed=0)



labels = (train_generator.class_indices)

labels = dict((v,k) for k,v in labels.items())



print(labels)
model = Sequential([

    Conv2D(filters=32, kernel_size=3, padding='same', activation='relu', input_shape=(300, 300, 3)),

    MaxPooling2D(pool_size=2),



    Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'),

    MaxPooling2D(pool_size=2),

    

    Conv2D(filters=32, kernel_size=3, padding='same', activation='relu'),

    MaxPooling2D(pool_size=2),

    

    Conv2D(filters=32, kernel_size=3, padding='same', activation='relu'),

    MaxPooling2D(pool_size=2),



    Flatten(),



    Dense(64, activation='relu'),



    Dense(6, activation='softmax')

])
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
model.fit_generator(train_generator, epochs=100, steps_per_epoch=2276//32,validation_data=validation_generator,

                    validation_steps=251//32)
test_x, test_y = validation_generator.__getitem__(1)



preds = model.predict(test_x)



plt.figure(figsize=(16, 16))

for i in range(16):

    plt.subplot(4, 4, i+1)

    plt.title('pred:%s / truth:%s' % (labels[np.argmax(preds[i])], labels[np.argmax(test_y[i])]))

    plt.imshow(test_x[i])