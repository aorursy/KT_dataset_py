validation_dir = "../input/plantdiesease/Untrained_Testing_Images/"

train_dir = "../input/training-dataset-complete/train/"
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(

      rescale=1./255,

       rotation_range=20,

#       width_shift_range=0.2,

#       height_shift_range=0.2,

       horizontal_flip=True,

      fill_mode='nearest',

    validation_split=0.3,

 

)



validation_datagen = ImageDataGenerator(rescale=1./255 ,    )



train_batchsize = 128

val_batchsize = 128

 

train_generator = train_datagen.flow_from_directory(

        train_dir,

    subset ="training",

        target_size=(224, 224),

        batch_size=train_batchsize,

        class_mode='categorical')

validation_generator = train_datagen.flow_from_directory(

        train_dir,

    subset ="validation",

        target_size=(224, 224),

        batch_size=train_batchsize,

        class_mode='categorical')

#validation_generator = validation_datagen.flow_from_directory(

#   validation_dir,

#    

#   target_size=(224, 224),

#   batch_size=val_batchsize,

#   class_mode='categorical',

#   shuffle=True)

from keras.applications import VGG16

#Load the VGG model

vgg_conv = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))



for layer in vgg_conv.layers[:-4]:

    layer.trainable = False

 

# Check the trainable status of the individual layers

for layer in vgg_conv.layers:

    print(layer, layer.trainable)


from keras import models

from keras import layers

from keras import optimizers

 

# Create the model

model = models.Sequential()

 

# Add the vgg convolutional base model

model.add(vgg_conv)

 

# Add new layers

model.add(layers.Flatten())

model.add(layers.Dense(1024, activation='relu'))

model.add(layers.Dense(512, activation='relu'))

model.add(layers.Dropout(0.5))

model.add(layers.Dense(38, activation='softmax'))
model.compile(loss='categorical_crossentropy',

              optimizer=optimizers.RMSprop(lr=1e-4),

              metrics=['acc'])

# Train the model

history = model.fit_generator(

      train_generator,

      steps_per_epoch=train_generator.samples/train_generator.batch_size ,

      epochs=2,

      validation_data=validation_generator,

      validation_steps=validation_generator.samples/validation_generator.batch_size,

      verbose=1)