%autosave 10
from keras.layers import Input, Lambda, Dense, Flatten
from keras.models import Model
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
import os
import matplotlib.pyplot as plt
from glob import glob
import numpy as np
base_path = "../input/kermany2018/oct2017/OCT2017 /"
# re-size all the images to this
IMAGE_SIZE = 100

# training config:
epochs = 5
batch_size = 32
train_path = base_path + 'train'
valid_path = base_path + 'val'
test_path = base_path + 'test'
image_files = glob(train_path + '/*/*.jp*g')
valid_image_files = glob(valid_path + '/*/*.jp*g')
test_image_files = glob(test_path + '/*/*.jp*g')
folders = glob(train_path + '/*')
# look at an image for fun
plt.imshow(image.load_img(np.random.choice(image_files)))
plt.axis('off')
plt.show()
weights_file = "../input/keras-pretrained-models/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5"
vgg = VGG16(input_shape=[IMAGE_SIZE,IMAGE_SIZE] + [3], weights=weights_file, include_top=False)


# don't train existing weights
for layer in vgg.layers:
  layer.trainable = False

# our layers - you can add more if you want
x = Flatten()(vgg.output)
# x = Dense(1000, activation='relu')(x)
prediction = Dense(len(folders), activation='softmax')(x)


# create a model object
model = Model(inputs=vgg.input, outputs=prediction)

# view the structure of the model
model.summary()

# tell the model what cost and optimization method to use
model.compile(
  loss='categorical_crossentropy',
  optimizer='rmsprop',
  metrics=['accuracy']
)
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
image_size = 100
data_generator_with_aug = ImageDataGenerator(
                                            rotation_range=40,
                                            width_shift_range=0.2,
                                            height_shift_range=0.2,
                                            rescale=1./255,
                                            shear_range=0.2,
                                            zoom_range=0.2,
                                            horizontal_flip=True,
                                            fill_mode='nearest'
)
            
data_generator_no_aug = ImageDataGenerator(rescale = 1./255)

train_generator = data_generator_with_aug.flow_from_directory(
       directory = base_path + "train/",
       target_size = (IMAGE_SIZE, IMAGE_SIZE),
       batch_size = batch_size,
       class_mode = 'categorical')

validation_generator = data_generator_no_aug.flow_from_directory(
       directory = base_path + "test/",
       target_size = (IMAGE_SIZE, IMAGE_SIZE),
       class_mode = 'categorical',
       shuffle = False)

fitted_model = model.fit_generator(
       train_generator, # specify where model gets training data
       epochs = epochs,
       steps_per_epoch = 83484/batch_size,
       validation_steps=len(validation_generator),
       validation_data = validation_generator) # specify where model gets validation data
model.save('m1.h5')
def get_confusion_matrix(data_path, N):
  # we need to see the data in the same order
  # for both predictions and targets
  print("Generating confusion matrix", N)
  predictions = []
  targets = []
  i = 0
  for x, y in gen.flow_from_directory(data_path, target_size=[IMAGE_SIZE,IMAGE_SIZE], shuffle=False, batch_size=batch_size):
    i += 1
    if i % 50 == 0:
      print(i)
    p = model.predict(x)
    p = np.argmax(p, axis=1)
    y = np.argmax(y, axis=1)
    predictions = np.concatenate((predictions, p))
    targets = np.concatenate((targets, y))
    if len(targets) >= N:
      break

  cm = confusion_matrix(targets, predictions)
  return cm
# valid_cm = get_confusion_matrix(valid_path, len(valid_image_files))
# print(valid_cm)
test_cm = get_confusion_matrix(test_path, len(test_image_files))
print(test_cm)
result = np.round(model.predict_generator(validation_generator, steps=len(validation_generator)))
import random
test_files = []
actual_res = []
test_res = []
for i in range(0, 3):
  rng = random.randint(0, len(validation_generator.filenames))
  test_files.append(test_path + '/' + validation_generator.filenames[rng])
  actual_res.append(validation_generator.filenames[rng].split('/')[0])
  test_res.append(labels[np.argmax(result[rng])])
  
for i in range(0, 3):
    plt.imshow(plt.imread(test_files[i]), cmap='gray')
    plt.axis('off')
    plt.show()
    print("Actual class: " + str(actual_res[i]))
    print("Predicted class: " + str(test_res[i]))
