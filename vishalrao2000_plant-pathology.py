import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
train_csv = pd.read_csv('../input/plant-pathology-2020-fgvc7/train.csv')
train_csv.head()
train_csv.shape
img_name = []
for img_id in train_csv['image_id']:
    img_name.append(str(img_id)+'.jpg')
    
train_csv['image_id'] = img_name
train_csv.head()
test_csv = pd.read_csv('../input/plant-pathology-2020-fgvc7/test.csv')
test_csv.head()
img_name = []
for img_id in test_csv['image_id']:
    img_name.append(str(img_id)+'.jpg')
    
test_csv['image_id'] = img_name
test_csv.head()
from keras.models import Sequential
from keras_preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras import regularizers, optimizers
train_csv.shape
test_csv.shape
# 1721 => train and 100 => validation
columns = ['healthy', 'multiple_diseases', 'rust', 'scab']

datagen = ImageDataGenerator(rescale=1./255.)
test_datagen = ImageDataGenerator(rescale=1./255.)

# train data generator
train_generator = datagen.flow_from_dataframe(dataframe=train_csv[:1721],
                                             directory="../input/plant-pathology-2020-fgvc7/images/",
                                             x_col='image_id',
                                             y_col=columns,
                                             batch_size=32,
                                             seed=42,
                                             shuffle=True,
                                             class_mode='raw',
                                             target_size=(100, 100))
# validation generator
valid_generator = test_datagen.flow_from_dataframe(dataframe=train_csv[1721:],
                                                  directory="../input/plant-pathology-2020-fgvc7/images/",
                                                  x_col='image_id',
                                                  y_col=columns,
                                                  batch_size=32,
                                                  seed=42,
                                                  shuffle=True,
                                                  class_mode='raw',
                                                  target_size=(100, 100))

# test generator
test_generator = test_datagen.flow_from_dataframe(dataframe=test_csv,
                                                 directory="../input/plant-pathology-2020-fgvc7/images/",
                                                 x_col='image_id',
                                                 batch_size=1,
                                                 seed=42,
                                                 shuffle=False,
                                                 class_mode=None,
                                                 target_size=(100, 100))
### modelling

model = Sequential()
model.add(Conv2D(32, (3,3), padding='same', input_shape=(100,100,3), activation='relu'))
model.add(Conv2D(32, (3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3,3), padding='same', activation='relu'))
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(4, activation='sigmoid'))

### compiling 
model.compile(optimizers.rmsprop(lr=0.0001, decay=1e-6),
             loss='binary_crossentropy',
             metrics=['accuracy'])

model.summary()
### fitting the model

STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
STEP_SIZE_VALID=valid_generator.n//valid_generator.batch_size
STEP_SIZE_TEST=test_generator.n//test_generator.batch_size
model.fit_generator(generator=train_generator,
                    steps_per_epoch=STEP_SIZE_TRAIN,
                    validation_data=valid_generator,
                    epochs=10,
                    validation_steps=STEP_SIZE_VALID
)
### predict the output

test_generator.reset()
pred = model.predict_generator(test_generator, steps=STEP_SIZE_TEST, verbose=1)
pred[0]
pred_bool = (pred>0.5)
pred_bool[0]
predictions = pred_bool.astype(int)
columns = ['healthy', 'multiple_diseases', 'rust', 'scab']

results = pd.DataFrame(predictions, columns=columns)
### add image id column
results['image_id'] = test_generator.filenames
ordered_cols = ['image_id'] + columns
results = results[ordered_cols]
results
# we need to remove .jpg
test_tmp = pd.read_csv('../input/plant-pathology-2020-fgvc7/test.csv')
results['image_id'] = test_tmp['image_id']
results
pred[0]
test_filenames = []
for im_id in results['image_id']:
    test_filenames.append('../input/plant-pathology-2020-fgvc7/images/'+im_id+'.jpg')
# let's visualize one of them

plt.imshow(plt.imread(test_filenames[0]))
### lets plot some predictions
# i_multiplier = 0
num_rows = 3
num_cols = 2
num_images = num_rows*num_cols
plt.figure(figsize=(5*2*num_cols, 5*num_rows))
for i in range(num_images):
    plt.subplot(num_rows, 2*num_cols, 2*i+1)
    plt.imshow(plt.imread(test_filenames[i]))
    
    plt.subplot(num_rows, 2*num_cols, 2*i+2)
    plt.bar(np.arange(4), pred[i])
    plt.xticks(np.arange(4), labels=['healthy', 'multiple_diseases', 'rust', 'scab'], rotation='vertical')
plt.tight_layout(h_pad=1.0)
plt.show()
