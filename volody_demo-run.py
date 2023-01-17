import numpy as np

import pandas as pd

import os

# apply ignore

import warnings

warnings.filterwarnings('ignore')
# show size of train and test data

train_images = os.listdir('../input/understanding_cloud_organization/train_images')

print(len(train_images))

test_images = os.listdir('../input/understanding_cloud_organization/test_images')

print(len(test_images))
train_data = pd.read_csv('../input/understanding_cloud_organization/train.csv')



# Split Image_Label into ImageId and Label

split = train_data['Image_Label'].str.split('_', n = 1, expand = True)

train_data['id'] = split[0]

train_data['label'] = split[1]



# Select columns 

selected_features = [cname for cname in train_data.columns if cname not in ['Image_Label']]



train_data.head()
# count unique labels 

train_data['label'].value_counts()
# Dropping old Image_Label columns 

df = train_data[['id', 'label']]

from tensorflow.python.keras.applications import ResNet50

from tensorflow.python.keras.models import Sequential

from tensorflow.python.keras.layers import Dense
num_classes = 4

# imagenet easy to debug

my_new_model = Sequential()

# resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5

my_new_model.add(ResNet50(include_top=False, pooling='avg', weights='imagenet'))

my_new_model.add(Dense(num_classes, activation='softmax'))



# Say not to train first layer (ResNet) model. It is already trained

my_new_model.layers[0].trainable = False
my_new_model.compile(optimizer='sgd', 

                     loss='categorical_crossentropy', 

                     metrics=['accuracy'])
my_new_model.summary()
from sklearn.model_selection import train_test_split

# for proto modelling, lets grab only part of the data



# Break off validation set from training data

train_df, valid_df, _, _ = train_test_split(df, df.label, train_size=0.1, 

                                                      test_size=0.1,random_state=2)
from tensorflow.keras.applications.resnet50 import preprocess_input

from tensorflow.keras.preprocessing.image import ImageDataGenerator



image_size = 224

data_generator = ImageDataGenerator(preprocess_input, validation_split=0.20)

train_data_dir = '../input/understanding_cloud_organization/train_images'



train_generator = data_generator.flow_from_dataframe(

                                        dataframe=train_df,

                                        directory=train_data_dir,

                                        x_col="id",

                                        y_col="label",

                                        target_size=(image_size, image_size),

                                        batch_size=1000,

                                        class_mode='categorical',

                                        subset='training')



validation_generator = data_generator.flow_from_dataframe(

                                        dataframe=valid_df,

                                        directory=train_data_dir,

                                        x_col="id",

                                        y_col="label",

                                        target_size=(image_size, image_size),

                                        batch_size=32,

                                        class_mode='categorical',

                                        subset='validation')
# fit_stats below saves some statistics describing how model fitting went

# the key role of the following line is how it changes my_new_model by fitting to data

fit_stats = my_new_model.fit_generator(train_generator,

                                       steps_per_epoch=2,

                                       validation_data=validation_generator,

                                       validation_steps=1)
# path to file for predictions

#test_data_path = '../input/test.csv'



# read test data

#test_data = pd.read_csv(test_data_path)



# create test_X which comes from test_data but includes only the columns used for prediction.

# The list of columns is stored in a variable called features

#test_X = test_data[forest_features]



# make predictions used to submit. 

#test_preds = np.round(forest_model.predict(test_X)).astype(int)



# The lines below shows how to save predictions in competition format



#output = pd.DataFrame({'id': test_data.Id,

#                       'label': test_preds})

#output.to_csv('submission.csv', index=False)