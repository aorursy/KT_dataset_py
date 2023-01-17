import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.applications import ResNet50
from sklearn.model_selection import train_test_split
import os
print(os.listdir("../input"))
train_df = pd.read_csv("../input/train_labels.csv")
train,valid = train_test_split(train_df, test_size=0.3)
num_classes = 2
resnet_weights_path = '../input/resnet50/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'
model = Sequential()
model.add(ResNet50(include_top=False, pooling='avg', weights="imagenet"))
model.add(Dense(num_classes, activation='softmax'))
model.layers[0].trainable = False
model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
from tensorflow.python.keras.applications.resnet50 import preprocess_input
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator

image_size = 224
train_datagen = ImageDataGenerator(preprocess_input)
validation_datagen = ImageDataGenerator(preprocess_input)
train_generator = train_datagen.flow_from_dataframe(
    dataframe=train,
    directory='../input/train/',
    x_col='id',
    y_col='label',
    has_ext=False,
    shuffle=True
    )
validation_generator = validation_datagen.flow_from_dataframe(
    dataframe=valid,
    directory='../input/train/',
    x_col='id',
    y_col='label',
    has_ext=False,
    shuffle=False
)
model.fit_generator(train_generator,
                    steps_per_epoch=10,
                    validation_data=validation_generator,
                    validation_steps=10,
                    epochs=13)
from glob import glob
from skimage.io import imread

base_test_dir = '../input/test/'
test_files = glob(os.path.join(base_test_dir,'*.tif'))
submission = pd.DataFrame()
file_batch = 5000
max_idx = len(test_files)
for idx in range(0, max_idx, file_batch):
    print("Indexes: %i - %i"%(idx, idx+file_batch))
    test_df = pd.DataFrame({'path': test_files[idx:idx+file_batch]})
    test_df['id'] = test_df.path.map(lambda x: x.split('/')[3].split(".")[0])
    test_df['image'] = test_df['path'].map(imread)
    K_test = np.stack(test_df["image"].values)
    K_test = (K_test - K_test.mean()) / K_test.std()
    predictions = model.predict(K_test)
    test_df['label'] = predictions[:,1]
    submission = pd.concat([submission, test_df[["id", "label"]]])
submission.head()

submission.to_csv("submission.csv", index = False, header = True)

predictions
