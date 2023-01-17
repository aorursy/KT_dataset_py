import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import os
import math
import seaborn as sn
import albumentations as A
import tensorflow as tf
from tensorflow.keras.applications import mobilenet_v2 as tf_mobilenet_v2
from tensorflow.keras import layers as tf_layers
from tensorflow.keras import models as tf_models
from tensorflow.keras import callbacks as tf_callbacks
from sklearn import metrics as sk_metrics
from IPython.display import YouTubeVideo
YouTubeVideo('7vtpUklKlsk', width=800, height=450)
DATASET_PATH = '../input/lego-minifigures-classification'

df_index = pd.read_csv(os.path.join(DATASET_PATH, 'index.csv'), index_col=0)
df_metadata = pd.read_csv(os.path.join(DATASET_PATH, 'metadata.csv'), index_col=0)
df_index = pd.merge(df_index, df_metadata[['class_id', 'minifigure_name']], on='class_id')

df_index
YouTubeVideo('iedmZlFxjfA', width=800, height=450)
ax = df_index['minifigure_name'].value_counts().plot(
    kind='bar',
    figsize=(14,8),
    title="Count of each mini-figure",
)

ax.set_xlabel("Mini-figure")
ax.set_ylabel("Count")
plt.show()
YouTubeVideo('Ql8QPcp8818', width=800, height=450)
plt.figure(figsize=(16, 10))
for ind, el in enumerate(df_index.sample(15).iterrows(), 1):
    plt.subplot(3, 5, ind)
    image = cv2.imread(os.path.join(DATASET_PATH, el[1]['path']))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(image)
    plt.title(f"{el[1]['class_id']}: {el[1]['minifigure_name']}")
    plt.xticks([])
    plt.yticks([])
plt.figure(figsize=(16, 10))
for ind, el in enumerate(df_index[df_index['minifigure_name']=='SPIDER-MAN'].sample(15).iterrows(), 1):
    plt.subplot(3, 5, ind)
    image = cv2.imread(os.path.join(DATASET_PATH, el[1]['path']))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(image)
    plt.xticks([])
    plt.yticks([])
YouTubeVideo('oy5EeamF_M8', width=800, height=450)
class DataGenerator(tf.keras.utils.Sequence):
    def __init__(
        self, 
        paths, 
        targets, 
        image_size=(224, 224), 
        batch_size=64, 
        shuffle=True, 
        transforms=None
    ):
        # the list of paths to files
        self.paths = paths
        # the list with the true labels of each file
        self.targets = targets
        # images size
        self.image_size = image_size
        # batch size (the number of images)
        self.batch_size = batch_size
        # if we need to shuffle order of files
        # for validation we don't need to shuffle, for training - do
        self.shuffle = shuffle
        # Augmentations for our images. It is implemented with albumentations library
        self.transforms = transforms
        
        # Call function to create and shuffle (if needed) indices of files
        self.on_epoch_end()
        
    def on_epoch_end(self):
        # This function is called at the end of each epoch while training
        
        # Create as many indices as many files we have
        self.indexes = np.arange(len(self.paths))
        # Shuffle them if needed
        if self.shuffle:
            np.random.shuffle(self.indexes)
            
    def __len__(self):
        # We need that this function returns the number of steps in one epoch
        
        # How many batches we have
        return len(self.paths) // self.batch_size
    
    
    def __getitem__(self, index):
        # This function returns batch of pictures with their labels
        
        # Take in order as many indices as our batch size is
        indexes = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]
        
        # Take image file paths that are included in that batch
        batch_paths = [self.paths[k] for k in indexes]
        # Take labels for each image
        batch_y = [self.targets[k] - 1 for k in indexes]
        batch_X = []
        for i in range(self.batch_size):
            # Read the image
            img = cv2.imread(batch_paths[i])
            # Resize it to needed shape
            img = cv2.resize(img, self.image_size)
            # Convert image colors from BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # Normalize image
            img = img / 255.
            # Apply transforms (see albumentations library)
            if self.transforms:
                img = self.transforms(image=img)['image']
            
            batch_X.append(img)
            
        return np.array(batch_X), np.array(batch_y)
YouTubeVideo('hxLU32zhze0', width=800, height=450)
def get_train_transforms():
    return A.Compose(
        [
            A.Rotate(limit=30, border_mode=cv2.BORDER_REPLICATE, p=0.5),
            A.Cutout(num_holes=8, max_h_size=20, max_w_size=20, fill_value=0, p=0.5),
            A.Cutout(num_holes=8, max_h_size=20, max_w_size=20, fill_value=1, p=0.5),
            A.HorizontalFlip(p=0.5),
            A.RandomContrast(p=0.5),
            A.Blur(p=0.5),
        ], 
        p=1.0
    )
BASE_DIR = '../input/lego-minifigures-classification/'

# Read information about dataset
df = pd.read_csv('../input/lego-minifigures-classification/index.csv', index_col=0)

# Get only train rows
tmp_train = df[df['train-valid'] == 'train']
# Get train file paths
train_paths = tmp_train['path'].values
# Get train labels
train_targets = tmp_train['class_id'].values
# Create full train paths (base dir + concrete file)
train_paths = list(map(lambda x: os.path.join(BASE_DIR, x), train_paths))

# Get only valid rows
tmp_valid = df[df['train-valid'] == 'valid']
# Get valid file paths
valid_paths = tmp_valid['path'].values
# Get valid labels
valid_targets = tmp_valid['class_id'].values
# Create full valid paths (base dir + concrete file)
valid_paths = list(map(lambda x: os.path.join(BASE_DIR, x), valid_paths))
IMAGE_SIZE = (512, 512)

TRAIN_BATCH_SIZE = 4

VALID_BATCH_SIZE = 1 

# Initialize the train data generator
train_generator = DataGenerator(
    train_paths, 
    train_targets, 
    batch_size=TRAIN_BATCH_SIZE, 
    image_size=IMAGE_SIZE,
    shuffle=True, 
    transforms=get_train_transforms()
)

# Initialize the valid data generator
valid_generator = DataGenerator(
    valid_paths, 
    valid_targets, 
    image_size=IMAGE_SIZE,
    batch_size=VALID_BATCH_SIZE, 
    shuffle=False,
)
YouTubeVideo('OO4HD-1wRN8', width=800, height=450)
# We define the number of classes
N_CLASSES = 22

# We take pretrained MobileNetV2 (see Keras docs)
base_model = tf_mobilenet_v2.MobileNetV2()
# Take penultimate layer of the MobileNetV2 model and connect this layer with Dropout
x = tf_layers.Dropout(.5)(base_model.layers[-2].output)
# Add additional Dense layer, with number of neurons as number of our classes
# Use softmax activation because we have one class classification problem
outputs = tf_layers.Dense(N_CLASSES, activation='softmax')(x)
# Create model using MobileNetV2 input and our created output
model = tf_models.Model(base_model.inputs, outputs)


# Compile model using Adam optimizer and categorical crossentropy loss
model.compile(
    optimizer=tf.keras.optimizers.Adam(0.0001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
# checkpoint to saving the best model by validation loss
callback_save = tf_callbacks.ModelCheckpoint(
    'best.hdf5',
    monitor="val_loss",
    save_best_only=True,
    mode="min",
)

# checkpoint to stop training if model didn't improve valid loss for 3 epochs
callback_early_stopping = tf_callbacks.EarlyStopping(
    monitor="val_loss",
    patience=3,
)
EPOCHS = 25

# Train model using data generators
history = model.fit(
    train_generator,
    validation_data=valid_generator,
    epochs=EPOCHS,
    callbacks=[
        callback_save, 
        callback_early_stopping
    ],
)
YouTubeVideo('apmNSYWEEnw', width=800, height=450)
plt.figure(figsize=(16, 6))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='valid loss')
plt.grid()
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='train acc')
plt.plot(history.history['val_accuracy'], label='valid acc')
plt.grid()
plt.legend();
model = tf_models.load_model('best.hdf5')

y_pred = []
y_valid = []
for _X_valid, _y_valid in valid_generator:
    y_pred.extend(model.predict(_X_valid).argmax(axis=-1))
    y_valid.extend(_y_valid)

print(f'Accuracy score on validation data:  {sk_metrics.accuracy_score(y_valid, y_pred)}')
print(f'Macro F1 score on validation data:  {sk_metrics.f1_score(y_valid, y_pred, average="macro")}')
YouTubeVideo('Kdsp6soqA7o', width=800, height=450)
df_metadata = pd.read_csv('../input/lego-minifigures-classification/metadata.csv')
labels = df_metadata['minifigure_name'].tolist()

confusion_matrix = sk_metrics.confusion_matrix(y_valid, y_pred)
df_confusion_matrix = pd.DataFrame(confusion_matrix, index=labels, columns=labels)
plt.figure(figsize=(12, 12))
sn.heatmap(df_confusion_matrix, annot=True, cbar=False);
true_images = []
true_label = []
true_pred = []

for _X_valid, _y_valid in valid_generator:
    pred = model.predict(_X_valid).argmax(axis=-1)
    if pred[0] == _y_valid:
        true_images.extend(_X_valid)
        true_label.extend(_y_valid)
        true_pred.extend(pred)

true_images = true_images[:4]

for ind, image in enumerate(true_images):
    plt.subplot(math.ceil(len(true_images) / int(len(true_images) ** 0.5)), int(len(true_images) ** 0.5), ind + 1)
    plt.imshow(image)
    plt.title(f'Predicted: {labels[true_pred[ind]]} | Real: {labels[true_label[ind]]}')
    plt.axis('off')
error_images = []
error_label = []
error_pred = []

for _X_valid, _y_valid in valid_generator:
    pred = model.predict(_X_valid).argmax(axis=-1)
    if pred[0] != _y_valid:
        error_images.extend(_X_valid)
        error_label.extend(_y_valid)
        error_pred.extend(pred)

error_images = error_images[:4]

for ind, image in enumerate(error_images):
    plt.subplot(math.ceil(len(error_images) / int(len(error_images) ** 0.5)), int(len(error_images) ** 0.5), ind + 1)
    plt.imshow(image)
    plt.title(f'Predicted: {labels[error_pred[ind]]} | Real: {labels[error_label[ind]]}')
    plt.axis('off')