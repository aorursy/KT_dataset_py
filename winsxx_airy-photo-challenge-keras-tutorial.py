import os

import cv2 

import numpy as np

import pandas as pd



import warnings

warnings.filterwarnings('ignore')
# Change variables

TEST_DIR_PATH = '/kaggle/input/airy-photo/test'

DATA_DIR_PATH = '/kaggle/input/airy-photo/train'
# Read dataset



data_raw = []

for class_dir in os.listdir(DATA_DIR_PATH):

    class_absolute_dir = os.path.join(DATA_DIR_PATH, class_dir)

    for class_file in os.listdir(class_absolute_dir):

        absolute_file_path = os.path.join(class_absolute_dir, class_file)

        img = cv2.imread(absolute_file_path, cv2.IMREAD_COLOR)

        data_raw.append([

            class_file, # file name act as id

            absolute_file_path, 

            class_dir, # label / category 

            img.shape[0], # height

            img.shape[1] # width

        ])



data_pd = pd.DataFrame(data=data_raw, columns=['photo_id', 'file_path', 'category', 'height', 'width'])
# Read testset



test_raw = []

for test_file in os.listdir(TEST_DIR_PATH):

    absolute_file_path = os.path.join(TEST_DIR_PATH, test_file)

    img = cv2.imread(absolute_file_path, cv2.IMREAD_COLOR)

    test_raw.append([

        test_file, # file name act as id

        absolute_file_path, # absolute file path

        img.shape[0], #height

        img.shape[1] #width

    ])



test_pd = pd.DataFrame(data=test_raw, columns=['photo_id', 'file_path', 'height', 'width'])
# Read sample submission



sample_submission = pd.read_csv('/kaggle/input/airy-photo/sample_submission.csv')
from sklearn.metrics import f1_score, classification_report



def evaluation_report_airy_photo(prediction, target) -> None:

    '''

    This is the evaluation report for Airy Photo Classification.

    Accept two pandas DataFrame, both contains 2 columns (photo_id, category)

    '''

    assert prediction.size == target.size, 'Prediction and target should have equal length.'

    

    print(classification_report(y_true=target, y_pred=prediction))

    print('-----------------------------------------------------------')

    evaluation_score = f1_score(y_true=target, y_pred=prediction, average='micro')

    print('\tMicro F1-score: {:.3f}'.format(evaluation_score))

    print('-----------------------------------------------------------')

    
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

%matplotlib inline
data_pd.head()
sample_submission.head()
# Change variables

VALIDATION_PROPORTION = 0.3

LABEL_ATTRIBUTE = 'category'
train_pd, val_pd = train_test_split(data_pd, test_size=VALIDATION_PROPORTION, stratify=data_pd.category, random_state=23)

train_X_pd = train_pd.loc[:, train_pd.columns != LABEL_ATTRIBUTE]

train_y_pd = train_pd.loc[:, LABEL_ATTRIBUTE]

val_X_pd = val_pd.loc[:, val_pd.columns != LABEL_ATTRIBUTE]

val_y_pd = val_pd.loc[:, LABEL_ATTRIBUTE]
def show_image(img: np.array, ax = None) -> None:

    '''

    Display an image. Image is an numpy array (height, width, channel)

    This function expect channel is in RGB color space.

    '''

    if ax is not None:

        ax.imshow(img)

    else:

        plt.imshow(img)

    

def read_image(file_path: str) -> np.array:

    '''

    Read an image form given a file name. Image is an numpy array (height, width, channel). 

    Channel is in RGB color space.

    '''

    img = cv2.imread(file_path, cv2.IMREAD_COLOR)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img



def pad_and_crop_image(img: np.array, width: int, height:int, pad_value = [0, 0, 0]) -> np.array:

    '''

    This function receive a numpy array of img and the target size.

    Add padding to bottom and right of the image.

    '''

    # Crop if current image size exceed target size

    target_img = img[:height,:width]

    # Pad the image if the size is not match the target size

    border_bottom = height-target_img.shape[0]

    border_right = width-target_img.shape[1]

    target_img = cv2.copyMakeBorder(target_img, 0, border_bottom, 0, border_right, cv2.BORDER_CONSTANT, value = pad_value)

    

    return target_img



def resize_image(img: np.array, scale_ratio:float=1) -> np.array:

    '''

    Scale image to have size = original_size * scale

    '''

    height = int(round(img.shape[0] * scale_ratio))

    width = int(round(img.shape[1] * scale_ratio))

    target_img = cv2.resize(img,(width, height)) 

    return target_img



def preprocess_image(img, target_width, target_height):

    '''

    Resizing, pad or crop the image to match the target size

    '''

    # Choose to crop than pad

    target_aspect_ratio = target_width / target_height

    current_aspect_ratio = img.shape[1] / img.shape[0] 

    

    if (current_aspect_ratio > target_aspect_ratio): # current width > target width

        scale_ratio = target_height / img.shape[0]

    else:

        scale_ratio = target_width / img.shape[1]

    

    # Resize and crop

    current_img = resize_image(img, scale_ratio)

    current_img = pad_and_crop_image(current_img, width=target_width, height=target_height)

    return current_img
train_y_pd.value_counts().plot(kind='bar');
SAMPLE_SIZE = 4



# Sample SAMPLE_SIZE images for each category

category_image_samples = train_pd.groupby('category')['file_path'].apply(lambda df: df.sample(SAMPLE_SIZE, random_state=23))



# Prepare the figure

num_category = category_image_samples.index.get_level_values(0).unique().size

fig, axes = plt.subplots(num_category, SAMPLE_SIZE)

fig.set_size_inches(SAMPLE_SIZE * 6, num_category * 4)



for idx, (category, df_select) in enumerate(category_image_samples.groupby(level=0)):

    row_ax = axes[idx]

    

    # Draw images

    for ax, filename in zip(row_ax, df_select):

        ax.axis('off')

        img = read_image(filename)

        show_image(img, ax)

        ax.set_title(category, fontsize=20)



fig.tight_layout()

plt.show()
bedroom_proportion = float((train_y_pd == 'bedroom').sum()) / train_y_pd.size

print('Bedroom proportion: {:.2f}'.format( bedroom_proportion ))

print('If we predict all as bedroom, we will have micro recall = {:.2f}, and precision = {:.2f}'.format(bedroom_proportion, bedroom_proportion))

print('Thus the F1-score = {:.2f}'.format( 2 * bedroom_proportion * bedroom_proportion / (bedroom_proportion + bedroom_proportion) ))
benchmark_prediction = pd.DataFrame(

    data = {

        'photo_id': val_X_pd.photo_id,

        'category': 'bedroom'

    })



evaluation_report_airy_photo(

    prediction=benchmark_prediction.category, 

    target=val_y_pd)



benchmark_prediction = pd.DataFrame(

    data = {

        'photo_id': test_pd.photo_id,

        'category': 'bedroom'

    })



benchmark_prediction.to_csv('benchmark_prediction.csv', index=False)
# Change variables

TARGET_WIDTH = 256

TARGET_HEIGHT = 160
print( train_X_pd.width.value_counts() )

print( train_X_pd.height.value_counts() )
from keras.models import Sequential

from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Dropout, Flatten, Dense

from keras.callbacks import ModelCheckpoint, EarlyStopping

from sklearn.preprocessing import OneHotEncoder

from tensorflow import set_random_seed

from numpy.random import seed

import tensorflow as tf

import keras.backend as K
class AiryPhotoModel():

    BATCH_SIZE = 16

    EPOCH = 50

    

    def __init__(self, model: Sequential, 

                 image_size = (256, 160), 

                 model_file_name = 'temp_model.h5' ):

        self.is_trained = False

        self.model = model

        self.image_width = image_size[0]

        self.image_height = image_size[1]

        self.model_file_name = model_file_name

        

    def fit(self, file_paths: pd.Series, categories: pd.Series, val_file_paths: pd.Series, val_categories: pd.Series):

        # Label encoding

        self.label_encoder = OneHotEncoder(categories='auto')

        self.label_encoder.fit(categories.values.reshape(-1,1))

        

        # Data

        train_images = self._get_images(file_paths)

        train_label_one_hot = self._transform_labels(categories)

        

        val_images = self._get_images(val_file_paths)

        val_label_one_hot = self._transform_labels(val_categories)

        

        # Train model

        earlystoping_callback = self._get_earlystoping_cb()

        history = self.model.fit(

            x=train_images, 

            y=train_label_one_hot, 

            batch_size=self.BATCH_SIZE, 

            epochs=self.EPOCH, 

            callbacks=[earlystoping_callback],

            validation_data=(val_images, val_label_one_hot)

        )

        self.is_trained = True

        

        # Validation result

        pred_one_hot = self.model.predict(val_images)

        pred_label = self._inverse_transform_label(pred_one_hot)

        evaluation_report_airy_photo(prediction=pred_label, target=val_categories)

        

        return history

        

    def predict(self, file_paths: pd.Series):

        assert self.is_trained, 'Shoud train model before predict'

        images = self._get_images(file_paths)

        pred_one_hot = self.model.predict(images)

        pred_label = self._inverse_transform_label(pred_one_hot)

        return pred_label

    

    def _get_images(self, file_paths: pd.Series):

        images = file_paths.apply(lambda fp: read_image(fp))

        images = images.apply(lambda img: preprocess_image(img, target_height=self.image_height, target_width=self.image_width))

        images = np.stack(images, axis=0) 

        images = images / 255.0

        return images

    

    def _transform_labels(self, categories: pd.Series) -> np.array:

        categories_one_hot = self.label_encoder.transform(categories.values.reshape(-1,1))

        return categories_one_hot

    

    def _inverse_transform_label(self, categories_one_hot: np.array) -> pd.Series: 

        categories = self.label_encoder.inverse_transform(categories_one_hot)

        categories = pd.Series(categories.reshape(-1))

        return categories

    

    def _get_checkpoint_cb(self) -> ModelCheckpoint:

        return  ModelCheckpoint(

            self.model_file_name, 

            monitor='val_loss', 

            verbose=0, 

            save_best_only=True, 

            save_weights_only=False, 

            mode='auto', 

            period=1

        )

    

    def _get_earlystoping_cb(self) -> EarlyStopping:

        return EarlyStopping(

            monitor='val_loss', 

            min_delta=0, 

            patience=5, 

            verbose=0, 

            mode='auto',  

            restore_best_weights=True

        )
def plot_keras_train_history(history):

    history_df = pd.DataFrame(history.history)

    epochs = range(history_df.shape[0])



    fig, ax = plt.subplots(1,2,figsize=(18,4))





    ax[0].plot(epochs, history_df['loss'], label='Training loss')

    ax[0].plot(epochs, history_df['val_loss'], label='Validation loss')

    ax[0].set_title('Training and validation loss')

    ax[0].legend()



    ax[1].plot(epochs, history_df['acc'],label='Training accuracy')

    ax[1].plot(epochs, history_df['val_acc'], label='Validation accuracy')

    ax[1].set_title('Training and validation accuracy')

    ax[1].legend()



    plt.show()
from keras.applications.densenet import DenseNet121
# Ensure reproducibility

seed(23)

set_random_seed(23)



# Define model, no transfer learning (weights=None)

model = DenseNet121(include_top=True, weights=None, input_shape=(160, 256, 3), classes=10)

model.compile(optimizer="adam", loss='categorical_crossentropy', metrics=['accuracy'])



# Train model

airy_photo_model = AiryPhotoModel(model=model, image_size=(256, 160), model_file_name='airy_photo_densenet121.h5')

history = airy_photo_model.fit(

    file_paths = train_X_pd.file_path, 

    categories = train_y_pd,

    val_file_paths = val_X_pd.file_path, 

    val_categories = val_y_pd)



plot_keras_train_history(history)
pred_y = airy_photo_model.predict(test_pd.file_path)

pred_pd = pd.DataFrame(

    data = {

        'photo_id': test_pd.photo_id,

        'category': pred_y

    })



pred_pd.to_csv('airy_photo_densenet121.csv', index=False)