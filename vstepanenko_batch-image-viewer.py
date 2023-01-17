import pandas as pd
import pathlib
from PIL import Image
import IPython.display as display
import tensorflow as tf
import matplotlib.pyplot as plt
data_train_dir = pathlib.Path('../input/landmark-retrieval-2020/train')
train_df = pd.read_csv('../input/landmark-retrieval-2020/train.csv')

# Add extra column with respective path (upto the train data folder)
train_df['id_path']=train_df['id'].map(lambda x: '/'.join(list(x[:3])) + f'/{x}.jpg')
# show the slice of lables
train_df['landmark_id'].unique()[2060:2070]
# Select images with the label 5139 (landmark_id)
img_group = train_df[train_df['landmark_id'] == 5139]['id'].values
img_group[:5]
def img_path(img_id, data_dir):
    """Returns the path for a img_id."""
    return data_dir/pathlib.Path('/'.join(list(img_id[:3])) + f'/{img_id}.jpg')
# Given the path, display up to 5 images (or change the slicing for more) in original size
for image_path in [img_path(i, data_train_dir) for i in img_group][:5]:
    display.display(Image.open(str(image_path)))
# top 10 landmark_id
train_df[['landmark_id', 'id']].groupby('landmark_id').count().sort_values(by='id', ascending=False).head(10)
#Resizing/rescaling images
image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
BATCH_SIZE = 25
IMG_HEIGHT = 224
IMG_WIDTH = 224
# sample n pictures
n = 1000

train_data_gen = image_generator.flow_from_dataframe(
    directory=data_train_dir,
    dataframe=train_df.sample(n=n),
    class_mode='raw',
    x_col='id_path', y_col='landmark_id',
    batch_size=BATCH_SIZE,
    shuffle=True,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
)
def show_batch(image_batch, label_batch):
  plt.figure(figsize=(20,20))
  for n in range(len(image_batch)):
      ax = plt.subplot(5,5,n+1)
      plt.imshow(image_batch[n])
      plt.title(label_batch[n])
      plt.axis('off')
# Now you can re-run this cell multiple times to get next batch of images
# Since we have n=1000 and BATCH_SIZE = 25, you can run the cell 40 times
# After that, it will start over again.

image_batch, label_batch = next(train_data_gen)
show_batch(image_batch, label_batch)
# sample pictures with given landmark_id
landmark_id = 4239

train_data_gen_label = image_generator.flow_from_dataframe(
    directory=data_train_dir,
    dataframe=train_df[train_df['landmark_id'] == landmark_id],
    class_mode='raw',
    x_col='id_path', y_col='landmark_id',
    batch_size=BATCH_SIZE,
    shuffle=True,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
)
# You can re-run this cell multiple times to get next batch of images
image_batch, label_batch = next(train_data_gen_label)
show_batch(image_batch, label_batch)