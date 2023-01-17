import os
import cv2 as cv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image
DATA_FOLDER = '../input/deepfake-detection-challenge'
TRAIN_SAMPLE_FOLDER = 'train_sample_videos'
TEST_FOLDER = 'test_videos'

print(f"Train samples: {len(os.listdir(os.path.join(DATA_FOLDER, TRAIN_SAMPLE_FOLDER)))}")
print(f"Test samples: {len(os.listdir(os.path.join(DATA_FOLDER, TEST_FOLDER)))}")
train_list = list(os.listdir(os.path.join(DATA_FOLDER, TRAIN_SAMPLE_FOLDER)))
json_file = [file for file in train_list if  file.endswith('json')][0]
print(f"JSON file: {json_file}")
def get_meta_from_json(path):
    df = pd.read_json(os.path.join(DATA_FOLDER, path, json_file))
    df = df.T
    return df

meta_train_df = get_meta_from_json(TRAIN_SAMPLE_FOLDER)
meta_train_df.head()
def read_image_from_video(video_file):
    capture_image = cv.VideoCapture(os.path.join(DATA_FOLDER, TRAIN_SAMPLE_FOLDER, video_file)) 
    ret, frame = capture_image.read()
    return ret, frame

def display_image_from_video(video_file):
    ret, frame = read_image_from_video(video_file)
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111)
    frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    ax.set_title(video_file)
    ax.imshow(frame)
fake_train_video = ["aelzhcnwgf.mp4", "adylbeequz.mp4"]
for video_file in fake_train_video:
    display_image_from_video(video_file)
def save_image(image, filename, label):
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)
    image.save("./images/%s/%s.jpg" % (label, filename))
    print("Saved as %s.jpg" % filename)

if not os.path.exists("./images"):
    os.mkdir("./images")
    
if not os.path.exists("./images/real"):
    os.mkdir("./images/real")
    
if not os.path.exists("./images/fake"):
    os.mkdir("./images/fake")
    
for video_file, label in zip(meta_train_df.index, meta_train_df.label):
    ret, frame = read_image_from_video(video_file)
    if ret:
        save_image(frame, os.path.splitext(video_file)[0], label.lower())
    else:
        print("Failed to read image from video %s" % video_file)
meta_train_df['image'] = meta_train_df.index.str.replace("mp4", "jpg")
meta_train_df.head()
IMAGE_DATA_FOLDER = "./images"

def read_image(image_path):
    return np.asarray(Image.open(image_path))

def display_image(image):
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111)
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    ax.imshow(image)

image = read_image(os.path.join(IMAGE_DATA_FOLDER, "fake", "aelzhcnwgf.jpg"))
display_image(image)
def downscale_image(image):
    image_size = [image.shape[1], image.shape[0]]
    return np.asarray(Image.fromarray(image)
        .resize([image_size[0] // 4, image_size[1] // 4], Image.BOX))

lr_image = downscale_image(image)
display_image(lr_image)
lr_image = tf.expand_dims(lr_image, 0)
lr_image = tf.cast(lr_image, tf.float32)
hr_image = model(lr_image)
hr_image = tf.cast(hr_image, tf.uint8)
display_image(tf.squeeze(hr_image).numpy())
IMAGE_DATA_FOLDER = "./images"

def read_image(image_path):
    return np.asarray(Image.open(image_path))

def display_image(image):
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111)
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    ax.imshow(image)

image = read_image(os.path.join(IMAGE_DATA_FOLDER, "fake", "adylbeequz.jpg"))
display_image(image)
def downscale_image(image):
    image_size = [image.shape[1], image.shape[0]]
    return np.asarray(Image.fromarray(image)
        .resize([image_size[0] // 4, image_size[1] // 4], Image.BOX))

lr_image = downscale_image(image)
display_image(lr_image)
lr_image = tf.expand_dims(lr_image, 0)
lr_image = tf.cast(lr_image, tf.float32)
hr_image = model(lr_image)
hr_image = tf.cast(hr_image, tf.uint8)
display_image(tf.squeeze(hr_image).numpy())