import pandas as pd
import numpy as np
with open('../input/flickr-8k/Flickr_8k.trainImages.txt','r') as tr_imgs:
    train_imgs = tr_imgs.read().splitlines()
    
with open('../input/flickr-8k/Flickr_8k.devImages.txt','r') as dv_imgs:
    dev_imgs = dv_imgs.read().splitlines()
    
with open('../input/flickr-8k/Flickr_8k.testImages.txt','r') as ts_imgs:
    test_imgs = ts_imgs.read().splitlines()
    
with open('../input/flickr-8k/Flickr8k.token.txt','r') as img_tkns:
    captions = img_tkns.read().splitlines()
train_imgs = train_imgs + dev_imgs
from collections import defaultdict

caption_map = defaultdict(list)

for record in captions:
    record = record.split('\t')
    img_name = record[0][:-2]
    img_caption = record[1].strip()
    caption_map[img_name].append(img_caption)
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input as preprocess_vgg16_input
def process_image2arr(path, img_dims=(224, 224)):
    img = image.load_img(path, target_size=img_dims)
    img_arr = image.img_to_array(img)
    img_arr = np.expand_dims(img_arr, axis=0)
    img_arr = preprocess_vgg16_input(img_arr)
    return img_arr
from keras.applications import vgg16
from keras.models import Model


vgg_model = vgg16.VGG16(include_top=True, weights='imagenet', 
                        input_shape=(224, 224, 3))
vgg_model.layers.pop()
output = vgg_model.layers[-1].output
vgg_model = Model(vgg_model.input, output)
vgg_model.trainable = False
vgg_model.summary()