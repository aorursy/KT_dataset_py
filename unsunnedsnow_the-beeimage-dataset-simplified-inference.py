import numpy as np

import pandas as pd

import random, datetime, os, shutil, math

import matplotlib.pyplot as plt

from keras.preprocessing import image

from keras.preprocessing.image import ImageDataGenerator

from keras import models

import os



image_size = (150, 150)



model_filename = "../input/modelcache/hbhc-simple-val_loss-20190206-6-acc100.h5"

csv_filename = "../input/modelcache/hbhc-simple-val_loss-20190206-6-acc100.csv"

#images in:

input_dir = '../input/honeybees-simplified/bees-simple/bees/'

train_dir = input_dir + "train"

validate_dir = input_dir + "validate"

test_dir = input_dir + "test"



log_filename = "hbhc_infer_log.txt"

log_file = open(log_filename, "a")



# timestamp and then write the msg to both the console and the log file:

def logprint(msg):

  msg_str = "["+str(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))+"] "+str(msg)

  print(msg_str)

  log_file = open(log_filename, "a")

  log_file.write(msg_str+"\n")

  log_file.close()



logprint("Reopened log file "+log_filename)



#display a sample of bee photos in an auto-sized grid:

def show_bees(bzz):

  numbees = len(bzz)

  if numbees == 0:

    return None

  rows = int(math.sqrt(numbees))

  cols = (numbees+1)//rows

  f, axs = plt.subplots(rows, cols)

  fig = 0

  for b in bzz:

    img = image.load_img(b)

    row = fig // cols

    col = fig % cols

    axs[row, col].imshow(img)

    fig += 1

  plt.show()

  

#show some sample images:

dir_name = os.path.join(test_dir,"category0")

all_images = [os.path.join(dir_name, fname) for fname in os.listdir(dir_name)]

show_bees(all_images[:6])
# This is the graph of the training run which generated this model:'

df_hist = pd.read_csv(csv_filename)

ax = df_hist.plot(x='epoch',y=['acc','val_acc'])

ax = df_hist.plot(x='epoch',y=['loss','val_loss'])
# evaluate each of the train, validate and test subsets:

model = models.load_model(model_filename)



logprint("evaluating the test subset")

test_datagen = ImageDataGenerator(rescale=1./255)

test_flow = test_datagen.flow_from_directory(

        test_dir,

        target_size=image_size,

        batch_size=20,

        class_mode='categorical')

test_loss, test_acc = model.evaluate_generator(test_flow, steps=40)

logprint('test acc:'+str(test_acc))



logprint("evaluating the validation subset")

validate_datagen = ImageDataGenerator(rescale=1./255)

validate_flow = validate_datagen.flow_from_directory(

        validate_dir,

        target_size=image_size,

        batch_size=20,

        class_mode='categorical')

validate_loss, validate_acc = model.evaluate_generator(validate_flow, steps=40)

logprint('validate acc:'+str(validate_acc))
