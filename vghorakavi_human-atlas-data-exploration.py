# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import csv

from PIL import Image

import numpy as np

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

#print(os.listdir("../input"))

#train and test are directories with images of type png

#print(os.listdir("../input/train/"))

#print(os.listdir("../input/test/"))

# Any results you write to the current directory are saved as output.
def read_csv(filename):

    with open(filename, 'r') as csv_reader:

        csv_reader = csv.reader(csv_reader)

        for line in csv_reader:

            print(line)
def view_image():

    #implement a counter to track images in directory

    figure = plt.figure(figsize=(20, 20))

    train_list = os.listdir("../input/train")

    for i, image in enumerate(np.random.choice(train_list, 5)):

        image_axis = figure.add_subplot(2, 20//2, i+1, xticks=[], yticks=[])

        image = Image.open("../input/train/" + image)

        plt.imshow(image)

        
def main():

    #read_csv("../input/train.csv")

    #read_csv("../input/sample_submission.csv")

    view_image()



main()