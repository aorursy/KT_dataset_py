import os



import numpy as np

import pandas as pd 

import imageio



import matplotlib as mpl

import matplotlib.pyplot as plt

import matplotlib.image as mpimg

from PIL import Image, ImageOps

import scipy.ndimage as ndi
#here we need to get to the data

dirname = '/kaggle/input/herbarium-2020-fgvc7/nybg2020/train/images'

dir_001 = os.path.join(dirname, '001')

dir_001_19 = os.path.join(dir_001, '19')
os.listdir(dir_001)
os.listdir(dir_001_19)
def plot_imgs_one_dir(item_dir, num_imgs=25):

    all_item_dirs = os.listdir(item_dir)

    item_files = [os.path.join(item_dir, file) for file in all_item_dirs][:num_imgs]



    plt.figure(figsize=(10, 10))

    for idx, img_path in enumerate(item_files):

        plt.subplot(5, 5, idx+1)



        img = plt.imread(img_path)

        plt.imshow(img)



    plt.tight_layout()
plot_imgs_one_dir(dir_001_19)
def plot_img_hist_3channel(item_dir, num_img=6):

    all_item_dirs = os.listdir(item_dir)

    item_files = [os.path.join(item_dir, file) for file in all_item_dirs][:num_img]



    #plt.figure(figsize=(10, 10))

    for idx, img_path in enumerate(item_files):

        fig1 = plt.figure(idx,figsize=(10, 10))

        fig1.add_subplot(2, 2, 1)

        img = mpimg.imread(img_path, )

        plt.imshow(img)

        fig1.add_subplot(2, 2, 2)

        plt.hist(img.ravel(), bins = 256, color = 'orange')

        plt.hist(img[:, :, 0].ravel(), bins = 256, color = 'red')

        plt.hist(img[:, :, 1].ravel(), bins = 256, color = 'green')

        plt.hist(img[:, :, 2].ravel(), bins = 256, color = 'blue')

        plt.xlabel('Intensity')

        plt.ylabel('Count')

        plt.legend(['Total', 'Red Channel', 'Green Channel', 'Blue Channel'])

        plt.show()

    

    plt.tight_layout()
plot_img_hist_3channel(dir_001_19)
os.mkdir('/kaggle/working/work_imgs')
dir_output = os.path.join('/kaggle/working/','work_imgs');
def  hist_equal(path_from, path_to, stop=4):

    i=1

    files = os.listdir(path_from)

    

    for file in files[:stop]: 

        try:

            file_dir = os.path.join(path_from, file)

            file_dir_save = os.path.join(path_to, file)

            img = Image.open(file_dir)

            img = ImageOps.equalize(img)

            #img = img.convert("RGB") #konwersja z RGBA do RGB, usuniecie kanału alfa zeby zapisać do jpg

            img.save(file_dir_save) 

            i=i+1

        except:

            continue
hist_equal(dir_001_19, dir_output)
plot_img_hist_3channel(dir_output, 4)
def plot_cdf_comparison(item_dir_before,item_dir_after, num_img=1):

    all_item_dirs = os.listdir(item_dir_before)

    item_files_before = [os.path.join(item_dir_before, file) for file in all_item_dirs][:num_img]

    item_files_after = [os.path.join(item_dir_after, file) for file in all_item_dirs][:num_img]

  

  #plt.figure(figsize=(10, 10))

    for idx, img_path in enumerate(item_files_before):

        im_b = imageio.imread(img_path)

        hist_b = ndi.histogram(im_b, min=0, max=255, bins=256)

        cdf_b = hist_b.cumsum() / hist_b.sum()

        

        img_path_a = item_files_after[idx]

        im_a = imageio.imread(img_path_a)

        hist_a = ndi.histogram(im_a, min=0, max=255, bins=256)

        cdf_a = hist_a.cumsum() / hist_a.sum()



        fig1 = plt.figure(idx,figsize=(10, 10))

        fig1.add_subplot(2, 4, 1)

        img_b = mpimg.imread(img_path, )

        plt.title("Before. {}".format(idx))

        plt.imshow(img_b, cmap='gray')

        fig1.add_subplot(2, 4, 4)

        plt.title("CDF comparison")

        plt.plot(cdf_b)

        

        fig2 = plt.figure(idx,figsize=(10, 10))

        fig2.add_subplot(2, 4, 2)

        img_a = mpimg.imread(img_path_a, )

        plt.title("Before. {}".format(idx))

        plt.imshow(img_a)

        fig1.add_subplot(2, 4, 4)

        plt.plot(cdf_a)

        plt.legend(['CDF before', 'CDF after'])



    plt.tight_layout()
plot_cdf_comparison(dir_001_19, dir_output, 4)