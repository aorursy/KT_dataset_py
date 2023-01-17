SYSTEM = "Kaggle" # "Paperspace" # 
import numpy as np

import pandas as pd

import os

import cv2

import matplotlib.pyplot as plt
if SYSTEM == "Kaggle":

    Image_Path = "../input/Train/"

    Crop_Path  = "./TestLargeCrop/"

else:

    Image_Path = "/home/paperspace/Project/Sealion/TestLargeRaw/"

    Crop_Path  = "/home/paperspace/Project/Sealion/TestLargeCrop/"
Crop_Size = (416,416)
file_names = os.listdir(Image_Path)

file_names = sorted(file_names, key=lambda 

                    item: (int(item.partition('.')[0]) if item[0].isdigit() else float('inf'), item))

file_names = file_names[:1]
print(file_names)
def ceil_devide(Big, Small):

    result = int(Big/Small)

    if(Big%Small != 0):

        result += 1

    return result
def create_crop_template(filename):

    ### remove existing template

    if os.path.exists('crop_template.jpg'):

        os.remove('crop_template.jpg')

        

    image = cv2.imread(Image_Path + filename)

    image = image[:Crop_Size[1],:Crop_Size[0],:]

    image = cv2.absdiff(image,image)



    cv2.imwrite('crop_template.jpg',image)
def delete_crop_template():

    os.remove('crop_template.jpg')
def create_crop_file():

    if(SYSTEM == "Kaggle"):

        if not os.path.exists("./TestLargeCrop/"):

            os.makedirs("./TestLargeCrop/")

        if not os.path.exists("./TestLargeCrop/JPEGImages/"):

            os.makedirs("./TestLargeCrop/JPEGImages/")        

    else:

        if not os.path.exists("/home/paperspace/Project/Sealion/TestLargeCrop/"):

            os.makedirs("/home/paperspace/Project/Sealion/TestLargeCrop/")

        if not os.path.exists("/home/paperspace/Project/Sealion/TestLargeCrop/JPEGImages/"):

            os.makedirs("/home/paperspace/Project/Sealion/TestLargeCrop/JPEGImages/")
def delete_crop_image_names():

    if os.path.exists(Crop_Path + "TestLargeCrop.txt"):

        os.remove(Crop_Path + "TestLargeCrop.txt")
create_crop_template(file_names[0])

create_crop_file()



delete_crop_image_names()

crop_image_names = open(Crop_Path + "TestLargeCrop.txt", 'w')



for filename in file_names:

    ### skip if file is not image

    if(filename[-3:] != 'jpg'):

        continue

        

    ### read origin image

    print("Cropping {0}".format(filename))

    ori_image = cv2.imread(Image_Path + filename)

    Shape = ori_image.shape

    X_Len = Shape[1]

    Y_Len = Shape[0]

    

    X_Amt = ceil_devide(X_Len, Crop_Size[0])

    Y_Amt = ceil_devide(Y_Len, Crop_Size[1])

    

    cnt = 0

    for j in range(Y_Amt):

        for i in range(X_Amt):

            # counting

            cnt += 1

            

            # create crop image

            crop_image = cv2.imread('crop_template.jpg')

            tmp_image  = ori_image[j*Crop_Size[1]:(j+1)*Crop_Size[1], i*Crop_Size[0]:(i+1)*Crop_Size[0], :]

            crop_image[:tmp_image.shape[0], :tmp_image.shape[1], :] = tmp_image

            

            # save crop image

            Name = Crop_Path + "JPEGImages/" + filename.split('.')[0] + '_' + str(cnt) + '.jpg'

            cv2.imwrite(Name, crop_image) 

            

            # write crop images in txt

            crop_image_names.write(Name)

            crop_image_names.write("\n")



crop_image_names.close()

delete_crop_template()
create_crop_file()


