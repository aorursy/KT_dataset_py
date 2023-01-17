import numpy as np

import pandas as pd

import os

import cv2

import matplotlib.pyplot as plt

import skimage.feature

#%matplotlib inline



#from subprocess import check_output

#print(check_output(["ls", "../input"]).decode("utf8"))
import sys

arg = sys.argv[1:2]
print(arg)


if not os.path.exists("./labels"):

    os.makedirs("./labels")

if not os.path.exists("./JPEGImages"):

    os.makedirs("./JPEGImages")
if(arg[0] == '-b'):

    Path_Type = "Basic_{0}".format(arg[1])

elif(arg[0] == '-a'):

    Path_Type = "Advance_{0}".format(arg[1])

else:

    Path_Type = "Basic_32"
Path_Sealion = "./" # "/home/paperspace/Project/Sealion/{0}/".format(Path_Type) # 

Path_Train  = "../input/Train/" # "/home/paperspace/Project/Sealion/TrainSmall2/Train/"  #  

Path_Dotted = "../input/TrainDotted/" # "/home/paperspace/Project/Sealion/TrainSmall2/TrainDotted/"  # 

file_names = os.listdir(Path_Train)

file_names = sorted(file_names, key=lambda 

                    item: (int(item.partition('.')[0]) if item[0].isdigit() else float('inf'), item))



file_names = file_names[1:2]

print(file_names)

# select a subset of files to run on

#file_names = file_names[0:1]
"""

if not os.path.exists("/home/paperspace/Project/Sealion/{0}".format(Path_Type)):

    os.makedirs("/home/paperspace/Project/Sealion/{0}".format(Path_Type))

if not os.path.exists("/home/paperspace/Project/Sealion/{0}/labels".format(Path_Type)):

    os.makedirs("/home/paperspace/Project/Sealion/{0}/labels".format(Path_Type))

if not os.path.exists("/home/paperspace/Project/Sealion/{0}/JPEGImages".format(Path_Type)):

    os.makedirs("/home/paperspace/Project/Sealion/{0}/JPEGImages".format(Path_Type))

"""
Sub_Im_Size = (416,416)



image_tmp = cv2.imread(Path_Train + file_names[0])

image_tmp = image_tmp[:Sub_Im_Size[1],:Sub_Im_Size[0],:]

image_tmp = cv2.absdiff(image_tmp,image_tmp)



plt.imshow(cv2.cvtColor(image_tmp, cv2.COLOR_BGR2RGB))

cv2.imwrite('sub_im_template.jpg',image_tmp)
def get_blobs(filename):

    # read the Train and Train Dotted images

    image_1 = cv2.imread(Path_Dotted + filename)

    image_2 = cv2.imread(Path_Train + filename)

    

    # absolute difference between Train and Train Dotted

    image_3 = cv2.absdiff(image_1,image_2)

    

    # mask out blackened regions from Train Dotted

    mask_1 = cv2.cvtColor(image_1, cv2.COLOR_BGR2GRAY)

    mask_1[mask_1 < 20] = 0

    mask_1[mask_1 > 0] = 255

    

    mask_2 = cv2.cvtColor(image_2, cv2.COLOR_BGR2GRAY)

    mask_2[mask_2 < 20] = 0

    mask_2[mask_2 > 0] = 255

    

    image_3 = cv2.bitwise_or(image_3, image_3, mask=mask_1)

    image_3 = cv2.bitwise_or(image_3, image_3, mask=mask_2) 

    

    # convert to grayscale to be accepted by skimage.feature.blob_log

    image_3 = cv2.cvtColor(image_3, cv2.COLOR_BGR2GRAY)

    

    # detect blobs

    blobs = skimage.feature.blob_log(image_3, min_sigma=3, max_sigma=4, num_sigma=1, threshold=0.02)

    

    return blobs
def get_xy_range_basic(x, y, x_max, y_max, size):

    ### x_left, x_right, y_up, y_down

    x_left  = min(size, x)

    x_right = min(size, x_max-x-1)

    y_up    = min(size, y)

    y_down  = min(size, y_max-y-1)

    return (x_left, x_right, y_up, y_down)
# classes = ["adult_males", "subadult_males", "adult_females", "juveniles", "pups"]



def get_species(r,g,b):    

    if r > 200 and g < 50 and b < 50: # RED

        return 0        

    elif r > 200 and g > 200 and b < 50: # MAGENTA

        return 1         

    elif r < 100 and g < 100 and 150 < b < 200: # GREEN

        return 2

    elif r < 100 and  100 < g and b < 100: # BLUE

        return 3

    elif r < 150 and g < 50 and b < 100:  # BROWN

        return 4
def parse_image(filename):

    ## open sub_image_names file

    sub_image_names = open(Path_Sealion + "Train.txt", 'a')

    

    ### get original image

    ori_image = cv2.imread(Path_Dotted + filename)

    dot_image = cv2.imread(Path_Dotted + filename)

    cnt = 0

    

    ### get coordinate of all sea lions

    Dict_range = {}

    blobs = get_blobs(filename)

    

    for blob in blobs:

        # get the coordinates for each blob

        y, x, s = blob

        

        # decide get_xy_range method

        if(arg[0] == '-b' and arg[1]):

            xy_range = get_xy_range_basic(x=x, y=y, x_max=ori_image.shape[1], y_max=ori_image.shape[0], size=int(arg[1])/2)

        elif(arg[0] == '-a' and arg[1]):

            xy_range = get_xy_range_advance()

        else:

            xy_range = get_xy_range_basic(x=x, y=y, x_max=ori_image.shape[1], y_max=ori_image.shape[0], size=16)

            

        # save range info    

        Dict_range[(x,y)] = xy_range

    

    ### output sub_image and annotation file for each blob

    Delete_Key_List = []

    for key in list(Dict_range.keys()):       

        if(key in Dict_range):

            # add cnt for new sub_image name

            cnt += 1

            

            # get x, y, xy_range in original image

            main_x = int(key[0])

            main_y = int(key[1])

            xy_range = Dict_range[key]

            

            ### get basic sub_image

            sub_image = cv2.imread('sub_im_template.jpg')            

            sub_x_center = int(sub_image.shape[1]/2)

            sub_y_center = int(sub_image.shape[0]/2)

            sub_image[int(sub_y_center-xy_range[2]):int(sub_y_center+xy_range[3]), int(sub_x_center-xy_range[0]):int(sub_x_center+xy_range[1]), :] = ori_image[int(main_y-xy_range[2]):int(main_y+xy_range[3]), int(main_x-xy_range[0]):int(main_x+xy_range[1]), :]

            del Dict_range[key]

                    

            ### get species

            g,b,r = dot_image[int(main_y)][int(main_x)][:]

            species = get_species(r,g,b)

            

            # get pos info for annotation file

            x_pos = float(sub_x_center)/float(Sub_Im_Size[0])

            y_pos = float(sub_y_center)/float(Sub_Im_Size[0])

            x_len = float(xy_range[0]+xy_range[1])/float(Sub_Im_Size[0])

            y_len = float(xy_range[2]+xy_range[3])/float(Sub_Im_Size[0])

            element = [x_pos, y_pos, x_len, y_len]

            

            # save species info in annotation file

            ant_file = open(Path_Sealion + 'labels/{0}_{1}.txt'.format(filename[:-4], cnt), 'w')

            ant_file.write(str(species) + " " + " ".join([str(x) for x in element]) + '\n')

            

            

            ### include other sea lion

            # max min coordinate for including image based on origin image

            x_min = max(main_x - sub_image.shape[1]/2 + 1, 0)

            x_max = min(main_x + sub_image.shape[1]/2 - 1, ori_image.shape[1])

            y_min = max(main_y - sub_image.shape[0]/2 + 1, 0)

            y_max = min(main_y + sub_image.shape[0]/2 - 1, ori_image.shape[0])

            

            for ex_key in list(Dict_range.keys()):

                if(ex_key[0] > x_min and ex_key[0] < x_max and ex_key[1] > y_min and ex_key[1] < y_max):

                    ### coordinate of ex_sea_lion in origin image

                    ex_range = Dict_range[ex_key]

                    ex_left  = int(ex_key[0] - ex_range[0])

                    ex_right = int(ex_key[0] + ex_range[1])

                    ex_up    = int(ex_key[1] - ex_range[2])

                    ex_down  = int(ex_key[1] + ex_range[3])

                    if(ex_left > x_min and ex_right < x_max and ex_up > y_min and ex_down < y_max):

                        ### sub_image's coordinate where ex_sea_lion put  

                        in_up    = int(sub_y_center - main_y + ex_key[1] - ex_range[2])

                        in_down  = int(sub_y_center - main_y + ex_key[1] + ex_range[3])

                        in_left  = int(sub_x_center - main_x + ex_key[0] - ex_range[0])

                        in_right = int(sub_x_center - main_x + ex_key[0] + ex_range[1])

                        sub_image[ in_up:in_down, in_left:in_right, :] = ori_image[ex_up:ex_down, ex_left:ex_right, :]

                        del Dict_range[ex_key]

                        

                        ### get species for include sea_lion

                        g,b,r = dot_image[int(ex_key[1])][int(ex_key[0])][:]

                        species = get_species(r,g,b)

            

                        # get pos info for annotation file

                        x_pos = float(sub_x_center - main_x + ex_key[0])/float(Sub_Im_Size[0])

                        y_pos = float(sub_y_center - main_y + ex_key[1])/float(Sub_Im_Size[0])

                        x_len = float(ex_range[0]+ex_range[1])/float(Sub_Im_Size[0])

                        y_len = float(ex_range[2]+ex_range[3])/float(Sub_Im_Size[0])

                        element = [x_pos, y_pos, x_len, y_len]

            

                        # save species info in annotation file

                        ant_file.write(str(species) + " " + " ".join([str(x) for x in element]) + '\n')                          

            

            cv2.imwrite(Path_Sealion + 'JPEGImages/{0}_{1}.jpg'.format(filename[:-4], cnt),sub_image)

            #cv2.imwrite('{0}_{1}.png'.format(filename[:-4], cnt),sub_image)

            sub_image_names.write(Path_Sealion + 'JPEGImages/{0}_{1}.jpg'.format(filename[:-4], cnt))

            sub_image_names.write("\n")

            ant_file.close()

    sub_image_names.close()
### remove Train.txt if exist

if os.path.exists(Path_Sealion + "Train.txt"):

    os.remove(Path_Sealion + "Train.txt")



for filename in file_names:

    if filename[-3:] == 'jpg':

        parse_image(filename)
os.remove('sub_im_template.jpg')