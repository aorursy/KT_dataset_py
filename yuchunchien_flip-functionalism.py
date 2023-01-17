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

arg = sys.argv[1:]



if('-f' in arg):

    arg = ['Basic', '32', 'Whole', 'Train']

else:

    new_arg = []

    if('Basic' in arg):

        idx = arg.index('Basic')

        new_arg.append('Basic')

        new_arg.append(arg[idx+1])

    elif('Advance' in arg):

        idx = arg.index('Advance')

        new_arg.append('Advance')

        new_arg.append(arg[idx+1])

    else:

        print("Please enter \"Basic\" or \"Advance\".")

    

    if('Whole' in arg):

        new_arg.append('Whole')

    elif('Black' in arg):

        new_arg.append('Black')

    else:

        print("Please enter \"Whole\" or \"Black\".")

    

    if('Test' in arg):

        new_arg.append('Test')

    elif('Train' in arg):

        new_arg.append('Train')

    else:

        print("Please enter \"Test\" or \"Train\".")

        

    arg = new_arg
Path_Type = "{0}/{1}_{2}_{3}".format(arg[3], arg[0], arg[1], arg[2])

Path_Type
Path_Sealion = "./" # "/home/paperspace/Project/Sealion/{0}/".format(Path_Type) # 

Path_Train   = "../input/Train/" # "/home/paperspace/Project/Sealion/TrainSmall2/Train/"  # 

Path_Dotted  = "../input/TrainDotted/" # "/home/paperspace/Project/Sealion/TrainSmall2/TrainDotted/"  # 

file_names = os.listdir(Path_Train)

file_names = sorted(file_names, key=lambda 

                    item: (int(item.partition('.')[0]) if item[0].isdigit() else float('inf'), item))



file_names = file_names[1:2]

print(file_names)
"""

if not os.path.exists("/home/paperspace/Project/Sealion/{0}/Augment".format(Path_Type)):

    os.makedirs("/home/paperspace/Project/Sealion/{0}/Augment".format(Path_Type))

if not os.path.exists("/home/paperspace/Project/Sealion/{0}/Augment/Flip".format(Path_Type)):

    os.makedirs("/home/paperspace/Project/Sealion/{0}/Augment/Flip".format(Path_Type))

if not os.path.exists("/home/paperspace/Project/Sealion/{0}/Augment/Flip/JPEGImages".format(Path_Type)):

    os.makedirs("/home/paperspace/Project/Sealion/{0}/Augment/Flip/JPEGImages".format(Path_Type))

if not os.path.exists("/home/paperspace/Project/Sealion/{0}/Augment/Flip/DotImages".format(Path_Type)):

    os.makedirs("/home/paperspace/Project/Sealion/{0}/Augment/Flip/DotImages".format(Path_Type))

if not os.path.exists("/home/paperspace/Project/Sealion/{0}/Augment/Flip/labels".format(Path_Type)):

    os.makedirs("/home/paperspace/Project/Sealion/{0}/Augment/Flip/labels".format(Path_Type))

"""


if not os.path.exists("./Augment"):

    os.makedirs("./Augment")     

if not os.path.exists("./Augment/Flip"):

    os.makedirs("./Augment/Flip") 

if not os.path.exists("./Augment/Flip/JPEGImages"):

    os.makedirs("./Augment/Flip/JPEGImages")  

if not os.path.exists("./Augment/Flip/DotImages"):

    os.makedirs("./Augment/Flip/DotImages")  

if not os.path.exists("./Augment/Flip/labels"):

    os.makedirs("./Augment/Flip/labels")

Sub_Im_Size = (416,416)



image_tmp = cv2.imread(Path_Train + file_names[0])

image_tmp = image_tmp[:Sub_Im_Size[1],:Sub_Im_Size[0],:]

image_tmp = cv2.absdiff(image_tmp,image_tmp)



plt.imshow(cv2.cvtColor(image_tmp, cv2.COLOR_BGR2RGB))

cv2.imwrite('sub_im_template.jpg',image_tmp)
AUG_METHOD = "flip"



def augment(image, method):

    if(method == "flip"):

        image = image[::-1]

    

    return image
def get_blobs(filename):

    # read the Train and Train Dotted images

    image_1 = cv2.imread(Path_Dotted + filename)

    image_2 = cv2.imread(Path_Train + filename)

    

    # flip image

    image_1 = augment(image_1, AUG_METHOD)

    image_2 = augment(image_2, AUG_METHOD)

    

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
# classes = ["adult_males", "subadult_males", "pups", "juveniles", "adult_females"]



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

    else:

        return -1
def get_xy_range_basic(x, y, x_max, y_max, size):

    ### x_left, x_right, y_up, y_down

    x_left  = min(size, x)

    x_right = min(size, x_max-x-1)

    y_up    = min(size, y)

    y_down  = min(size, y_max-y-1)

    return (x_left, x_right, y_up, y_down)
def get_dict_range_basic(filename, blobs):

    Dict_range = {}

    ori_image = cv2.imread(Path_Train + filename)

    

    ### flip image

    ori_image = augment(ori_image, AUG_METHOD)

    

    for blob in blobs:

        # get the coordinates for each blob

        y, x, s = blob

        

        # get basic xy_range

        xy_range = get_xy_range_basic(x=x, y=y, x_max=ori_image.shape[1], y_max=ori_image.shape[0], size=int(arg[1])/2)

        

        # save range info    

        Dict_range[(x,y)] = xy_range

        

    return Dict_range
MAX_SUB_IM_HALF_LEN = 48



def get_dict_range_advance(filename, blobs):

    Dict_range = {}

    ori_image = cv2.imread(Path_Train + filename)

    dot_image = cv2.imread(Path_Dotted + filename)

    s_size = int(arg[1])/8

    

    ### flip image

    ori_image = augment(ori_image, AUG_METHOD)

    dot_image = augment(dot_image, AUG_METHOD)

    

    for blob in blobs:

        # get the coordinates for each blob

        y, x, s = blob

        

        # get species

        g,b,r = dot_image[int(y)][int(x)][:]

        species = get_species(r,g,b)

         

        # get basic xy_range

        xy_range = get_xy_range_basic(x=x, y=y, x_max=ori_image.shape[1], y_max=ori_image.shape[0], size=int(arg[1])/2)

        x_left  = xy_range[0] 

        x_right = xy_range[1]

        y_up    = xy_range[2]

        y_down  = xy_range[3]

        

        # set augmenting threshold for sea_lion

        level = 45

        

        # augment xy_range

        if(species != 2):

            mid_im = ori_image[int(y-s_size):int(y+s_size), int(x-s_size):int(x+s_size), :]

            mid_abs = np.linalg.norm(np.mean(mid_im, axis=(0,1)))

            

            # augment left side

            while(x_left != x and x_left != MAX_SUB_IM_HALF_LEN):

                # get three sub_sub_image on left

                left_mid_im  = ori_image[  int(y-s_size):int(y+s_size), int(x-x_left):int(x-x_left+2*s_size), :]

                left_up_im   = ori_image[int(y-3*s_size):int(y-s_size), int(x-x_left):int(x-x_left+2*s_size), :]

                left_down_im = ori_image[int(y+s_size):int(y+3*s_size), int(x-x_left):int(x-x_left+2*s_size), :]

                

                # calculate distance

                left_mid_abs  = np.linalg.norm(np.mean(  left_mid_im, axis=(0,1)))

                left_up_abs   = np.linalg.norm(np.mean(   left_up_im, axis=(0,1)))

                left_down_abs = np.linalg.norm(np.mean( left_down_im, axis=(0,1)))

                

                # augment if proper

                if(abs(mid_abs-left_mid_abs) < level or abs(mid_abs-left_up_abs) < level or abs(mid_abs-left_down_abs) < level):  

                    x_left = min(x_left+s_size, MAX_SUB_IM_HALF_LEN, x)

                    #print("Augment x_left {0},{1}".format(x,y))

                else:

                    break



            # augment right side

            while(x_right != ori_image.shape[1]-x and x_right != MAX_SUB_IM_HALF_LEN):

                # get three sub_sub_image on right

                right_mid_im  = ori_image[  int(y-s_size):int(y+s_size), int(x+x_right-2*s_size):int(x+x_right), :]

                right_up_im   = ori_image[int(y-3*s_size):int(y-s_size), int(x+x_right-2*s_size):int(x+x_right), :]

                right_down_im = ori_image[int(y+s_size):int(y+3*s_size), int(x+x_right-2*s_size):int(x+x_right), :]

                

                # calculate distance

                right_mid_abs  = np.linalg.norm(np.mean(  right_mid_im, axis=(0,1)))

                right_up_abs   = np.linalg.norm(np.mean(   right_up_im, axis=(0,1)))

                right_down_abs = np.linalg.norm(np.mean( right_down_im, axis=(0,1)))

                

                # augment if proper

                if(abs(mid_abs-right_mid_abs) < level or abs(mid_abs-right_up_abs) < level or abs(mid_abs-right_down_abs) < level):

                    x_right = min(x_right+s_size, MAX_SUB_IM_HALF_LEN, ori_image.shape[1]-x)

                    #print("Augment x_right {0},{1}".format(x,y))

                else:

                    break

            

            # augment up side

            while(y_up != y and y_up != MAX_SUB_IM_HALF_LEN):

                # get three sub_sub_image on up

                up_mid_im   = ori_image[int(y-y_up):int(y-y_up+2*s_size),   int(x-s_size):int(x+s_size), :]

                up_left_im  = ori_image[int(y-y_up):int(y-y_up+2*s_size), int(x-3*s_size):int(x-s_size), :]

                up_right_im = ori_image[int(y-y_up):int(y-y_up+2*s_size), int(x+s_size):int(x+3*s_size), :]

                

                # calculate distance

                up_mid_abs   = np.linalg.norm(np.mean(   up_mid_im, axis=(0,1)))

                up_left_abs  = np.linalg.norm(np.mean(  up_left_im, axis=(0,1)))

                up_right_abs = np.linalg.norm(np.mean( up_right_im, axis=(0,1)))

            

                # augment if proper

                if(abs(mid_abs-up_mid_abs) < level or abs(mid_abs-up_left_abs) < level or abs(mid_abs-up_right_abs) < level):

                    y_up = min(y_up+s_size, MAX_SUB_IM_HALF_LEN, y)

                    #print("Augment y_up {0},{1}".format(x,y))

                else:

                    break

            

            # augment down side

            while(y_down != ori_image.shape[0]-y and y_down != MAX_SUB_IM_HALF_LEN):

                # get three sub_sub_image on down

                down_mid_im   = ori_image[int(y+y_down-2*s_size):int(y+y_down),   int(x-s_size):int(x+s_size), :]

                down_left_im  = ori_image[int(y+y_down-2*s_size):int(y+y_down), int(x-3*s_size):int(x-s_size), :]

                down_right_im = ori_image[int(y+y_down-2*s_size):int(y+y_down), int(x+s_size):int(x+3*s_size), :]

                

                # calculate distance

                down_mid_abs   = np.linalg.norm(np.mean(   down_mid_im, axis=(0,1)))

                down_left_abs  = np.linalg.norm(np.mean(  down_left_im, axis=(0,1)))

                down_right_abs = np.linalg.norm(np.mean( down_right_im, axis=(0,1)))

                

                # augment if proper

                if(abs(mid_abs-down_mid_abs) < level or abs(mid_abs-down_left_abs) < level or abs(mid_abs-down_right_abs) < level):

                    y_down = min(y_down+s_size, MAX_SUB_IM_HALF_LEN, ori_image.shape[0]-y)

                    #print("Augment y_down {0},{1}".format(x,y))

                else:

                    break

                    

        # save range info    

        Dict_range[(x,y)] = (x_left, x_right, y_up, y_down)

    

    return Dict_range
def parse_image_black(filename):

    ## open sub_image_names file

    sub_image_names = open(Path_Sealion + "Augment/Flip/Train.txt", 'a')

    

    ### get original image

    ori_image = cv2.imread(Path_Train + filename)

    dot_image = cv2.imread(Path_Dotted + filename)

    cnt = 0

    

    ### flip origin image

    ori_image = augment(ori_image, AUG_METHOD)

    dot_image = augment(dot_image, AUG_METHOD)

    

    ### get coordinate of all sea lions

    Dict_range = {}

    blobs = get_blobs(filename)

    

    ### get xy_range info for all sea_lion

    if(arg[0] == 'Basic'):

        Dict_range = get_dict_range_basic(filename, blobs)

    elif(arg[0] == 'Advance'):

        Dict_range = get_dict_range_advance(filename, blobs)

    

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

            if(species == -1):

                continue

            

            # get pos info for annotation file

            x_pos = float(sub_x_center)/float(Sub_Im_Size[0])

            y_pos = float(sub_y_center)/float(Sub_Im_Size[0])

            x_len = float(xy_range[0]+xy_range[1])/float(Sub_Im_Size[0])

            y_len = float(xy_range[2]+xy_range[3])/float(Sub_Im_Size[0])

            element = [x_pos, y_pos, x_len, y_len]

            

            # save species info in annotation file

            ant_file = open(Path_Sealion + 'Augment/Flip/labels/{0}_{1}.txt'.format(filename[:-4], cnt), 'w')

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

            

            cv2.imwrite(Path_Sealion + 'Augment/Flip/JPEGImages/{0}_{1}.jpg'.format(filename[:-4], cnt),sub_image)

            #cv2.imwrite('{0}_{1}.png'.format(filename[:-4], cnt),sub_image)

            sub_image_names.write(Path_Sealion + 'Augment/Flip/JPEGImages/{0}_{1}.jpg'.format(filename[:-4], cnt))

            sub_image_names.write("\n")

            ant_file.close()

    sub_image_names.close()
def parse_image_whole(filename):

    ## open sub_image_names file

    sub_image_names = open(Path_Sealion + "Augment/Flip/Train.txt", 'a')

    

    ### get original image

    ori_image = cv2.imread(Path_Train + filename)

    dot_image = cv2.imread(Path_Dotted + filename)

    cnt = 0

    

    ### flip origin image

    ori_image = augment(ori_image, AUG_METHOD)

    dot_image = augment(dot_image, AUG_METHOD)

    

    ### get coordinate of all sea lions

    blobs = get_blobs(filename)

    

    ### get xy_range info for all sea_lion

    if(arg[0] == 'Basic'):

        Dict_range = get_dict_range_basic(filename, blobs)

    elif(arg[0] == 'Advance'):

        Dict_range = get_dict_range_advance(filename, blobs)

    

    ### output sub_image and annotation file for each blob

    Used_Key_Set = set()

    for key in Dict_range.keys():       

        if(key not in Used_Key_Set):

            # add cnt for new sub_image name

            cnt += 1

            

            # get x, y, xy_range in original image

            main_x = int(key[0])

            main_y = int(key[1])

            xy_range = Dict_range[key]

            

            # adjust main_x main_y if sea_lion close to boundary

            if(main_x - Sub_Im_Size[0]/2 < 0):

                main_x = Sub_Im_Size[0]/2

            if(main_x + Sub_Im_Size[0]/2 > ori_image.shape[1]):

                main_x = ori_image.shape[1] - Sub_Im_Size[0]/2

            if(main_y - Sub_Im_Size[1]/2 < 0):

                main_y = Sub_Im_Size[1]/2

            if(main_y + Sub_Im_Size[1]/2 > ori_image.shape[0]):

                main_y = ori_image.shape[0] - Sub_Im_Size[1]/2

            

            # get_sub_image

            sub_image     = ori_image[int(main_y - Sub_Im_Size[1]/2) : int(main_y + Sub_Im_Size[1]/2), int(main_x - Sub_Im_Size[0]/2) : int(main_x + Sub_Im_Size[0]/2), :]

            sub_image_dot = dot_image[int(main_y - Sub_Im_Size[1]/2) : int(main_y + Sub_Im_Size[1]/2), int(main_x - Sub_Im_Size[0]/2) : int(main_x + Sub_Im_Size[0]/2), :]

            

            # record key as used_key

            Used_Key_Set.add(key)

            

            ### get species

            g,b,r = dot_image[int(key[1])][int(key[0])][:]

            species = get_species(r,g,b)

            if(species == -1):

                continue



            # get pos info for annotation file

            x_pos = float(key[0]-(main_x-Sub_Im_Size[0]/2))/float(Sub_Im_Size[0])

            y_pos = float(key[1]-(main_y-Sub_Im_Size[1]/2))/float(Sub_Im_Size[1])

            x_len = float(xy_range[0]+xy_range[1])/float(Sub_Im_Size[0])

            y_len = float(xy_range[2]+xy_range[3])/float(Sub_Im_Size[1])

            element = [x_pos, y_pos, x_len, y_len]

            

            # save species info in annotation file

            ant_file = open(Path_Sealion + 'Augment/Flip/labels/{0}_{1}.txt'.format(filename[:-4], cnt), 'w')

            ant_file.write(str(species) + " " + " ".join([str(x) for x in element]) + '\n')

            

            

            ### include other sea lion

            # max min coordinate for including image based on origin image

            x_min = max(main_x - sub_image.shape[1]/2 + 1, 0)

            x_max = min(main_x + sub_image.shape[1]/2 - 1, ori_image.shape[1])

            y_min = max(main_y - sub_image.shape[0]/2 + 1, 0)

            y_max = min(main_y + sub_image.shape[0]/2 - 1, ori_image.shape[0])

            

            for ex_key in Dict_range.keys():

                if(ex_key != key and ex_key[0] > x_min and ex_key[0] < x_max and ex_key[1] > y_min and ex_key[1] < y_max):

                    ### coordinate of ex_sea_lion in origin image

                    ex_range = Dict_range[ex_key]

                    ex_left  = int(ex_key[0] - ex_range[0])

                    ex_right = int(ex_key[0] + ex_range[1])

                    ex_up    = int(ex_key[1] - ex_range[2])

                    ex_down  = int(ex_key[1] + ex_range[3])

                    

                    ### sub_image's coordinate where ex_sea_lion put  

                    Used_Key_Set.add(ex_key)



                    ### get species for include sea_lion

                    g,b,r = dot_image[int(ex_key[1])][int(ex_key[0])][:]

                    species = get_species(r,g,b)

            

                    # get pos info for annotation file

                    x_pos = float(ex_key[0]-(main_x-Sub_Im_Size[0]/2))/float(Sub_Im_Size[0])

                    y_pos = float(ex_key[1]-(main_y-Sub_Im_Size[1]/2))/float(Sub_Im_Size[1])

                    

                    # adjust real size of include sea_lion due to be close to boarder

                    ex_left_size  = min(ex_range[0], ex_key[0]-(main_x-Sub_Im_Size[0]/2))

                    ex_right_size = min(ex_range[1], (main_x+Sub_Im_Size[0]/2)-ex_key[0])

                    ex_up_size    = min(ex_range[2], ex_key[1]-(main_y-Sub_Im_Size[1]/2))

                    ex_down_size  = min(ex_range[3], (main_y+Sub_Im_Size[1]/2)-ex_key[1])

                    

                    x_len = float(ex_left_size+ex_right_size)/float(Sub_Im_Size[0])

                    y_len = float(ex_up_size+ex_down_size)/float(Sub_Im_Size[1])

                    element = [x_pos, y_pos, x_len, y_len]

            

                    # save species info in annotation file

                    ant_file.write(str(species) + " " + " ".join([str(x) for x in element]) + '\n')                          

            

            cv2.imwrite(Path_Sealion + 'Augment/Flip/JPEGImages/{0}_{1}.jpg'.format(filename[:-4], cnt), sub_image)

            cv2.imwrite(Path_Sealion +  'Augment/Flip/DotImages/{0}_{1}.jpg'.format(filename[:-4], cnt), sub_image_dot)

            #cv2.imwrite('{0}_{1}.png'.format(filename[:-4], cnt),sub_image)

            sub_image_names.write(Path_Sealion + 'Augment/Flip/JPEGImages/{0}_{1}.jpg'.format(filename[:-4], cnt))

            sub_image_names.write("\n")

            ant_file.close()

    sub_image_names.close()
if os.path.exists(Path_Sealion + "Augment/Flip/Train.txt"):

    os.remove(Path_Sealion + "Augment/Flip/Train.txt")



for filename in file_names:

    if filename[-3:] == 'jpg':

        parse_image_whole(filename)

        

        """

        print("Parsing {0}".format(filename))

        if(arg[2] == 'Black'):

            parse_image_black(filename)

        elif(arg[2] == 'Whole'):

            parse_image_whole(filename)

        else:

            print("Wrong in argument")

        """
os.remove('sub_im_template.jpg')