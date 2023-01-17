import numpy as np 

import pandas as pd 



import os
# Basic library

import numpy as np 

import pandas as pd 



# Data preprocessing

import cv2# Open cv

from sklearn.model_selection import train_test_split



# Visualization

from matplotlib import pyplot as plt

import seaborn as sns

plt.style.use('fivethirtyeight')
sample = pd.read_csv("/kaggle/input/alaska2-image-steganalysis/sample_submission.csv")
# path

path_Test = "/kaggle/input/alaska2-image-steganalysis/Test/"

path_juni =  "/kaggle/input/alaska2-image-steganalysis/JUNIWARD/"

path_jmi =  "/kaggle/input/alaska2-image-steganalysis/JMiPOD"

path_cov =  "/kaggle/input/alaska2-image-steganalysis/Cover/"

path_uer = "/kaggle/input/alaska2-image-steganalysis/UERD/"
dir_name = os.listdir("/kaggle/input/alaska2-image-steganalysis/")

dir_name
# Drop csv from dir_name

dir_name = ['Test', 'JUNIWARD', 'JMiPOD', 'Cover', 'UERD']



# Create empty dataframe and list

df = pd.DataFrame({})

lists = []

cate = []



# get the filenames

for dir_ in dir_name:

    # file name

    list_ = os.listdir("/kaggle/input/alaska2-image-steganalysis/"+dir_+"/")

    lists = lists+list_

    # category name

    cate_ = np.tile(dir_,len(list_))

    cate = np.concatenate([cate,cate_])

    

# insert dataframe

df["cate"] = cate

df["name"] = lists
df = df.sample(1000)
# data loading

# Define data size

size = 256



# Create image data list

img_data = []



# Data loading

for dir_ in dir_name:

    for name in df[df["cate"]==dir_]["name"]:

        path = "/kaggle/input/alaska2-image-steganalysis/"+dir_+"/"+name+""

        img = cv2.imread(path)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Change to color array, BGR⇒RGB

        image = cv2.resize(img, (size,size), interpolation=cv2.INTER_AREA)

        img_data.append(image)



# Add to dataframe

df["img"] = img_data
df.head()
# image check

cate_name = df["cate"].value_counts().index



fig, ax = plt.subplots(5,4, figsize=(20,30))

for i in range(5):

    for j in range(4):

        ax[i,j].imshow(df[df["cate"]==cate_name[i]]["img"].values[j])

        ax[i,j].set_title(cate_name[i])

        ax[i,j].grid()
# 154631

# sample image and Red,Green,Blue image

sample_img = df.reset_index()["img"][10]



red = sample_img.copy()

red[:,:,(1, 2)] = 0 #Cancel green and blue value to 0

green = sample_img.copy()

green[:,:,(0, 2)] = 0

blue = sample_img.copy()

blue[:,:,(0, 1)] = 0





# imshow

fig, ax = plt.subplots(1,4, figsize=(20,6))

ax[0].imshow(sample_img)

ax[0].set_title("row data")

ax[1].imshow(red)

ax[1].set_title("red")

ax[2].imshow(green)

ax[2].set_title("green")

ax[3].imshow(blue)

ax[3].set_title("blue")
# Conversion list

flags = [i for i in dir(cv2) if i.startswith('COLOR_')]

print( flags )
# gray, hsv, rgba, hls

gray = cv2.cvtColor(sample_img, cv2.COLOR_RGB2GRAY)

hsv = cv2.cvtColor(sample_img, cv2.COLOR_RGB2HSV)

xyz = cv2.cvtColor(sample_img, cv2.COLOR_RGB2XYZ)

hls = cv2.cvtColor(sample_img, cv2.COLOR_RGB2HLS)



# imshow

fig, ax = plt.subplots(1,5, figsize=(20,4))

ax[0].imshow(sample_img)

ax[0].set_title("sample_img")

ax[1].imshow(gray, cmap=plt.get_cmap('gray'), vmin=0, vmax=255) # with gray scale, it need to specify color map.

ax[1].set_title("gray")

ax[2].imshow(hsv)

ax[2].set_title("hsv")

ax[3].imshow(xyz)

ax[3].set_title("xyz")

ax[4].imshow(hls)

ax[4].set_title("hls")
# to visualize hsv image with imshow, need to separate, and gray scale view

h, s, v = cv2.split(hsv)



# imshow

fig, ax = plt.subplots(1,4, figsize=(20,6))

ax[0].imshow(hsv)

ax[0].set_title("total hsv image")

ax[1].imshow(h, cmap=plt.get_cmap('gray'), vmin=0, vmax=255)

ax[1].set_title("Hue")

ax[2].imshow(s, cmap=plt.get_cmap('gray'), vmin=0, vmax=255)

ax[2].set_title("Saturation")

ax[3].imshow(v, cmap=plt.get_cmap('gray'), vmin=0, vmax=255)

ax[3].set_title("Value")
# to visualize hls image with imshow, need to separate, and gray scale view

h, l, s = cv2.split(hls)



# imshow

fig, ax = plt.subplots(1,4, figsize=(20,6))

ax[0].imshow(hls)

ax[0].set_title("total hls image")

ax[1].imshow(h, cmap=plt.get_cmap('gray'), vmin=0, vmax=255)

ax[1].set_title("Hue")

ax[2].imshow(l, cmap=plt.get_cmap('gray'), vmin=0, vmax=255)

ax[2].set_title("Lightness")

ax[3].imshow(s, cmap=plt.get_cmap('gray'), vmin=0, vmax=255)

ax[3].set_title("Saturation")
lower_blue = np.array([20,50,50])

upper_blue = np.array([130,255,255])



mask = cv2.inRange(hsv, lower_blue, upper_blue)

res = cv2.bitwise_and(hsv,hsv, mask=mask)



fig, ax = plt.subplots(1,4, figsize=(20,6))

ax[0].imshow(sample_img)

ax[0].set_title("sample_img")

ax[1].imshow(hsv)

ax[1].set_title("hsv")

ax[2].imshow(mask)

ax[2].set_title("mask")

ax[3].imshow(res)

ax[3].set_title("res")
# Scaling 

scale = cv2.resize(sample_img, (64,64), interpolation=cv2.INTER_AREA)

# Transformation

M = np.float32([[1,0,100],[0,1,50]])

trans= cv2.warpAffine(sample_img, M, (128,128))

# Rotation

M = cv2.getRotationMatrix2D(((128-1)/2.0,(128-1)/2.0),90,1)

rotation = cv2.warpAffine(sample_img, M, (128,128))

# Affine Transformation 3point ⇒ 3point

pts1 = np.float32([[50,50],[128,50],[50,100]])

pts2 = np.float32([[10,100],[128,50],[70,128]])

M = cv2.getAffineTransform(pts1,pts2)

Affine = cv2.warpAffine(sample_img, M, (128,128))

# Perspective Transformation 4point ⇒ 4point

pts1 = np.float32([[56,65],[128,52],[28,128],[128,128]])

pts2 = np.float32([[0,0],[128,0],[0,100],[100,100]])

M = cv2.getPerspectiveTransform(pts1,pts2)

perspec = cv2.warpPerspective(sample_img, M, (128,128))



# imshow

fig, ax = plt.subplots(1,5, figsize=(20,4))

ax[0].imshow(sample_img)

ax[0].set_title("sample_img")

ax[1].imshow(scale)

ax[1].set_title("scale")

ax[2].imshow(rotation)

ax[2].set_title("rotation")

ax[3].imshow(Affine)

ax[3].set_title("Affine")

ax[4].imshow(perspec)

ax[4].set_title("perspec")
# binary

ret,th1 = cv2.threshold(gray, 164, 255, cv2.THRESH_BINARY)



# mean_c

th2 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)



# goussian_c

th3 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)



# imshow

fig, ax = plt.subplots(1,4, figsize=(20,6))

ax[0].imshow(gray, cmap=plt.get_cmap('gray'), vmin=0, vmax=255)

ax[0].set_title("gray")

ax[1].imshow(th1, cmap=plt.get_cmap('gray'), vmin=0, vmax=255)

ax[1].set_title("binary")

ax[2].imshow(th2, cmap=plt.get_cmap('gray'), vmin=0, vmax=255)

ax[2].set_title("mean_c")

ax[3].imshow(th3, cmap=plt.get_cmap('gray'), vmin=0, vmax=255)

ax[3].set_title("goussian_c")
# 2D Convolution ( Image Filtering )

kernel = np.ones((5,5),np.float32)/25

dst = cv2.filter2D(sample_img, -1, kernel)



# Averaging blur

blur = cv2.blur(sample_img, (5,5))



# Gaussian Blurring

blur_g = cv2.GaussianBlur(sample_img, (5,5), 0)



# Median Blurring

median = cv2.medianBlur(sample_img, 5)



# Bilateral Filtering

blur_b = cv2.bilateralFilter(sample_img, 9, 75, 75)



# imshow

fig, ax = plt.subplots(2,3, figsize=(16,10))

ax[0,0].imshow(sample_img)

ax[0,0].set_title("sample_img")

ax[0,1].imshow(dst)

ax[0,1].set_title("2D Convolution")

ax[0,2].imshow(blur)

ax[0,2].set_title("Averaging blur")

ax[1,0].imshow(blur_g)

ax[1,0].set_title("Gaussian Blurring")

ax[1,1].imshow(median)

ax[1,1].set_title("Median Blurring")

ax[1,2].imshow(blur_b)

ax[1,2].set_title("Bilateral Filtering")
# Erosion

kernel = np.ones((5,5), np.float32)

ero = cv2.erode(gray, kernel, iterations=1)



# Dilation

dila = cv2.dilate(gray, kernel, iterations=1)



# Opening

opening = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)



# Closing

closing = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)



# Morphological Gradient

grad = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel)



# Top Hat

tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)



# Black Hat

blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)



# imshow

fig, ax = plt.subplots(2,4, figsize=(20,10))

ax[0,0].imshow(gray, cmap=plt.get_cmap('gray'), vmin=0, vmax=255)

ax[0,0].set_title("gray")

ax[0,1].imshow(ero, cmap=plt.get_cmap('gray'), vmin=0, vmax=255)

ax[0,1].set_title("Erosion")

ax[0,2].imshow(dila, cmap=plt.get_cmap('gray'), vmin=0, vmax=255)

ax[0,2].set_title("Dilation")

ax[0,3].imshow(opening, cmap=plt.get_cmap('gray'), vmin=0, vmax=255)

ax[0,3].set_title("Opening")

ax[1,0].imshow(closing, cmap=plt.get_cmap('gray'), vmin=0, vmax=255)

ax[1,0].set_title("Closing")

ax[1,1].imshow(grad, cmap=plt.get_cmap('gray'), vmin=0, vmax=255)

ax[1,1].set_title("Morphological Gradient")

ax[1,2].imshow(tophat, cmap=plt.get_cmap('gray'), vmin=0, vmax=255)

ax[1,2].set_title("Top Hatr")

ax[1,3].imshow(blackhat, cmap=plt.get_cmap('gray'), vmin=0, vmax=255)

ax[1,3].set_title("Black Hat")
# Laplacian Derivatives

lap = cv2.Laplacian(gray, cv2.CV_64F)



# sobelx

sobelx = cv2.Sobel(gray, cv2.CV_64F,1,0,ksize=5)



# sobely

sobely = cv2.Sobel(gray, cv2.CV_64F,0,1,ksize=5)



# imshow

fig, ax = plt.subplots(1,4, figsize=(20,6))

ax[0].imshow(gray, cmap=plt.get_cmap('gray'), vmin=0, vmax=255)

ax[0].set_title("gray")

ax[1].imshow(lap, cmap=plt.get_cmap('gray'), vmin=0, vmax=255)

ax[1].set_title("Laplacian")

ax[2].imshow(sobelx, cmap=plt.get_cmap('gray'), vmin=0, vmax=255)

ax[2].set_title("sobelx")

ax[3].imshow(sobely, cmap=plt.get_cmap('gray'), vmin=0, vmax=255)

ax[3].set_title("sobely")
# min_val100, max_val200

edge1 = cv2.Canny(sample_img, 100, 200)



# min_val50, max_val200

edge2 = cv2.Canny(sample_img, 50, 200)



# min_val100, max_val300

edge3 = cv2.Canny(sample_img, 100, 300)



# imshow

fig, ax = plt.subplots(1,4, figsize=(20,4))

ax[0].imshow(gray, cmap=plt.get_cmap('gray'), vmin=0, vmax=255)

ax[0].set_title("gray")

ax[1].imshow(edge1, cmap=plt.get_cmap('gray'), vmin=0, vmax=255)

ax[1].set_title("min_val100, max_val200")

ax[2].imshow(edge2, cmap=plt.get_cmap('gray'), vmin=0, vmax=255)

ax[2].set_title("min_val50, max_val200")

ax[3].imshow(edge3, cmap=plt.get_cmap('gray'), vmin=0, vmax=255)

ax[3].set_title("min_val100, max_val300")
# calculate histgrams

# cv.calcHist(images, channels, mask, histSize, ranges[, hist[, accumulate]])

# channels, if gray [0], if color [0] or [1] or [2]

hist = cv2.calcHist([sample_img],[0],None, [256],[0,256])
# Visualization

fig, ax = plt.subplots(1,2,figsize=(15,6))

color=["red", "green", "blue"]

ax[0].imshow(sample_img)

ax[0].grid()

for i in range(3):

    ax[1].plot(cv2.calcHist([sample_img],[i], None, [256],[0,256]), color=color[i])

ax[1].set_xlim([0,256])
# calculate histgrams for gray image

hist = cv2.calcHist([gray],[0],None, [256],[0,256])
# Visualization

fig, ax = plt.subplots(1,2,figsize=(15,6))

ax[0].imshow(gray, cmap=plt.get_cmap('gray'), vmin=0, vmax=255)

ax[0].grid()

ax[1].hist(cv2.calcHist([gray],[0], None, [256],[0,256]), color="gray", bins=128)

ax[1].set_xlim([0,256])
# Equalization

equ = cv2.equalizeHist(gray)



# Visualization

fig, ax = plt.subplots(1,2,figsize=(15,6))

ax[0].imshow(equ, cmap=plt.get_cmap('gray'), vmin=0, vmax=255)

ax[0].grid()

ax[1].hist(cv2.calcHist([equ],[0], None, [256],[0,256]), color="gray", bins=128)

ax[1].set_xlim([0,256])
# Equalization

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

cl1 = clahe.apply(gray)



# Visualization

fig, ax = plt.subplots(1,2,figsize=(15,6))

ax[0].imshow(cl1, cmap=plt.get_cmap('gray'), vmin=0, vmax=255)

ax[0].grid()

ax[1].hist(cv2.calcHist([cl1],[0], None, [256],[0,256]), color="gray", bins=128)

ax[1].set_xlim([0,256])
# calculate histgram, after change to hsv

hsv = cv2.cvtColor(sample_img, cv2.COLOR_RGB2HSV)



hist = cv2.calcHist([hsv], [0,1], None, [180,256], [0,180,0,256])
# Visualization

fig, ax = plt.subplots(1,2,figsize=(15,6))

ax[0].imshow(hsv)

ax[0].grid()

ax[1].imshow(hist) # X axis shows S values and Y axis shows Hue

ax[1].grid()
# preparing the image

img = np.float32(sample_img)



# define criteria, number of cluster(K) and apply kmeans.

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1)

K = 8

ret, label, center = cv2.kmeans(img, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)



# cnvert bak into unit8, and make original image

center = center.astype("uint8")

res = center[label.flatten()]

res2 = res.reshape((img.shape))



fig, ax = plt.subplots(1, 4, figsize=(20,6))

ax[0].imshow(sample_img)

ax[0].set_title("sample image")

ax[1].imshow(img)

ax[1].set_title("convert float32 image")

ax[2].imshow(res)

ax[2].set_title("center")

ax[3].imshow(res2)

ax[3].set_title("result")
# calculate dst

dst = cv2.fastNlMeansDenoisingColored(sample_img, None, 10, 10, 7, 21)



# visalization

fig, ax = plt.subplots(1,2, figsize=(12,6))



ax[0].imshow(sample_img)

ax[0].set_title("sample_image")

ax[1].imshow(dst)

ax[1].set_title("image denoising")
# Library

from sklearn.decomposition import PCA
# sample of gray image

img = gray.copy()



# Create instance

pca = PCA()



# Fitting, Holds 99% of variance

pca = PCA(0.99).fit(img)



# components

components = pca.transform(img)

filterd = pca.inverse_transform(components)
# visalization

fig, ax = plt.subplots(1,2, figsize=(12,12))



ax[0].imshow(gray, cmap=plt.get_cmap('gray'), vmin=0, vmax=255)

ax[0].set_title("gray")

ax[1].imshow(filterd, cmap=plt.get_cmap('gray'), vmin=0, vmax=255)

ax[1].set_title("pca filtered")