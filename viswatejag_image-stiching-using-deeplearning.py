# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
from PIL import Image
from PIL import ImageFilter
import multiprocessing
import random; random.seed(2016);
import cv2
import re
import os, glob
sample_sub = pd.read_csv('../input/draper-satellite-image-chronology/sample_submission.csv')
train_files = pd.DataFrame([[f,f.split("/")[4].split(".")[0].split("_")[0],f.split("/")[4].split(".")[0].split("_")[1]] for f in glob.glob("../input/draper-satellite-image-chronology/train_sm/*.jpeg")])
train_files.columns = ['path', 'group', 'pic_no']
test_files = pd.DataFrame([[f,f.split("/")[4].split(".")[0].split("_")[0],f.split("/")[4].split(".")[0].split("_")[1]] for f in glob.glob("../input/draper-satellite-image-chronology/test_sm/*.jpeg")])
test_files.columns = ['path', 'group', 'pic_no']
print(len(train_files),len(test_files),len(sample_sub))
train_images = train_files[train_files["group"]=='set107']
train_images = train_images.sort_values(by=["pic_no"], ascending=[1]).reset_index(drop=True)
plt.rcParams['figure.figsize'] = (12.0, 12.0)
plt.subplots_adjust(wspace=0, hspace=0)
i_ = 0
a = []
for l in train_images.path:
    im = cv2.imread(l)
    plt.subplot(5, 2, i_+1).set_title(l)
    plt.hist(im.ravel(),256,[0,256]); plt.axis('off')
    a.append([im.mean(),im.max(),im.min()])
    plt.subplot(5, 2, i_+2).set_title(l)
    plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)); plt.axis('off')
    i_ += 2
print(a)
kaze = cv2.KAZE_create()
akaze = cv2.AKAZE_create()
brisk = cv2.BRISK_create()

plt.rcParams['figure.figsize'] = (7.0, 18.0)
plt.subplots_adjust(wspace=0, hspace=0)
i = 0
for detector in [kaze, akaze, brisk]:
    start_time = time.time()
    im = cv2.imread(train_images.path[1])
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    (kps, descs) = detector.detectAndCompute(gray, None)       
    cv2.drawKeypoints(im, kps, im, (0, 255, 0))
    plt.subplot(3, 1, i+1).set_title(list(['kaze','akaze','brisk'])[i] + " " + str(round(((time.time() - start_time)/60),5)))
    plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)); plt.axis('off')
    i+=1
print(cv2.__version__)

img1 = cv2.imread(train_images.path[1], 0)
img2 = cv2.imread(train_images.path[2], 0)
brisk = cv2.BRISK_create()
kp1, des1 = brisk.detectAndCompute(img1,None)
kp2, des2 = brisk.detectAndCompute(img2,None)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1,des2)
matches = sorted(matches, key = lambda x:x.distance)
img1 = cv2.imread(train_images.path[1])
img2 = cv2.imread(train_images.path[2])
img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches[:100], flags=2, outImg=img2, matchColor = (0,255,0))
plt.rcParams['figure.figsize'] = (14.0, 8.0)
plt.imshow(cv2.cvtColor(img3, cv2.COLOR_BGR2RGB)); plt.axis('off')

brisk = cv2.BRISK_create()
dm = cv2.DescriptorMatcher_create("BruteForce")

def c_resize(img, ratio):
    wh = (int(img.shape[1] * ratio), int(img.shape[0] * ratio))
    img = cv2.resize(img, wh, interpolation = cv2.INTER_AREA)
    return img
    
def im_stitcher(imp1, imp2, imsr = 1.0, withTransparency=False):
    img1 = cv2.imread(imp1, 0)
    img2 = cv2.imread(imp2, 0)
    if imsr < 1.0:
        img1 = c_resize(img1,imsr); img2 = c_resize(img2,imsr)
    h1,w1 = img1.shape[:2]
    h2,w2 = img2.shape[:2]
    kp1, des1 = brisk.detectAndCompute(img1,None)
    kp2, des2 = brisk.detectAndCompute(img2,None)
    matches = dm.knnMatch(des1,des2, 2)
    matches_ = []
    for m in matches:
        if len(m) == 2 and m[0].distance < m[1].distance * 0.75:
            matches_.append((m[0].trainIdx, m[0].queryIdx))
    kp1_ = np.float32([kp1[m[1]].pt for m in matches_]).reshape(-1,1,2)
    kp2_ = np.float32([kp2[m[0]].pt for m in matches_]).reshape(-1,1,2)
    H, mask = cv2.findHomography(kp2_,kp1_, cv2.RANSAC, 4.0)
    pts1 = np.float32([[0,0],[0,h1],[w1,h1],[w1,0]]).reshape(-1,1,2)
    pts2 = np.float32([[0,0],[0,h2],[w2,h2],[w2,0]]).reshape(-1,1,2)
    pts2_ = cv2.perspectiveTransform(pts2, H)
    pts = np.concatenate((pts1, pts2_), axis=0)
    [xmin, ymin] = np.int32(pts.min(axis=0).ravel() - 0.5)
    [xmax, ymax] = np.int32(pts.max(axis=0).ravel() + 0.5)
    t = [-xmin,-ymin]
    Ht = np.array([[1,0,t[0]],[0,1,t[1]],[0,0,1]])
    img1 = cv2.imread(imp1)
    img2 = cv2.imread(imp2)
    if imsr < 1.0:
        img1 = c_resize(img1,imsr); img2 = c_resize(img2,imsr)
    im = cv2.warpPerspective(img2, Ht.dot(H), (xmax-xmin, ymax-ymin))
    if withTransparency == True:
        h3,w3 = im.shape[:2]
        bim = np.zeros((h3,w3,3), np.uint8)
        bim[t[1]:h1+t[1],t[0]:w1+t[0]] = img1
        im = cv2.addWeighted(im,1.0,bim,0.9,0)
    else:
        im[t[1]:h1+t[1],t[0]:w1+t[0]] = img1
    return im
img = im_stitcher(train_images.path[1], train_images.path[4], 0.5, True)
plt.rcParams['figure.figsize'] = (12.0, 12.0)
img[np.where((img < [20,20,20]).all(axis = 2))] = [255,255,255]
plt.imshow(img); plt.axis('off')
img = cv2.imread(train_images.path[1])
cv2.imwrite('panoramic.jpeg',img)
plt.rcParams['figure.figsize'] = (12.0, 12.0)
for i in range(1,5):
    img = im_stitcher(train_images.path[i], 'panoramic.jpeg', 0.5, False)
    cv2.imwrite('panoramic.jpeg',img)
img[np.where((img < [20,20,20]).all(axis = 2))] = [255,255,255]
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)); plt.axis('off')
files = pd.DataFrame([[f,int(f.split("/")[6][3:-5])] for f in glob.glob("../input/under-vachle/images/IMAGES/IMAGE SET 1/img*.jpeg")])
files.columns = ['path', 'pic_no']
files = files.sort_values(by=["pic_no"], ascending=[1]).reset_index(drop=True)
plt.rcParams['figure.figsize'] = (12.0, 12.0)
plt.subplots_adjust(wspace=0, hspace=0)
i_ = 0
a = []
for l in files.path[0:10]:
    im = cv2.imread(l)
    plt.subplot(10, 2, i_+1)
    plt.hist(im.ravel(),256,[0,256]); plt.axis('off')
    a.append([im.mean(),im.max(),im.min()])
    plt.subplot(10, 2, i_+2)
    plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)); plt.axis('off')
    i_ += 2
print(a)

i_ = 0
a = []
for l in files.path[10:20]:
    im = cv2.imread(l)
    plt.subplot(10, 2, i_+1)
    plt.hist(im.ravel(),256,[0,256]); plt.axis('off')
    a.append([im.mean(),im.max(),im.min()])
    plt.subplot(10, 2, i_+2)
    plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)); plt.axis('off')
    i_ += 2
print(a)
i_ = 0
a = []
for l in files.path[20:30]:
    im = cv2.imread(l)
    plt.subplot(10, 2, i_+1)
    plt.hist(im.ravel(),256,[0,256]); plt.axis('off')
    a.append([im.mean(),im.max(),im.min()])
    plt.subplot(10, 2, i_+2)
    plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)); plt.axis('off')
    i_ += 2
print(a)
i_ = 0
a = []
for l in files.path[30:40]:
    im = cv2.imread(l)
    plt.subplot(10, 2, i_+1)
    plt.hist(im.ravel(),256,[0,256]); plt.axis('off')
    a.append([im.mean(),im.max(),im.min()])
    plt.subplot(10, 2, i_+2)
    plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)); plt.axis('off')
    i_ += 2
print(a)
i_ = 0
a = []
for l in files.path[40:50]:
    im = cv2.imread(l)
    plt.subplot(10, 2, i_+1)
    plt.hist(im.ravel(),256,[0,256]); plt.axis('off')
    a.append([im.mean(),im.max(),im.min()])
    plt.subplot(10, 2, i_+2)
    plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)); plt.axis('off')
    i_ += 2
print(a)
kaze = cv2.KAZE_create()
akaze = cv2.AKAZE_create()
brisk = cv2.BRISK_create()

plt.rcParams['figure.figsize'] = (7.0, 18.0)
plt.subplots_adjust(wspace=0, hspace=0)
i = 0
for detector in [kaze, akaze, brisk]:
    start_time = time.time()
    im = cv2.imread(files.path[20])
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    (kps, descs) = detector.detectAndCompute(gray, None)       
    cv2.drawKeypoints(im, kps, im, (0, 255, 0))
    plt.subplot(3, 1, i+1).set_title(list(['kaze','akaze','brisk'])[i] + " " + str(round(((time.time() - start_time)/60),5)))
    plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)); plt.axis('off')
    i+=1
img1 = cv2.imread("../input/under-vachle/images/IMAGES/IMAGE SET 1/img10.jpeg", 0)
img2 = cv2.imread("../input/under-vachle/images/IMAGES/IMAGE SET 1/img11.jpeg", 0)
brisk = cv2.BRISK_create()
kp1, des1 = brisk.detectAndCompute(img1,None)
kp2, des2 = brisk.detectAndCompute(img2,None)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1,des2)
matches = sorted(matches, key = lambda x:x.distance)
img1 = cv2.imread("../input/under-vachle/images/IMAGES/IMAGE SET 1/img20.jpeg")
img2 = cv2.imread("../input/under-vachle/images/IMAGES/IMAGE SET 1/img21.jpeg")
img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches[:100], flags=2, outImg=img2, matchColor = (0,255,0))
plt.rcParams['figure.figsize'] = (14.0, 8.0)
plt.imshow(cv2.cvtColor(img3, cv2.COLOR_BGR2RGB)); plt.axis('off')

files = pd.DataFrame([[f,f.split("/")[6]] for f in glob.glob("../input/under-vachle/images/IMAGES/IMAGE SET 1/img*.jpeg")])
img = cv2.imread("../input/under-vachle/images/IMAGES/IMAGE SET 1/img9.jpeg")
cv2.imwrite('panoramic.jpeg',img)
plt.rcParams['figure.figsize'] = (12.0, 12.0)
for i in range(10,11):
    img = im_stitcher( '../input/under-vachle/images/IMAGES/IMAGE SET 1/img%d.jpeg' % i , 'panoramic.jpeg', 0.5, False)
    cv2.imwrite('panoramic.jpeg',img)
img[np.where((img < [20,20,20]).all(axis = 2))] = [255,255,255]
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)); plt.axis('off')