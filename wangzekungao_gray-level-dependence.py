import cv2
import numpy as np
import csv
import pandas as pd
import matplotlib as plt
import matplotlib.cm as cm
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import chardet
from skimage.feature import local_binary_pattern
import copy
demoindex = 30
info = pd.read_csv("/kaggle/input/ultrasound-nerve-segmentation/train_masks.csv") 
image_path = "/kaggle/input/ultrasound-nerve-segmentation/train/{}_{}.tif"
mask_path = "/kaggle/input/ultrasound-nerve-segmentation/train/{}_{}_mask.tif"
rows = 420
cols = 580
window_height = 120
window_width = 140
step_size = [window_height, window_width]
label = []
features = []
li = []
i = 0
tr = 150
while i < tr:  #note the index of images that have nerve
    mask = cv2.imread(mask_path.format(info['subject'][i], info['img'][i]), 2)
    if (np.sum(mask) > 0):
        li.append(i)
    i = i + 1    
nums = len(li)
contours = np.zeros((nums, rows, cols))
imgwithoutline = np.zeros((nums, rows, cols))
cor = np.zeros((nums, 2))
pospatch = np.zeros((nums, window_height, window_width))
poswithoutline = np.zeros((nums, window_height, window_width))

for n in range(0,nums):
    img = cv2.imread(image_path.format(info['subject'][li[n]], info['img'][li[n]]), 2)
    mask = cv2.imread(mask_path.format(info['subject'][li[n]], info['img'][li[n]]), 2)
    imgm=cv2.medianBlur(img,5)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    equ = clahe.apply(imgm)
    imgwithoutline[n,:,:] = copy.deepcopy(equ)
    mask_outline = cv2.blur(mask, (3,3))
    mask_outline = mask_outline * ((mask_outline < 255) & (mask_outline > 0))
    contours[n,:,:] = mask_outline > 0
    temp = np.where(mask.astype(np.bool))
    if temp[0].size != 0:         # center of the nerve
        xs = temp[0]
        ys = temp[1]
        x = int((min(xs) + max(xs)) / 2)
        y = int((min(ys) + max(ys)) / 2)
        cor[n,0] = x
        cor[n,1] = y
        pospatch[n,:,:] = img[x - int(window_height / 2) : x + int(window_height / 2),y - int(window_width / 2):y + int(window_width / 2)]
        temp = copy.deepcopy(pospatch[n,:,:])
        contour = contours[n,x - int(window_height / 2) : x + int(window_height / 2),y - int(window_width / 2):y + int(window_width / 2)]
        temp[contour>0] = 255
        poswithoutline[n] = temp
        imgwithoutline[n][contours[n] > 0]=255
neg = []
allpatch = []
patcheswithoutline = []
labels = []
step_size = [window_height, window_width]
for n in range(0, tr):
    curneg = []
    curpatch = []
    curoutline = []
    img = cv2.imread(image_path.format(info['subject'][n], info['img'][n]), 2)
    mask = cv2.imread(mask_path.format(info['subject'][n], info['img'][n]), 2)
    imgm=cv2.medianBlur(img,5)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    equ = clahe.apply(imgm)
    mask_outline = cv2.blur(mask, (3,3))
    mask_outline = mask_outline * ((mask_outline < 255) & (mask_outline > 0))
    for i in range(0, rows - window_height + 1, step_size[0]):
        for j in range(0, cols - window_width + 1, step_size[1]):
            temp = equ[i:i+window_height, j: j + window_width]
            m = mask[i:i+window_height, j:j+window_width]
            o = mask_outline[i:i+window_height, j:j+window_width]
            curpatch.append(temp)
            if (np.sum(m) == 0):    
                curneg.append(temp)
                labels.append(0)
            else:
                labels.append(1)
            curoutline.append(o)
    curneg = np.stack(curneg)
    curpatch = np.stack(curpatch)
    curoutline = np.stack(curoutline)
    neg.append(curneg)
    allpatch.append(curpatch)
    patcheswithoutline.append(curoutline)
neg = np.array(neg)
allpatch = np.array(allpatch)
# labels = np.array(labels, dtype = np.bool)
patcheswithoutline = np.array(patcheswithoutline)
import random
negpatch = np.zeros((2 * tr, window_height, window_width))
for i in range(0, tr):
    curneg = neg[i]
    randomindex = random.sample(range(0,curneg.shape[0]),2)
    negpatch[2 * i] = curneg[randomindex[0]]
    negpatch[2 * i + 1] = curneg[randomindex[1]]
al = np.concatenate((pospatch, negpatch), axis = 0)
labels = np.zeros(al.shape[0], dtype = np.bool)
labels[: pospatch.shape[0]] = 1
from scipy.stats import kurtosis,skew
from skimage.feature import greycomatrix,greycoprops
features=[]
for i in range(0, al.shape[0]):
    vector = np.zeros(4)
    cur = np.array(al[i],dtype = np.int16)
    cormatrix = greycomatrix(cur, [1], [np.pi/4],levels=256)
    contrast = greycoprops(cormatrix, 'contrast')
    energy = greycoprops(cormatrix, 'ASM')
    corelation = greycoprops(cormatrix, 'correlation')
    homogeneity = greycoprops(cormatrix, 'homogeneity')
    vector[0] = contrast
    vector[1] = energy
    vector[2] = corelation
    vector[3] = homogeneity
    features.append(vector)
features = np.stack(features)

from sklearn import svm
import joblib
filename = 'finalized_model.sav'
clf = svm.SVC(kernel="linear", gamma="scale", C=5.0, probability=True)
clf.fit(features, labels)
joblib.dump(clf, filename)
testnums = 50
testpatches = []
testlabels = []
testout = []
test_step_size = [window_height, window_width]
testimg = []
resultimg = np.zeros((testnums, rows, cols))
corordinate = set()
for n in range(tr, tr + testnums ):
    img = cv2.imread(image_path.format(info['subject'][n], info['img'][n]), 2)
    mask = cv2.imread(mask_path.format(info['subject'][n], info['img'][n]), 2)
    testimg.append(img)
    imgm=cv2.medianBlur(img,5)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    equ = clahe.apply(imgm)
    for i in range(50, int(rows - window_height + 1), test_step_size[0]):
        for j in range(100, int(cols - window_width + 1), test_step_size[1]):
            temp = imgm[i:i+window_height, j: j + window_width]
            m = mask[i:i+window_height, j:j+window_width]
            if (np.sum(m) > 0):
                testlabels.append(1)
            else:
                testlabels.append(0)
            testpatches.append(temp)
            corordinate.add((i, j))
testpatches = np.array(testpatches)
testlabels = np.array(testlabels, dtype = np.bool)
testfeatures = []
for i in range(0, testpatches.shape[0]):
    vector = np.zeros(4)
    cur = np.array(testpatches[i],dtype = np.int16)
    cormatrix = greycomatrix(cur, [1], [np.pi/4],levels=256)
    contrast = greycoprops(cormatrix, 'contrast')
    energy = greycoprops(cormatrix, 'ASM')
    corelation = greycoprops(cormatrix, 'correlation')
    homogeneity = greycoprops(cormatrix, 'homogeneity')
    vector[0] = contrast
    vector[1] = energy
    vector[2] = corelation
    vector[3] = homogeneity
    testfeatures.append(vector)
corordinate = list(corordinate)
testfeatures = np.stack(testfeatures)
from sklearn.metrics import confusion_matrix
output = clf.predict(testfeatures)
confusion = confusion_matrix(testlabels, output)
p = clf.predict_proba(testfeatures)
confusion