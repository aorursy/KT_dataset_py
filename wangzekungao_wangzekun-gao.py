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
demoindex = 1
def downsample(img):
    row = img.shape[0]
    col = img.shape[1]
    newimg = np.zeros((int(row / 2), int(col / 2)))
    for i in range(0, row - 1, 2):
        for j in range(0, col - 1, 2):
            s = 0
            s = s + img[i,j] + img[i+1,j] + img[i,j+1] + img[i+1,j+1]
            s = int(s/4)
            r = int(i / 2)
            c = int(j / 2)
            newimg[r][c] = s
    return newimg
info = pd.read_csv("/kaggle/input/ultrasound-nerve-segmentation/train_masks.csv") 
image_path = "/kaggle/input/ultrasound-nerve-segmentation/train/{}_{}.tif"
mask_path = "/kaggle/input/ultrasound-nerve-segmentation/train/{}_{}_mask.tif"
rows = 420
cols = 580
window_height = 60
window_width = 70
step_size = [window_height, window_width]
nums = 120
li = []
i = 0
while len(li) < nums:  #note the index of images that have nerve
    mask = cv2.imread(mask_path.format(info['subject'][i], info['img'][i]), 2)
    if (np.sum(mask) > 0):
        li.append(i)
    i = i + 1    
imgs = np.zeros((nums, int(rows / 2), int(cols / 2)))
masks = np.zeros((nums, int(rows / 2), int(cols / 2)))
contours = np.zeros((nums, int(rows/2), int(cols/2)))
imgwithoutline = np.zeros((nums, int(rows/2), int(cols/2)))
cor = np.zeros((nums, 2))
centeredoutline = np.zeros((nums, int(window_height), int(window_width)))
maps = np.zeros((nums, 5, int(window_height), int(window_width)))
map1 = np.zeros((nums, int(rows/2), int(cols/2)))
map2 = np.zeros((nums, int(rows/2), int(cols/2)))
map3 = np.zeros((nums, int(rows/2), int(cols/2)))
map4 = np.zeros((nums, int(rows/2), int(cols/2)))
map5 = np.zeros((nums, int(rows/2), int(cols/2)))


L5 = np.array((1,4,6,4,1))
L5 = L5.reshape(1,5)
E5 = np.array((-1,-2,0,2,1))
E5 = E5.reshape(1,5)
S5 = np.array((-1,0,2,0,-1))
S5 = S5.reshape(1,5)
R5 = np.array((1,-4,6,-4,1))
R5 = R5.reshape(1,5)
W5 = np.array((-1,2,0,-2,1))
W5 = W5.reshape(1,5)
kernel1 = np.kron(L5.T, L5)
kernel2 = np.kron(E5.T, E5)
kernel3 = np.kron(S5.T,S5)
kernel4 = np.kron(W5.T,W5)
kernel5 = np.kron(R5.T,R5)
from scipy import signal
for n in range(0,nums):
    img = cv2.imread(image_path.format(info['subject'][n], info['img'][n]), 2)
    mask = cv2.imread(mask_path.format(info['subject'][n], info['img'][n]), 2)
    imgm=cv2.medianBlur(img,5)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    equ = clahe.apply(imgm)
    equ = downsample(equ)
    mask = downsample(mask)
    imgs[n,:,:] = equ
    map1[n] = signal.convolve2d(equ, kernel1, mode = 'same')
    map2[n] = signal.convolve2d(equ, kernel2, mode = 'same')
    map3[n] = signal.convolve2d(equ, kernel3, mode = 'same')
    map4[n] = signal.convolve2d(equ, kernel4, mode = 'same')
    map5[n] = signal.convolve2d(equ, kernel5, mode = 'same')
    masks[n,:,:] = mask.astype(np.bool)
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
        contour = contours[n,x - int(window_height / 2) : x + int(window_height / 2),y - int(window_width / 2):y + int(window_width / 2)]
        centeredoutline[n,:,:] = contour
        maps[n,0] = map1[n,x - int(window_height / 2) : x + int(window_height / 2),y - int(window_width / 2):y + int(window_width / 2)]
        maps[n,1] = map2[n,x - int(window_height / 2) : x + int(window_height / 2),y - int(window_width / 2):y + int(window_width / 2)]
        maps[n,2] = map3[n,x - int(window_height / 2) : x + int(window_height / 2),y - int(window_width / 2):y + int(window_width / 2)]
        maps[n,3] = map4[n,x - int(window_height / 2) : x + int(window_height / 2),y - int(window_width / 2):y + int(window_width / 2)]
        maps[n,4] = map5[n,x - int(window_height / 2) : x + int(window_height / 2),y - int(window_width / 2):y + int(window_width / 2)]
imgwithmask = imgs * masks
plt.imshow(imgs[demoindex],cmap = 'gray')
f,axarr = plt.subplots(5,1, figsize = (6, 5 * 6))
axarr[0].imshow(maps[demoindex,0],cmap = 'gray')
axarr[0].imshow(centeredoutline[demoindex],alpha = 0.2)
axarr[0].set_title('detect average intensity')
axarr[1].imshow(maps[demoindex,1],cmap = 'gray')
axarr[1].imshow(centeredoutline[demoindex],alpha = 0.2)
axarr[1].set_title('detect edges')
axarr[2].imshow(maps[demoindex,2],cmap = 'gray')
axarr[2].imshow(centeredoutline[demoindex],alpha = 0.2)
axarr[2].set_title('detect spots')
axarr[3].imshow(maps[demoindex,3],cmap = 'gray')
axarr[3].imshow(centeredoutline[demoindex],alpha = 0.2)
axarr[3].set_title('detect wave texture')
axarr[4].imshow(maps[demoindex,4],cmap = 'gray')
axarr[4].imshow(centeredoutline[demoindex],alpha = 0.2)
axarr[4].set_title('detect ripple texture')
plt.show()
map1.shape
neg_patch = []
allpatch = []
patcheswithoutline = []
labels = []
step_size = [window_height, window_width]
features = []

for n in range(0, nums):
    curneg = []
    img = imgs[n]
    mask = masks[n]
    for i in range(0, int(rows/2) - window_height + 1, step_size[0]):
        for j in range(0, int(cols/2) - window_width + 1, step_size[1]):
            curpatch = np.zeros((5, window_height, window_width))
            m = mask[i:i+window_height, j:j+window_width]
            m1 = map1[n, i:i+window_height, j: j + window_width]
            m2 = map2[n, i:i+window_height, j: j + window_width]
            m3 = map3[n, i:i+window_height, j: j + window_width]
            m4 = map4[n, i:i+window_height, j: j + window_width]
            m5 = map5[n, i:i+window_height, j: j + window_width]
            curpatch[0,:,:]=m1
            curpatch[1,:,:]=m2
            curpatch[2,:,:]=m3
            curpatch[3,:,:]=m4
            curpatch[4,:,:]=m5
            curimage.append(curpatch)
            if (np.sum(m) == 0):    
                curneg.append(curpatch)
    curneg = np.array(curneg)
    neg_patch.append(curneg)
    allpatch.append(curimage)
neg_patch = np.array(neg_patch)
#             temp = withline[i:i + window_height, j:j + window_width]
#             curoutline.append(temp)
#     curneg = np.stack(curneg)
#     curpatch = np.stack(curpatch)
#     curoutline = np.stack(curoutline)
#     neg.append(curneg)
#             allpatch.append(curpatch)
#     patcheswithoutline.append(curoutline)
# neg = np.array(neg)
# allpatch = np.array(allpatch)
# labels = np.array(labels, dtype = np.bool)
# patcheswithoutline = np.array(patcheswithoutline)
import random
negpatch = np.zeros((5 * nums, 5, window_height, window_width))
cur = 0
for i in range(0, nums):
    curneg = neg_patch[i]
    randomindex = random.sample(range(0,curneg.shape[0]),5)
    negpatch[5 * i] = curneg[randomindex[0]]
    negpatch[5 * i + 1] = curneg[randomindex[1]]
    negpatch[5 * i + 2] = curneg[randomindex[2]]
    negpatch[5 * i + 3] = curneg[randomindex[3]]
    negpatch[5 * i + 4] = curneg[randomindex[4]]

f,axarr = plt.subplots(5,1, figsize = (6, 5 * 6))
axarr[0].imshow(negpatch[demoindex,0],cmap = 'gray')
axarr[0].set_title('detect average intensity')
axarr[1].imshow(negpatch[demoindex,1],cmap = 'gray')
axarr[1].set_title('detect edges')
axarr[2].imshow(negpatch[demoindex,2],cmap = 'gray')
axarr[2].set_title('detect spots')
axarr[3].imshow(negpatch[demoindex,3],cmap = 'gray')
axarr[3].set_title('detect wave texture')
axarr[4].imshow(negpatch[demoindex,4],cmap = 'gray')
axarr[4].set_title('detect ripple texture')
plt.show()
maps.shape
al = np.concatenate((maps, negpatch), axis = 0)
labels = np.zeros(al.shape[0], dtype = np.bool)
labels[: maps.shape[0]] = 1
features = al.reshape(al.shape[0], -1)
scores = np.zeros(features.shape[1])
for i in range(features.shape[1]):
    f = features[:, i]
    fa = f[labels == 1]
    fb = f[labels == 0]
    mi = np.mean(f)
    mia = np.mean(fa)
    mib = np.mean(fb)
    na = fa.shape[0]
    nb = fb.shape[0]
    va = np.var(fa)
    vb = np.var(fb)
    scores[i] = (na * np.square((mia - mi))+nb * np.square((mib-mi)))/(na*va+nb*vb)
indexes = np.argsort(scores) 
np.where(scores > 0.1)[0].size
testnums = 50
testpatches = []
testlabels = []
testout = []
test_step_size = [window_height, window_width]
testmaps = np.zeros((testnums, 5, window_height, window_width))
corordinate = set()
for n in range(li[nums - 1] + 1, li[nums - 1] + testnums ):
    img = cv2.imread(image_path.format(info['subject'][n], info['img'][n]), 2)
    mask = cv2.imread(mask_path.format(info['subject'][n], info['img'][n]), 2)
    imgm=cv2.medianBlur(img,5)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    equ = clahe.apply(imgm)
    equ = downsample(equ)
    mask = downsample(mask)
    for i in range(0, int(rows/2 - window_height + 1), test_step_size[0]):
        for j in range(0, int(cols/2 - window_width + 1), test_step_size[1]):
            map1 = signal.convolve2d(equ[i:i+window_height, j: j + window_width], kernel1, mode = 'same')
            map2 = signal.convolve2d(equ[i:i+window_height, j: j + window_width], kernel2, mode = 'same')
            map3 = signal.convolve2d(equ[i:i+window_height, j: j + window_width], kernel3, mode = 'same')
            map4 = signal.convolve2d(equ[i:i+window_height, j: j + window_width], kernel4, mode = 'same')
            map5 = signal.convolve2d(equ[i:i+window_height, j: j + window_width], kernel5, mode = 'same')
            m = mask[i:i+window_height, j:j+window_width]
            testmaps[n,0] = map1
            testmaps[n,1] = map2
            testmaps[n,2] = map3
            testmaps[n,3] = map4
            testmaps[n,4] = map5
            corordinate.add((i, j))
            if (np.sum(m) > 0):
                testlabels.append(1)
            else:
                testlabels.append(0)
testlabels = np.array(testlabels, dtype = np.bool)
testfeatures = testmaps.reshape(testmaps.shape[0], -1)


from sklearn.metrics import confusion_matrix
output = clf.predict(testfeatures)
confusion = confusion_matrix(testlabels, output)
p = clf.predict_proba(testfeatures)
confusion