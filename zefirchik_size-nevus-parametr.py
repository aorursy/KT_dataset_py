!pip install imutils
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cv2
import math
from skimage import data, img_as_float
from skimage import exposure
from skimage.io import imsave
from imutils import perspective
from imutils import contours
import imutils
from scipy.spatial import distance as dist
from keras.layers import AveragePooling2D, MaxPooling2D, Input
from keras.models import Model
import keras.backend as K
from keras.preprocessing import image as ik
def midpoint(ptA, ptB):
    return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)
import matplotlib.gridspec as gridspec
COUNTLEN = 10 #how many sizes from the center of the mole to the edge to return
def HAIR_SORRY_REMOVE(image, clip_hist_percent=1):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    kernel = cv2.getStructuringElement(1,(17,17))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    _,threshold = cv2.threshold(blackhat,10,255,cv2.THRESH_BINARY)
    final_image = cv2.inpaint(image,threshold,1,cv2.INPAINT_TELEA)
    return (final_image)
def step_assimetry(img):
    img = HAIR_SORRY_REMOVE(img)
#     img = cv2.bilateralFilter(img,50,15,15)
    
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret, thresh =     cv2.threshold(gray,100,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    
    kernel = np.ones((2,2),np.uint8)
    opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)
    kernel = np.ones((2,2),np.uint8)
    dilate = cv2.erode(opening,kernel,iterations=3)
#     dilate = cv2.dilate(opening,kernel,iterations=3)
    blur = dilate
   
#     blur = cv2.blur(opening,(15,15))
    ret, thresh =     cv2.threshold(blur,100,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    
    contours, hierarchy =     cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
#     images = cv2.drawContours( img, contours, -1, (255,0,0), 3, cv2.LINE_AA, hierarchy, 1 )
#     plt.imshow(images)
    
    delta = 40
    if len(contours)<3:
        delta = 70
    S = list()
    for c in contours:
        box = cv2.minAreaRect(c)
        box = cv2.cv.BoxPoints(box)  if imutils.is_cv2() else cv2.boxPoints( box)
        box = np.array(box, dtype="int")
        box = perspective.order_points( box)
        center = img.shape[0]/2
       
        ( tl, tr, br, bl) = box
        (centerX, centerY) = midpoint( tl, br)
        (midlx, midly) = midpoint( tl, bl)
#         print(midlx,"---")
        if (tl[0]>0 and tr[0]<250 and br[1]<255) or (midlx>0 and br[0]<252 and centerX>center-delta and centerX<center+delta):
#             if (centerX>center-delta and centerX<center+delta) and (centerY>center-delta and centerY<center+delta):
                S.append(c)
    
    return S
SHAPE = 256
def get_Assimetry2(im):
    example = [(105, 24), (78, 84), (93, 90), (105, 112), (110, 148), (93, 179), (76, 228), (44, 225), (112, 255), (184, 252), (217, 255), (255, 224), (255, 123), (207, 110), (209, 56), (153, 78)] 
    result = im.copy()
    orig2 = im.copy()
    img = im.copy()
#     img = segmentation_color3(img.copy())
   
    S = step_assimetry(img)
    shape_orig = orig2.shape[0]
    if len(S)>0:        
        cnt = max(S, key=cv2.contourArea)
    else:
        return "err", result, True,"var1"
    box = cv2.minAreaRect(cnt)
    box = cv2.cv.BoxPoints(box)  if imutils.is_cv2() else cv2.boxPoints( box)
    box = np.array(box, dtype="int")
    box = perspective.order_points( box)
    orig = cv2.drawContours(result, [box.astype("int")] , -1 , ( 0 , 255 , 0 ) , 2)
    for ( x, y)  in box:
        cv2.circle(result, (int(x), int(y)) , 5 , ( 0 , 0 , 255 ) , -1)
    ( tl, tr, br, bl) = box
    ( centerXX, centerYY) = midpoint( tl, br)
    (midlx, midly) = midpoint( tl, bl)
#     print(int(midly),int(midlx))
    cv2.circle(result,(int(midlx), int(midly)),15,(255,45,12),2)
    start = (int(centerXX), int( centerYY))
    count_len = COUNTLEN
    shape = cnt.shape[0]
    run = int(shape/count_len)
    if shape<count_len:
        run = 1
    LENGTH = list()
    sss = []
    previos = start
    
    for i,r in enumerate(range(0,shape,run)):
        if i<count_len:
            end = (int(cnt[r][0][0]),int(cnt[r][0][1]))
            dA = dist.euclidean(start, end)
            
            cv2.line(result, start, end,(0, 0, 0), 2)
            previos = end
            
            
            
            LENGTH.append(dA)
        else:
            break
    max_LENGTN = max(LENGTH)
    if len(LENGTH)<count_len:
        razn = count_len - len(LENGTH)
        for i in range(razn):
            LENGTH.append(0)
    if max_LENGTN<25:
        cv2.circle(result,(int(midlx), int(midly)),40,(255,45,12),2)
        return LENGTH, result, True,"var2"
#         max_LENGTN=55#80
#     max_LENGTN+=10#15
    top_crop = centerYY-max_LENGTN
    bottom_crop = centerYY+max_LENGTN
    left_crop = centerXX-max_LENGTN
    right_crop = centerXX+max_LENGTN
    if top_crop<0:
        top_crop=0
    if bottom_crop>256:
        bottom_crop=256
    if left_crop<0:
        left_crop=0
    if right_crop>256:
        right_crop=256
#     result = result[int(top_crop):int(bottom_crop),int(left_crop):int(right_crop),:]
#     result = cv2.bilateralFilter(result,9,100,100)
#     result = segmentation_color3(result)
#     print (sss)
    return LENGTH, result, False, "var0" #return length border
img = cv2.imread(train_dir+test_image_name_arr[0]+".jpg")
_,im,_,_ = get_Assimetry2(img)

plt.imshow(im)

plt.imshow(im)
train_dir = "../input/jpeg-melanoma-256x256/train/"
train = pd.read_csv("../input/jpeg-melanoma-256x256/train.csv")
col = ["length_"+str(i) for i in range(COUNTLEN)]
data = pd.DataFrame(columns = col)
test_image_name_arr = train[train.target==1]["image_name"].values
# fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(15,15), gridspec_kw={'wspace':0.1, 'hspace':0})
j=0
ran = 240
plt.figure(figsize = (15,15))
gs1 = gridspec.GridSpec(6, 6)
gs1.update(wspace=0.00, hspace=0.0)
setka = 0
for i in range(36):
    img = cv2.imread(train_dir+test_image_name_arr[ran+i]+".jpg")
    length,image_result,_,_ = get_Assimetry2(img)
    img = cv2.cvtColor(image_result,cv2.COLOR_BGR2RGB)
    data.loc[i,col] = length
    ax1 = plt.subplot(gs1[setka])
    setka+=1
    plt.axis('on')
    ax1.set_xticklabels([])
    ax1.set_yticklabels([])
    ax1.set_aspect('equal')
    ax1.imshow(img)
   
#     ax[j].imshow(img)
#     ax[j].axis("off")
    j+=1
display(data)
