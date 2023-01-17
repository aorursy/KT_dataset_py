# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/train_v2.csv")
tags = df["tags"].apply(lambda x: x.split(' '))
   
end = len(tags)
id_haze = []
id_cloudy = []
id_partly = []
id_clear = []

for i in range (0,end):
    for x in tags[i]:
        if x == 'haze':
            id_haze.append(i)
        elif x == 'cloudy':
            id_cloudy.append(i)
        elif x == 'partly_cloudy':
            id_partly.append(i)
        elif x == 'clear':
            id_clear.append(i)
print (len(id_haze))
print (len(id_cloudy))
print(len(id_partly))
print (len(id_clear))
import cv2
import matplotlib.pyplot as plt
import random

new_style = {'grid': True}
plt.rc('axes', **new_style)
_, ax = plt.subplots(3, 3, sharex='col', sharey='row', figsize=(12, 12))
index = []
for i in range(0,9):
 
    if i <3:
        l = random.choice(id_cloudy)
        index.append(l)
    elif (i>=3 and i<6):
        l = random.choice(id_partly)
        index.append(l)
    elif (i>=6 and i<9):
        l = random.choice(id_haze)
        index.append(l)
    
    img = cv2.imread('../input/train-jpg/train_'+str(l)+'.jpg')
    ax[i // 3, i % 3].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

plt.show()
print (index)
import cv2
import matplotlib.pyplot as plt
import numpy as np

plt.figure(figsize=(8,8))
l_cloudy = random.choice(id_cloudy)
im = cv2.imread('../input/train-jpg/train_'+str(l_cloudy)+'.jpg')
im_array = np.array(im)
var_cloudy = np.var(im_array)
im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

r = im[:,:,0]
g = im[:,:,1]
b = im[:,:,2]
    
plt.subplot(1,2,1)
plt.imshow(im)
plt.title("image_cloudy")
plt.subplot(1,2,2)
plt.hist(r.ravel(), bins=256, range=(0., 255))
plt.hist(g.ravel(), bins=256, range=(0., 255))
plt.hist(b.ravel(), bins=256, range=(0., 255))
plt.show()


print(var_cloudy) #Variance of image cloudy complete
import cv2
import matplotlib.pyplot as plt
import numpy as np

plt.figure(figsize=(8,8))
l_partly = random.choice(id_partly)
im = cv2.imread('../input/train-jpg/train_'+str(l_partly)+'.jpg')
im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
im_array = np.array(im)
var_partly = np.var(im_array)

r = im[:,:,0]
g = im[:,:,1]
b = im[:,:,2]
    
plt.subplot(1,2,1)
plt.imshow(im)
plt.title("image_partly")
plt.subplot(1,2,2)
plt.hist(r.ravel(), bins=256, range=(0., 255),color='red')
plt.hist(g.ravel(), bins=256, range=(0., 255),color='green')
plt.hist(b.ravel(), bins=256, range=(0., 255),color='blue')
plt.show()

print(var_partly)

import cv2
import matplotlib.pyplot as plt
import numpy as np

plt.figure(figsize=(8,8))
l_haze = random.choice(id_haze)
im = cv2.imread('../input/train-jpg/train_'+str(l_haze)+'.jpg')
im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
im_array = np.array(im)
var_haze = np.var(im_array)

r = im[:,:,0]
g = im[:,:,1]
b = im[:,:,2]
    
plt.subplot(1,2,1)
plt.imshow(im)
plt.title("image_haze")
plt.subplot(1,2,2)
plt.hist(r.ravel(), bins=256, range=(0., 255))
plt.hist(g.ravel(), bins=256, range=(0., 255))
plt.hist(b.ravel(), bins=256, range=(0., 255))
plt.show()
print(var_haze)
import cv2
import matplotlib.pyplot as plt
import numpy as np

plt.figure(figsize=(8,8))
l_clear = random.choice(id_clear)
im = cv2.imread('../input/train-jpg/train_'+str(l_clear)+'.jpg')
im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
im_array = np.array(im)
var_clear = np.var(im_array)

r = im[:,:,0]
g = im[:,:,1]
b = im[:,:,2]
    
plt.subplot(1,2,1)
plt.imshow(im)
plt.title("image_clear")
plt.subplot(1,2,2)
plt.hist(r.ravel(), bins=256, range=(0., 255))
plt.hist(g.ravel(), bins=256, range=(0., 255))
plt.hist(b.ravel(), bins=256, range=(0., 255))
plt.show()
print(var_clear)
# take the variance of image complete
print("variance of image cloudy complete is %d"%var_cloudy)
print("variance of image haze complete is %d"%var_haze)
print("variance of image partly complete is %d"%var_partly)
print("variance of image clear complete is %d"%var_clear)
### PART2 : PLOT IMAGE cloudy
### l = random.choice(id_cloudy)
im = cv2.imread('../input/train-jpg/train_'+str(l_cloudy)+'.jpg')
im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
plt.imshow(im)
plt.show()
### SLINDING WINDOW Image cloudy 
### l = random.choice(id_cloudy)
im = cv2.imread('../input/train-jpg/train_'+str(l_cloudy)+'.jpg')
im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)


def sliding_window(image, stepSize, windowSize):
	# slide a window across the image
	for y in range(0, image.shape[0], stepSize):
		for x in range(0, image.shape[1], stepSize):
			# yield the current window
			yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0],:])

#SIZE OF WINDOW

winW = 20
winH = 20 

# APPLY SLINDING WINDOWS
fenetre = []
for (x, y, window) in sliding_window(im, stepSize=32, windowSize=(winW, winH)):
	# if the window does not meet our desired window size, ignore it
	if window.shape[0] == winH and window.shape[1] == winW:
		fenetre.append(window)
        
L = len(fenetre)
idd = range(0,L,1)

l1 = random.choice(idd)
l2 = random.choice(idd)
l3 = random.choice(idd)
l4 = random.choice(idd)
l5 = random.choice(idd)
l6 = random.choice(idd)

# print all window
plt.subplot(2,3,1)
plt.imshow(fenetre[l1])
plt.subplot(2,3,2)
plt.imshow(fenetre[l2])
plt.subplot(2,3,3)
plt.imshow(fenetre[l3])
plt.subplot(2,3,4)
plt.imshow(fenetre[l4])
plt.subplot(2,3,5)
plt.imshow(fenetre[l5])
plt.subplot(2,3,6)
plt.imshow(fenetre[l6])

print (l1)
print (l2)
print (l3)
print (l4)
print (l5)
print (l6)

im_array1 = np.array(fenetre[l1])
im_array2 = np.array(fenetre[l2])
im_array3 = np.array(fenetre[l3])
im_array4 = np.array(fenetre[l4])
im_array5 = np.array(fenetre[l5])
im_array6 = np.array(fenetre[l6])
var1 = np.var(im_array1)
var2 = np.var(im_array2)
var3 = np.var(im_array3)
var4 = np.var(im_array4)
var5 = np.var(im_array5)
var6 = np.var(im_array6)
print(var1,var2,var3,var3,var4,var5,var6) #variance of window image
win_cloudy = [var1,var2,var3,var4,var5,var6]
var_win_cloudy = np.var(win_cloudy)
mean_var_win_cloudy = np.mean(win_cloudy)
# take id cloudy 
wind_cloudy = [l1, l2 ,l3 ,l4,l5,l6]
wind_not= [30,50,52,63]
### PART3 : PLOT IMAGE haze
### l = random.choice(id_haze)
im = cv2.imread('../input/train-jpg/train_'+str(l_haze)+'.jpg')
im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
plt.imshow(im)
plt.show()
### SLINDING WINDOW Image cloudy 
### l = random.choice(id_haze)
im = cv2.imread('../input/train-jpg/train_'+str(l_haze)+'.jpg')
im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)


def sliding_window(image, stepSize, windowSize):
	# slide a window across the image
	for y in range(0, image.shape[0], stepSize):
		for x in range(0, image.shape[1], stepSize):
			# yield the current window
			yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0],:])

#SIZE OF WINDOW

winW = 20
winH = 20 

# APPLY SLINDING WINDOWS
fenetre = []
for (x, y, window) in sliding_window(im, stepSize=32, windowSize=(winW, winH)):
	# if the window does not meet our desired window size, ignore it
	if window.shape[0] == winH and window.shape[1] == winW:
		fenetre.append(window)
        
L = len(fenetre)
idd = range(0,L,1)

l1 = random.choice(idd)
l2 = random.choice(idd)
l3 = random.choice(idd)
l4 = random.choice(idd)
l5 = random.choice(idd)
l6 = random.choice(idd)

# print all window
plt.subplot(2,3,1)
plt.imshow(fenetre[l1])
plt.subplot(2,3,2)
plt.imshow(fenetre[l2])
plt.subplot(2,3,3)
plt.imshow(fenetre[l3])
plt.subplot(2,3,4)
plt.imshow(fenetre[l4])
plt.subplot(2,3,5)
plt.imshow(fenetre[l5])
plt.subplot(2,3,6)
plt.imshow(fenetre[l6])

print (l1)
print (l2)
print (l3)
print (l4)
print (l5)
print (l6)

im_array1 = np.array(fenetre[l1])
im_array2 = np.array(fenetre[l2])
im_array3 = np.array(fenetre[l3])
im_array4 = np.array(fenetre[l4])
im_array5 = np.array(fenetre[l5])
im_array6 = np.array(fenetre[l6])
var1 = np.var(im_array1)
var2 = np.var(im_array2)
var3 = np.var(im_array3)
var4 = np.var(im_array4)
var5 = np.var(im_array5)
var6 = np.var(im_array6)
print(var1,var2,var3,var3,var4,var5,var6) #variance of window image
win_haze= [var1,var2,var3,var4,var5,var6]
var_win_haze= np.var(win_haze)
mean_var_win_haze = np.mean(win_haze)
# take id haze and not haze
wind_haze = [l1, l2 ,l3 ,l4,l5,l6]
wind_not= [30,50,52,63]
### PART4 : PLOT IMAGE clear
### l = random.choice(id_clear)
im = cv2.imread('../input/train-jpg/train_'+str(l_clear)+'.jpg')
im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
plt.imshow(im)
plt.show()
### SLINDING WINDOW Image clear
### l = random.choice(id_clear)
im = cv2.imread('../input/train-jpg/train_'+str(l_clear)+'.jpg')
im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)


def sliding_window(image, stepSize, windowSize):
	# slide a window across the image
	for y in range(0, image.shape[0], stepSize):
		for x in range(0, image.shape[1], stepSize):
			# yield the current window
			yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0],:])

#SIZE OF WINDOW

winW = 20
winH = 20 

# APPLY SLINDING WINDOWS
fenetre = []
for (x, y, window) in sliding_window(im, stepSize=32, windowSize=(winW, winH)):
	# if the window does not meet our desired window size, ignore it
	if window.shape[0] == winH and window.shape[1] == winW:
		fenetre.append(window)
        
L = len(fenetre)
idd = range(0,L,1)

l1 = random.choice(idd)
l2 = random.choice(idd)
l3 = random.choice(idd)
l4 = random.choice(idd)
l5 = random.choice(idd)
l6 = random.choice(idd)

# print all window
plt.subplot(2,3,1)
plt.imshow(fenetre[l1])
plt.subplot(2,3,2)
plt.imshow(fenetre[l2])
plt.subplot(2,3,3)
plt.imshow(fenetre[l3])
plt.subplot(2,3,4)
plt.imshow(fenetre[l4])
plt.subplot(2,3,5)
plt.imshow(fenetre[l5])
plt.subplot(2,3,6)
plt.imshow(fenetre[l6])

print (l1)
print (l2)
print (l3)
print (l4)
print (l5)
print (l6)

im_array1 = np.array(fenetre[l1])
im_array2 = np.array(fenetre[l2])
im_array3 = np.array(fenetre[l3])
im_array4 = np.array(fenetre[l4])
im_array5 = np.array(fenetre[l5])
im_array6 = np.array(fenetre[l6])
var1 = np.var(im_array1)
var2 = np.var(im_array2)
var3 = np.var(im_array3)
var4 = np.var(im_array4)
var5 = np.var(im_array5)
var6 = np.var(im_array6)
print(var1,var2,var3,var3,var4,var5,var6) #variance of window image
win_clear= [var1,var2,var3,var4,var5,var6]
var_win_clear= np.var(win_clear)
mean_var_win_clear = np.mean(win_clear)
# take id clear 
wind_clear = [11, l2 ,l3 ,l4,l5,l6]
wind_not= [30,50,52,63]
### PART5 : PLOT IMAGE partly
### l = random.choice(id_partly)
im = cv2.imread('../input/train-jpg/train_'+str(l_partly)+'.jpg')
im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
plt.imshow(im)
plt.show()
### SLINDING WINDOW Image partly
### l = random.choice(id_partly)
im = cv2.imread('../input/train-jpg/train_'+str(l_partly)+'.jpg')
im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)


def sliding_window(image, stepSize, windowSize):
	# slide a window across the image
	for y in range(0, image.shape[0], stepSize):
		for x in range(0, image.shape[1], stepSize):
			# yield the current window
			yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0],:])

#SIZE OF WINDOW

winW = 20
winH = 20 

# APPLY SLINDING WINDOWS
fenetre = []
for (x, y, window) in sliding_window(im, stepSize=32, windowSize=(winW, winH)):
	# if the window does not meet our desired window size, ignore it
	if window.shape[0] == winH and window.shape[1] == winW:
		fenetre.append(window)
        
L = len(fenetre)
idd = range(0,L,1)

l1 = random.choice(idd)
l2 = random.choice(idd)
l3 = random.choice(idd)
l4 = random.choice(idd)
l5 = random.choice(idd)
l6 = random.choice(idd)

# print all window
plt.subplot(2,3,1)
plt.imshow(fenetre[l1])
plt.subplot(2,3,2)
plt.imshow(fenetre[l2])
plt.subplot(2,3,3)
plt.imshow(fenetre[l3])
plt.subplot(2,3,4)
plt.imshow(fenetre[l4])
plt.subplot(2,3,5)
plt.imshow(fenetre[l5])
plt.subplot(2,3,6)
plt.imshow(fenetre[l6])

print (l1)
print (l2)
print (l3)
print (l4)
print (l5)
print (l6)

im_array1 = np.array(fenetre[l1])
im_array2 = np.array(fenetre[l2])
im_array3 = np.array(fenetre[l3])
im_array4 = np.array(fenetre[l4])
im_array5 = np.array(fenetre[l5])
im_array6 = np.array(fenetre[l6])
var1 = np.var(im_array1)
var2 = np.var(im_array2)
var3 = np.var(im_array3)
var4 = np.var(im_array4)
var5 = np.var(im_array5)
var6 = np.var(im_array6)
print(var1,var2,var3,var3,var4,var5,var6) #variance of window image
win_partly = [var1,var2,var3,var4,var5,var6]
var_win_partly = np.var(win_partly)
mean_var_win_partly = np.mean(win_partly)
# take id partly
wind_partly = [11, l2 ,l3 ,l4,l5,l6]
wind_not= [30,50,52,63]
# feature extractor 
from skimage.feature import local_binary_pattern
from skimage.feature import hog 
H = []
S = []
V = []
HOG = []
LBP = []
P = 8
R = 4
L = len (fenetre)
for i in range(0,L):
    window = cv2.cvtColor(fenetre[i], cv2.COLOR_BGR2HSV)
    H.append (window[:,:,0])
    S.append ( window[:,:,1])
    V.append (window[:,:,2])

for i in range(0,L):
    window = cv2.cvtColor(fenetre[i], cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(window,P,R)
    hog_ft = hog(window, orientations=9, pixels_per_cell=(4, 4), cells_per_block=(3, 3))
    LBP.append(lbp)
    HOG.append(hog_ft)
         
  # Plot result
#index cloudy, haze
id1 = random.choice(wind_cloudy)
id2 = random.choice(wind_clear)


# CLOUDY
plt.figure(figsize=(12,12))
col = 4
row = 2
plt.subplot(row,col,1)
plt.imshow(fenetre[id1])
plt.title('cloudy')
plt.subplot(row,col,2)
plt.hist(H[id1].ravel(), bins=256, range=(0., 255),color='red')
plt.hist(S[id1].ravel(), bins=256, range=(0., 255),color='green')
plt.hist(V[id1].ravel(), bins=256, range=(0., 255),color='blue')
plt.title('histo HSV')
plt.subplot(row,col,3)
plt.hist(LBP[id1].ravel(), bins=256, range=(0., 255),color='blue')
plt.title('histo LBP')
plt.subplot(row,col,4)
plt.hist(HOG[id1])
plt.title('HOG')

#haze
plt.subplot(row,col,5)
plt.imshow(fenetre[id2])
plt.title('clear')
plt.subplot(row,col,6)
plt.hist(H[id2].ravel(), bins=256, range=(0., 255),color='red')
plt.hist(S[id2].ravel(), bins=256, range=(0., 255),color='green')
plt.hist(V[id2].ravel(), bins=256, range=(0., 255),color='blue')
plt.title('histo HSV')
plt.subplot(row,col,7)
plt.hist(LBP[id2].ravel(), bins=256, range=(0., 255),color='blue')
plt.title('histo LBP')
plt.subplot(row,col,8)
plt.hist(HOG[id2])
plt.title('HOG')

plt.show()
# Plot result
#index cloudy, 
id1 = random.choice(wind_clear)
id2 = random.choice(wind_partly)

# clear
plt.figure(figsize=(12,12))
col = 4
row = 2
plt.subplot(row,col,1)
plt.imshow(fenetre[id1])
plt.title('clear')
plt.subplot(row,col,2)
plt.hist(H[id1].ravel(), bins=256, range=(0., 255),color='red')
plt.hist(S[id1].ravel(), bins=256, range=(0., 255),color='green')
plt.hist(V[id1].ravel(), bins=256, range=(0., 255),color='blue')
plt.title('histo HSV')
plt.subplot(row,col,3)
plt.hist(LBP[id1].ravel(), bins=256, range=(0., 255),color='blue')
plt.title('histo LBP')
plt.subplot(row,col,4)
plt.hist(HOG[id1])
plt.title('HOG')

# partly
plt.subplot(row,col,5)
plt.imshow(fenetre[id2])
plt.title('partly')
plt.subplot(row,col,6)
plt.hist(H[id2].ravel(), bins=256, range=(0., 255),color='red')
plt.hist(S[id2].ravel(), bins=256, range=(0., 255),color='green')
plt.hist(V[id2].ravel(), bins=256, range=(0., 255),color='blue')
plt.title('histo HSV')
plt.subplot(row,col,7)
plt.hist(LBP[id2].ravel(), bins=256, range=(0., 255),color='blue')
plt.title('histo LBP')
plt.subplot(row,col,8)
plt.hist(HOG[id2])
plt.title('HOG')

plt.show()
# take the variance of image complete
print("variance of image cloudy complete is %d"%var_cloudy)
print("variance of image haze complete is %d"%var_haze)
print("variance of image partly complete is %d"%var_partly)
print("variance of image clear complete is %d"%var_clear)

# take the variance of image window
print("MEANS of VAEIANCE OF image cloudy WINDOW  is %d"%mean_var_win_cloudy)
print("MEANS of VAEIANCE OF image haze WINDOW  is %d"%mean_var_win_haze)
print("MEANS of VAEIANCE OF image partly WINDOW  is %d"%mean_var_win_partly)
print("MEANS of VAEIANCE OF image clear WINDOW  is %d"%mean_var_win_clear)