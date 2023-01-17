#using 2D filters as in OpenCV

import cv2

import numpy as np

from matplotlib import pyplot as plt

img = cv2.imread('../input/Beli Vit River - Village of Ribaritsa.jpg')

kernel = np.ones((5,5),np.float32)/25

dst = cv2.filter2D(img,-1,kernel)

plt.subplot(121),plt.imshow(img),plt.title('Original')

plt.xticks([]), plt.yticks([])

plt.subplot(122),plt.imshow(dst),plt.title('Averaging')

plt.xticks([]), plt.yticks([])

plt.show()
img = cv2.imread('../input/Beli Vit River - Village of Ribaritsa.jpg')

color = ('b','g','r')

for i,col in enumerate(color):

    histr = cv2.calcHist([img],[i],None,[256],[0,256])

    plt.plot(histr,color = col)

    plt.xlim([0,256])

plt.show()


img = cv2.imread('../input/Beli Vit River - Village of Ribaritsa.jpg',0)

plt.hist(img.ravel(),256,[0,256]); plt.show()

import cv2

import numpy as np

from matplotlib import pyplot as plt

img = cv2.imread('../input/Beli Vit River - Village of Ribaritsa.jpg')

blur = cv2.blur(img,(5,5))

plt.subplot(121),plt.imshow(img),plt.title('Original')

plt.xticks([]), plt.yticks([])

plt.subplot(122),plt.imshow(blur),plt.title('Blurred')

plt.xticks([]), plt.yticks([])

plt.show()

img = blur

color = ('b','g','r')

for i,col in enumerate(color):

    histr = cv2.calcHist([img],[i],None,[256],[0,256])

    plt.plot(histr,color = col)

    plt.xlim([0,256])

plt.show()

img = blur

plt.hist(img.ravel(),256,[0,256]); plt.show()