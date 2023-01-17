# se carga las librerias
import numpy as np
import cv2
from matplotlib import pyplot as plt

img = cv2.imread('../input/Dulima1.jpg')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
gray = np.float32(gray)

k = 0.01 # parametro de rechazo

dst = cv2.cornerHarris(gray,2,3,k)
dst2 = cv2.dilate(dst,None)

height, width = dst.shape
color = (0, 255, 0)
temp=0

for y in range(0, height):
    for x in range(0, width):
        if dst.item(y, x) > k * dst.max():
            cv2.circle(img, (x, y), 1, color, -1)
            temp = temp+1

#print('Esquinas identificadas:', temp)

plt.figure(figsize=(20,100))
plt.subplot(1,2,1)
plt.imshow(img,cmap = 'gray')
plt.title('Esquinas con Harris')
plt.subplot(1,2,2)
plt.imshow(dst2, cmap='gray')
plt.title('imagen dilatada')
img = cv2.imread('../input/Dulima1.jpg')
gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
gray = np.float32(gray)

cornersShi = cv2.goodFeaturesToTrack(gray,200,0.05,10)
cornersShi = np.int0(cornersShi)
for i in cornersShi:
    x,y = i.ravel()
    cv2.circle(img,(x,y),2,255,-1)

plt.figure(figsize=(20,100))
plt.subplot(1,2,1)
plt.imshow(img)
plt.title('Esquinas con Shi-Tomasi')
plt.subplot(1,2,2)
plt.imshow(gray, cmap='gray')
plt.title('Imagen en escala de grises')
