import matplotlib.pyplot as plt

import skimage

import skimage.io

import scipy.misc

import imageio
original_img = skimage.io.imread("../input/train/iPhone-6/(iP6)2.jpg")

print(original_img.shape)
plt.figure(figsize=(6,6))

plt.imshow(original_img)

plt.title("original")

plt.show()
new_img = scipy.misc.imresize(original_img, 0.5, interp='bicubic')

print(new_img.shape)
plt.figure(figsize=(3,3))

plt.imshow(new_img)

plt.title("resize 0.5")

plt.show()
new_img = scipy.misc.imresize(original_img, 2.0, interp='bicubic')

print(new_img.shape)
plt.figure(figsize=(12,12))

plt.imshow(new_img)

plt.title("resize 2.0")

plt.show()
new_img = skimage.exposure.adjust_gamma(original_img, gamma=0.8)



plt.figure(figsize=(6,6))

plt.imshow(new_img)

plt.title("gamma 0.8")

plt.show()
new_img = skimage.exposure.adjust_gamma(original_img, gamma=1.2)



plt.figure(figsize=(6,6))

plt.imshow(new_img)

plt.title("gamma 1.2")

plt.show()
imageio.imwrite('quality-70.jpg', original_img, quality=70)



new_img = skimage.io.imread('quality-70.jpg')

plt.figure(figsize=(6,6))

plt.imshow(new_img)

plt.title("quality-70")

plt.show()
imageio.imwrite('quality-90.jpg', original_img, quality=90)



new_img = skimage.io.imread('quality-90.jpg')

plt.figure(figsize=(6,6))

plt.imshow(new_img)

plt.title("quality-90")

plt.show()
imageio.imwrite('quality-5.jpg', original_img, quality=5)



new_img = skimage.io.imread('quality-5.jpg')

plt.figure(figsize=(6,6))

plt.imshow(new_img)

plt.title("quality-5")

plt.show()