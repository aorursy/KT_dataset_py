#Importing the Numpy Package
import numpy as np
#Array with just zeros
#zeros(shape, dtype=float, order='C')
a = np.zeros(5)
a
#Type of the Numpy object
type(a)
#Type of the individual array element
type(a[0])
#shape of the array
a.shape
#We can change the dimension with shape
a.shape = (5,1)
a
#Array with ones
b = np.ones(10)
b
#Return a new array of given shape and type, without initializing entries. Almost similar to np.zeros
c = np.empty(3)
c
#from 10 to 50 with 5 elements. Helpful in plots
d = np.linspace(10,50,5) 
d
#Defining array with our elements
e = np.array([10,20])
e
# Returns number of dimensions of the array
e.ndim
#Two Dimensional Array
list1 = [[1,2,3,4],[5,6,7,8,]]
f = np.array(list1)
f
# Returns number of dimensions of the array
f.ndim
#Total number of elements
f.size
#3 dimensional array
b = np.array([[[1,2,3,4],[5,6,7,8],[11,12,13,14]],[[15,16,17,18],[2,3,4,5],[7,8,9,0]]])
b
b.ndim
#Array filled with described value
np.full((3,3),99,dtype='float16')
# array with random numbers in a range
#randint(low, high=None, size=None, dtype='l')
np.random.seed(0)
x = np.random.randint(10,size=6)
x
# array with random numbers in a range
#randint(low, high=None, size=None, dtype='l')
np.random.seed(0)
x = np.random.randint(5,10,size=6)
x
x[0]
x[:-1]
x[-1]
#Example of Broadcasting with single element getting added to every other element of the first list
x[:-1]+x[-1]
a = [5,0,3,3,7,9]
a[-1]
a[:-1]
#We cannot add an elementary object with a list. Hence we will pass the element in a list
#List addition just results in appending the values and no broadcasting similar to Numpy ndarray
a[:-1] + [a[-1]]
list1 = [1,2,3,4,5]
list1 + [6]
array1 = np.array([1,2,3,4,5])
array1 + 6
array1 = np.array([1,2,3,4,5])
array2 = np.array([6])
array1 + array2
print(np.sum(array1))
print(np.prod(array1))
print(np.mean(array1))
print(np.std(array1))
print(np.var(array1))
print(np.min(array1))
print(np.max(array1))
print(np.argmin(array1))
print(np.argmax(array1))
#Matrix Multiplication
a = np.ones((2,3))
print(a)

b = np.full((3,2),2)
print(b)

#a*b 
# Trying to multiply two arrays like above will result in error.Hence use Matmul function
np.matmul(a,b)
#Identitiy matrix
c = np.identity(3)
c
#determinant
np.linalg.det(c)
#Dot operation
array1 = np.array([1,2,3,4,5],dtype='int8')
array2 = np.ones(5,dtype='int8')
array1 @ array2
z = np.array([1,2,3,4,5])
z
#Below operation gives us Boolean output with condition TRUE or FALSE
z<3
#We can provie the above condition within and filter out the values accordingly
z[z<3]
#STEP1 -> Creating full 5*5 matrix filled with Ones
a = np.ones((5,5),dtype='int8')
a
#STEP2 -> Creating a 3*3 matrix with zeroes which can fit in our pattern
z = np.zeros((3,3))
z
#Replacing the centre element
z[1,1]=5
z
#Creating the pattern with replacing the elements by Slicing
a[1:4,1:4] = z
a
#Load data from a text file
#data.txt contains values in a comma seperated fashion
#We can pullin those values with genfromtxt function
data1 = np.genfromtxt('../input/data-for-intro-to-numpy-kernel/data.txt',delimiter=',')
data1 = data1.astype('int32')
data1
#pip install skimage if it is not installed in your machine
from skimage import io
#Reading an image present in my Local computer
photo = io.imread("../input/data-for-intro-to-numpy-kernel/cards.jpg")
#pip install matplotlib if it is not installed in your machine
#Plotting the image read in the previous cell
import matplotlib.pyplot as plt
plt.imshow(photo)
#Cropping out a part of the image by slicing 
#You can provide a range for the rows and columns inorder to Slice the image
photo2 = photo[200:500,100:400]
plt.imshow(photo2)
#We can clearly see the image here is nothing but a nd array. Hence we can very well utilise Numpy package for operations.
type(photo)
#pip install opencv-python if it is not already present
#cv stands for computer vision
import cv2
img1 = cv2.imread("../input/data-for-intro-to-numpy-kernel/cards.jpg")
cv2.imshow("Output window",img1)
cv2.waitKey(0)
#Again we can see that the type of the image is numpy array
type(img1)
#Cropping a part of the iamge
img2 = img1[250:400]
cv2.imshow("Output window cropped",img2)
cv2.waitKey(0)
#Shape of the Image
img1.shape
#Creating an RGB image of shape 512 * 512
new_img = np.zeros((512,512,3))
new_img.shape
#We can see that we have created a RGB image now
cv2.imshow("new window",new_img)
cv2.waitKey(0)
new_img[:] = 0,255,0
#For OpenCV colors are in the format BGR. Hence 0,255,0 represents color Green ON
cv2.imshow("new window",new_img)
cv2.waitKey(0)
