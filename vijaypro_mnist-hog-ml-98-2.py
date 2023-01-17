# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import glob
import cv2
import sklearn
import tqdm
import random
import matplotlib.pyplot as plt
import seaborn as sns
os.chdir('/kaggle/input')
# Deskewing images
def deskew(img):
    m = cv2.moments(img)
    SZ=28 # Size of the image
    if abs(m['mu02']) < 1e-2: 
        return img.copy()
    skew = m['mu11']/m['mu02']
    M = np.float32([[1, skew, -0.5*SZ*skew], [0, 1, 0]])
    img = cv2.warpAffine(img, M, (SZ, SZ), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)
    return img
# Test "deskew" function on a Sample Image
skew_img=cv2.imread("mnistasjpg/testSet/testSet/img_10001.jpg",0)
deskew_img=deskew(skew_img)
fig=plt.figure(figsize=(8,8))
columns=1
rows=2
fig.add_subplot(columns, rows, 1)
plt.imshow(skew_img,cmap='gray')
plt.title("Skewed Image")
fig.add_subplot(columns, rows, 2)
plt.imshow(deskew_img,cmap='gray')
plt.title("De-Skewed Image")
plt.show()
# Computer HOG descriptor for all the images.
def HistOfGrad(img):
    img=deskew(img)
    winSize = (28,28)
    blockSize = (14,14)
    blockStride = (7,7)
    cellSize = (14,14)
    nbins = 9
    derivAperture = 1
    winSigma = -1.
    histogramNormType = 0
    L2HysThreshold = 0.2
    gammaCorrection = 1
    nlevels = 64
    signedGradients = True
    hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,
                            cellSize,nbins,derivAperture,winSigma,
                            histogramNormType,L2HysThreshold,
                            gammaCorrection,nlevels, signedGradients)
    descriptor = hog.compute(img)
    return descriptor
# Test the model on 12 random test images
def test_on_random_images(model=None):
    testing_files=glob.glob('mnistasjpg/testSet/testSet/*.jpg')
    test_images=random.choices(testing_files,k=12)
    fig=plt.figure(figsize=(15,15))
    columns=3
    rows=4
    for i in range(len(test_images)):
        img=cv2.imread(test_images[i],0)
        fig.add_subplot(columns, rows, i + 1)
        if model:
            desc=HistOfGrad(img)
            desc=np.array(desc)
            desc=np.resize(desc,(desc.shape[1],desc.shape[0]))
            pred=model.predict(desc)
            text="The Predicted Number is: \n"+ str(pred[0])
        else:
            text=""
        img=cv2.resize(img,(224,224)) # Resizing for displaying
        plt.text(100,-10, text, size=12, ha="center")
        plt.imshow(img,cmap='gray')
    plt.show()
# Test the "test_on_random_images" function
# Running the function each time gives a different set of images
test_on_random_images()
folders=glob.glob('mnistasjpg/trainingSet/trainingSet/*')
X=[]
y=[]
count=0
for i in tqdm.tqdm(folders):
    count=int(os.path.basename(i))
    images=glob.glob(i+"/*.jpg")
    for j in images:
        image=cv2.imread(j,0)
        desc=HistOfGrad(image)
        X.append(desc)
        y.append(count)
assert len(X)==len(y)
# Visualizing Number of Labels
plt.figure(figsize=(10,10))
sns.countplot(y)
# Importing Required Library
from sklearn import datasets 
from sklearn.metrics import confusion_matrix 
from sklearn.model_selection import train_test_split
# Train and Test Splits
X=np.array(X)
y=np.array(y)
X=np.reshape(X,(X.shape[0],X.shape[1]))
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0) 
# Support Vector Machine
from sklearn.svm import SVC
def SVM(X_train, X_test, y_train, y_test):
    svm = SVC(kernel = 'poly', C = 1).fit(X_train, y_train)
    svm_predictions = svm.predict(X_test)
    accuracy = svm.score(X_test, y_test)
    cm = confusion_matrix(y_test, svm_predictions)
    print(accuracy)
    print(cm)
    return svm
model=SVM(X_train, X_test, y_train, y_test)
test_on_random_images(model)
# K-Nearest Neighbours
from sklearn.neighbors import KNeighborsClassifier 
def KNN(X_train, X_test, y_train, y_test):
    knn = KNeighborsClassifier(n_neighbors = 7).fit(X_train, y_train)
    accuracy = knn.score(X_test, y_test)
    knn_predictions=knn.predict(X_test)
    cm=confusion_matrix(knn_predictions,y_test)
    print(accuracy)
    print(cm)
    return knn
model=KNN(X_train, X_test, y_train, y_test)
test_on_random_images(model)
# Decision Tree
from sklearn.tree import DecisionTreeClassifier
def DecisionTree(X_train, X_test, y_train, y_test):
    dtree = DecisionTreeClassifier(max_depth = 25).fit(X_train, y_train)
    dtree_predictions=dtree.predict(X_test)
    accuracy=dtree.score(X_test,y_test)
    cm=confusion_matrix(y_test,dtree_predictions)
    print(accuracy)
    print(cm)
    return dtree
model=DecisionTree(X_train, X_test, y_train, y_test)
test_on_random_images(model)
# Neural Network (Multi Layer Perceptron)
from sklearn.neural_network import MLPClassifier
def NeuralNetwork(X_train, X_test, y_train, y_test):
    nn = MLPClassifier(random_state=1, max_iter=300, learning_rate='adaptive').fit(X_train, y_train)
    accuracy = nn.score(X_test, y_test)
    nn_predictions = nn.predict(X_test)
    cm=confusion_matrix(nn_predictions,y_test)
    print(accuracy)
    print(cm)
    return nn
model=NeuralNetwork(X_train, X_test, y_train, y_test)
test_on_random_images(model)