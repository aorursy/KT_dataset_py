# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import matplotlib.pyplot as plt

import cv2

import pywt

import pywt.data

import os

from sklearn.neural_network import MLPClassifier

from sklearn.model_selection import train_test_split

from sklearn.utils import shuffle

from sklearn.preprocessing import MaxAbsScaler

from sklearn.neighbors import KNeighborsClassifier

from sklearn.decomposition import PCA

from sklearn.svm import SVC

from sklearn import metrics

from matplotlib.colors import ListedColormap

from sklearn.preprocessing import StandardScaler

from sklearn.datasets import make_moons, make_circles, make_classification

from sklearn.neural_network import MLPClassifier

from sklearn.gaussian_process import GaussianProcessClassifier

from sklearn.gaussian_process.kernels import RBF

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
IMG_SIZE = 128

db1 = pywt.Wavelet('db1')

db2 = pywt.Wavelet('db2')

level = pywt.dwt_max_level(IMG_SIZE, db2)

print('Level :',level)
N = 17792

M = N//20
M = 500
randM = np.random.randint(2, size=(M, N))
def encryptionPhi(ls, randM):

    return np.dot(randM,ls)
data_path = '../input/modified-jaffe-facial-expression-dataset/jaffe/jaffe'

data_dir_list = os.listdir(data_path)



img_data_list=[]



for dataset in np.sort(data_dir_list):

    img_list=os.listdir(data_path+'/'+ dataset)

    print ('Loaded the images of dataset-'+'{}\n'.format(dataset))

    for img in img_list:

        input_img=cv2.imread(data_path + '/'+ dataset + '/'+ img )

        input_img=cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)

        input_img_resize=cv2.resize(input_img,(IMG_SIZE,IMG_SIZE))

        coeffs = pywt.wavedec(input_img_resize, db2, mode='constant', level=level)

        ls = []

        for array in coeffs:

            ls = ls + array.flatten().tolist()

        ls = np.array(ls)

        img = encryptionPhi(ls, randM)

        img_data_list.append(img)

        

img_data = np.array(img_data_list)

#img_data = img_data.astype('float32')



#img_data = np.expand_dims(img_data, axis=1)

img_data.shape
#img_data[0]
'''

num_classes = 7



num_of_samples = img_data.shape[0]

labels = np.ones((num_of_samples,),dtype='int64')



labels[0:29]=0 #30

labels[30:59]=1 #30

labels[60:88]=2 #29

labels[89:120]=3 #32

labels[121:151]=4 #31

labels[152:182]=5 #31

labels[183:]=6 #30



names = ['ANGRY','DISGUST','FEAR','HAPPY','NEUTRAL','SAD','SURPRISE']



def getLabel(id):

    return ['ANGRY','DISGUST','FEAR','HAPPY','NEUTRAL','SAD','SURPRISE'][id]

'''
num_classes = 3 #positive,negative,neutral



num_of_samples = img_data.shape[0]

labels = np.ones((num_of_samples,),dtype='int64')



labels[0:29]=0 #30

labels[30:59]=0 #30

labels[60:88]=0 #29

labels[89:120]=2 #32

labels[121:151]=1 #31

labels[152:182]=1 #31

labels[183:]=2 #30



names = ['NEGATIVE','NEUTRAL','POSITIVE']



def getLabel(id):

    return ['NEGATIVE','NEUTRAL','POSITIVE'][id]
np.unique(labels)
'''

X = img_data

#X = MaxAbsScaler().fit_transform(X)

#Shuffle the dataset



#pca=PCA(n_components=2)

#pca.fit(X)



x,y = shuffle(X,labels)

# Split the dataset

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.15, random_state=2)

x_test=X_test

'''
#from sklearn.decomposition import PCA

#pca=PCA(n_components=2)

#pca.fit(X)
'''

Y = img_data

pca=PCA()

pca.fit(Y)



plt.figure(1, figsize=(12,8))



plt.plot(pca.explained_variance_, linewidth=2)

 

plt.xlabel('Components')

plt.ylabel('Explained Variaces')

plt.show()

'''
names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Gaussian Process",

         "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",

         "Naive Bayes", "QDA"]



classifiers = [

    KNeighborsClassifier(3),

    SVC(kernel="linear", C=0.025),

    SVC(gamma=2, C=1),

    GaussianProcessClassifier(1.0 * RBF(1.0)),

    DecisionTreeClassifier(max_depth=5),

    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),

    MLPClassifier(alpha=1, max_iter=1000),

    AdaBoostClassifier(),

    GaussianNB(),

    QuadraticDiscriminantAnalysis()]
# preprocess dataset, split into training and test part

X, y = shuffle(img_data,labels)

X = StandardScaler().fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42)





# iterate over classifiers

for name, clf in zip(names, classifiers):

    clf.fit(X_train, y_train)

    score = clf.score(X_test, y_test)

    print(name+":",score)
from sklearn.model_selection import KFold

mlp = MLPClassifier(max_iter=1000, alpha=1)

k_fold = KFold(5)

avg_score = 0

for k, (train, test) in enumerate(k_fold.split(X, y)):

    mlp.fit(X[train], y[train])

    score = mlp.score(X[test], y[test])

    print('Fold :',k)

    print('Score :',score)

    avg_score+=score



print('Average_Score : ', avg_score/5)
mlp = MLPClassifier(max_iter=1000, alpha=1)



mlp.fit(X_train, y_train)

print("Training set score: %f" % mlp.score(X_train, y_train))

print("Test set score: %f" % mlp.score(X_test, y_test))

print(X_train.shape)
import pickle 

  

# Save the trained model as a pickle string. 

saved_model = pickle.dumps(mlp) 

  

# Load the pickled model 

#knn_from_pickle = pickle.loads(saved_model) 

  

# Use the loaded pickled model to make predictions 

#knn_from_pickle.predict(X_test)
from sklearn.externals import joblib 

  

# Save the model as a pickle in a file 

joblib.dump(mlp, 'mlp.pkl')

# Load the model from the file 

#knn_from_joblib = joblib.load('filename.pkl')  

# Use the loaded model to make predictions 

#knn_from_joblib.predict(X_test) 