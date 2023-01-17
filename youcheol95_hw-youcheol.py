# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages 

%pip install imutils

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



from sklearn.svm import SVC

from sklearn.model_selection import train_test_split

from imutils import paths

from glob import glob

import cv2 

import os

# Any results you write to the current directory are saved as output.
dataset_train = "../input/2019-fall-pr-project/train/train/"



def feature_vector(image, size=(32, 32)):   #이미지 resize 함수

    return cv2.resize(image, size).flatten()



print("[INFO] describing images...")

imagePaths = list(paths.list_images(dataset_train))





print(len(imagePaths))

print(imagePaths[0])



rawImages = []

labels = []



for (i, imagePath) in enumerate(imagePaths):

  # 이미지를 로드하고 클래스 레이블을 추출



    image = cv2.imread(imagePath)

    label = imagePath.split(os.path.sep)[-1].split(".")[0] #라벨 추출

 

    pixels = feature_vector(image) #이미지 resize

    

    rawImages.append(pixels)

    labels.append(label)

 

  # 이미지 1000개당 업데이트

    if i > 0 and i % 1000 == 0:

        print("[INFO] processed {}/{}".format(i, len(imagePaths)))

  

(trainImage, testImage, trainLabel, testLabel) = train_test_split(

    rawImages, labels, test_size=0.25, random_state=42) #이미지와 라벨 서브 샘플링



rawImages_subset = rawImages[:2000]

labels_subset= labels[:2000]



(subtrainImage, subtestImage, subtrainLabel, subtestLabel) = train_test_split(

    rawImages_subset, labels_subset, test_size=0.25, random_state=42)
print("[INFO] evaluating raw pixel accuracy...")



clf = SVC(kernel = 'rbf',gamma='auto', C = 10)



clf.fit(subtrainImage, subtrainLabel)
dataset_test = "../input/2019-fall-pr-project/test1/test1/"



imagePaths = list(paths.list_images(dataset_test))



rawImages = []

labels = []



for (i, imagePath) in enumerate(imagePaths):

    # 이미지를 로드하고 클래스 레이블을 추출

    # 위의 트레인 데이터와 동일하게 작성

    image = cv2.imread(imagePath)

 

    pixels = feature_vector(image) #이미지 resize

    label = imagePath.split(os.path.sep)[-1].split(".")[0]



    rawImages.append(pixels)

    labels.append(label)

 

  # 이미지 1000개당 업데이트

    if i > 0 and i % 1000 == 0:

        print("[INFO] processed {}/{}".format(i, len(imagePaths)))
from sklearn.metrics import accuracy_score

print(len(testImage),len(testLabel))

acc = clf.score(testImage, testLabel)

labels=np.array(labels)

print(labels.shape)

print("[INFO] raw pixel accuracy: {:.2f}%".format(acc * 100))
# numpy 를 Pandas 이용하여 결과 파일로 저장



import pandas as pd



testLabel=np.array(testLabel)

print(testLabel.shape)



df = pd.DataFrame([testLabel])

df = df.replace('dog',1)

df = df.replace('cat',0)



df.T.to_csv('results-yk-v2.csv',index=True, header=False)