# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
os.listdir('./')
import cv2

import numpy as np

from typing import *

import random

import json

import os

import pickle

from matplotlib import pyplot as plt

from sklearn.svm import SVC

from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

def ImageResize(img_array: 'np.ndarray', resize: int) -> np.ndarray:

    """

    :param img_array: 输入的图片，格式为ndarray

    :param resize: 缩放大小

    :return:

    """

    return cv2.resize(img_array, (resize, resize))





def fft(img):

    """

    img 是图片矩阵

    """

    img = ImageResize(img, 1024)  # resize成400



    # 傅里叶变换

    dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)



    # 将频谱低频从左上角移动至中心位置

    dft_shift = np.fft.fftshift(dft)



    # 频谱图像双通道复数转换为0-255区间

    result = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1])+1e-5) # 加上1e-5防止除0

    return result





def azimuthalAverage(image, center=None):

    """

    Calculate the azimuthally averaged radial profile.



    image - The 2D image

    center - The [x,y] pixel coordinates used as the center. The default is

             None, which then uses the center of the image (including

             fracitonal pixels).



    """

    # Calculate the indices from the image

    y, x = np.indices(image.shape)



    if not center:

        center = np.array([(x.max() - x.min()) / 2.0, (y.max() - y.min()) / 2.0])



    r = np.hypot(x - center[0], y - center[1])



    # Get sorted radii

    ind = np.argsort(r.flat)

    r_sorted = r.flat[ind]

    i_sorted = image.flat[ind]



    # Get the integer part of the radii (bin size = 1)

    r_int = r_sorted.astype(int)



    # Find all pixels that fall within each radial bin.

    deltar = r_int[1:] - r_int[:-1]  # Assumes all radii represented

    rind = np.where(deltar)[0]  # location of changed radius

    nr = rind[1:] - rind[:-1]  # number of radius bin



    # Cumulative sum to figure out sums for each radius bin

    csim = np.cumsum(i_sorted, dtype=float)

    tbin = csim[rind[1:]] - csim[rind[:-1]]



    radial_prof = tbin / nr



    return radial_prof





def metadataReader(label_path) -> List:

    """

    数据预处理

    将meatadata里的数据转化成一个列表

    列表里面每个元素都是一个字典，形如{'video_file':'xxxx', 'label':0/1}

    label = 0 -> FAKE

    label = 1 -> REAL

    :param label_path: metadata 文件路径

    :return: record

    """

    record = []



    with open(label_path) as f:

        file = f.read()

        jsonfile = json.loads(file)

        for key, val in zip(jsonfile.keys(), jsonfile.values()):

            rec = {'video_file': key,

                   'label': 1 if val['label'] == "REAL" else 0}

            record.append(rec)



    return record





def CaptureVideoImage(

        videoFile: str,

        totalFrame=20,

) -> np.ndarray:

    """

    读取videoFile，生成一个[totalFrame, resolution, resolution, channel]形式的数组

    :param videoFile: 视频文件

    :param totalFrame: 截取总共帧数，作为一个batch

    :return: [totalFrame, resolution, resolution, channel] ndarray



    """

    result_list = []

    # cnt = 0  # 数组计数器



    vidcap = cv2.VideoCapture(videoFile)  # 视频流截图

    frame_all = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))  # 总帧数

    frame_start = random.randint(0, frame_all // 2)  # 起始帧



    frame_interval = 5

    

    if vidcap.isOpened():

        for i in range(frame_start, frame_start + totalFrame * frame_interval, frame_interval):

            vidcap.set(cv2.CAP_PROP_POS_FRAMES, i)  # set方法获取指定帧

            success, img = vidcap.read()

            if success:

                img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

                # print(img.shape)

                fft_img = fft(img)

                spectral = azimuthalAverage(fft_img)



                result_list.append(spectral)



    result_array = np.array(result_list)

    print(result_array.shape)

    result_array = np.mean(result_array, axis=0)



    return result_array

pkl_file = open('/kaggle/input/trainframe/train_reproduce.pkl', 'rb')

data = pickle.load(pkl_file)

pkl_file.close()

X_train = data["data"]

Y_train = data["label"]

for data in X_train:

    plt.plot(data)
X_train_trainsplit, X_train_testsplit, Y_train_split, Y_test_split = train_test_split(X_train, Y_train, test_size=.3)

svclassifier_r = SVC(C=120, kernel='rbf', gamma=0.001, probability=True)



svclassifier_r.fit(X_train_trainsplit, Y_train_split)



svclassifier_r.predict_proba(X_train_testsplit)



SVM = svclassifier_r.score(X_train_testsplit, Y_test_split)



print("RBF SVM SCORE IS ", SVM)
logreg = LogisticRegression(solver='liblinear', max_iter=1000)

logreg.fit(X_train_trainsplit, Y_train_split)

logreg.predict_proba(X_train_testsplit)



logregScore = logreg.score(X_train_testsplit, Y_test_split)

print("Logistic Score is ", logregScore)
# label_path = "/kaggle/input/deepfake-detection-challenge/train_sample_videos/metadata.json"

# video_dir = "/kaggle/input/deepfake-detection-challenge/train_sample_videos"

# records = metadataReader(label_path)



# pickle_dict = {'data':[], 'label':[]}

# cnt = 0

# for i in range(2):

#     for record in records:

#         video_file_name = os.path.join(video_dir + '/', record['video_file'])

#         video_file_label = record['label']

#         if video_file_label == 0:

#             randnum = random.randint(1, 100) / 100

#             if randnum > 0.25:

#                 continue

#         video_frame_data = CaptureVideoImage(video_file_name)

#         pickle_dict['data'].append(video_frame_data)

#         pickle_dict['label'].append(video_file_label)

#         cnt += 1

#         print(cnt)



# output = open('/kaggle/output/kaggle/working/train.pkl', 'wb')

# pickle.dump(pickle_dict, output)
pkl_test_file = open('/kaggle/input/test-data/test.pkl', 'rb')

test_data = pickle.load(pkl_test_file)

pkl_test_file.close()

X_test = test_data["data"]

filename = test_data["file_name"]



for data in X_test:

    plt.plot(data)

    

predict = svclassifier_r.predict_proba(X_test)

predict
max_val_index = np.argmax(predict, -1)

resList = []

for i in range(len(predict)):

    resList.append(predict[i][max_val_index[i]])



filename = np.array(filename)

data = {'filename':filename, 'label':resList}

df = pd.DataFrame(data, columns=['filename', 'label'])

df.sort_values(by='filename', inplace=True)

df
df.to_csv('submission.csv', index=False)
# simple_predict = []

# for data in X_test:

#     if data[500] > 150:

#         simple_predict.append(0.85)

#     else:

#         simple_predict.append(0.1)

        

# data = {'filename':filename, 'label':simple_predict}

# df = pd.DataFrame(data, columns=['filename', 'label'])
# sample_submit = pd.read_csv('/kaggle/input/deepfake-detection-challenge/sample_submission.csv')

# filename = list(filename)

# final_predict = []

# for submit_filename in sample_submit['filename']:

#     index = filename.index(submit_filename)

#     final_predict.append(simple_predict[index])

    

# finaldata = {'filename':sample_submit['filename'], 'label':final_predict}

# finaldf = pd.DataFrame(finaldata, columns=['filename', 'label'])
# finaldf
# try:



#     finaldf.to_csv('submission.csv', index=False)

# except:



#     finaldf.to_csv('submission.csv', index=False)



#     print("Exception writing submission.csv, wrote it again...") 
# submission = pd.read_csv('/kaggle/input/deepfake-detection-challenge/sample_submission.csv')

# submission['label'] = 0.5

# submission.set_index('filename')

# # for path in paths:

# #     try:

# #         # load video and predict

# #         submission.loc[path] = prediction

# #     except:

# #         pass

# submission.to_csv('submission.csv', index=True)