# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



# import numpy as np # linear algebra

# import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



# import os

# for dirname, _, filenames in os.walk('/kaggle/input/'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import json

import os

import numpy as np

import pandas as pd

import torch

import torch.nn as nn

import torch.nn.functional as F

import torchvision.transforms.functional as t_F

import torchvision.models as models

import torchvision.transforms as transforms

import torch.utils.data as data

import torchvision

from torch.autograd import Variable

from torch.utils.data import Dataset

import cv2



import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import OneHotEncoder, LabelEncoder

from sklearn.metrics import accuracy_score
# set path

data_path = "/kaggle/input/deepfake-detection-challenge/test_videos"

save_model_path = "/kaggle/input/single-frame/"

meta_data = "metadata.json"



res_size = 224        # ResNet image size



# training parameters

k = 2             # number of target category

epochs = 30        # training epochs

batch_size = 32

learning_rate = 1e-3

log_interval = 10   # interval for displaying training info
class FrameDataset(Dataset):

    """Dataset Class for Loading Video"""



    def __init__(self, files, labels, num_frames, transform=None, test=False):

        """

        """

        self.files = files

        self.labels  = labels

        self.num_frames = num_frames

        self.max_num_frames = 60

        self.transform = transform

        self.test = test

        self.frame_no = num_frames

        self.face_cascade = cv2.CascadeClassifier('/kaggle/input/single-frame/haarcascade_frontalface_default.xml')



    def face_detect(self, frame):

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Resize frame of video to 1/4 size for faster face detection processing

        small_frame = cv2.resize(gray, (0, 0), fx=0.25, fy=0.25)

        # Detect the faces

        faces = self.face_cascade.detectMultiScale(small_frame, 1.1, 4)

        return faces





    def __len__(self):

        return len(self.files)





    def readVideo(self, videoFile):



        # Load the cascade



        # Open the video file

        cap = cv2.VideoCapture(videoFile)

        # cap.set(1, self.frame_no)

        # nFrames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # frames = torch.FloatTensor(self.channels, self.timeDepth, self.xSize, self.ySize)



        attempts = 0

        while attempts < self.max_num_frames:

            ret, frame = cap.read()

            attempts += 1

            if ret:

                last_good_frame = frame

                try:

                    faces = self.face_detect(frame)

                    # Face detected

                    if len(faces) > 0:

                        # Get the face, if more than two, use the whole frame

                        if len(faces) > 1:

                            break

                        x, y, w, h = faces[0] * 4

                        face_img = frame[y: y + h, x: x + w]

                        frame = torch.from_numpy(face_img)

                        # HWC2CHW

                        frame = frame.permute(2, 0, 1)

                        if self.transform is not None:

                            frame = t_F.to_pil_image(frame)

                            frame = self.transform(frame)

                            cap.release()

                            return frame

                except:

                    print("Face detection error")

            else:

                break



        frame = torch.from_numpy(last_good_frame)

        # HWC2CHW

        frame = frame.permute(2, 0, 1)

        if self.transform is not None:

            frame = t_F.to_pil_image(frame)

            frame = self.transform(frame)

        cap.release()

        return frame



    def __getitem__(self, index):



        file = self.files[index]

        X = self.readVideo(file)

        if self.test:

            y = self.labels[index]

        else:

            y = torch.LongTensor([self.labels[index]])  # (labels) LongTensor are for int64 instead of FloatTensor



        return X, y
def test(model, device, test_loader):

    # set model as testing mode

    output_file = 'submission.csv'

    if os.path.exists(output_file):

        os.remove(output_file)      

    cnn_encoder= model

    cnn_encoder.eval()



    results = {}

    with torch.no_grad():

        for X, y in test_loader:

            # distribute data to device

            X = X.to(device)

            # y = y.to(device).view(-1, )

            output = cnn_encoder(X)

            output_prob = F.softmax(output, dim=1)

            for i, item in enumerate(output_prob):

                file_name = y[i].split('/')[-1]

                #file_name = y[i] 

                prob = output_prob[i][1].item()

                results[file_name] = prob

                

    df =  pd.DataFrame([results.keys(), results.values()]).T

    df.columns = ['filename', 'label']

    df.fillna(0.5)

    df.to_csv(output_file, sep=',', index=False)

    print("Finished prediction!!!")
def get_X(data_folder, valid=False):

    X = []

    y = []

    videos = os.listdir(data_folder)

    if valid:

         with open(os.path.join(data_folder, meta_data)) as json_file:

            label_data = json.load(json_file)

    for v in videos:

        if v.endswith('mp4'):

            X.append(os.path.join(data_folder, v))

            if valid:

                if label_data[v]['label'] == 'FAKE':

                    y.append(1)

                else:

                    y.append(0)

    return X, y
# Detect devices

use_cuda = torch.cuda.is_available()                   # check if GPU exists

device = torch.device("cuda" if use_cuda else "cpu")   # use CPU or GPU



# Data loading parameters

params = {'batch_size': batch_size, 'shuffle': True, 'pin_memory': True} if use_cuda else {}



test_X, test_y = get_X(data_path)
transform = transforms.Compose([transforms.Resize([res_size, res_size]),

                                transforms.ToTensor(),

                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])



# selected_frames = np.arange(begin_frame, end_frame, skip_frame).tolist()

num_frames = 60



test_set = FrameDataset(test_X, test_X, num_frames, transform=transform, test=True)

test_loader = data.DataLoader(test_set, **params)
# Create model

model_ft = models.resnet18()

num_ftrs = model_ft.fc.in_features

model_ft.fc = nn.Linear(num_ftrs, 2)

model_ft = model_ft.to(device)



# Load model

encoder_model_path = os.path.join(save_model_path, 'cnn_encoder_epoch1.pth')

model_ft.load_state_dict(torch.load(encoder_model_path))
# Predict

test(model_ft, device, test_loader)