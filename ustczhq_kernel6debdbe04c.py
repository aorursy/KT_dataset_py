import os, sys, time

import cv2

import numpy as np

import pandas as pd

import torch

import torch.nn as nn

import torch.nn.functional as F
test_dir = "/kaggle/input/deepfake-detection-challenge/test_videos/"



test_videos = sorted([x for x in os.listdir(test_dir) if x[-4:] == ".mp4"])
gpu = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
import sys

sys.path.insert(0, "/kaggle/input/xxxxxx/pytorchretinaface")

sys.path.insert(0, "/kaggle/input/zzzzzz/mymodel")
from __future__ import print_function

import os

import argparse

import torch

import torch.backends.cudnn as cudnn

import numpy as np

from PIL import Image

from skimage import transform as trans

from data import cfg_mnet, cfg_re50

from layers.functions.prior_box import PriorBox

from utils.nms.py_cpu_nms import py_cpu_nms

import cv2

from models.retinaface import RetinaFace

from utils.box_utils import decode, decode_landm

import time



arcface_src = np.array([

  [122.5, 141.25],

  [197.5, 141.25],

  [160., 178.75],

  [137.5, 225.25],

  [182.5, 225.25] ], dtype=np.float32 ) # Ziyu

arcface_src = np.expand_dims(arcface_src, axis=0)



def estimate_norm(lmk, image_size = 112, mode='arcface'):

    assert lmk.shape==(5,2)

    tform = trans.SimilarityTransform()

    lmk_tran = np.insert(lmk, 2, values=np.ones(5), axis=1)

    min_M = []

    min_index = []

    min_error = float('inf') 

    if mode=='arcface':

        src = arcface_src

    else:

        src = src_map[image_size]

    for i in np.arange(src.shape[0]):

        tform.estimate(lmk, src[i])

    M = tform.params[0:2,:]

    results = np.dot(M, lmk_tran.T)

    results = results.T

    error = np.sum(np.sqrt(np.sum((results - src[i]) ** 2,axis=1)))

#         print(error)

    if error< min_error:

        min_error = error

        min_M = M

        min_index = i

    return min_M, min_index



def norm_crop(img, landmark, image_size=112, mode='arcface'):

    M, pose_index = estimate_norm(landmark, image_size, mode)

    warped = cv2.warpAffine(img,M, (image_size, image_size), borderValue = 0.0)

    return warped



def check_keys(model, pretrained_state_dict):

    ckpt_keys = set(pretrained_state_dict.keys())

    model_keys = set(model.state_dict().keys())

    used_pretrained_keys = model_keys & ckpt_keys

    unused_pretrained_keys = ckpt_keys - model_keys

    missing_keys = model_keys - ckpt_keys

    print('Missing keys:{}'.format(len(missing_keys)))

    print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))

    print('Used keys:{}'.format(len(used_pretrained_keys)))

    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'

    return True





def remove_prefix(state_dict, prefix):

    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''

    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x

    return {f(key): value for key, value in state_dict.items()}





def load_model(model, pretrained_path, load_to_cpu):

    print('Loading pretrained model from {}'.format(pretrained_path))

    if load_to_cpu:

        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)

    else:

        device = torch.cuda.current_device()

        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))

    if "state_dict" in pretrained_dict.keys():

        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')

    else:

        pretrained_dict = remove_prefix(pretrained_dict, 'module.')

    check_keys(model, pretrained_dict)

    model.load_state_dict(pretrained_dict, strict=False)

    return model



torch.set_grad_enabled(False)

cfg = cfg_re50

cfg['pretrain']=False

# net and model

!cp /kaggle/input/xxxxxx/pytorchretinaface/weights ./ -rf

net = RetinaFace(cfg=cfg, phase = 'test')

net = load_model(net, './weights/Resnet50_Final.pth', False)

net.eval()

print('Finished loading model!')

cudnn.benchmark = True

device = torch.device("cuda")

net = net.to(device)
from torchvision import transforms

from model import WSDAN

from util import  batch_augment

tts=transforms.Compose([

            transforms.ToTensor(),

            transforms.Normalize(mean=[0.4479, 0.3744, 0.3473],std=[0.2537, 0.2502, 0.2424])

        ])





def extract_frames(data_path, method='cv2'):

    """Method to extract frames, either with ffmpeg or opencv. FFmpeg won't

    start from 0 so we would have to rename if we want to keep the filenames

    coherent."""

    data_path=test_dir+data_path

    if method == 'cv2':

        reader = cv2.VideoCapture(data_path)

        outputbuff=[]

        frames = 0

        count = 0

        resize=1

        while reader.isOpened():

            success, img = reader.read()

            img_raw = img

            if not success:

                break

            



            frames +=1

            if frames==1:

                im_height, im_width, _ = img.shape

                scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])

                scale = scale.to(device)

                priorbox = PriorBox(cfg, image_size=(im_height, im_width))

                priors = priorbox.forward()

                priors = priors.to(device)

                prior_data = priors.data

                scale1 = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0],

                               img.shape[1], img.shape[0], img.shape[1], img.shape[0],

                               img.shape[1], img.shape[0]])

                scale1 = scale1.to(device)

            if frames%10==0:

                img = img.astype(np.int8)

                img -= (104, 117, 123)

                img = img.transpose(2, 0, 1)

                img = torch.from_numpy(img).unsqueeze(0)

                img=img.to(device,dtype=torch.float32)

                loc, conf, landms = net(img)

                boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance'])

                boxes = boxes * scale / resize

                boxes = boxes.cpu().numpy()

                scores = conf.squeeze(0).data.cpu().numpy()[:, 1]

                landms = decode_landm(landms.data.squeeze(0), prior_data, cfg['variance'])

                landms = landms * scale1 / resize

                landms = landms.cpu().numpy()

                inds = np.where(scores > 0.8)[0]

                if inds.shape[0]==0:

                    continue

                boxes = boxes[inds]

                landms = landms[inds]

                scores = scores[inds]

                areas = scores

                for it in range(areas.shape[0]):

                    areas[it] = (boxes[it][3]-boxes[it][1])*(boxes[it][2]-boxes[it][0])

                order = areas.argsort()[::-1][:1]

                boxes = boxes[order]

                landms = landms[order]

                scores = scores[order]

                landmarks = landms.reshape(5,2).astype(np.int)

                img=norm_crop(img_raw,landmarks,image_size=320)

                aligned=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

                outputbuff.append(aligned)

                count+=1

                if count==20:

                    break



        reader.release()

        return outputbuff



def predict_on_video_set(video_paths,model,model2):

    predictions = []

    for num in range(len(video_paths)):

        try:

            stime=time.time()

            frames=extract_frames( video_paths[num])

            frames=torch.cat([tts(i).unsqueeze(0) for i in  frames])

            images=frames.view(-1,3,320,320).cuda()

            print(time.time()-stime)

            y_pred_raw, _,_ = model(images)

            logits=torch.mean(F.softmax(y_pred_raw,dim=1),dim=0)

            pred=logits[1].item()

            images_b=F.interpolate(images,size=300,mode='bilinear')

            y_pred_raw2, _,_ = model2(images_b)

            logits2=torch.mean(F.softmax(y_pred_raw2,dim=1),dim=0)

            pred2=logits2[1].item()

            pred=(pred*0.7+pred2*0.3)

            if pred>0.99:

                pred=0.99

            if pred<0.01:

                pred=0.01

            predictions.append(pred)

            print(time.time()-stime,pred)

        except Exception as e:

            print(e)

            predictions.append(0.5)            

    return predictions
modelx=WSDAN(num_classes=2, M=8, net='xception', pretrained=False).cuda()

modelx.load_state_dict(torch.load('/kaggle/input/zzzzzz/mymodel/ckpt_x.pth')['state_dict'])

modely=WSDAN(num_classes=2, M=8, net='efficientnet', pretrained=False).cuda()

modely.load_state_dict(torch.load('/kaggle/input/zzzzzz/mymodel/ckpt_e.pth')['state_dict'])

modelx.eval()

modely.eval()

predictions = predict_on_video_set(test_videos,modelx,modely)

submission_df = pd.DataFrame({"filename": test_videos, "label": predictions})

submission_df.to_csv("submission.csv", index=False)