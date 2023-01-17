%matplotlib inline

from matplotlib import pyplot as plt

import matplotlib.gridspec as gridspec

from matplotlib.patches import Rectangle



import os

import json

import time

import numpy as np

from PIL import Image

import torch

from tqdm import tqdm

import cv2

import sys



# sys.path.append('/kaggle/input/retinaface/RetinaFace')

# from retinaface import RetinaFace



# sys.path.append('/kaggle/input/yolov2face')

# from yolov2 import load_mobilenetv2_224_075_detector, FaceDetector_yolo, get_boxes_points



# sys.path.append('/kaggle/input/s3fdface/s3fd')

# from detection.sfd import FaceDetector



sys.path.append('/kaggle/input/retinafacetorch')

from retina import retinaface_model, detect_images
device = 'cuda' if torch.cuda.is_available() else 'cpu'

retinaface_model = retinaface_model(model_path='/kaggle/input/retinafacetorch/Resnet50_Final.pth',device=device)
video_path = '/kaggle/input/deepfake-detection-challenge/train_sample_videos/'
# n_frames = 10

# for vi in os.listdir(video_path):

#     start = time.time()

#     imgs = []

    

#     cap = cv2.VideoCapture(os.path.join(video_path, vi))

#     v_len = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

#     video_length = 0

#     for j in range(v_len):

#         success = cap.grab()

#         if success:

#             video_length += 1

#         else:

#             break

#     cap.release()



#     sample = np.linspace(0, video_length-1, n_frames).astype(int)

#     cap = cv2.VideoCapture(os.path.join(video_path, vi))

#     for j in range(video_length):

#         succ, image = cap.read()

#         if j in sample and succ:

#             # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

#             imgs.append(np.float32(image))

#         if len(imgs) == n_frames:

#             break

#     if len(imgs) != 0:

#         detect_images(imgs=imgs, net=retinaface_model, thresh=0.94, device=device)

#         print(time.time()-start)
def show_sequence(sequence, num_frames):

    columns = 3

    rows = (num_frames + 1) // (columns)

    fig = plt.figure(figsize = (32,(16 // columns) * rows))

    gs = gridspec.GridSpec(rows, columns)

    for j in range(rows*columns):

        plt.subplot(gs[j])

        plt.axis("off")

        plt.imshow(sequence[j])



sample_video = '/kaggle/input/deepfake-detection-challenge/train_sample_videos/apatcsqejh.mp4'
imgs = []

cap = cv2.VideoCapture(sample_video)

video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

sample = np.linspace(0, video_length-1, 9).astype(int)

for j in range(video_length):

    success = cap.grab()

    if j in sample:

        success, image = cap.retrieve()

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if not success:

            continue

        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        imgs.append(image)

cap.release()

bboxes = detect_images(imgs=[np.float32(img) for img in imgs], net=retinaface_model, thresh=0.94, device=device)

pyretina_final_list = []

red = (255,0,0)

for i in range(len(imgs)):

    for b in bboxes[i]:

        lx, ly, rx, ry = b[0], b[1], b[2], b[3]

        cv2.rectangle(imgs[i], (int(round(lx)),int(round(ly))), (int(round(rx)), int(round(ry))), red, 2)

    pyretina_final_list.append(imgs[i])

show_sequence(pyretina_final_list, 9)
# retina_detector = RetinaFace('/kaggle/input/retinaface/RetinaFace/models/R50', 0, 0, 'net3')

# for vi in os.listdir(video_path):

#     start = time.time()

#     imgs = []

#     cap = cv2.VideoCapture(os.path.join(video_path, vi))

#     video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

#     sample = np.linspace(0, video_length-1, 10).astype(int)

#     for j in range(video_length):

#         success = cap.grab()

#         if j in sample:

#             success, image = cap.retrieve()

#             if not success:

#                 continue

#             # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

#             imgs.append(image)

#     cap.release()

#     thresh = 0.95

#     scales = [1024, 1980]

#     im_shape = imgs[0].shape

#     target_size = scales[0]

#     max_size = scales[1]

#     im_size_min = np.min(im_shape[0:2])

#     im_size_max = np.max(im_shape[0:2])

#     #im_scale = 1.0

#     #if im_size_min>target_size or im_size_max>max_size:

#     im_scale = float(target_size) / float(im_size_min)

#     # prevent bigger axis from being more than max_size:

#     if np.round(im_scale * im_size_max) > max_size:

#         im_scale = float(max_size) / float(im_size_max)

#     scales = [im_scale]

#     flip = False

#     faces, _ = retina_detector.detect(imgs, thresh, scales=scales, do_flip=flip)

#     print(len(faces))

#     print(type(faces))

#     print(faces)

#     # print(faces)

#     # print(time.time()-start)
# device = 'cuda' if torch.cuda.is_available() else 'cpu'

# sfd_detector = FaceDetector(device=device, path_to_detector='/kaggle/input/s3fdface/s3fd/s3fd-619a316812.pth', verbose=False)

# for vi in os.listdir(video_path):

#     start = time.time()

#     imgs = []

#     cap = cv2.VideoCapture(os.path.join(video_path, vi))

#     video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

#     sample = np.linspace(0, video_length-1, 10).astype(int)

#     for j in range(video_length):

#         success = cap.grab()

#         if j in sample:

#             success, image = cap.retrieve()

#             image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

#             if not success:

#                 continue

#             # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

#             imgs.append(image)

#     cap.release()

#     for im in imgs:

#         detected_faces = sfd_detector.detect_from_image(im, rgb=True)

#         # print(faces)

#     print(time.time()-start)

#     break
# mobilenetv2 = load_mobilenetv2_224_075_detector("/kaggle/input/yolov2face/facedetection-mobilenetv2-size224-alpha0.75.h5")

# yolo_model = FaceDetector_yolo(mobilenetv2)

# for vi in os.listdir(video_path):

#     start = time.time()

#     imgs = []

#     cap = cv2.VideoCapture(os.path.join(video_path, vi))

#     video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

#     sample = np.linspace(0, video_length-1, 10).astype(int)

#     for j in range(video_length):

#         success = cap.grab()

#         if j in sample:

#             success, image = cap.retrieve()

#             image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

#             if not success:

#                 continue

#             # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

#             imgs.append(image)

#     cap.release()

#     for im in imgs:

#         yolo_boxes = yolo_model.detect(im, 0.9)

#         # print(faces)

#     print(time.time()-start)

#     break
# def show_sequence(sequence, num_frames):

#     columns = 3

#     rows = (num_frames + 1) // (columns)

#     fig = plt.figure(figsize = (32,(16 // columns) * rows))

#     gs = gridspec.GridSpec(rows, columns)

#     for j in range(rows*columns):

#         plt.subplot(gs[j])

#         plt.axis("off")

#         plt.imshow(sequence[j])



# sample_video = '/kaggle/input/deepfake-detection-challenge/train_sample_videos/apatcsqejh.mp4'
# imgs = []

# cap = cv2.VideoCapture(sample_video)

# video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# sample = np.linspace(0, video_length-1, 9).astype(int)

# for j in range(video_length):

#     success = cap.grab()

#     if j in sample:

#         success, image = cap.retrieve()

#         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

#         if not success:

#             continue

#         # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

#         imgs.append(image)

# cap.release()
# retina_detector = RetinaFace('/kaggle/input/retinaface/RetinaFace/models/R50', 0, -1, 'net3')

# thresh = 0.94

# scales = [1024, 1980]

# im_shape = imgs[0].shape

# target_size = scales[0]

# max_size = scales[1]

# im_size_min = np.min(im_shape[0:2])

# im_size_max = np.max(im_shape[0:2])

# #im_scale = 1.0

# #if im_size_min>target_size or im_size_max>max_size:

# im_scale = float(target_size) / float(im_size_min)

# # prevent bigger axis from being more than max_size:

# if np.round(im_scale * im_size_max) > max_size:

#     im_scale = float(max_size) / float(im_size_max)

# scales = [im_scale]

# flip = False

# retina_final_list = []

# red = (255,0,0)

# for im in imgs:

#     faces, _ = retina_detector.detect(im, thresh, scales=scales, do_flip=flip)

#     if faces is not None:

#         for i in range(faces.shape[0]):

#             box = faces[i].astype(np.int)

#             lx, ly, rx, ry = box[0], box[1], box[2], box[3]

#             cv2.rectangle(im, (int(round(lx)),int(round(ly))), (int(round(rx)), int(round(ry))), red, 2)

#     retina_final_list.append(im)

# show_sequence(retina_final_list, 9)
# sfd_final_list = []

# red = (255,0,0)

# for im in imgs:

#     detected_faces = sfd_detector.detect_from_image(im, rgb=True)

#     for b in detected_faces:

#         lx, ly, rx, ry, _ = b

#         cv2.rectangle(im, (int(round(lx)),int(round(ly))), (int(round(rx)), int(round(ry))), red, 2)

#     sfd_final_list.append(im)

# show_sequence(sfd_final_list, 9)
# mobilenetv2 = load_mobilenetv2_224_075_detector("/kaggle/input/yolov2face/facedetection-mobilenetv2-size224-alpha0.75.h5")

# yolo_model = FaceDetector_yolo(mobilenetv2)

# yolo_final_list = []

# red = (255,0,0)

# for im in imgs:

#     yolo_boxes = yolo_model.detect(im, 0.75)

#     print(yolo_boxes)

#     yb = get_boxes_points(yolo_boxes, im.shape) 

#     for b in yb:

#         lx, ly, rx, ry = b

#         cv2.rectangle(im, (int(round(lx)),int(round(ly))), (int(round(rx)), int(round(ry))), red, 2)

#     yolo_final_list.append(im)

# show_sequence(yolo_final_list, 9)