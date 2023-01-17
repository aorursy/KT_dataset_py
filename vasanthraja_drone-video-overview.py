%matplotlib inline

import os

from glob import glob

import numpy as np

import cv2

import matplotlib.pyplot as plt

VIDEO_DIR = os.path.join('..', 'input')

!ls ../input
all_videos = [x for x in glob(os.path.join(VIDEO_DIR, '*')) if x.upper().endswith('MP4') or x.upper().endswith('.MOV')]

print('Found', len(all_videos), 'videos')
def read_video_segment(in_path, vid_seg = None):

    cap = cv2.VideoCapture(in_path)

    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1

    frames = []

    if cap.isOpened() and video_length > 0:

        frame_ids = [0]

        if vid_seg is None:

            vid_seg = np.array([0, 0.25, 0.5, 0.75, 1])

        else:

            vid_seg = np.clip(vid_seg, 0, 1)

            

        frame_ids = np.clip(video_length*vid_seg, 0, video_length-1).astype(int)

        count = 0

        success, image = cap.read()

        print('Loaded', video_length, 'frames at', image.shape, 'resolution')

        while success:

            if count in frame_ids:

                frames.append(image)

            success, image = cap.read()

            count += 1

    return frames
fig, m_axs = plt.subplots(len(all_videos), 4, figsize = (20, len(all_videos)*4))

for c_path, c_axs in zip(all_videos, m_axs):

    for c_frame, c_ax in zip(read_video_segment(c_path), 

                             c_axs):

        c_ax.imshow(c_frame[:, :, ::-1])

        c_ax.set_title(os.path.basename(c_path))

        c_ax.axis('off')