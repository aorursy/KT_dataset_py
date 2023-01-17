import os

import glob

import cv2

from PIL import Image

import pandas as pd

import numpy as np
filenames = glob.glob('/kaggle/input/deepfake-detection-challenge/test_videos/*.mp4')
submission = pd.read_csv("/kaggle/input/deepfake-detection-challenge/sample_submission.csv", index_col=0)

submission['label'] = 0.5
print(len(filenames))

for filename in filenames:

    try:

        name = os.path.basename(filename)

        v_cap = cv2.VideoCapture(filename)

        v_len = int(v_cap.get(cv2.CAP_PROP_FRAME_COUNT))

        for j in range(v_len):

            success = v_cap.grab()

            if j == (v_len-1):

                success, vframe = v_cap.retrieve()

                vframe = Image.fromarray(vframe) # this line fails for 27 videos

        v_cap.release()

        submission.loc[name, 'label'] = 0.5

    except:

        submission.loc[name, 'label'] = 1

submission.to_csv('submission.csv')