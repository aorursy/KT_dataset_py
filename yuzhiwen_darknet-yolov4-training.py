# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os

!cp -r ../input/darknet/* .
!ls

!ls data/VOCdevkit/VOC2007/Annotations
!mkdir data/VOCdevkit/VOC2007/JPEGImage/
!ls data/VOCdevkit/VOC2007/
!mkdir data/VOCdevkit/VOC2007/ImageSets/
!mkdir data/VOCdevkit/VOC2007/ImageSets/Main/
!python voc2yolo4.py
!python  ./build/darknet/x64/data/voc/voc_label.py
!ls
!./darknet detector train cfg/wheat.data data/yolov4-obj.cfg yolov4.conv.137