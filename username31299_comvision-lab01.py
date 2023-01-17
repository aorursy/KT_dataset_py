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

import cv2
import matplotlib.pyplot as plt
import glob

link_pic = '../kaggle/input/'

images = glob.glob(link_pic)
images
# image = cv2.imread(link_pic)
# _image= image[:,:,::-1]
# plt.imshow(_image)

import matplotlib.pyplot as plt
%matplotlib inline
plt.imshow(plt.imread('/kaggle/input/aaa.jpg'))
import PIL
import matplotlib.pyplot as plt
link_pic = '/kaggle/input/aaa.jpg'
img = PIL.Image.open(link_pic)

img.show()
# plt.imshow(img)
import PIL
import cv2
import matplotlib.pyplot as plt
import numpy as np

link_pic = '/kaggle/input/aaa.jpg'

img1 = PIL.Image.open(link_pic)
img2 = cv2.imread(link_pic)

print('PIL.Image.open: ',type(img1))
print('cv2.imread:     ',type(img2))

print('\n---------------------------------\n')

img3 = np.asarray(img1)
img4 = PIL.Image.fromarray(img2)
print('After PIL.Image.open: ',type(img3))
print('After cv2.imread:     ',type(img4))


import PIL
import cv2
import matplotlib.pyplot as plt
import numpy as np

link_pic = '/kaggle/input/aaa.jpg'

img1 = PIL.Image.open(link_pic)
img2 = cv2.imread(link_pic)

print(img1.size) # width, height
print(img2.shape) # width, height, channel
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt

link_pic = '/kaggle/input/somethings/aaa.jpg'
font = ImageFont.truetype('/kaggle/input/userfont/Ominus.otf',150)

img = Image.open(link_pic)

draw = ImageDraw.Draw(img)
draw.line((0,0,img.size),fill=(181,230,29),width=15)
draw.ellipse((tuple(int(e/2-50) for e in img.size),(tuple(int(e/2+50) for e in img.size))),fill=(255,136,4))
draw.text((10,10),'Hello World', fill='red', font=font)

plt.imshow(img)
