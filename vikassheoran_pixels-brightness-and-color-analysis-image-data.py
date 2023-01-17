import os

import numpy as np

path = '/kaggle/input/utkface-new/UTKFace'

os.chdir(path)

images = os.listdir()
print('No. of Images : '+str(len(images)))
!pip install git+https://github.com/Soongja/basic-image-eda
from basic_image_eda import BasicImageEDA



extensions = ['jpg']

threads = 0

dimension_plot = True

channel_hist = True

nonzero = False

hw_division_factor = 1.0



BasicImageEDA.explore(path, extensions, threads, dimension_plot, 

                      channel_hist, nonzero, hw_division_factor)
from PIL.ImageStat import Stat

from PIL import ImageStat

from PIL import Image



#Converting Images to PIL images

IMAGES = []

for img in images:

  im = Image.open(img)

  IMAGES.append(im)
type(IMAGES[0]),len(IMAGES)
def print_stats(img):

    """ prints stats, remember that img should already have been opened """

    stat = Stat(img)

    print("extrema    : ", stat.extrema)

    print("count      : ", stat.count)

    print("sum        : ", stat.sum)

    print("sum2       : ", stat.sum2)

    print("mean       : ", stat.mean)

    print("median     : ", stat.median)

    print("rms        : ", stat.rms)

    print("var        : ", stat.var)

    print("stddev     : ", stat.stddev)
print_stats(IMAGES[0]) #Stats for first Image
import math



def brightness(im):

    """ Computes brightness """

    stat = ImageStat.Stat(im)

    r,g,b = stat.mean

    return math.sqrt(0.241*(r**2) + 0.691*(g**2) + 0.068*(b**2))
print('Brigthness of Image 1: '+str(brightness(IMAGES[0]))+

      '\nBrigthness of Image 2: '+str(brightness(IMAGES[1]))+

      '\nBrigthness of Image 3: '+str(brightness(IMAGES[2])))
brightness_images = [brightness(img) for img in IMAGES]
import matplotlib.pyplot as plt

plt.scatter(np.arange(0,len(IMAGES),1),brightness_images,alpha=0.01)

plt.title('Brightness of ith image')

plt.xlabel('Image Index')

plt.ylabel('Perceived Brightness')

plt.show()
import numpy as np



plt.hist(brightness_images,bins=int(np.sqrt(len(IMAGES))))

plt.title('Brightness Distribution')

plt.xlabel('Brightness')

plt.ylabel('Frequency')

plt.show();



print('Mean: ' + str(np.mean(brightness_images)),

      '\nStd : ' + str(np.std(brightness_images)))
images[:5]
#Further I discovered 3 images were unlabelled



remove  = ['20170109150557335.jpg.chip.jpg',

           '20170116174525125.jpg.chip.jpg',

           '20170109142408075.jpg.chip.jpg']
Age, Gender, Race = [], [], []

for img in images:

  if img.split('_')[2] not in remove:

    Age.append(int(img.split('_')[0]))

    Gender.append(int(img.split('_')[1]))

    Race.append(int(img.split('_')[2]))
len(Age), len(Gender), len(Race)
import matplotlib.pyplot as plt 

import math

import numpy as np



print('(Age, Gender, Race)')

fig=plt.figure(figsize=(16, 16))

display_first = 20

columns = 5

rows = math.ceil(display_first/10)

for i in range(1,display_first+1):

    img = Image.open(images[i-1])

    fig.add_subplot(10, 10, i)

    plt.title((Age[i-1],Gender[i-1],Race[i-1]))

    plt.axis('off')

    plt.imshow(img)

plt.show()
def is_grayscale(img):



    im = img.convert("RGB")

    stat = ImageStat.Stat(im)



    if sum(stat.sum)/3 == stat.sum[0]:

        return True

    else:

        return False
img_type = [is_grayscale(img) for img in IMAGES]

print(img_type[:10]) 
import pandas as pd

df = pd.DataFrame(img_type)

df[0].value_counts()
gray_images = df[df[0]==True]

gray_images.columns =['Gray Scale ?']

print(gray_images)
fig=plt.figure(figsize=(16, 16))

fig.tight_layout()

columns = 20

rows = math.ceil(838/20)

j = 1

for i in gray_images.index:

    img = IMAGES[i]

    fig.add_subplot(20, 20, j)

    j+=1

    plt.axis('off')

    plt.imshow(img)

    if j>100:

      break

plt.show()