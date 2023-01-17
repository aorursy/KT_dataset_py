import numpy as np # line

import pandas as pd 







import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

from PIL import Image
############################################

# Kindly UpVote this beautiful data set    #

############################################
def display_Image(path, save):

    img1 = Image.open(path)

    display(img1)

    if save == True:

        img1.save('Hubble.jpg')

    
display_Image('/kaggle/input/top-100-hubble-telescope-images/heic0814a.tif', 0)
display_Image('/kaggle/input/top-100-hubble-telescope-images/potw1818a.tif',0)
display_Image('/kaggle/input/top-100-hubble-telescope-images/heic0822b.tif',True)
display_Image('/kaggle/input/top-100-hubble-telescope-images/opo0006a.tif',0)
display_Image('/kaggle/input/top-100-hubble-telescope-images/heic1406a.tif',0)