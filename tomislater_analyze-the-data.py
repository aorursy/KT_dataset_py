import time

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from ipywidgets import interact, widgets
from IPython.display import clear_output

import seaborn as sns
sns.set(color_codes=True)

%matplotlib inline
df_train = pd.read_csv("../input/train.csv")
df_train.head()
@interact(ix=widgets.IntSlider(min=0, max=len(df_train), step=1, value=0, continuous_update=False))
def show_train_images(ix):
    
    clear_output(wait=True)
    
    how_many = 9
    hm_sq = int(np.sqrt(how_many))
    
    f, axes = plt.subplots(hm_sq, hm_sq)
    f.set_size_inches(18, 12)
    
    for nr, i in enumerate(range(ix, ix + how_many)):
        image_path = df_train["Image"][i]
        
        axes[int(nr / hm_sq)][nr % hm_sq].imshow(
            mpimg.imread("../input/train/" + image_path)
        )
        
    plt.show()
from os import listdir
from os.path import isfile, join

my_path = "../input/test"

only_images = [f for f in listdir(my_path) if isfile(join(my_path, f))]
@interact(ix=widgets.IntSlider(min=0, max=len(only_images), step=1, value=0, continuous_update=False))
def show_test_images(ix):
    
    clear_output(wait=True)
    
    how_many = 9
    hm_sq = int(np.sqrt(how_many))
    
    f, axes = plt.subplots(hm_sq, hm_sq)
    f.set_size_inches(18, 12)
    
    for nr, i in enumerate(range(ix, ix + how_many)):
        image_path = "../input/test/" + only_images[i]
        
        axes[int(nr / hm_sq)][nr % hm_sq].imshow(
            mpimg.imread(image_path)
        )
        
    plt.show()
from PIL import Image
train_widths = []
train_heights = []

for img_path in df_train["Image"]:
    with Image.open("../input/train/" + img_path) as img:
        width, height = img.size
        train_widths.append(width)
        train_heights.append(height)
test_widths = []
test_heights = []

for img_path in only_images:
    with Image.open("../input/test/" + img_path) as img:
        width, height = img.size
        test_widths.append(width)
        test_heights.append(height)
f, axes = plt.subplots(2, 2)
f.set_size_inches(18, 12)
axes[0][0].hist(train_widths)
axes[0][0].set_title("Widths (train)")

axes[1][0].hist(test_widths)
axes[1][0].set_title("Widths (test)")

axes[0][1].hist(train_heights)
axes[0][1].set_title("Heights (train)")

axes[1][1].hist(test_heights)
axes[1][1].set_title("Heights (test)")

plt.show()
len(df_train.groupby(["Id"]).count())
df_train.groupby(["Id"]).count().sort_values(by=["Image"], ascending=False)
