import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from skimage import feature

from sklearn import preprocessing

import seaborn as sns



from subprocess import check_output

print(check_output(["ls",'-lsa', "../input/"]).decode("utf8"))
data = np.load('../input/img_array_train_6k_22.npy')
data.shape
image_1 = data[0,:,:]

plt.imshow(image_1)
edge_img = feature.canny(data[1,:,:]).astype(int)
plt.imshow(edge_img)
df = pd.read_csv('../input/adni_demographic_master_kaggle.csv')
df.head()
df.shape


sns.kdeplot(image_1.flatten())
bin_im1 = preprocessing.binarize(image_1,600)

plt.imshow(bin_im1)