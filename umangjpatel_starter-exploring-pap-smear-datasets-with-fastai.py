%matplotlib inline

from fastai import *

from fastai.vision import *

import pandas as pd

import matplotlib.image as mpimg
!ls ../input/
data_path = Path("../input/")

herlev_path = data_path/"herlev_pap_smear"

sipakmed_path = data_path/"sipakmed_fci_pap_smear"
herlev_path.ls()
class_paths = herlev_path.ls()

for c in class_paths:

    img_in_c_paths = get_image_files(c)

    print(f"Number of images in '{c.name}' : {len(img_in_c_paths)}")
sample_path = Path("../input/herlev_pap_smear/normal_superficiel")

imgs = sorted(get_image_files(sample_path))[:10]



for img_item in imgs:

    img = open_image(img_item)

    img.show()
sample_path = Path("../input/herlev_pap_smear/normal_superficiel")

imgs = sorted(get_image_files(sample_path))

imgs
mask, img = imgs[:2]

mask = open_image(mask)

img = open_image(img)

mask.show()

img.show()
sipakmed_path.ls()
classes = sipakmed_path.ls()

classes[0].ls()
one_class = classes[0]

one_class
dat_and_img_files = sorted(one_class.ls())

dat_and_img_files
sample_data = dat_and_img_files[:3]

sample_data
sample_img = sample_data[0]

open_image(sample_img)
nuc_data = pd.read_csv(sample_data[1], header=None)

nuc_data.head()
cyto_data = pd.read_csv(sample_data[2], header=None)

cyto_data.head()
img = mpimg.imread(sample_img)

plt.imshow(img)

plt.scatter(nuc_data.iloc[:, 0], nuc_data.iloc[:, 1], c="red")

plt.scatter(cyto_data.iloc[:, 0], cyto_data.iloc[:, 1], c="green")

plt.show()
tfms = get_transforms(flip_vert=True, max_warp=0.0, max_zoom=0.)

herlev_data_block = (ImageList.from_folder(herlev_path)

                    .filter_by_func(lambda fname: "-d" not in fname.name)

                    .split_by_rand_pct(valid_pct=0.2, seed=0)

                    .label_from_func(lambda fname: "abnormal" if "abnormal" in fname.parent.name else "normal")

                    .transform(tfms, size=128)

                    .databunch(bs=16)

                    .normalize(imagenet_stats))
herlev_data_block
herlev_data_block.show_batch(rows=4, figsize=(10 ,10))
def labelling_func(fname):

    c = fname.parent.name

    if "abnormal" in c:

        return "abnormal"

    elif "benign" in c:

        return "abnormal"

    else:

        return "normal"



tfms = get_transforms(flip_vert=True, max_warp=0.0, max_zoom=0.9)



sipakmed_data_block = (ImageList.from_folder(sipakmed_path)

                      .split_by_rand_pct(valid_pct=0.2, seed=42)

                      .label_from_func(labelling_func)

                      .transform(tfms, size=128)

                      .databunch(bs=16)

                      .normalize(imagenet_stats))
sipakmed_data_block
sipakmed_data_block.show_batch(rows=4, figsize=(10, 10))