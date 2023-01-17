import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

from PIL import Image

import os

import torch

from torch.utils.data import Dataset, random_split, DataLoader

import torchvision.transforms as tt

from torchvision.utils import make_grid
! ls /kaggle/input/human-protein-atlas-image-classification
DATA_DIR = '/kaggle/input/human-protein-atlas-image-classification'



TRAIN_CSV = DATA_DIR + '/train.csv'

TRAIN_DIR = DATA_DIR + '/train'
df = pd.read_csv(TRAIN_CSV)

display(df.head())

print(f"df.shape: {df.shape}")
len(df['Id'].value_counts())
text_labels = {

0:  "Nucleoplasm", 

1:  "Nuclear membrane",   

2:  "Nucleoli",   

3:  "Nucleoli fibrillar center" ,  

4:  "Nuclear speckles",

5:  "Nuclear bodies",

6:  "Endoplasmic reticulum",   

7:  "Golgi apparatus",

8:  "Peroxisomes",

9:  "Endosomes",

10:  "Lysosomes",

11:  "Intermediate filaments",   

12:  "Actin filaments",

13:  "Focal adhesion sites",   

14:  "Microtubules",

15:  "Microtubule ends",   

16:  "Cytokinetic bridge",   

17:  "Mitotic spindle",

18:  "Microtubule organizing center",  

19:  "Centrosome",

20:  "Lipid droplets",   

21:  "Plasma membrane",   

22:  "Cell junctions", 

23:  "Mitochondria",

24:  "Aggresome",

25:  "Cytosol",

26:  "Cytoplasmic bodies",   

27:  "Rods & rings" 

}



NUM_LABELS = len(text_labels)

print(f"There are {NUM_LABELS} labels")
FILTERS = ['red', 'green', 'blue','yellow']
def get_image(ddir, filename):

    r = Image.open(f'{ddir}/{filename}_red.png')

    g = Image.open(f'{ddir}/{filename}_green.png')

    b = Image.open(f'{ddir}/{filename}_blue.png')

    y = Image.open(f'{ddir}/{filename}_yellow.png')

    return r, g, b, y





def display_image(image, ax):

    [a.axis('off') for a in ax]

    r, g, b, y = image

    ax[0].imshow(r,cmap='Reds')

    ax[0].set_title('Microtubules')

    ax[1].imshow(g,cmap='Greens')

    ax[1].set_title('Protein of Interest')

    ax[2].imshow(b,cmap='Blues')

    ax[2].set_title('Nucleus')

    ax[3].imshow(y,cmap='Oranges') 

    ax[3].set_title('Endoplasmic Reticulum')

    return ax
filename = df.Id.sample(1, random_state=9473).values[0]

imgs = get_image(TRAIN_DIR, filename)



fig, ax = plt.subplots(figsize=(15,5),nrows=1, ncols=4)

display_image(imgs, ax);
for key in text_labels.keys():

    df[key] = df['Target'].apply(lambda x: int(str(key) in x.split()))



targets_df = df.drop(labels=['Id', 'Target'], axis=1)



# targets_df.head()



target_counts = pd.DataFrame({'Localization': [v + ' ' + str(k) for k, v in text_labels.items()],

                              'Count': targets_df[text_labels.keys()].sum().values})

target_counts.sort_values('Count', inplace=True)

ax = target_counts.plot.barh(x='Localization', y='Count',figsize=(15,10), legend=False)



for i, v in enumerate(target_counts['Count']):

    ax.text(v + 3, i - 0.25, str(v) + ', ' + str(round(v / len(df) * 100, 2)) + '%')

ax.set_xlabel('Count');

ax.set_ylabel('');

# plt.axis('off')
targets_df['num_labels'] = targets_df.sum(axis=1)

targets_df.head()
label_counts = pd.DataFrame({'image_count': targets_df['num_labels'].value_counts(),

                             'pct_of_dataset': targets_df['num_labels'].value_counts() / len(df) * 100})

label_counts.columns = ['image_count','pct_of_dataset']

label_counts
plt.figure(figsize=(10, 10))

sns.heatmap(targets_df[targets_df['num_labels']>1].drop(['num_labels'], axis=1).rename(

    columns={k: f"{v} ({str(k)})" for k, v in text_labels.items()}

).corr(), cmap='YlGnBu');
class LocalizationDataLoader():

    def __init__(self, labels, batch_size, ddir):

        self.labels = labels

        self.batch_size = batch_size

        self.ddir = ddir

        self.get_image_ids()

    

    

    def are_labels_subset_of_targets(self, s):

        targets = [int(i) for i in s.split()]

        return np.where(set(self.labels).issubset(targets), 1, 0)

    

    

    def get_image_ids(self):

        df['check_condition'] = df.Target.apply(lambda s: self.are_labels_subset_of_targets(s))

        self.identified_image_ids = df[df['check_condition'] == 1].Id.values

        df.drop('check_condition', axis=1, inplace=True)

        

    

    def get_loader(self):

        idx = 0

        batch_images = []

        batch_image_ids = []

        for image_id in self.identified_image_ids:

            idx += 1

            batch_images.append(get_image(self.ddir, image_id))

            batch_image_ids.append(image_id)

            if idx == self.batch_size:

                yield batch_images, batch_image_ids

                idx = 0

                batch_images = []

                batch_image_ids = []

        if batch_images != []:

            yield batch_images, batch_image_ids
def get_image_targets(image_id):

    targets = df[df.Id==image_id].Target.values[0]

    targets = ', '.join([f"{text_labels[int(t)]} {t}" for t in targets.split()])

    return f"{targets}"
batch_size = 5

endo_lyso = LocalizationDataLoader([9, 10], batch_size, TRAIN_DIR)

loader = endo_lyso.get_loader()
imgs, img_ids = next(loader)



fig, ax = plt.subplots(nrows=len(imgs), ncols=4, figsize=(15, 5 * len(imgs)))

for i, img in enumerate(imgs):

    display_image(img, ax[i])

#     ax[i][1].set_title(get_image_targets(img_ids[i]), y=-0.1)