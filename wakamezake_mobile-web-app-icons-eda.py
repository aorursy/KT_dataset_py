%ls ../input
import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

from pathlib import Path

from PIL import Image



dataset_path = Path("../input/common-mobile-web-app-icons/")
# https://docs.python.org/ja/3/library/pathlib.html

# 106 classes

dirs = [x for x in dataset_path.iterdir() if x.is_dir()]

len(dirs)
dirs
image_paths = [path for path in dataset_path.glob("*/*")]
# number of images

len(image_paths)
# unique extensions

unique_ext = set([image_path.suffix for image_path in image_paths])

unique_ext
df = pd.DataFrame({"image_path": image_paths})
df["class"] = df["image_path"].map(lambda x: x.parent.stem)
df["ext"] = df["image_path"].map(lambda x: x.suffix)
df["ext"].value_counts()
# delete svg image_path

df.drop(df[df["ext"] == ".svg"].index, inplace=True)
df["image_size"] = df["image_path"].map(lambda x: Image.open(x).size)

df["width"] = df["image_size"].map(lambda x: x[0])

df["height"] = df["image_size"].map(lambda x: x[1])

df.drop(columns=["image_size"], inplace=True)
df.head()
class_counts = df["class"].value_counts()

class_counts
width_counts = df["width"].value_counts()

width_counts
height_counts = df["height"].value_counts()

height_counts
# 30 types from the top

plt.figure(figsize=(12,10))

sns.barplot(class_counts[:30].index, class_counts[:30].values, alpha=0.8)

plt.ylabel('Number of images', fontsize=12)

plt.xlabel('class', fontsize=12)

plt.xticks(rotation=90)
# 30 types from the end

plt.figure(figsize=(12,10))

sns.barplot(class_counts[30:].index, class_counts[30:].values, alpha=0.8)

plt.ylabel('Number of images', fontsize=12)

plt.xlabel('class', fontsize=12)

plt.xticks(rotation=90)
# 30 types from the top

plt.figure(figsize=(12,10))

sns.barplot(width_counts[:30].index, width_counts[:30].values, alpha=0.8)

plt.ylabel('Number of images', fontsize=12)

plt.xlabel('width', fontsize=12)

plt.xticks(rotation=90)
# 30 types from the top

plt.figure(figsize=(12,10))

sns.barplot(height_counts[:30].index, height_counts[:30].values, alpha=0.8)

plt.ylabel('Number of images', fontsize=12)

plt.xlabel('height', fontsize=12)

plt.xticks(rotation=90)
import numpy as np

import torchvision.transforms as transforms

from torch.utils.data import DataLoader

from torch.utils.data import Dataset

from torchvision.utils import make_grid
def imshow(img, title=""):

    plt.figure(figsize=(12,10))

    npimg = img.numpy()

    plt.title(title)

    plt.imshow(np.transpose(npimg, (1, 2, 0)))
class AppIconDataset(Dataset):

    def __init__(self, img_path):

        self.img_path = img_path

        self.img_paths = list(self.img_path.glob("*jpg"))

        self.label_length = 5

        self.transform = transforms.Compose([

            transforms.Resize((128, 128)),

            transforms.ToTensor()])

    

    def __getitem__(self, index):

        """ Get a sample from the dataset

        """

        img_path = self.img_paths[index]

        image = Image.open(img_path).convert('RGB')

        return self.transform(image), img_path.parent.stem



    def __len__(self):

        """

        Total number of samples in the dataset

        """

        return len(self.img_paths)

    
for _dir in dirs:

    app_icons = AppIconDataset(img_path=_dir)

    batch_size = 64

    loader = DataLoader(app_icons, batch_size=batch_size, shuffle=False, num_workers=0)

    dataiter = iter(loader)

    images, labels = dataiter.next()

    imshow(make_grid(images), title=labels[0])