
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import sys
sys.path.insert(0, "../input")
import cv2

import torch
import pandas as pd
from matplotlib import pyplot as plt


import torchvision
import torchvision.models as models
from torchvision import datasets, transforms as T
from PIL import Image
from torchvision import transforms
from random import randint, choice, choices
import json
from torch.utils.data import DataLoader, Dataset
from tqdm.notebook import tqdm


with open("../input/imagenet-class-index/imagenet_class_index.json") as json_file:
    class_idx = json.load(json_file)
idx2label = [class_idx[str(k)][1] for k in range(len(class_idx))]
def load_img(img_id, img_dir=f"../input/landmark-recognition-2020/train", PIL=True):
    f1, f2, f3 = img_id[:3]
    filename = f"{img_dir}/{f1}/{f2}/{f3}/{img_id}.jpg"
    if PIL:
        img = Image.open(filename)
    else:
        img = cv2.imread(filename)
    return img


def plot_img(img, size=(7, 7), title=""):
    if isinstance(img, str):
        img = load_img(img)
    plt.figure(figsize=size)
    plt.imshow(img)
    plt.suptitle(title)
    plt.show()
    
    
def plot_imgs(imgs, cols=5, size=7, title="", title_list=None):
    rows = len(imgs)//cols + 1
    fig = plt.figure(figsize=(cols*size, rows*size))
    for i, img in enumerate(imgs):
        if isinstance(img, str):
            img = load_img(img)
        fig.add_subplot(rows, cols, i+1)
        plt.imshow(img)
        if title_list is not None:
            plt.title(title_list[i])
    plt.suptitle(title)
    plt.show()

def toTensor(array, axis=(2,0,1)):
    if isinstance(array, torch.Tensor):
        return array
    return torch.tensor(array).permute(axis)

def toNumpy(tensor, axis=(1,2,0)):
    if isinstance(tensor, np.ndarray):
        return tensor
    return tensor.detach().cpu().permute(axis).numpy()
class LandmarkDataset(Dataset):
    def __init__(self, landmark_ids, img_dir, dataframe, transforms=None):
        self.landmark_ids = landmark_ids
        self.img_dir = img_dir
        self.df = dataframe
        self.transforms = transforms
        
    def __getitem__(self, idx):
        landmark_id = self.landmark_ids[idx]
        df = self.df
        img_id = df[df["landmark_id"]==landmark_id]["id"].values[0]
        img = load_img(img_id, self.img_dir)
        if self.transforms:
            img = self.transforms(img)
        return landmark_id, img_id, img
        
    def __len__(self):
        return len(self.landmark_ids)
img_dir=f"../input/landmark-recognition-2020/train"
df = pd.read_csv("../input/landmark-recognition-2020/train.csv")
df.head()
landmark_ids = df["landmark_id"].unique()
sample_ids = []

for lmark_id in tqdm(landmark_ids):
    sample_id = df[df["landmark_id"]==lmark_id]["id"].values[0]
    sample_ids.append(sample_id)
    
landmark_df = pd.DataFrame({
    "landmark_id": landmark_ids,
    "id" : sample_ids
})
n = len(df)
img_ids = list(df["id"])
transforms_ = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

landmark_dataset = LandmarkDataset(landmark_ids=landmark_ids, img_dir=img_dir, dataframe=landmark_df, transforms=transforms_)
landmark_loader = DataLoader(landmark_dataset, batch_size=80, shuffle=False, drop_last=False)
# lmark_ids, ids, img = next(iter(landmark_loader))

model_path = "../input/pretrained-pytorch-models/resnet50-19c8e357.pth"
model = models.resnet50()
model.eval()
model.load_state_dict(torch.load(model_path))
import pickle

def save_label(id_labels, filename):
    with open(filename, "w+") as f:
        pickle.dump(id_labels, f)
        f.close()

device = torch.device("cuda")
model.to(device)
model.eval()
result = []
i = 0


for lmark_ids, ids, img_patch in tqdm(landmark_loader):
    i += 1
    img_patch = img_patch.to(device)
    labels = model(img_patch)
    labels = labels.detach().cpu().numpy().argmax(1)
    
    for lmark_id, label in zip(lmark_ids, labels):
        result.append([lmark_id, label])
lmark_ids = [int(e[0]) for e in result]
label_ids = [e[1] for e in result]
labels = [idx2label[l_id] for l_id in label_ids]

classify_df = pd.DataFrame({
    "landmark_id": lmark_ids,
    "label_id": label_ids,
    "label": labels
})

classify_df.to_csv("classify_dataframe.csv")
classify_df.head()

imgs = []
labels = []

for i,(lmark_id, _, img) in enumerate(landmark_dataset):
    labels.append(classify_df[classify_df["landmark_id"] == lmark_id]["label"].values[0])
    imgs.append(toNumpy(img))
    if i == 10:
        break
plot_imgs(imgs, title_list=labels)
from bokeh.plotting import figure as bokeh_figure
from bokeh.io import output_notebook, show, output_file
from bokeh.models import ColumnDataSource, HoverTool, Panel
from bokeh.models.widgets import Tabs


def hist_hover(dataframe, column, color=["#94c8d8", "#ea5e51"], bins=30, title="", value_range=None):
    hist, edges = np.histogram(dataframe[column], bins=bins, range=value_range)
    hist_frame = pd.DataFrame({
        column: hist,
        "left": edges[:-1],
        "right": edges[1:]
    })
    hist_frame["interval"] = ["%d to %d" %
                              (left, right) for left, right in zip(edges[:-1], edges[1:])]
    src = ColumnDataSource(hist_frame)
    plot = bokeh_figure(
        plot_height=600, plot_width=1000,
        title=title, x_axis_label=column,
        y_axis_label="Count"
    )
    plot.quad(
        bottom=0, top=column, left="left", right="right",
        source=src, fill_color=color[0], line_color="#35838d",
        fill_alpha=0.7, hover_fill_alpha=0.7,
        hover_fill_color=color[1]
    )
    hover = HoverTool(
        tooltips=[("Interval", "@interval"), ("Count", str(f"@{column}"))]
    )
    plot.add_tools(hover)
    output_notebook()
    show(plot)


hist_hover(classify_df, column="label_id", bins=1000, title="frequency of landmark type")
lmark_labels = classify_df["label"]
labels, counts = np.unique(lmark_labels,return_counts=True)
label_counts = list(zip(labels, counts))
label_counts.sort(key=lambda x: x[1])

print("top 20 COMMON")
print(label_counts[-20:])

print("top 20 NOT COMMON")
print(label_counts[:20])