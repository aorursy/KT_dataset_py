import torch

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from PIL import Image

from pathlib import Path

import torchvision

from torchvision import transforms

from torch.utils.data import DataLoader

import torch.optim as optim

import torch.nn.functional as F

from torch import nn

import random

import gc
DATA_DIR = Path("../input/plant-pathology-2020-fgvc7")

CLASS_NAMES = np.array(["healthy", "multiple_diseases", "rust", "scab"])

BATCH_SIZE = 8

IMAGE_SIZE = (512, 512)

TEST_SPLIT = 0.2
device = "cuda" if torch.cuda.is_available() else "cpu"
class MyModel(nn.Module):

    def __init__(self):

        super(MyModel, self).__init__()

        self.backbone = torchvision.models.densenet201(pretrained=False)

        self.fc = nn.Linear(1000, 4)

        # ReLU(inplace=False)

        self.relu = nn.ReLU()

        

    def forward(self, x):

        x = self.backbone(x)

        x = self.fc(x)

        return F.log_softmax(x, dim=1)
model = MyModel().to(device)
model.load_state_dict(torch.load("../input/plant-pathology-models/acc_98_size_512.pth"))
model.eval()
class PlantPathologyTestDataset():

    def __init__(self, root, data_df, transform = None, preload=False):

        self.data_df = data_df

        self.images = None

        self.transform = transform

        if preload:

            self.images = []

            for idx in range(len(data_df)):

                image_path = DATA_DIR/"images"/(data_df["image_id"][idx] + ".jpg")

                image = Image.open(str(image_path))

                image = image.resize(IMAGE_SIZE)

                self.images.append(image.copy())

                

    def __len__(self):

        return len(self.images)

    

    def __getitem__(self, idx):

        if self.images is None:

            data_df = self.data_df

            image_path = DATA_DIR/"images"/(data_df["image_id"][idx] + ".jpg")

            image = Image.open(str(image_path))

            image = image.resize(IMAGE_SIZE)

        else:

            image = self.images[idx]

            

        if self.transform:

            image = self.transform(image)

        return self.data_df["image_id"][idx], image
data_df = pd.read_csv(DATA_DIR/"test.csv")

submission_data = PlantPathologyTestDataset(DATA_DIR, data_df, transform=transforms.ToTensor(), preload=True)

submission_loader = DataLoader(submission_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=1)
image_ids = []

h, m, r, s = [], [], [], []

with torch.no_grad():

  for batch_idx, (image_id_batch, data) in enumerate(submission_loader):

        data = data.to(device)   

        h_fliped_data = data.flip(2)

        output2 = model(h_fliped_data)

        del h_fliped_data

        gc.collect()

        v_fliped_data = data.flip(3)

        output3 = model(v_fliped_data)

        del v_fliped_data

        gc.collect()

        pred_batch = (model(data) +  output2 + output3)/3

        del data

        gc.collect()

        for image_id, pred in zip(image_id_batch, pred_batch):

          res = [0, 0, 0, 0]

          res[np.argmax( pred.cpu().numpy())] = 1

          h.append(res[0])

          m.append(res[1])

          r.append(res[2])

          s.append(res[3])

          image_ids.append(image_id)

        
sub = pd.DataFrame({"image_id":image_ids, "healthy": h, 'multiple_diseases': m, "rust": r, "scab": s})
sub.to_csv('submission.csv', index=False)

sub.head()