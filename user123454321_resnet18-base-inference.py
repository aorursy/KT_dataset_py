import numpy as np 

import pandas as pd 

import torchvision

import torch

import torch.nn as nn

import torch.nn.functional as F

from PIL import Image

import os

from functools import reduce
test_df = pd.read_csv('/kaggle/input/bengaliai-cv19/test.csv')

submission_df = pd.read_csv('/kaggle/input/bengaliai-cv19/sample_submission.csv')
class TestDataset(torch.utils.data.Dataset):



    def __init__(self, test_images, image_ids, transforms=None):

        super(TestDataset, self).__init__()

        self.image_ids = image_ids

        self.test_images = test_images

        self.transforms = transforms



    def __getitem__(self, index):

        image_id = self.image_ids.iloc[index]

        img_array = np.zeros((137, 236, 3), dtype='uint8')

        img_array[:, :, 0] = self.test_images[index].reshape(137, 236)

        img_array[:, :, 1] = img_array[:, :, 0]

        img_array[:, :, 2] = img_array[:, :, 0]

        img = Image.fromarray(img_array)

        if self.transforms:

            img = self.transforms(img)



        return img



    def __len__(self,):

        return len(self.image_ids)
class GraphemeModel(nn.Module):



    def __init__(self):

        super(GraphemeModel, self).__init__()

        self.base_model = torchvision.models.resnet18(pretrained=False) # use resnet18 as the base model

#         self.fc = nn.Linear(1000, 256) 

        self.fc_root = nn.Linear(1000, 168)

        self.fc_vowel = nn.Linear(1000, 11)

        self.fc_consonant = nn.Linear(1000, 7)

        

    def forward(self, inp):

        x = self.base_model(inp)

        x = x.view(x.shape[0], -1)

#         x = F.relu(self.fc(x))

        root_output = self.fc_root(x)

        vowel_output = self.fc_vowel(x)

        consonant_output = self.fc_consonant(x)



        return (root_output, vowel_output, consonant_output)
model = torch.load('/kaggle/input/bengaligraphememodel3/model.pth')
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
model = model.to(device)
transforms = torchvision.transforms.Compose([

                              torchvision.transforms.ToTensor(),

                              torchvision.transforms.Normalize((0, 0, 0), (1., 1., 1.))                

            ])
model.eval()

predictions = []



for i in range(4):

    test_image_data = pd.read_parquet(f'/kaggle/input/bengaliai-cv19/test_image_data_{i}.parquet') # read ith test parquet file

    test_matrix = test_image_data.drop(columns=['image_id']).values

    image_ids = test_image_data.image_id

    test_dataset = TestDataset(test_images=test_matrix, image_ids=image_ids, transforms=transforms)

    test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=128, shuffle=False)

    for x in test_dataloader:

        root, vowel, consonant = model(x.to(device)) # get prediction for the batch

        root = root.argmax(1).detach().cpu().numpy() # convert to numpy

        vowel = vowel.argmax(1).detach().cpu().numpy()

        consonant = consonant.argmax(1).detach().cpu().numpy()

        predictions += list(reduce(lambda a, b: a + b, zip(consonant, root, vowel)))

    
submission_df['target'] = predictions
submission_df.head()
submission_df.to_csv('submission.csv', index=False)