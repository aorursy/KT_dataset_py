import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

from torch.utils.data import Dataset,DataLoader

import albumentations as A

from albumentations.pytorch.transforms import ToTensorV2

import cv2

from matplotlib import pyplot as plt

import torch

#

# Super cool Dataset from https://www.kaggle.com/shonenkov/training-efficientdet

class DatasetRetriever(Dataset):



    def __init__(self, marking, image_ids, transforms=None, test=False):

        super().__init__()



        self.image_ids = image_ids

        self.marking = marking

        self.transforms = transforms

        self.test = test

        self.alpha = 1.0



    def __getitem__(self, index: int):

        image_id = self.image_ids[index]

        if self.test or random.random() > 0.55:

            image, boxes = self.load_image_and_boxes(index)

        else:

            #if random.random() > 0.70:

            image, boxes = self.load_cutmix_image_and_boxes(index)

            #else:

            #    image, boxes = self.load_mixup_v1(index)

                



        # there is only one class

        labels = torch.ones((boxes.shape[0],), dtype=torch.int64)

        

        target = {}

        target['boxes'] = boxes

        target['labels'] = labels

        target['image_id'] = torch.tensor([index])



        if self.transforms:

            for i in range(10):

                sample = self.transforms(**{

                    'image': image,

                    'bboxes': target['boxes'],

                    'labels': labels

                })

                if len(sample['bboxes']) > 0:

                    image = sample['image']

                    target['boxes'] = torch.stack(tuple(map(torch.tensor, zip(*sample['bboxes'])))).permute(1, 0)

                    target['boxes'][:,[0,1,2,3]] = target['boxes'][:,[1,0,3,2]]  #yxyx: be warning

                    break



        return image, target, image_id



    def __len__(self) -> int:

        return self.image_ids.shape[0]

    

    

    def clahe(self, bgr, gridsize=8):

        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)

        lab_planes = cv2.split(lab)

        clahe = cv2.createCLAHE(clipLimit=4.0,tileGridSize=(gridsize,gridsize))

        lab_planes[0] = clahe.apply(lab_planes[0])

        lab = cv2.merge(lab_planes)

        bgr_e = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        return bgr_e



    def load_image_and_boxes(self, index):

        image_id = self.image_ids[index]

        records = self.marking[self.marking['image_id'] == image_id]

        base_path = records['base_path'].values[0]

        

        image = cv2.imread(f'{base_path}/{image_id}.jpg', cv2.IMREAD_COLOR)

        ##

        ##adde clahe one out of three!!

        if np.random.rand() < 0.3 and not self.test:

            image = self.clahe(image, np.random.randint(6, 11))

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)

        image /= 255.0

        boxes = records[['x', 'y', 'w', 'h']].values

        boxes[:, 2] = boxes[:, 0] + boxes[:, 2]

        boxes[:, 3] = boxes[:, 1] + boxes[:, 3]

        return image, boxes

    

    

    def load_mixup_v1(self, index):

        lam = np.random.beta(self.alpha, self.alpha)



        image, boxes = self.load_image_and_boxes(index)

        r_image, r_boxes = self.load_image_and_boxes(random.randint(0, self.image_ids.shape[0] - 1))

        mixup_image = lam*image+(1-lam)*r_image



        mixup_boxes = []

        for box in boxes.astype(int):

            mixup_boxes.append(box)



        for box in r_boxes.astype(int):

            mixup_boxes.append(box)

        mixup_boxes = np.array(mixup_boxes)  

        return mixup_image, mixup_boxes



    def load_cutmix_image_and_boxes(self, index, imsize=1024):

        """ 

        This implementation of cutmix author:  https://www.kaggle.com/nvnnghia 

        Refactoring and adaptation: https://www.kaggle.com/shonenkov

        """

        w, h = imsize, imsize

        s = imsize // 2

    

        xc, yc = [int(random.uniform(imsize * 0.25, imsize * 0.75)) for _ in range(2)]  # center x, y

        indexes = [index] + [random.randint(0, self.image_ids.shape[0] - 1) for _ in range(3)]



        result_image = np.full((imsize, imsize, 3), 1, dtype=np.float32)

        result_boxes = []



        for i, index in enumerate(indexes):

            image, boxes = self.load_image_and_boxes(index)

            if i == 0:

                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)

                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)

            elif i == 1:  # top right

                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc

                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h

            elif i == 2:  # bottom left

                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)

                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, max(xc, w), min(y2a - y1a, h)

            elif i == 3:  # bottom right

                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)

                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

            result_image[y1a:y2a, x1a:x2a] = image[y1b:y2b, x1b:x2b]

            padw = x1a - x1b

            padh = y1a - y1b



            boxes[:, 0] += padw

            boxes[:, 1] += padh

            boxes[:, 2] += padw

            boxes[:, 3] += padh



            result_boxes.append(boxes)



        result_boxes = np.concatenate(result_boxes, 0)

        np.clip(result_boxes[:, 0:], 0, 2 * s, out=result_boxes[:, 0:])

        result_boxes = result_boxes.astype(np.int32)

        result_boxes = result_boxes[np.where((result_boxes[:,2]-result_boxes[:,0])*(result_boxes[:,3]-result_boxes[:,1]) > 0)]

        return result_image, result_boxes
SIZE = 512

def get_valid_transforms():

    return A.Compose(

        [

            A.Resize(height=SIZE, width=SIZE, p=1.0),

            ToTensorV2(p=1.0),

        ], 

        p=1.0, 

        bbox_params=A.BboxParams(

            format='pascal_voc',

            min_area=0, 

            min_visibility=0,

            label_fields=['labels']

        )

    )
#

# wheat original version



original_size = 1024

skpike_valid_images = '../input/wheat-dataset-original/convertor/images/'

skpike_valid_labels = '../input/wheat-dataset-original/convertor/labels/'

we_images = os.listdir(skpike_valid_images)

we_labels = os.listdir(skpike_valid_labels)

we_images = np.sort(we_images)

we_labels = np.sort(we_labels)

spike_df_train_orig = None

for (lab, img) in zip(we_labels, we_images):

    df = pd.read_csv(skpike_valid_labels + '/' + lab, sep=' ', header=None)

    df.columns = ['class', 'x', 'y', 'w', 'h']

    df['x'] = (1024 * df['x'])

    df['y'] = np.ceil(1024 * df['y'])

    df['w'] = np.floor(1024 * df['w'])

    df['h'] = np.floor(1024 * df['h'])

    df['x'] = np.ceil(df['x'] - df['w']/2 - 1)

    df['y'] = np.ceil(df['y'] - df['h']/2 - 1)

    

    df['x'] = df['x'].clip(0.1, 1023)

    df['y'] = df['y'].clip(0.1, 1023)

    keep_idx = df['w'] > 1

    df = df[keep_idx]

    keep_idx = df['h'] > 1

    df = df[keep_idx]

    

    

    

    df['image_id'] = img.split('.')[0]

    df['base_path'] = '../input/wheat-dataset-original/convertor/images/'

    df['width'] = 1024

    df['height'] = 1024

    df['source'] = 'spike'

    df = df.drop(['class'], axis=1)

    df = df[['image_id', 'width', 'height', 'source', 'x', 'y', 'w', 'h', 'base_path']]

    

    

    #print ( lab, img)

    if spike_df_train_orig is None:

        spike_df_train_orig = df.copy()

    else:

        spike_df_train_orig = pd.concat((spike_df_train_orig, df))

spike_df_train_orig.head()    


dataset_spike = DatasetRetriever(

    image_ids=spike_df_train_orig['image_id'].unique(),

    marking=spike_df_train_orig,

    transforms=get_valid_transforms(),

    test=True,

)

print (f'There are {len(dataset_spike)} images')
for i in range(5):

    image, target, image_id = dataset_spike[5*i]

    boxes = target['boxes'].cpu().numpy().astype(np.int32)



    numpy_image = image.permute(1,2,0).cpu().numpy()



    fig, ax = plt.subplots(1, 1, figsize=(16, 8))



    for box in boxes:

        cv2.rectangle(numpy_image, (box[1], box[0]), (box[3],  box[2]), (0, 1, 0), 2)



    ax.set_axis_off()

    ax.imshow(numpy_image);
#

# wheat spike split (2nd) version



original_size = 1024

skpike_valid_images = '../input/spike-dataset/images/train/'

skpike_valid_labels = '../input/spike-dataset/labels/train/'

we_images = os.listdir(skpike_valid_images)

we_labels = os.listdir(skpike_valid_labels)

we_images = np.sort(we_images)

we_labels = np.sort(we_labels)

spike_df_train = None

for (lab, img) in zip(we_labels, we_images):

    df = pd.read_csv(skpike_valid_labels + '/' + lab, sep=' ', header=None)

    df.columns = ['class', 'x', 'y', 'w', 'h']

    df['x'] = (1024 * df['x'])

    df['y'] = np.ceil(1024 * df['y'])

    df['w'] = np.floor(1024 * df['w'])

    df['h'] = np.floor(1024 * df['h'])

    df['x'] = np.ceil(df['x'] - df['w']/2 - 1)

    df['y'] = np.ceil(df['y'] - df['h']/2 - 1)

    

    df['x'] = df['x'].clip(0.1, 1023)

    df['y'] = df['y'].clip(0.1, 1023)

    keep_idx = df['w'] > 1

    df = df[keep_idx]

    keep_idx = df['h'] > 1

    df = df[keep_idx]

    

    

    

    df['image_id'] = img.split('.')[0]

    df['base_path'] = '../input/spike-dataset/images/train/'

    df['width'] = 1024

    df['height'] = 1024

    df['source'] = 'spike'

    df = df.drop(['class'], axis=1)

    df = df[['image_id', 'width', 'height', 'source', 'x', 'y', 'w', 'h', 'base_path']]

    

    

    #print ( lab, img)

    if spike_df_train is None:

        spike_df_train = df.copy()

    else:

        spike_df_train = pd.concat((spike_df_train, df))

spike_df_train.head()    


dataset_spike = DatasetRetriever(

    image_ids=spike_df_train['image_id'].unique(),

    marking=spike_df_train,

    transforms=get_valid_transforms(),

    test=True,

)

print (f'There are {len(dataset_spike)} images')
for i in range(5):

    image, target, image_id = dataset_spike[5*i]

    boxes = target['boxes'].cpu().numpy().astype(np.int32)



    numpy_image = image.permute(1,2,0).cpu().numpy()



    fig, ax = plt.subplots(1, 1, figsize=(16, 8))



    for box in boxes:

        cv2.rectangle(numpy_image, (box[1], box[0]), (box[3],  box[2]), (0, 1, 0), 2)



    ax.set_axis_off()

    ax.imshow(numpy_image);