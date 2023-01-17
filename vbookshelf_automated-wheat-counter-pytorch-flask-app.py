! pip install segmentation-models-pytorch
import pandas as pd

import numpy as np

import os



import ast





import cv2

import time



import torch

import torch.nn as nn

import torch.optim as optim

import torch.nn.functional as F



import torchvision

import torchvision.transforms as transforms



from torch.utils.data import Dataset, DataLoader

from torchvision import transforms, utils



# set a seed value

torch.manual_seed(555)



from torch.optim.lr_scheduler import ReduceLROnPlateau

from torch.optim.lr_scheduler import StepLR





import albumentations as albu

from albumentations import Compose





from sklearn.model_selection import train_test_split

from sklearn.utils import shuffle





import matplotlib.pyplot as plt

%matplotlib inline



# Don't Show Warning Messages

import warnings

warnings.filterwarnings('ignore')



# Note: Pytorch uses a channels-first format:

# [batch_size, num_channels, height, width]



print(torch.__version__)

print(torchvision.__version__)
os.listdir('../input/')
BACKBONE = 'resnet34'



IMAGE_HEIGHT_ORIG = 1024

IMAGE_WIDTH_ORIG = 1024

IMAGE_CHANNELS_ORIG = 3



IMAGE_HEIGHT = 512

IMAGE_WIDTH = 512

IMAGE_CHANNELS = 3



BATCH_SIZE = 8



SAMPLE_SIZE = 15



NUM_EPOCHS = 30



THRESHOLD = 0.7



LRATE = 0.0001



# Check the number of available cpu cores.

# This variable is used to set the num workers in the data loader.

NUM_CORES = os.cpu_count()



NUM_CORES
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



print(device)


def create_mask(image_id, image_height, image_width):

    """

    Input: image_id

    Output: Image mask containing all the bbox masks.

    """

    

    # filter out all rows with the image_id

    df = df_data[df_data['image_id'] == image_id]



    # create blank mask

    mask = np.zeros((image_height, image_width, 1))



    # get a list of all bbox values

    bbox_list = list(df['bbox'])





    # loop over the bbox list

    for bbox_string in bbox_list:



        bbox = ast.literal_eval(bbox_string) 

        

        # First we check if a mask exists.

        # If not a blank mask is returned.

        if (bbox_string[0] != '[') or (len(bbox) == 0):



            pass



        else:



            # get the bounding box

            # ast.literal_eval converts '[...]' to [...]

            bbox = ast.literal_eval(bbox_string) 



            x = int(bbox[0])

            y = int(bbox[1])

            w = int(bbox[2])

            h = int(bbox[3])



            # add the bbox mask to the blank mask created above

            mask[y:y+h, x:x+w] = 1

    

    return mask











def multiply_masks_and_images(images, thresh_masks):

    

    """

    Trying to do this multiplication with Pytorch tensors

    did not produce the result that I wanted. Therefore, here I am

    converting the tensors to numpy, doing the multiplication, and 

    then converting back to pytorch.

    

    """



    # convert from torch tensors to numpy

    np_images = images.cpu().numpy()

    np_thresh_masks = thresh_masks.cpu().numpy()



    # reshape

    np_images = np_images.reshape((-1, 512, 512, 3))

    np_thresh_masks = np_thresh_masks.reshape((-1, 512, 512, 1))





    # multiply the mask by the image

    modified_images = np_thresh_masks * np_images



    # change shape to channels first to suit pytorch

    #modified_images = modified_images.transpose((2, 0, 1))

    modified_images = modified_images.reshape((-1, 3, 512, 512))



    # convert to torch tensor

    modified_images = torch.tensor(modified_images, dtype=torch.float)



    return modified_images



# I saved the original competition data as a Kaggle dataset.

# Here I'm using that dataset as the data source.



base_path = '../input/wheat-detection-comp-original-data/global-wheat-detection/'



os.listdir(base_path)
path = base_path + 'train.csv'



df_data = pd.read_csv(path)



print(df_data.shape)



df_data.head()
# [xmin, ymin, width, height]





def extract_width(x):

    

    # convert the string to a python list

    bbox = ast.literal_eval(x) 

    

    # get the width

    w = int(bbox[2])

    

    return w







def extract_height(x):

    

    # convert the string to a python list

    bbox = ast.literal_eval(x) 

    

    # get the width

    h = int(bbox[3])

    

    return h







# Create new columns

df_data['w'] = df_data['bbox'].apply(extract_width)

df_data['h'] = df_data['bbox'].apply(extract_height)



df_data.head()
# Filter out the big masks.

# Use 500px as the limit.



df_data = df_data[df_data['w'] < 500]

df_data = df_data[df_data['h'] < 500]
# Filter out the very small masks.

# Use 20px as the limit.



df_data = df_data[df_data['w'] >= 20]

df_data = df_data[df_data['h'] >= 20]
# Check how many rows now exist



df_data.shape
df_masks = df_data.copy()



# Create a new column

df_masks['num_masks'] = 1



df_masks = df_masks[['image_id', 'num_masks']]



print(df_masks.shape)



df_masks.head()
# Create a dataframe showing num wheat heads on each unique image



df_masks = df_masks.groupby('image_id').sum()



df_masks = df_masks.reset_index()



df_masks.head()
# List the data sources



df_data['source'].unique()
# Filter out all images

df_usask_1 = df_data[df_data['source'] == 'usask_1']

# Drop duplicate image_id values

df_usask_1 = df_usask_1.drop_duplicates(subset='image_id', keep='first', inplace=False)



# Filter out all images

df_arvalis_1 = df_data[df_data['source'] == 'arvalis_1']

# Drop duplicate image_id values

df_arvalis_1 = df_arvalis_1.drop_duplicates(subset='image_id', keep='first', inplace=False)



# Filter out all images

df_arvalis_2 = df_data[df_data['source'] == 'arvalis_2']

# Drop duplicate image_id values

df_arvalis_2 = df_arvalis_2.drop_duplicates(subset='image_id', keep='first', inplace=False)



# Filter out all images

df_arvalis_3 = df_data[df_data['source'] == 'arvalis_3']

# Drop duplicate image_id values

df_arvalis_3 = df_arvalis_3.drop_duplicates(subset='image_id', keep='first', inplace=False)



# Filter out all images

df_inrae_1 = df_data[df_data['source'] == 'inrae_1']

# Drop duplicate image_id values

df_inrae_1 = df_inrae_1.drop_duplicates(subset='image_id', keep='first', inplace=False)



# Filter out all images

df_ethz_1 = df_data[df_data['source'] == 'ethz_1']

# Drop duplicate image_id values

df_ethz_1 = df_ethz_1.drop_duplicates(subset='image_id', keep='first', inplace=False)



# Filter out all images

df_rres_1 = df_data[df_data['source'] == 'rres_1']

# Drop duplicate image_id values

df_rres_1 = df_rres_1.drop_duplicates(subset='image_id', keep='first', inplace=False)







print('usask_1:', df_usask_1.shape) # Use for validation

print('arvalis_1:', df_arvalis_1.shape)

print('arvalis_2:', df_arvalis_2.shape) # Use for validation

print('arvalis_3:', df_arvalis_3.shape)

print('inrae_1:', df_inrae_1.shape)

print('ethz_1:', df_ethz_1.shape)

print('rres_1:', df_rres_1.shape)

# Print one image



image_id = '42e6efaaa'



path = base_path + 'train/' + image_id + '.jpg'





mask = create_mask(image_id, 1024, 1024)

mask = mask[:,:,0]



image = plt.imread(path)



print(image.shape)



plt.imshow(image, cmap='Greys')

plt.imshow(mask, cmap='Reds', alpha=0.3)



plt.show()
# Train

    # df_arvalis_1

    # df_arvalis_3

    # df_inrae_1

    # df_ethz_1

    # df_rres_1







# Val

    # df_usask_1

    # df_arvalis_2



    

    

# Create the train set

df_train = pd.concat([df_arvalis_1, df_arvalis_3, df_inrae_1, df_ethz_1, df_rres_1], axis=0)





# Create the val set

df_val = pd.concat([df_usask_1, df_arvalis_2], axis=0)





print(df_train.shape)

print(df_val.shape)
# We perform an inner join (intersection).

# Only image_id values that are common to both dataframes will be kept. Other rows will be deleted.

# Codebasics tutorial: https://www.youtube.com/watch?v=h4hOPGo4UVU



df_train = pd.merge(df_train, df_masks, on='image_id', how='inner')



df_val = pd.merge(df_val, df_masks, on='image_id', how='inner')



# Select the columns we want.

cols = ['image_id', 'source', 'num_masks']

df_train = df_train[cols]

df_val = df_val[cols]



print(df_train.shape)

print(df_val.shape)
df_train.head()
df_val.head()
df_test = df_val.sample(n=SAMPLE_SIZE, random_state=101)



print(df_test.shape)



df_test.head(15)
df_val.shape
# Remove the test images from df_val



test_images_list = list(df_test['image_id'])

 

df_val = df_val[~df_val['image_id'].isin(test_images_list)] # This line means: is not in



df_val.shape
# Shuffle



df_train = shuffle(df_train)



df_val = shuffle(df_val)
df_data.to_csv('df_data.csv.gz', compression='gzip', index=False)



df_train.to_csv('df_train.csv.gz', compression='gzip', index=False)

df_val.to_csv('df_val.csv.gz', compression='gzip', index=False)



df_test.to_csv('df_test.csv.gz', compression='gzip', index=False)
# Check that the dataframes have been saved.

!ls


import albumentations as albu



# Define the augmentations



def get_training_augmentation():

    

    train_transform = [

    albu.Flip(always_apply=False, p=0.8),

    albu.RandomRotate90(always_apply=False, p=0.8),

    albu.Blur(blur_limit=7, always_apply=False, p=0.3),

    albu.OneOf([

        albu.RandomContrast(),

        albu.RandomGamma(),

        albu.RandomBrightness(),

        ], p=0.5),

    albu.OneOf([

        albu.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),

        albu.GridDistortion(),

        albu.OpticalDistortion(distort_limit=2, shift_limit=0.5),

        ], p=0.5),

   

    ]

  

    return albu.Compose(train_transform)









def get_preprocessing(preprocessing_fn):

    """Construct preprocessing transform

    

    Args:

        preprocessing_fn (callbale): data normalization function 

            (can be specific for each pretrained neural network)

    Return:

        transform: albumentations.Compose

    

    """

    

    _transform = [

        albu.Lambda(image=preprocessing_fn),

    ]

    return albu.Compose(_transform)

   
# Display an image and mask

# ==========================



image_id = '7d5af5b74'



path = '../input/global-wheat-detection/train/' + image_id + '.jpg'





mask = create_mask(image_id, 1024, 1024)

mask = mask[:,:,0]



image = plt.imread(path)



print(image.min())

print(image.max())

print(image.shape)



plt.imshow(image, cmap='Greys')

plt.imshow(mask, cmap='Reds', alpha=0.3)



plt.show()
# Display an AUGMENTED image

# =============================



image_id = '7d5af5b74'

path = '../input/global-wheat-detection/train/' + image_id + '.jpg'



image = plt.imread(path)

mask = create_mask(image_id, 1024, 1024)



augmentation = get_training_augmentation()



sample = augmentation(image=image, mask=mask)

image, mask = sample['image'], sample['mask']





print(image.min())

print(image.max())



print(image.shape)

print(mask.shape)







plt.imshow(image, cmap='Greys')

plt.imshow(np.squeeze(mask), cmap='Reds', alpha=0.3)



plt.show()
# Display a PRE-PROCESSED image

# ==============================





image_id = '7d5af5b74'

path = '../input/global-wheat-detection/train/' + image_id + '.jpg'



image = plt.imread(path)

mask = create_mask(image_id, 1024, 1024)









from segmentation_models_pytorch.encoders import get_preprocessing_fn



# Initialize the preprocessing function

preprocessing_fn = get_preprocessing_fn(BACKBONE, pretrained='imagenet')



preprocessing = get_preprocessing(preprocessing_fn)



sample = preprocessing(image=image, mask=mask)

image, mask = sample['image'], sample['mask']







print(image.min())

print(image.max())



print(image.shape)

print(mask.shape)







plt.imshow(image)



# Uncomment the next line to see a mask overlayed on the pre-processed image.

# plt.imshow(np.squeeze(mask), cmap='Reds', alpha=0.3)



plt.show()
# Reset the indices



df_train = df_train.reset_index(drop=True)

df_val = df_val.reset_index(drop=True)
class CompDataset(Dataset):

    

    def __init__(self, df, augmentation=None, preprocessing=None):

        self.df_data = df

        self.augmentation = augmentation

        self.preprocessing = preprocessing

        

        

        

    def __getitem__(self, index):

        image_id = self.df_data.loc[index, 'image_id']

        

        image_path = base_path + 'train/'

        



        # set the path to the image

        path = image_path + image_id + '.jpg'

        

        # Create the image

        # ------------------





        # read the image

        image = cv2.imread(path)



        # convert to from BGR to RGB

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)



        # resize the image

        image = cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT))



        

        

        # Create the mask

        # ------------------

        

        # create the mask

        mask = create_mask(image_id, 1024, 1024)



        # resize the mask

        mask = cv2.resize(mask, (IMAGE_WIDTH, IMAGE_HEIGHT))

        

        # create a channel dimension

        mask = np.expand_dims(mask, axis=-1)

        

  

        

        # apply augmentations

        if self.augmentation:

            sample = self.augmentation(image=image, mask=mask)

            image, mask = sample['image'], sample['mask']

        

        # apply preprocessing

        if self.preprocessing:

            sample = self.preprocessing(image=image, mask=mask)

            image, mask = sample['image'], sample['mask']

            



    

        # Swap color axis

        # numpy image: H x W x C

        # torch image: C X H X W

        image = image.transpose((2, 0, 1))

        mask = mask.transpose((2, 0, 1))

        

        # convert to torch tensor

        image = torch.tensor(image, dtype=torch.float)

        mask = torch.tensor(mask, dtype=torch.float)

        

        

        # Create the regression target.

        cols = ['num_masks']

        

        target = torch.tensor(self.df_data.loc[index, cols], dtype=torch.float)

        

        

        sample = (image, mask, target)

        

        

        return sample

    

    

    def __len__(self):

        return len(self.df_data)

    
# Test the dataloaders



from segmentation_models_pytorch.encoders import get_preprocessing_fn



# Initialize the preprocessing function

preprocessing_fn = get_preprocessing_fn(BACKBONE, pretrained='imagenet')





train_data = CompDataset(df_train, augmentation=get_training_augmentation(), 

                        preprocessing=get_preprocessing(preprocessing_fn))



val_data = CompDataset(df_val, augmentation=None, 

                        preprocessing=get_preprocessing(preprocessing_fn))





train_loader = torch.utils.data.DataLoader(train_data,

                                            batch_size=BATCH_SIZE,

                                            shuffle=True,

                                           num_workers=NUM_CORES

                                            )



val_loader = torch.utils.data.DataLoader(val_data,

                                            batch_size=BATCH_SIZE,

                                            shuffle=False,

                                           num_workers=NUM_CORES

                                            )



# get one of train batch

images, masks, targets = next(iter(train_loader))



print(images.shape)

print(masks.shape)

print(targets.shape)
# display an image from the batch

image = images[0, 0, :, :]

mask = masks[0, 0, :, :]



print(image.min())

print(image.max())



plt.imshow(image)

plt.imshow(mask, cmap='Reds', alpha=0.3)



plt.show()
# Display the image batch in a grid



grid = torchvision.utils.make_grid(images) # nrows is the number of classes



plt.figure(figsize=(15,15))

plt.imshow(np.transpose(grid, (1,2,0)))

plt.show()
# Display the mask batch in a grid



grid = torchvision.utils.make_grid(masks) # nrows is the number of classes



plt.figure(figsize=(15,15))

plt.imshow(np.transpose(grid, (1,2,0)))

plt.show()
# Model Workflow





# (1) Segmentation Model

# .......................



# seg_model Input: 

# 3x512x512 RGB pre-processed image



# seg_model Output:

# Mask with float values in range 0 to 1







# (2) Intermediate Step

# .......................



# Threshold the seg_model output to obtain a binary mask.

# Take the image that was used as the input to the seg_model and multiply it by the binary mask.







# (3) Regression Model

# .......................

# Resnet34 was used as the seg_model encoder therefore, we are using resnet34 in the reg_model also

# so that the image pre-processing will not need to be done again.



# reg_model Input: 

# 3x512x512 pre-processed image that has been multiplied by a binary mask.



# reg_model Output: 

# Number of wheat heads on the image.

import segmentation_models_pytorch as smp
# seg_model Input: 

# .................



# 3x512x512 RGB pre-processed image





# seg_model Output:

# .................



# Mask with values in range 0 to 1





ENCODER_WEIGHTS = 'imagenet'

ACTIVATION = 'sigmoid' 



# create segmentation model with pretrained encoder

seg_model = smp.Unet(

    encoder_name=BACKBONE, 

    encoder_weights=ENCODER_WEIGHTS, 

    classes=1, 

    activation=ACTIVATION,

)



preprocessing_fn = smp.encoders.get_preprocessing_fn(BACKBONE, ENCODER_WEIGHTS)



print(seg_model)
from torchvision import models
# reg_model Input: 

# .................



# 3x512x512 pre-processed image that has been multiplied by a binary mask.



# reg_model Output: 

# ..................



# Number of wheat heads on the image.





reg_model = models.resnet34(pretrained=True)

in_features = reg_model.fc.in_features # If we print the architecture we see this number in the last layer.



reg_model.fc = nn.Linear(in_features, 1)

#reg_model.out = nn.ReLU(inplace=True)





print(reg_model)
# get one train batch

images, masks, targets = next(iter(train_loader))



print(images.shape)

print(masks.shape)

print(targets.shape)
# Here we simply pass the input images through the model like we are making a prediction.

seg_preds = seg_model(images)



# loss

seg_criterion = smp.utils.losses.DiceLoss()

seg_loss = seg_criterion(seg_preds, masks).item()





print(seg_preds.shape)

print(seg_loss)
np_seg_preds = seg_preds.detach().numpy()





# reshape

np_seg_preds = np_seg_preds.reshape(-1, 512, 512, 1)



print(type(np_seg_preds))

print(np_seg_preds.min())

print(np_seg_preds.max())

print(np_seg_preds.shape)



# The range is 0 to 1 because of the sigmoid layer.
# Print a predicted mask.



pred_mask = np_seg_preds[1, :, :, 0]



print(pred_mask.shape)



plt.imshow(pred_mask)

plt.show()
# Threshold the predicted mask

# Note that seg_preds is of type torch tensor because we will use it as input for the reg_model.



threshold = 0.7



thresh_masks = (seg_preds >= threshold).int() # change the dtype of the torch tensor to int32



print(thresh_masks.min())

print(thresh_masks.max())

print(thresh_masks.shape)



plt.imshow(thresh_masks[1, 0, :, :])

plt.show()
# Create the input by multiplying the thresh_masks and the seg_model input images.

# When we mutiply we are selecting only parts of the image that are inside the predicted segmentations.

# All other parts of the image, outside the segmentations, are set to zero i.e. they become black.



# reg_input = thresh_masks * images # This line multiplies torch tensors. 

reg_input = multiply_masks_and_images(images, thresh_masks) # Here the multiplication is done using numpy.



print(reg_input.min())

print(reg_input.max())

print(reg_input.shape)

# Convert to numpy so we can use plt to display an image



np_reg_input = reg_input.numpy()



np_reg_input = np_reg_input.reshape((-1, 512, 512, 3))



print(np_reg_input.shape)



image = np_reg_input[1, :, :, :]



plt.imshow(image)



plt.show()
# Here we pass the processed seg_model preds through reg_model.



reg_preds = reg_model(reg_input)



# define the layer as a function

#reg_output = nn.ReLU(inplace=True)

#postive_preds = reg_output(reg_preds)



print(reg_preds.shape)



reg_preds
# loss

reg_criterion = nn.MSELoss()

reg_loss = reg_criterion(reg_preds, targets).item()



reg_loss
# Initialize the dataloaders



train_data = CompDataset(df_train, augmentation=get_training_augmentation(), 

                        preprocessing=get_preprocessing(preprocessing_fn))



# Note that we are not augmenting the validation images.

val_data = CompDataset(df_val, augmentation=None, 

                        preprocessing=get_preprocessing(preprocessing_fn))





train_loader = torch.utils.data.DataLoader(train_data,

                                            batch_size=BATCH_SIZE,

                                            shuffle=True,

                                           num_workers=NUM_CORES

                                            )



val_loader = torch.utils.data.DataLoader(val_data,

                                            batch_size=BATCH_SIZE,

                                            shuffle=False,

                                           num_workers=NUM_CORES

                                            )
# send the model to the device

seg_model.to(device)

reg_model.to(device)



# instantiate the optimizers

seg_optimizer = torch.optim.Adam(seg_model.parameters(), lr=LRATE)

reg_optimizer = torch.optim.Adam(reg_model.parameters(), lr=LRATE)



# define the loss functions

seg_criterion = smp.utils.losses.DiceLoss()

reg_criterion = nn.MSELoss()





seg_val_loss_list = []

reg_val_loss_list = []





for epoch in range(0, NUM_EPOCHS):

    

    print('\n') 

    print('Epoch:', epoch)

    print('Train steps:', len(train_loader))

    

    

    # ====================

    # TRAINING

    # ====================

    

    # Set the Mode

    seg_model.train()

    reg_model.train()

    

    # Turn gradient calculations on.

    torch.set_grad_enabled(True)

    

    epoch_loss = 0

    seg_epoch_loss = 0

    reg_epoch_loss = 0

    

    for i, batch in enumerate(train_loader):

        

        

        print(i, end="\r") 



        

        # Get a batch and send to device

        images, masks, reg_targets = batch

        

        images = images.to(device, dtype=torch.float)

        masks = masks.to(device, dtype=torch.float)

        reg_targets = reg_targets.squeeze(dim=1)

        reg_targets = reg_targets.to(device, dtype=torch.float)

        

        

        

        # (1) SEGMENTATION MODEL

        # .......................

        

        

        # pass the input through the model

        seg_preds = seg_model(images)

        

        # calculate the loss

        seg_loss = seg_criterion(seg_preds, masks)

        

        

        seg_optimizer.zero_grad()

        

        seg_loss.backward() # Calculate Gradients

        seg_optimizer.step() # Update Weights

        

        # accumulate the loss for the epoch

        seg_epoch_loss = seg_epoch_loss + seg_loss.item()

        

        

        # (2) PROCESS SEGMENTATION MODEL OUTPUT

        # ......................................

        

        # threshold the predicted segmentation masks

        thresh_masks = (seg_preds >= THRESHOLD).int()

        

        # Multiply the thresholded masks by the RGB images.

        # The result will be an image with 3 channels.

        # The wheat heads that are inside the segmentation will

        # be visible. Everything else on the image will be black.

        #seg_output_masks = thresh_masks * images

        

        # do the multiplication with numpy

        seg_output_masks = multiply_masks_and_images(images, thresh_masks)

        

        # send the masks to the device

        seg_output_masks = seg_output_masks.to(device, dtype=torch.float)

        

        

        # (3) REGRESSION MODEL

        # .....................

        

        # pass the input through the model

        reg_preds = reg_model(seg_output_masks)

        

        # calculate the loss

        reg_preds = reg_preds.squeeze(dim=1)

        

        reg_loss = reg_criterion(reg_preds, reg_targets)

        

        

        reg_optimizer.zero_grad()

        

        reg_loss.backward() # Calculate Gradients

        reg_optimizer.step() # Update Weights

        

        # accumulate the loss for the epoch

        reg_epoch_loss = reg_epoch_loss + reg_loss.item()

        

      

    # get the avarage loss for the epoch

    seg_avg_loss = seg_epoch_loss/len(train_loader)

    reg_avg_loss = reg_epoch_loss/len(train_loader)



    print('Train dice loss:', seg_avg_loss)

    print('Train mse loss:', reg_avg_loss)

    #print('\n')

    

    

    

    

    

    

    # ====================

    # VALIDATION

    # ====================

    

    # Set the Mode

    seg_model.eval()

    reg_model.eval()

    

    # Turn gradient calculations off.

    torch.set_grad_enabled(False)

    

    epoch_loss = 0

    seg_epoch_loss = 0

    reg_epoch_loss = 0

    

    print('---')

    print('Val steps:', len(val_loader))

    

    for i, batch in enumerate(val_loader):

        

        print(i, end="\r") 



        

        # Get a batch and send to device

        images, masks, reg_targets = batch

        

        images = images.to(device, dtype=torch.float)

        masks = masks.to(device, dtype=torch.float)

        reg_targets = reg_targets.squeeze(dim=1)

        reg_targets = reg_targets.to(device, dtype=torch.float)

        

        

        

        # (1) SEGMENTATION MODEL

        # .......................

        

        

        # pass the input through the model

        seg_preds = seg_model(images)

        

        # calculate the loss

        seg_loss = seg_criterion(seg_preds, masks)

        

        

        # accumulate the loss for the epoch

        seg_epoch_loss = seg_epoch_loss + seg_loss.item()

        

        

        # (2) PROCESS SEGMENTATION MODEL OUTPUT

        # ......................................

        

        # threshold the predicted segmentation masks

        thresh_masks = (seg_preds >= THRESHOLD).int()

        

        # Multiply the thresholded masks by the RGB images.

        # The result will be an image with 3 channels.

        # The wheat heads that are inside the segmentation will

        # be visible. Everything else on the image will be black.

        #seg_output_masks = thresh_masks * images

        

        # do the multiplication with numpy

        seg_output_masks = multiply_masks_and_images(images, thresh_masks)

        

        # send the masks to the device

        seg_output_masks = seg_output_masks.to(device, dtype=torch.float)

        

        

        # (3) REGRESSION MODEL

        # .....................

        

        # pass the input through the model

        reg_preds = reg_model(seg_output_masks)

        

        # calculate the loss

        reg_preds = reg_preds.squeeze(dim=1)

        

        reg_loss = reg_criterion(reg_preds, reg_targets)

        

        

        # accumulate the loss for the epoch

        reg_epoch_loss = reg_epoch_loss + reg_loss.item()

        

      

    # get the avarage loss for the epoch

    seg_avg_loss = seg_epoch_loss/len(val_loader)

    reg_avg_loss = reg_epoch_loss/len(val_loader)

    



    print('Val dice loss:', seg_avg_loss)

    print('Val mse loss:', reg_avg_loss)

    

    

    

    # Save the models

    # ----------------

    

    if epoch == 0:

        

        # save both models

        torch.save(seg_model.state_dict(), 'seg_model.pt')

        torch.save(reg_model.state_dict(), 'reg_model.pt')

        print('Both models saved.')

        

        

    if epoch != 0:

        

        # Be sure to calculate these variables before 

        # appending the new loss values to the lists.

        best_val_seg_avg_loss = min(seg_val_loss_list)

        best_val_reg_avg_loss = min(reg_val_loss_list)

 

        if seg_avg_loss < best_val_seg_avg_loss:

            # save the model

            torch.save(seg_model.state_dict(), 'seg_model.pt')

            print('Val dice loss improved. Saved model as seg_model.pt')

      

        if reg_avg_loss < best_val_reg_avg_loss:

            # save the model

            torch.save(reg_model.state_dict(), 'reg_model.pt')

            print('Val mse loss improved. Saved model as reg_model.pt')

            

            

    # append the loss values to the lists   

    seg_val_loss_list.append(seg_avg_loss)

    reg_val_loss_list.append(reg_avg_loss)

    
# Check that the models have been saved.



!ls
# Load the saved models

seg_model.load_state_dict(torch.load('seg_model.pt'))

reg_model.load_state_dict(torch.load('reg_model.pt'))



# Make a prediction on the val set

for i, batch in enumerate(val_loader):

        

        print(i, end="\r") 



        

        # Get a batch and send to device

        images, masks, reg_targets = batch

        

        images = images.to(device, dtype=torch.float)

        masks = masks.to(device, dtype=torch.float)

        reg_targets = reg_targets.squeeze(dim=1)

        reg_targets = reg_targets.to(device, dtype=torch.float)

        

        

        

        # (1) SEGMENTATION MODEL

        # .......................

        

        

        # pass the input through the model

        seg_preds = seg_model(images)

        

        

        

        # (2) PROCESS SEGMENTATION MODEL OUTPUT

        # ......................................

        

        # threshold the predicted segmentation masks

        thresh_masks = (seg_preds >= THRESHOLD).int()

        

        # Multiply the thresholded masks by the RGB images.

        # The result will be an image with 3 channels.

        # The wheat heads that are inside the segmentation will

        # be visible. Everything else on the image will be black.

        #seg_output_masks = thresh_masks * images

        

        # do the multiplication with numpy

        seg_output_masks = multiply_masks_and_images(images, thresh_masks)

        

        # send the masks to the device

        seg_output_masks = seg_output_masks.to(device, dtype=torch.float)

        

        

        # (3) REGRESSION MODEL

        # .....................

        

        # pass the input through the model

        reg_preds = reg_model(seg_output_masks)

        

        

        

         # Stack the predictions from each batch

        if i == 0:

            stacked_images = images

            stacked_masks = masks

            #stacked_thresh_masks = thresh_masks

            stacked_reg_targets = reg_targets

            

            

            stacked_seg_preds = seg_preds

            stacked_seg_output_masks = seg_output_masks

            stacked_reg_preds = reg_preds

            

        else:

            

            stacked_images = torch.cat((stacked_images, images), dim=0)

            stacked_masks = torch.cat((stacked_masks, masks), dim=0)

            #stacked_thresh_masks = torch.cat((stacked_thresh_masks, thresh_masks), dim=0)

            

            stacked_reg_targets = torch.cat((stacked_reg_targets, reg_targets), dim=0)

            

            stacked_seg_preds = torch.cat((stacked_seg_preds, seg_preds), dim=0)

            stacked_seg_output_masks = torch.cat((stacked_seg_output_masks, seg_output_masks), dim=0)

            stacked_reg_preds = torch.cat((stacked_reg_preds, reg_preds), dim=0)

            

# True    

print(stacked_images.shape)

print(stacked_masks.shape)

#print(stacked_thresh_masks.shape)

print(stacked_reg_targets.shape)

print('\n')



# Predicted

print(stacked_seg_preds.shape)

print(stacked_seg_output_masks.shape)

print(stacked_reg_preds.shape)
# Convert to numpy



np_stacked_images = stacked_images.cpu().numpy() ##

np_stacked_masks = stacked_masks.cpu().numpy() ##

#np_stacked_thresh_masks = stacked_thresh_masks.cpu().numpy()



np_stacked_seg_preds = stacked_seg_preds.cpu().numpy()

np_stacked_seg_output_masks = stacked_seg_output_masks.cpu().numpy() ##



# reshape to channels first

np_stacked_images = np_stacked_images.reshape((-1, 512, 512, 3)) ##

np_stacked_masks = np_stacked_masks.reshape((-1, 512, 512, 1)) ##

#np_stacked_thresh_masks = np_stacked_thresh_masks.reshape((-1, 512, 512, 1))



np_stacked_seg_preds = np_stacked_seg_preds.reshape((-1, 512, 512, 1))

np_stacked_seg_output_masks = np_stacked_seg_output_masks.reshape((-1, 512, 512, 3)) ##



#print(np_stacked_images.shape)

#print(np_stacked_masks.shape)

#print(np_stacked_thresh_masks.shape)

#print(np_stacked_seg_preds.shape)

#print(np_stacked_seg_output_masks.shape)
for index in range(1, 4):



    # set up the canvas for the subplots

    plt.figure(figsize=(15,15))

    plt.tight_layout()

    plt.axis('off')







    plt.subplot(1,4,1)

    true_image = np_stacked_images[index, :, :, :]

    plt.imshow(true_image)

    plt.title('Pre-processed Image', fontsize=14)

    plt.axis('off')



    plt.subplot(1,4,2)

    true_mask = np_stacked_masks[index, :, :, 0]

    plt.imshow(true_mask, cmap='Reds', alpha=0.3)

    plt.title('True Mask', fontsize=14)

    plt.axis('off')



    plt.subplot(1,4,3)

    pred_mask = np_stacked_seg_preds[index, :, :, 0]

    plt.imshow(pred_mask, cmap='Blues', alpha=0.3)

    plt.title('Seg model Pred Mask', fontsize=14)

    plt.axis('off')

    

    plt.subplot(1,4,4)

    pred_mask = np_stacked_seg_output_masks[index, :, :, :]

    #thresh_mask = np_stacked_thresh_masks[index, :, :, 0]

    

    plt.imshow(pred_mask)

    #plt.imshow(thresh_mask, cmap='Reds', alpha=0.3)

    plt.title('Reg model input', fontsize=14)

    plt.axis('off')



    plt.show()

    
# Change to numpy



np_stacked_reg_targets = stacked_reg_targets.cpu().numpy()



np_stacked_reg_preds = stacked_reg_preds.cpu().numpy()

np_stacked_reg_preds = np_stacked_reg_preds.squeeze()



#print(np_stacked_reg_targets.shape)

#print(np_stacked_reg_preds.shape)







# Add the predictions to a dataframe



df_val['true_count'] = np_stacked_reg_targets



df_val['pred_count'] = np_stacked_reg_preds





# Create more columns



df_val['pred_error'] = df_val['true_count'] - df_val['pred_count']



df_val['abs_error'] = df_val['pred_error'].apply(abs)



df_val['percent_error'] = (df_val['abs_error']/df_val['num_masks'])*100







cols = ['image_id', 'num_masks', 'true_count', 'pred_count', 'pred_error', 'abs_error']

df = df_val[cols]



#df.head(10)
df = df_val[df_val['percent_error'] < 10]



message = str(len(df)) + ' of 389 val images have < 10% error.'



print(message)



df = df.reset_index(drop=True)



# set up the canvas for the subplots

plt.figure(figsize=(15,15))



plt.subplot(3,4,1)



# Our subplot will contain 3 rows and 3 columns

# plt.subplot(nrows, ncols, plot_number)





for i in range(1, 10):

    

    # image

    plt.subplot(3,3,i)



    path = base_path + 'train/' + df.loc[i, 'image_id'] + '.jpg'

    image = plt.imread(path)



    true = df.loc[i, 'num_masks']



    # round the pred

    pred = round(df.loc[i, 'pred_count'], 0)

    # convert to type int

    pred = int(pred)



    source = df.loc[i, 'source']

    

    result = 'True: ' + str(true) + ' Pred: ' + str(pred) + ' -- ' + source

    

    plt.imshow(image)

    plt.title(result, fontsize=18)

    #plt.xlabel(source, fontsize=10)

    plt.tight_layout()

    plt.axis('off')

df = df_val[df_val['percent_error'] > 50]



message = str(len(df)) + ' of 389 val images have > 50% error.'



print(message)



df = df.reset_index(drop=True)



# set up the canvas for the subplots

plt.figure(figsize=(15,15))



plt.subplot(3,4,1)



# Our subplot will contain 3 rows and 3 columns

# plt.subplot(nrows, ncols, plot_number)





for i in range(1, 10):

    

    # image

    plt.subplot(3,3,i)



    path = base_path + 'train/' + df.loc[i, 'image_id'] + '.jpg'

    image = plt.imread(path)



    true = df.loc[i, 'num_masks']



    # round the pred

    pred = round(df.loc[i, 'pred_count'], 0)

    # convert to type int

    pred = int(pred)



    source = df.loc[i, 'source']

    

    result = 'True: ' + str(true) + ' Pred: ' + str(pred) + ' -- ' + source

    

    plt.imshow(image)

    plt.title(result, fontsize=18)

    #plt.xlabel(source, fontsize=10)

    plt.tight_layout()

    plt.axis('off')

print('Total val percentage error:', (df_val['abs_error'].sum()/df_val['true_count'].sum()) * 100)
# Put the rows from each region into separate dataframes



df_usask_1 = df_val[df_val['source'] == 'usask_1']

df_arvalis_2 = df_val[df_val['source'] == 'arvalis_2']
# Check the prediction error stats for the entire val set

print('MAE for all validation data:')

print('Val set MAE:', df_val['abs_error'].mean())

print('')

print('MAE by source:')

print('usask_1 MAE:', df_usask_1['abs_error'].mean())

print('arvalis_2 MAE:', df_arvalis_2['abs_error'].mean())
# Save df_val for future analysis



df_val.to_csv('df_val.csv.gz', compression='gzip', index=False)
# Check if the dataframe was saved.

!ls