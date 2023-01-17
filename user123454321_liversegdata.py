import os

import pandas as pd

import nibabel as nib

import matplotlib.pyplot as plt
images1 = '/kaggle/input/liversegtrainimages1'

train_images = [os.path.join(images1, image_name) for image_name in os.listdir(images1) if image_name.endswith('.nii')]
images2 = '/kaggle/input/liversegtrainimages2'

train_images += [os.path.join(images2, image_name) for image_name in os.listdir(images2) if image_name.endswith('.nii')]
labels = '/kaggle/input/liversegtrainlabels'

train_labels = [os.path.join(labels, image_name) for image_name in os.listdir(labels) if image_name.endswith('.nii')]
train_images = sorted(train_images, key=lambda path: path.split('/')[-1])
train_labels = sorted(train_labels, key=lambda path: path.split('/')[-1])
train_df = pd.DataFrame({

    'image_path': train_images,

    'label_path': train_labels

})
img = nib.load(train_df.iloc[0].image_path).get_fdata()

label = nib.load(train_df.iloc[0].label_path).get_fdata()
%matplotlib notebook

from matplotlib.colors import from_levels_and_colors

fig, ax = plt.subplots(1,1)

fig.set_figheight(8)

fig.set_figwidth(8)

curr_slice = 0

my_cmap, _ = from_levels_and_colors(levels=[0, 1, 2], colors=[[0., 0., 0., 0.],  [0., 1., 0., 1], [0.8, 0., 0.8, 1]], extend='max')

im1 = ax.imshow(img[:, :, curr_slice], cmap='gray')

im2 = ax.imshow(label[:, :, curr_slice].astype('int'), cmap=my_cmap, vmin=0., vmax=1.)

text = plt.text(img.shape[1] // 2 - 50, 0, "Slice 1", va="bottom", ha="left", fontsize=20)

def change_slice(e):

    global curr_slice

    global im

    global fig

    global img

    if e.key == 'right' and curr_slice < img.shape[2]:

        curr_slice += 1

    if e.key == 'left' and curr_slice > 0:

        curr_slice -= 1

    text.set_text('Slice {}'.format(curr_slice+1))

    im1.set_data(img[:, :, curr_slice])

    im2.set_data(label[:, :, curr_slice].astype('int'))

    fig.canvas.draw()

        

a = fig.canvas.mpl_connect('key_press_event', change_slice)