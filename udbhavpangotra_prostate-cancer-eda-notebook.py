#  Data Import

import numpy as np 
import pandas as pd 
import os
import matplotlib.pyplot as plt
import seaborn as sns
import PIL
from IPython.display import Image, display
from plotly import graph_objs as go
import plotly.express as px
import plotly.figure_factory as ff
import plotly.io as pio

import openslide

BASE_FOLDER = "/kaggle/input/prostate-cancer-grade-assessment/"
!ls {BASE_FOLDER}

mask_dir = f'{BASE_FOLDER}/train_label_masks'
train = pd.read_csv(BASE_FOLDER+"train.csv")
test = pd.read_csv(BASE_FOLDER+"test.csv")
sub = pd.read_csv(BASE_FOLDER+"sample_submission.csv")
train.head()
print("unique ids                 : ", len(train.image_id.unique()))
print("unique data provider       : ", len(train.data_provider.unique()))
print("unique isup_grade(target)  : ", len(train.isup_grade.unique()))
print("unique gleason_score       : ", len(train.gleason_score.unique()))
sns.countplot(train.data_provider,order = train['data_provider'].value_counts().index, palette='Set2')
plt.title('Data Providers')
feature='isup_grade'
hue='data_provider'
sns.countplot(x=feature, hue=hue, data=train, palette='Set2')
plt.title('Data Provider and ISUP Grade')
train.gleason_score.value_counts()
train.isup_grade.value_counts()
print(train[(train['gleason_score']=='3+4') | (train['gleason_score']=='4+3')]['isup_grade'].value_counts())
print()
print(train[(train['gleason_score']=='3+5') | (train['gleason_score']=='5+3')]['isup_grade'].value_counts())
print()
print(train[(train['gleason_score']=='5+4') | (train['gleason_score']=='4+5')]['isup_grade'].value_counts())
train[(train['isup_grade'] == 2) & (train['gleason_score'] == '4+3')]
train.drop([7273],inplace=True)
train['gleason_score'] = train['gleason_score'].apply(lambda x: "0+0" if x=="negative" else x)
# also removing the negative datapont and replacing it with a 0+0 score since that makes more sense.
train.gleason_score.value_counts()
print("shape                        : ", test.shape)
print("unique ids                   : ", len(test.image_id.unique()))
print("unique data provider         : ", len(test.data_provider.unique()))
temp = train.groupby('isup_grade').count()['image_id'].reset_index().sort_values(by='image_id',ascending=False)
temp.style.background_gradient(cmap='Greens')
fig = px.bar(temp, x='isup_grade', y='image_id',
             hover_data=['image_id', 'isup_grade'], color= 'isup_grade', height=400, width = 600)
fig.show()
temp = train.groupby('gleason_score').count()['image_id'].reset_index().sort_values(by='image_id',ascending=False)
temp.style.background_gradient(cmap='Blues')
fig = px.bar(temp, x='gleason_score', y='image_id',
             hover_data=['image_id', 'gleason_score'], color= 'gleason_score', height=400, width = 600)
fig.show()
data = [dict(
  type = 'scatter',
  x = train.isup_grade,
  y = train.gleason_score,
  mode = 'markers',
  transforms = [dict(
    type = 'groupby',
    groups = train.isup_grade,
    styles = [
        dict(target = '0', value = dict(marker = dict(color = 'blue'))),
        dict(target = '1', value = dict(marker = dict(color = 'red'))),
        dict(target = '2', value = dict(marker = dict(color = 'black'))),
        dict(target = '3', value = dict(marker = dict(color = 'green'))), 
        dict(target = '4', value = dict(marker = dict(color = 'brown'))),
        dict(target = '5', value = dict(marker = dict(color = 'violet'))  
              
            )
    ]
  )]
)]


fig_dict = dict(data=data)
pio.show(fig_dict, validate=False,width=600, height=150)
data_dir = f'{BASE_FOLDER}/train_images'
train = train.set_index('image_id')
train.head()
'''
Example for using Openslide to display an image
'''


# Open the image (does not yet read the image into memory)
example = openslide.OpenSlide(os.path.join(BASE_FOLDER+"train_images", '005e66f06bce9c2e49142536caf2f6ee.tiff'))

# Read a specific region of the image starting at upper left coordinate (x=17800, y=19500) on level 0 and extracting a 256*256 pixel patch.
# At this point image data is read from the file and loaded into memory.
patch = example.read_region((17800,19500), 0, (256, 256))

# Display the image
display(patch)

# Close the opened slide after use
example.close()
def get_values(image,max_size=(600,400)):
    slide = openslide.OpenSlide(os.path.join(BASE_FOLDER+"train_images", f'{image}.tiff'))
    
    # Here we compute the "pixel spacing": the physical size of a pixel in the image.
    # OpenSlide gives the resolution in centimeters so we convert this to microns.
    f,ax =  plt.subplots(2 ,figsize=(6,16))
    spacing = 1 / (float(slide.properties['tiff.XResolution']) / 10000)
    patch = slide.read_region((1780,1950), 0, (256, 256)) #ZOOMED FUGURE
    ax[0].imshow(patch) 
    ax[0].set_title('Zoomed Image')
    
    
    ax[1].imshow(slide.get_thumbnail(size=max_size)) #UNZOOMED FIGURE
    ax[1].set_title('Full Image')
    
    
    print(f"File id: {slide}")
    print(f"Dimensions: {slide.dimensions}")
    print(f"Microns per pixel / pixel spacing: {spacing:.3f}")
    print(f"Number of levels in the image: {slide.level_count}")
    print(f"Downsample factor per level: {slide.level_downsamples}")
    print(f"Dimensions of levels: {slide.level_dimensions}\n\n")
    
    print(f"ISUP grade: {train.loc[image, 'isup_grade']}")
    print(f"Gleason score: {train.loc[image, 'gleason_score']}")
get_values('07a7ef0ba3bb0d6564a73f4f3e1c2293')
def display_images_zoom(images):
    '''
    This function takes in input a list of images. It then iterates through the image making openslide objects , on which different functions
    for getting out information can be called later
    '''
    f, ax = plt.subplots(5,3, figsize=(15,25))
    for i, image in enumerate(images):
        slide = openslide.OpenSlide(os.path.join(BASE_FOLDER+"train_images", f'{image}.tiff')) # Making Openslide Object
        #Here we compute the "pixel spacing": the physical size of a pixel in the image,
        #OpenSlide gives the resolution in centimeters so we convert this to microns
        spacing = 1/(float(slide.properties['tiff.XResolution']) / 10000)
        patch = slide.read_region((1780,1950), 0, (256, 256)) #Reading the image as before betweeen x=1780 to y=1950 and of pixel size =256*256
        ax[i//3, i%3].imshow(patch) #Displaying Image
        slide.close()       
        ax[i//3, i%3].axis('off')
        

        
        image_id = image
        data_provider = train.loc[image, 'data_provider']
        isup_grade = train.loc[image, 'isup_grade']
        gleason_score = train.loc[image, 'gleason_score']
        ax[i//3, i%3].set_title(f"ID: {image_id}\nSource: {data_provider} ISUP: {isup_grade} Gleason: {gleason_score}")

    plt.show() 
def display_images_full(images):
    '''
    This function takes in input a list of images. It then iterates through the image making openslide objects , on which different functions
    for getting out information can be called later
    '''
    f, ax = plt.subplots(5,3, figsize=(15,25))
    for i, image in enumerate(images):
        slide = openslide.OpenSlide(os.path.join(BASE_FOLDER+"train_images", f'{image}.tiff')) # Making Openslide Object
        #Here we compute the "pixel spacing": the physical size of a pixel in the image,
        #OpenSlide gives the resolution in centimeters so we convert this to microns
        spacing = 1/(float(slide.properties['tiff.XResolution']) / 10000)

        max_size=(600,400)
        ax[i//3, i%3].imshow(slide.get_thumbnail(size=max_size))
        ax[i//3, i%3].set_title('Full Image')
                

        
        image_id = image
        data_provider = train.loc[image, 'data_provider']
        isup_grade = train.loc[image, 'isup_grade']
        gleason_score = train.loc[image, 'gleason_score']
        ax[i//3, i%3].set_title(f"ID: {image_id}\nSource: {data_provider} ISUP: {isup_grade} Gleason: {gleason_score}")

    plt.show() 


images = [
'07a7ef0ba3bb0d6564a73f4f3e1c2293',
    '037504061b9fba71ef6e24c48c6df44d',
    '035b1edd3d1aeeffc77ce5d248a01a53',
    '059cbf902c5e42972587c8d17d49efed',
    '06a0cbd8fd6320ef1aa6f19342af2e68',
    '06eda4a6faca84e84a781fee2d5f47e1',
    '0a4b7a7499ed55c71033cefb0765e93d',
    '0838c82917cd9af681df249264d2769c',
    '046b35ae95374bfb48cdca8d7c83233f',
    '074c3e01525681a275a42282cd21cbde',
    '05abe25c883d508ecc15b6e857e59f32',
    '05f4e9415af9fdabc19109c980daf5ad',
    '060121a06476ef401d8a21d6567dee6d',
    '068b0e3be4c35ea983f77accf8351cc8',
    '08f055372c7b8a7e1df97c6586542ac8'
]
display_images_zoom(images)
display_images_full(images)
os.path
!ls {BASE_FOLDER}

mask_dir = f'{BASE_FOLDER}/train_label_masks'
import matplotlib
def display_mask(slide):    
        
        max_size=(600,400)
        mask = openslide.OpenSlide(os.path.join(mask_dir, f'{slide}_mask.tiff'))
        mask_data = mask.read_region((0,0), mask.level_count - 1, mask.level_dimensions[-1])
        cmap = matplotlib.colors.ListedColormap(['black', 'gray', 'green', 'yellow', 'orange', 'red'])

        ax[0].imshow(np.asarray(mask_data)[:,:,0], cmap=cmap, interpolation='nearest', vmin=0, vmax=5)       
        ax[0].axis('off')
        
        ax[1].imshow(slide.get_thumbnail(size=max_size)) #UNZOOMED FIGURE
        ax[1].set_title('Full Image')
        
        data_provider = train.loc[slide, 'data_provider']
        isup_grade = train.loc[slide, 'isup_grade']
        gleason_score = train.loc[slide, 'gleason_score']
        ax[0].set_title(f"ID: {image_id}\nSource: {data_provider} ISUP: {isup_grade} Gleason: {gleason_score}")
        f.tight_layout()
        
        plt.show()

def mask_img(image,max_size=(600,400)):
    slide = openslide.OpenSlide(os.path.join(BASE_FOLDER+"train_images", f'{image}.tiff'))
    mask =  openslide.OpenSlide(os.path.join(mask_dir, f'{image}_mask.tiff'))
    # Here we compute the "pixel spacing": the physical size of a pixel in the image.
    # OpenSlide gives the resolution in centimeters so we convert this to microns.
    f,ax =  plt.subplots(1,2 ,figsize=(18,22))
    spacing = 1 / (float(slide.properties['tiff.XResolution']) / 10000)
    img = slide.get_thumbnail(size=(600,400)) #IMAGE 
    
    mask_data = mask.read_region((0,0), mask.level_count - 1, mask.level_dimensions[-1])
    cmap = matplotlib.colors.ListedColormap(['black', 'gray', 'green', 'yellow', 'orange', 'red'])
    
    ax[0].imshow(img) 
    #ax[0].set_title('Image')
    
    
    ax[1].imshow(np.asarray(mask_data)[:,:,0], cmap=cmap, interpolation='nearest', vmin=0, vmax=5) #IMAGE MASKS
    #ax[1].set_title('Image_MASK')
    
    
    image_id = image
    data_provider = train.loc[image, 'data_provider']
    isup_grade = train.loc[image, 'isup_grade']
    gleason_score = train.loc[image, 'gleason_score']
    ax[0].set_title(f"ID: {image_id}\nSource: {data_provider} ISUP: {isup_grade} Gleason: {gleason_score} IMAGE")
    ax[1].set_title(f"ID: {image_id}\nSource: {data_provider} ISUP: {isup_grade} Gleason: {gleason_score} IMAGE_MASK")
images_tmp= [
    '08ab45297bfe652cc0397f4b37719ba1',
    '090a77c517a7a2caa23e443a77a78bc7'
]

for image in images_tmp:
    mask_img(image)
IMG_PATH = '../input/prostate-cancer-grade-assessment/train_images/'
MASK_PATH = '../input/prostate-cancer-grade-assessment/train_label_masks/'

data = pd.read_csv('../input/prostate-cancer-grade-assessment/train.csv')


def read_mask(ID, path, level=2):
    mask_path = path + ID + '_mask.tiff'
    mask = skimage.io.MultiImage(mask_path)
    mask = mask[level]
    return mask[:, :, 0]

def read_img(ID, path, level=2):
    image_path = path + ID + '.tiff'
    img = skimage.io.MultiImage(image_path)
    img = img[level]
    return img

def compute_rects(mask, min_rect=32):
    
    contours, _ = cv2.findContours(
        mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    height, width = mask.shape

    boxes = []
    dims = []
    for contour in contours:
        rect = cv2.minAreaRect(contour)
        height = rect[1][1]
        width = rect[1][0]
        if width > min_rect and height > min_rect:
            dims.append((int(width), int(height)))
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            boxes.append(box)
    
    return dims, boxes

def warp_perspectives(img, dims, boxes):

    imgs = []
    for (width, height), box in zip(dims, boxes):
        src_pts = box.astype("float32")
        dst_pts = np.array([[0, height-1],
                            [0, 0],
                            [width-1, 0],
                            [width-1, height-1]], dtype="float32")
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        warped = cv2.warpPerspective(img, M, (width, height))
        imgs.append(warped)
    return imgs
ID = data.iloc[10606].image_id
mask_img(ID)


mask = read_mask(ID, MASK_PATH)
img = read_img(ID, IMG_PATH)
mask_background = np.where(mask == 0, 1, 0).astype(np.uint8)
mask_benign = np.where(mask == 1, 1, 0).astype(np.uint8)
mask_cancerous = np.where(mask == 2, 1, 0).astype(np.uint8)

fig, axes = plt.subplots(1, 3, figsize=(14, 7))

axes[0].imshow(mask_background.astype(float), cmap=plt.cm.gray)
axes[0].axis('off')
axes[0].set_title('background');
axes[1].imshow(mask_benign.astype(float), cmap=plt.cm.gray)
axes[1].axis('off')
axes[1].set_title('benign');
axes[2].imshow(mask_cancerous.astype(float), cmap=plt.cm.gray)
axes[2].axis('off')
axes[2].set_title('cancerous');



#plt.imshow(mask.astype(np.uint8));
ID = data.iloc[10612].image_id
mask_img(ID)

mask = read_mask(ID, MASK_PATH, level=1)
img = read_img(ID, IMG_PATH, level=1)
mask_background = np.where(mask == 0, 1, 0).astype(np.uint8)
mask_benign = np.where(mask == 1, 1, 0).astype(np.uint8)
mask_cancerous = np.where(mask == 5, 1, 0).astype(np.uint8)

fig, axes = plt.subplots(1,3, figsize=(14, 7))

axes[0].imshow(mask_background.astype(float), cmap=plt.cm.gray)
axes[0].axis('off')
axes[0].set_title('background');
axes[1].imshow(mask_benign.astype(float), cmap=plt.cm.gray)
axes[1].axis('off')
axes[1].set_title('benign');
axes[2].imshow(mask_cancerous.astype(float), cmap=plt.cm.gray)
axes[2].axis('off')
axes[2].set_title('cancerous');

ID
def overlay_mask_on_slide(images, center='radboud', alpha=0.8, max_size=(800, 800)):
    """Show a mask overlayed on a slide."""
    f, ax = plt.subplots(2,3, figsize=(18,22))
    
    
    for i, image_id in enumerate(images):
        slide = openslide.OpenSlide(os.path.join(BASE_FOLDER+"train_images", f'{image_id}.tiff'))
        mask = openslide.OpenSlide(os.path.join(mask_dir, f'{image_id}_mask.tiff'))
        slide_data = slide.read_region((0,0), slide.level_count - 1, slide.level_dimensions[-1])
        mask_data = mask.read_region((0,0), mask.level_count - 1, mask.level_dimensions[-1])
        mask_data = mask_data.split()[0]
        
        # Create alpha mask
        alpha_int = int(round(255*alpha))
        if center == 'radboud':
            alpha_content = np.less(mask_data.split()[0], 2).astype('uint8') * alpha_int + (255 - alpha_int)
        elif center == 'karolinska':
            alpha_content = np.less(mask_data.split()[0], 1).astype('uint8') * alpha_int + (255 - alpha_int)

        alpha_content = PIL.Image.fromarray(alpha_content)
        preview_palette = np.zeros(shape=768, dtype=int)

        if center == 'radboud':
            # Mapping: {0: background, 1: stroma, 2: benign epithelium, 3: Gleason 3, 4: Gleason 4, 5: Gleason 5}
            preview_palette[0:18] = (np.array([0, 0, 0, 0.5, 0.5, 0.5, 0, 1, 0, 1, 1, 0.7, 1, 0.5, 0, 1, 0, 0]) * 255).astype(int)
        elif center == 'karolinska':
            # Mapping: {0: background, 1: benign, 2: cancer}
            preview_palette[0:9] = (np.array([0, 0, 0, 0, 1, 0, 1, 0, 0]) * 255).astype(int)

        mask_data.putpalette(data=preview_palette.tolist())
        mask_rgb = mask_data.convert(mode='RGB')
        overlayed_image = PIL.Image.composite(image1=slide_data, image2=mask_rgb, mask=alpha_content)
        overlayed_image.thumbnail(size=max_size, resample=0)

        
        ax[i//3, i%3].imshow(overlayed_image) 
        slide.close()
        mask.close()       
        ax[i//3, i%3].axis('off')
        
        data_provider = train.loc[image_id, 'data_provider']
        isup_grade = train.loc[image_id, 'isup_grade']
        gleason_score = train.loc[image_id, 'gleason_score']
        ax[i//3, i%3].set_title(f"ID: {image_id}\nSource: {data_provider} ISUP: {isup_grade} Gleason: {gleason_score}")
images[:6]
overlay_mask_on_slide(images[:6])
images_tmp= [
    '07a7ef0ba3bb0d6564a73f4f3e1c2293',
    'ffdc59cd580a1468eac0e6a32dd1ff2d'
]

for image in images_tmp:
    mask_img(image)
pen_marked_images = [
    'fd6fe1a3985b17d067f2cb4d5bc1e6e1',
    'ebb6a080d72e09f6481721ef9f88c472',
    'ebb6d5ca45942536f78beb451ee43cc4',
    'ea9d52d65500acc9b9d89eb6b82cdcdf',
    'e726a8eac36c3d91c3c4f9edba8ba713',
    'e90abe191f61b6fed6d6781c8305fe4b'
]

overlay_mask_on_slide(pen_marked_images)
