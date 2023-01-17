import os

import cv2

import skimage.io

from tqdm.notebook import tqdm

import zipfile

import numpy as np

# All imports

import time



import matplotlib.pyplot as plt

import openslide

import pandas as pd

from skimage import morphology

import pprint

import json
TRAIN = '../input/prostate-cancer-grade-assessment/train_images/'

OUT_TRAIN = 'train.zip'



# Set up example slide and run pipeline on low resolution

slide_dir = "../input/prostate-cancer-grade-assessment/train_images/"

annotation_dir = "../input/prostate-cancer-grade-assessment/train_label_masks/"

example_id = "0032bfa835ce0f43a92ae0bbab6871cb"

example_slide = f"{slide_dir}{example_id}.tiff"
def get_disk_size(numpy_image):

    """Return disk size of a numpy array"""

    return (numpy_image.size * numpy_image.itemsize) / 1000000





def detect_tissue_external(input_slide, sensitivity=3000):

    

    """

    Description

    ----------

    Find RoIs containing tissue in WSI and only return the external most.

    Generate mask locating tissue in an WSI. Inspired by method used by

    Wang et al. [1]_.

    .. [1] Dayong Wang, Aditya Khosla, Rishab Gargeya, Humayun Irshad, Andrew

    H. Beck, "Deep Learning for Identifying Metastatic Breast Cancer",

    arXiv:1606.05718

    Credit: Github-wsipre

    

    Parameters

    ----------

    input_slide: numpy array

        Slide to detect tissue on.

    sensitivity: int

        The desired sensitivty of the model to detect tissue. The baseline is set

        at 3000 and should be adjusted down to capture more potential issue and

        adjusted up to be more agressive with trimming the slide.

        

    Returns (3)

    -------

    -Tissue binary mask as numpy 2D array, 

    -Tiers investigated,

    -Time Stamps from running tissue detection pipeline

    """

    

    # For timing

    time_stamps = {}

    time_stamps["start"] = time.time()



    # Convert from RGB to HSV color space

    slide_hsv = cv2.cvtColor(input_slide, cv2.COLOR_BGR2HSV)

    time_stamps["re-color"] = time.time()

    # Compute optimal threshold values in each channel using Otsu algorithm

    _, saturation, _ = np.split(slide_hsv, 3, axis=2)



    mask = otsu_filter(saturation, gaussian_blur=True)

    time_stamps["filter"] = time.time()

    # Make mask boolean

    mask = mask != 0



    mask = morphology.remove_small_holes(mask, area_threshold=sensitivity)

    mask = morphology.remove_small_objects(mask, min_size=sensitivity)

    time_stamps["morph"] = time.time()

    mask = mask.astype(np.uint8)

    mask_contours, tier = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    time_stamps["contour"] = time.time()

    time_stamps = {

        key: (value - time_stamps["start"]) * 1000 for key, value in time_stamps.items()

    }

    

    return mask_contours, tier, time_stamps





def new_detect_and_crop(image_location="",sensitivity: int = 3000, downsample_lvl = -1,

                        show_plots= "simple", out_lvl=-2):

    """

    Description

    ----------

    This method performs the pipeline as described in the notebook:

    https://www.kaggle.com/dannellyz/panda-tissue-detect-scaling-bounding-boxes-fast

    

    Parameters

    ----------

    image_location:str

        Location of the slide image to process

    sensitivity:int

        The desired sensitivty of the model to detect tissue. The baseline is set

        at 3000 and should be adjusted down to capture more potential issue and

        adjusted up to be more agressive with trimming the slide.

    downsample_lvl: int

        The level at which to downsample the slide. This can be referenced in

        reverse order to access the lowest resoltuion items first.

        [-1] = lowest resolution

        [0] = highest resolution

    show_plots: str (verbose|simple|none)

        The types of plots to display:

            - verbose - show all steps of process

            - simple - show only last step

            - none - show none of the plots

    out_lvl: int

        The level at which the final slide should sample at. This can be referenced in

        reverse order to access the lowest resoltuion items first.

        [-1] = lowest resolution

        [0] = highest resolution

    shape: touple

        (height, width) of the desired produciton(prod) image

        

    Returns (4)

    -------

    - Numpy array of final produciton(prod) slide

    - Percent memory reduciton from original slide

    - Time stamps from stages of the pipeline

    - Time stamps from the Tissue Detect pipeline

    """

    # For timing

    time_stamps = {}

    time_stamps["start"] = time.time()



    # Open Small Slide

    wsi_small = skimage.io.MultiImage(image_location)[downsample_lvl]

    time_stamps["open_small"] = time.time()



    # Get returns from detect_tissue() ons mall image

    (   tissue_contours,

        tier,

        time_stamps_detect,

    ) = detect_tissue_external(wsi_small, sensitivity)

    

    base_slide_mask = np.zeros(wsi_small.shape[:2])

    # Get minimal bounding rectangle for all tissue contours

    if len(tissue_contours) == 0:

        img_id = image_location.split("/")[-1]

        print(f"No Tissue Contours - ID: {img_id}")

        return None, 0, None, None

    

    # Open Big Slide

    wsi_big = skimage.io.MultiImage(image_location)[out_lvl]

    time_stamps["open_big"] = time.time()

    

    #Get small boudning rect and scale

    bounding_rect_small = cv2.minAreaRect(np.concatenate(tissue_contours))



    # Scale Rectagle to larger image

    scale = int(wsi_big.shape[0] / wsi_small.shape[0])

    scaled_rect = (

        (bounding_rect_small[0][0] * scale, bounding_rect_small[0][1] * scale),

        (bounding_rect_small[1][0] * scale, bounding_rect_small[1][1] * scale),

        bounding_rect_small[2],

    )

    # Crop bigger image with getSubImage()

    scaled_crop = getSubImage(wsi_big, scaled_rect)

    time_stamps["scale_bounding"] = time.time()

    

    #Cut out white

    white_cut = color_cut(scaled_crop)

    time_stamps["white_cut_big"] = time.time()

    



    # Get returns from detect_tissue() on small image

    (   tissue_contours_big,

        tier_big,

        time_stamps_detect,

    ) = detect_tissue_external(white_cut, sensitivity)

    prod_slide = tissue_cutout(white_cut, tissue_contours_big)

    time_stamps["remove_tissue"] = time.time()



    # Get size change

    base_size_high = get_disk_size(wsi_big)

    final_size = get_disk_size(prod_slide)

    pct_change = final_size / base_size_high

    

    if show_plots == "simple":

        print(f"Percent Reduced from Base Slide to Final: {(1- pct_change)*100:.2f}")

        plt.imshow(smart_bounding_crop)

        plt.show()

    elif show_plots == "verbose":

        # Set-up dictionary for plotting

        verbose_plots = {}

        # Add Base Slide to verbose print

        verbose_plots[f"Smaller Slide\n{get_disk_size(wsi_small):.2f}MB"] = wsi_small

        # Add Tissue Only to verbose print

        verbose_plots[f"Tissue Detect Low\nNo Change"] = wsi_big

        # Add Larger Plot cut with bounding boxes

        verbose_plots[f"Larger scaled\n{get_disk_size(scaled_crop):.2f}MB"] = scaled_crop

        # Add Bounding Boxes to verbose print

        verbose_plots[

            f"Final Produciton\n{get_disk_size(prod_slide):.2f}MB"

        ] = prod_slide

        print(f"Percent Reduced from Base Slide to Final: {(1- pct_change)*100:.2f}")

        plt = plot_figures(verbose_plots, 2, 2)

    elif show_plots == "none":

        pass

    else:

        pass

    time_stamps = {

        key: (value - time_stamps["start"]) * 1000 for key, value in time_stamps.items()

    }

    return prod_slide, (1 - pct_change), time_stamps, time_stamps_detect



def otsu_filter(channel, gaussian_blur=True):

    

    """Otsu filter."""

    

    if gaussian_blur:

        channel = cv2.GaussianBlur(channel, (5, 5), 0)

    channel = channel.reshape((channel.shape[0], channel.shape[1]))



    return cv2.threshold(channel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]



def getSubImage(input_slide, rect):

    

    """

    Description

    ----------

    Take a cv2 rectagle object and remove its contents from

    a source image.

    Credit: https://stackoverflow.com/a/48553593

    

    Parameters

    ----------

    input_slide: numpy array 

            Slide to pull subimage off 

    rect: cv2 rect

        cv2 rectagle object with a shape of-

            ((center_x,center_y), (hight,width), angle)

    

    Returns (1)

    -------

    - Numpy array of rectalge data cut from input slide

    """

    

    width = int(rect[1][0])

    height = int(rect[1][1])

    box = cv2.boxPoints(rect)



    src_pts = box.astype("float32")

    dst_pts = np.array(

        [[0, height - 1], [0, 0], [width - 1, 0], [width - 1, height - 1]],

        dtype="float32",

    )

    M = cv2.getPerspectiveTransform(src_pts, dst_pts)

    output_slide = cv2.warpPerspective(input_slide, M, (width, height))

    return output_slide



def color_cut(in_slide, color = [255,255,255]):

    

    """

    Description

    ----------

    Take a input image and remove all rows or columns that

    are only made of the input color [R,G,B]. The default color

    to cut from image is white.

    

    Parameters

    ----------

    input_slide: numpy array 

        Slide to cut white cols/rows 

    color: list

        List of [R,G,B] pixels to cut from the input slide

    

    Returns (1)

    -------

    - Numpy array of input_slide with white removed

    """

    #Remove by row

    row_not_blank = [row.all() for row in ~np.all(in_slide == color, axis=1)]

    output_slide = in_slide[row_not_blank, :]

    

    #Remove by col

    col_not_blank = [col.all() for col in ~np.all(output_slide == color, axis=0)]

    output_slide = output_slide[:, col_not_blank]

    return output_slide



def tissue_cutout(input_slide, tissue_contours):

    

    """

    Description

    ----------

    Set all parts of the in_slide to black except for those

    within the provided tissue contours

    Credit: https://stackoverflow.com/a/28759496

    

    Parameters

    ----------

    input_slide: numpy array

            Slide to cut non-tissue backgound out

    tissue_contours: numpy array 

            These are the identified tissue regions as cv2 contours

            

    Returns (1)

    -------

    - Numpy array of slide with non-tissue set to black

    """

    

    # Get intermediate slide

    base_slide_mask = np.zeros(input_slide.shape[:2])

    # Create mask where white is what we want, black otherwise

    crop_mask = np.zeros_like(base_slide_mask) 

    

    # Draw filled contour in mask

    cv2.drawContours(crop_mask, tissue_contours, -1, 255, -1) 

    

    # Extract out the object and place into output image

    tissue_only_slide = np.zeros_like(input_slide)  

    tissue_only_slide[crop_mask == 255] = input_slide[crop_mask == 255]

    

    return tissue_only_slide



def plot_figures(figures, nrows=1, ncols=1):

    

    """

    Description

    ----------

    Plot a dictionary of figures.

    Credit: https://stackoverflow.com/a/11172032



    Parameters

    ----------

    figures: dict 

        <title, figure> for those to plot

    ncols: int 

        number of columns of subplots wanted in the display

    nrows: int 

        number of rows of subplots wanted in the figure

    

    Returns(0)

    ----------

    """



    fig, axeslist = plt.subplots(ncols=ncols, nrows=nrows)

    for ind, title in enumerate(figures):

        axeslist.ravel()[ind].imshow(figures[title], aspect="auto")

        axeslist.ravel()[ind].set_title(title)

    plt.tight_layout()

    plt.show()

    return 
(processed_slide_med, pct_change_med, 

 time_stamps_pipeline_med, detect_time_med) = new_detect_and_crop(

                                        image_location=example_slide, out_lvl=-2,

                                        show_plots="verbose"

                                        )
(processed_slide_med, pct_change_med, 

 time_stamps_pipeline_med, detect_time_med) = new_detect_and_crop(

                                        image_location=example_slide, out_lvl=-1,

                                        show_plots="verbose"

                                        )
x_tot,x2_tot = [],[]

names = [name[:-5] for name in os.listdir(TRAIN)]

with zipfile.ZipFile(OUT_TRAIN, 'w') as img_out:

    for name in tqdm(names):

        img_path = f"{slide_dir}{name}.tiff"

        img,_,_,_ = new_detect_and_crop(image_location=img_path, show_plots=None, downsample_lvl=-1, out_lvl=-1)

        if img is None:

            continue

        x_tot.append((img/255.0).reshape(-1,3).mean(0))

        x2_tot.append(((img/255.0)**2).reshape(-1,3).mean(0)) 

         #if read with PIL RGB turns into BGR

        img = cv2.imencode('.png',cv2.cvtColor(img, cv2.COLOR_RGB2BGR))[1]

        img_out.writestr(f'{name}.png', img)
img_avr =  np.array(x_tot).mean(0)

img_std =  np.sqrt(np.array(x2_tot).mean(0) - img_avr**2)

print('mean:',img_avr, ', std:', np.sqrt(img_std))