# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
from collections import defaultdict

import cv2

from shapely.geometry import MultiPolygon, Polygon

import shapely.wkt

import shapely.affinity

import numpy as np

import tifffile as tiff

#import gdal

from osgeo import gdal

#import gdalconst



import matplotlib.pylab as plt

%matplotlib inline
gdal.AllRegister()

gdal.VersionInfo()
DATA_3_BANDS='../input/three_band/'

DATA_16_BANDS='../input/sixteen_band/'
image_id = "6020_2_2"



fname_3b = DATA_3_BANDS + "/{}.tif".format(image_id)

fname_pan = DATA_16_BANDS + "/{}_P.tif".format(image_id)

fname_ms = DATA_16_BANDS + "/{}_M.tif".format(image_id)

fname_swir = DATA_16_BANDS + "/{}_A.tif".format(image_id)



img_3b = gdal.Open(fname_3b, gdalconst.GA_ReadOnly)

assert img_3b, "WTF"

print("Image 3 bands info :", img_3b.RasterXSize, img_3b.RasterYSize,img_3b.RasterCount)

img_pan = gdal.Open(fname_pan, gdalconst.GA_ReadOnly)

assert img_pan, "WTF"

print("Image Pan info :", img_pan.RasterXSize, img_pan.RasterYSize,img_pan.RasterCount)

img_ms = gdal.Open(fname_ms, gdalconst.GA_ReadOnly)

assert img_ms, "WTF"

print("Image MS info :", img_ms.RasterXSize, img_ms.RasterYSize,img_ms.RasterCount)

img_swir = gdal.Open(fname_swir, gdalconst.GA_ReadOnly)

assert img_swir, "WTF"

print("Image SWIR info :", img_swir.RasterXSize, img_swir.RasterYSize,img_swir.RasterCount)
def scale_percentile(matrix):

    if len(matrix.shape) == 2:

        matrix = matrix.reshape(matrix.shape + (1,))

    w, h, d = matrix.shape

    matrix = np.reshape(matrix, [w * h, d]).astype(np.float64)

    # Get 2nd and 98th percentile

    mins = np.percentile(matrix, 1, axis=0)

    maxs = np.percentile(matrix, 99, axis=0) - mins

    matrix = 255*(matrix - mins[None, :]) / maxs[None, :]

    matrix = np.reshape(matrix, [w, h, d] if d > 1 else [w, h])

    matrix = matrix.clip(0, 255).astype(np.uint8)

    return matrix   

    



def display_img_1b(img_1b, roi=None):   

    img_1b_data = img_1b.ReadAsArray()

    if roi is not None:

        y,yh,x,xw = roi

        img_1b_data = img_1b_data[y:yh,x:xw]

    plt.figure(figsize=(8,4))

    plt.imshow(scale_percentile(img_1b_data), cmap='gray')



    

def display_img_3b(img_3b, roi=None):

    img_3b_data = img_3b.ReadAsArray()

    if roi is not None:

        y,yh,x,xw = roi

        img_3b_data = img_3b_data[:,y:yh,x:xw]

    plt.figure(figsize=(8,4))

    for i in [0,1,2]:

        plt.subplot(1,3,i+1)

        plt.imshow(scale_percentile(img_3b_data[i,:,:]), cmap='gray')

        plt.title("Channel %i" % i)

    

def display_img_8b(img_ms, roi=None):

    img_ms_data = img_ms.ReadAsArray()

    if roi is not None:

        y,yh,x,xw = roi

        img_ms_data = img_ms_data[:,y:yh,x:xw]

    plt.figure(figsize=(8,4))

    for i in range(8):

        plt.subplot(2,4,i+1)

        plt.imshow(scale_percentile(img_ms_data[i,:,:]), cmap='gray')

        plt.title("Channel %i" % i)
from ipywidgets import interact, IntSlider



fig = None

def interactive_visu(roi_x, roi_w, roi_y, roi_h):

    roi = [roi_x,roi_x+roi_w,roi_y,roi_y+roi_h]

    display_img_1b(img_pan, roi)

    plt.suptitle("Pansharpened image ROI")

    display_img_3b(img_3b, roi)

    plt.suptitle("3 bands image ROI")

    display_img_8b(img_ms, [int(r/4) for r in roi])

    plt.suptitle("8 bands image ROI")

    display_img_8b(img_swir, [int(r/4/6.2) for r in roi])

    _ = plt.suptitle("8 bands SWIR image ROI")



    

_ = interact(interactive_visu,           

         roi_x=IntSlider(value=100, min=0, max=3500, continuous_update=False), 

         roi_w=FloatSlider(value=200, min=150, max=350, continuous_update=False), 

         roi_y=FloatSlider(value=0, min=0, max=3500, continuous_update=False), 

         roi_h=FloatSlider(value=200, min=150, max=350, continuous_update=False))
def render_rgb(data_3_channels, bgr2rgb=False):    

    out = np.zeros_like(data_3_channels).astype(np.uint8)

    in_channels = [0,1,2]

    out_channels = [2,1,0] if bgr2rgb else in_channels

    for c, nc in zip(in_channels, out_channels):

        band = data_3_channels[:,:,c]

        min_value = np.percentile(band, 3)

        max_value = np.percentile(band, 97)     

        band[band < min_value] = min_value

        band[band > max_value] = max_value

        out[:,:,nc] = 255 * (band - min_value)/(max_value - min_value)    

    return out
!ls ../input/three_band/ | head
rgb_image_ids = ["6010_0_0", "6090_4_2", "6170_3_0", "6140_4_2", "6120_2_2"]

#rgb_image_ids = ["6120_2_2"]



for rgb_image_id in rgb_image_ids:

    

    test_rgb_image_filename = "../input/three_band/{}.tif".format(rgb_image_id)

    test_rgb_image = gdal.Open(test_rgb_image_filename)

    assert test_rgb_image is not None, "WTF"

    print("File : ", test_rgb_image_filename)

    print("Metadata list: ", test_rgb_image.GetMetadata_List())

    print("Metadata domaines: ", test_rgb_image.GetMetadataDomainList())

    print("Projection reference: ", test_rgb_image.GetProjectionRef())

    print("Geotransform: ", test_rgb_image.GetGeoTransform())

    band1 = test_rgb_image.GetRasterBand(1)

    print("Pixel depth: ", gdal.GetDataTypeName(band1.DataType))

    test_rgb_image_data = test_rgb_image.ReadAsArray().transpose([1,2,0])

    test_rgb_image_data = render_rgb(test_rgb_image_data)

    

    plt.figure(figsize=(10,4))

    plt.subplot(121)

    plt.imshow(test_rgb_image_data)

    plt.subplot(122)

    plt.imshow(test_rgb_image_data[2900:3200,2000:2300,:])
!ls ../input/sixteen_band/ | head
#mb_image_ids = ["6010_0_0_M", "6090_4_2_M", "6170_3_0_M", "6140_4_2_M", "6120_2_2_M"]

mb_image_ids = ["6120_2_2_M"]



rgb_channels = (0, 4, 2)

for mb_image_id in mb_image_ids:

    

    test_mb_image_filename = "../input/sixteen_band/{}.tif".format(mb_image_id)

    test_mb_image = gdal.Open(test_mb_image_filename)

    assert test_mb_image is not None, "WTF"

    print("File : ", test_mb_image_filename)

    print("Metadata list: ", test_mb_image.GetMetadata_List())

    print("Metadata domaines: ", test_mb_image.GetMetadataDomainList())

    print("Projection reference: ", test_mb_image.GetProjectionRef())

    print("Geotransform: ", test_mb_image.GetGeoTransform())

    band1 = test_mb_image.GetRasterBand(1)

    print("Pixel depth: ", gdal.GetDataTypeName(band1.DataType))



    test_mb_image_data = test_mb_image.ReadAsArray()  

    test_rgb_image_data = test_mb_image_data[rgb_channels,:,:].transpose([1,2,0])

    test_rgb_image_data = render_rgb(test_rgb_image_data)

    

    plt.figure(figsize=(10,4))

    plt.subplot(121)

    plt.imshow(test_rgb_image_data)

    plt.subplot(122)

    plt.imshow(test_rgb_image_data[200:300,200:300,:])
mb_image_ids = ["6120_2_2_M"]



for mb_image_id in mb_image_ids:

    

    test_mb_image_filename = "../input/sixteen_band/{}.tif".format(mb_image_id)

    test_mb_image = gdal.Open(test_mb_image_filename)

    assert test_mb_image is not None, "WTF"

    print("File : ", test_mb_image_filename)

    print("Metadata list: ", test_mb_image.GetMetadata_List())

    print("Metadata domaines: ", test_mb_image.GetMetadataDomainList())

    print("Projection reference: ", test_mb_image.GetProjectionRef())

    print("Geotransform: ", test_mb_image.GetGeoTransform())

    band1 = test_mb_image.GetRasterBand(1)

    print("Pixel depth: ", gdal.GetDataTypeName(band1.DataType))



    test_mb_image_data = test_mb_image.ReadAsArray()  

    nir = test_mb_image_data[7,:,:]

    vis = test_mb_image_data[6,:,:]

    ndvi = (nir - vis) / (nir + vis)

        

    plt.figure(figsize=(10,4))

    plt.subplot(121)

    plt.imshow(ndvi)

    plt.subplot(122)

    plt.imshow(ndvi[200:300,200:300])
labels = [

    None, 

    # 1

    "Buildings - large building, residential, non-residential, fuel storage facility, fortified building",

    # 2

    "Misc. Manmade structures", 

    # 3

    "Road", 

    # 4

    "Track - poor/dirt/cart track, footpath/trail",

    # 5

    "Trees - woodland, hedgerows, groups of trees, standalone trees",

    # 6

    "Crops - contour ploughing/cropland, grain (wheat) crops, row (potatoes, turnips) crops",

    # 7

    "Waterway", 

    # 8

    "Standing water",

    # 9

    "Vehicle Large - large vehicle (e.g. lorry, truck,bus), logistics vehicle",

    # 10

    "Vehicle Small - small vehicle (car, van), motorbike",    

]
train_wkt = pd.read_csv("../input/train_wkt_v4.csv")

train_wkt.head()
train_wkt[train_wkt["ImageId"] == "6120_2_2"]
GRID_SIZE = pd.read_csv('../input/grid_sizes.csv', names=['ImageId', 'Xmax', 'Ymin'], skiprows=1)

GRID_SIZE.columns = ['ImageId','Xmax','Ymin']

TRAIN_WKT = pd.read_csv('../input/train_wkt_v4.csv')



def get_grid_size(image_id):

    x_max = GRID_SIZE[GRID_SIZE['ImageId']==image_id].Xmax.values[0]

    y_min = GRID_SIZE[GRID_SIZE['ImageId']==image_id].Ymin.values[0]

    return x_max, y_min



def get_scalers(image_shape, x_max, y_min):

    h, w = image_shape  # they are flipped so that mask_for_polygons works correctly

    w_ = w * (w / (w + 1))

    h_ = h * (h / (h + 1))

    return w_ / x_max, h_ / y_min



def generate_image_mask(image_id, class_type):

    data_mask = (TRAIN_WKT["ImageId"] == image_id) & (TRAIN_WKT["ClassType"] == class_type)

    poly = TRAIN_WKT[data_mask]["MultipolygonWKT"].values[0]

    #print("poly=", poly)

    train_polygons = shapely.wkt.loads(poly)

    

    

    return train_polygons





poly = generate_image_mask("6120_2_2", 5, train_wkt)

poly