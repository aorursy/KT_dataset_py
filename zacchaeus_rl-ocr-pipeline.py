import json

import re

import os

import sys

import requests

import time

import numpy as np

import matplotlib.pyplot as plt

import copy

import glob

import pandas as pd

from matplotlib.patches import Polygon

from PIL import Image

from io import BytesIO

from sklearn.cluster import KMeans

import pickle

%matplotlib inline
with open('../input/msocrrawoutputs/ms_ocr_raw_output.pkl', 'rb') as f:

    raw = pickle.load(f)
# Enter the image root path here

IMG_ROOT_PATH = '../input/rlocrcroppedimg/work/main table structures'



# Enter an image index here

IMG_INDEX = 0 # 1

# IMG_ID = '1135'

IMG_ID = '0132b'
if IMG_ID is None:

    assert os.path.isdir(IMG_ROOT_PATH), 'Invalid Root Path.'

    img_paths = glob.glob(f'{IMG_ROOT_PATH}/*.JPG')

    assert len(img_paths) > 0, 'No Image in the Target Folder.'

    try:

        path = img_paths[IMG_INDEX]

    except IndexError:

        path = img_paths[0]

else:

    path = f'{IMG_ROOT_PATH}/{IMG_ID}.JPG'

img_id = path.split('/')[-1].split('.')[0]

print(f'img_id: {img_id}')

plt.rcParams['figure.figsize'] = [15, 15]

plt.axis('off')

plt.imshow(Image.open(path), cmap='Greys_r')

plt.show()
analysis = raw[img_id]

polygons = []

if ("analyzeResult" in analysis):

    # Extract the recognized text, with bounding boxes.

    polygons = [(line["boundingBox"], line["text"])

                for line in analysis["analyzeResult"]["readResults"][0]["lines"]]

polygons[:5]
# Display the image and overlay it with the extracted text.

image = Image.open(path)

plt.rcParams['figure.figsize'] = [15, 15]

ax = plt.imshow(image, cmap='Greys_r')

for polygon in polygons:

    vertices = [(polygon[0][i], polygon[0][i+1])

                for i in range(0, len(polygon[0]), 2)]

    text = polygon[1]

    patch = Polygon(vertices, closed=True, fill=False, linewidth=2, color='b')

    ax.axes.add_patch(patch)

    plt.text(vertices[1][0], vertices[1][1], text, fontsize=15, color='r', va="top")

plt.axis('off')

plt.show()
def calc_h(polygon):

    h1 = abs(int(polygon[1])-int(polygon[7]))

    h2 = abs(int(polygon[3])-int(polygon[5]))

    return (h1+h2)/2



def avg_h(polygons):

    hs = []

    for poly in polygons:

        hs.append(calc_h(poly[0]))

    return np.mean(np.array(hs))



def calc_center(polygon):

    '''

    calculate the center coordinate of a polygon.

    returns [x_center, y_center]

    '''

    x = np.array([polygon[0], polygon[2], polygon[4], polygon[6]])

    y = np.array([polygon[1], polygon[3], polygon[5], polygon[7]])

    return [np.mean(x), np.mean(y)]



def calc_iou(poly1, poly2):

    y1_upper = (poly1[1] + poly1[3]) / 2

    y1_lower = (poly1[5] + poly1[7]) / 2

    y2_upper = (poly2[1] + poly2[3]) / 2

    y2_lower = (poly2[5] + poly2[7]) / 2

    range1 = set(range(int(2*y1_upper), int(2*y1_lower)))

    range2 = set(range(int(2*y2_upper), int(2*y2_lower)))

    return len(range1.intersection(range2)) / len(range1.union(range2))    



def calc_area(polygon):

    '''

    calculate the area of a polygon.

    polygon -- a list of format: [

        upper_left_x, 

        upper_left_y, 

        upper_right_x,

        upper_right_y,

        lower_right_x,

        lower_right_y

        lower_left_x,

        lower_left_y]

    returns a float.

    '''

    x = np.array([polygon[0], polygon[2], polygon[4], polygon[6]])

    y = np.array([polygon[1], polygon[3], polygon[5], polygon[7]])

    return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))



def slope(polygons):

    '''

    calculate the mean and variance skewness of the boxes.

    returns a list [mean, var]

    '''

    n = len(polygons)

    ks = []

    for i, poly in enumerate(polygons):

        #filtering out too small boxes

        if calc_area(poly[0]) < 2200:

            n -= 1

            continue



        #avoiding zero-division error

        if (poly[0][2]-poly[0][0]<=0) or (poly[0][4]-poly[0][6]<=0):

            n -= 1

            continue



        upperK = (poly[0][3]-poly[0][1]) / (poly[0][2]-poly[0][0])

        lowerK = (poly[0][5]-poly[0][7]) / (poly[0][4]-poly[0][6])

        k = (upperK+lowerK) / 2



        #removing abnormal output

        if abs(k) > 1: # remove is degree larger than pi/4

            n -= 1

            continue



        ks.append(k)

    ks = np.array(ks)

        # print(i, k)

    return [np.mean(ks), np.var(ks)]



# def slope2dgree(slope):

#     '''

#     return the degree of skew (clockwise positive)

#     '''

#     return np.arctan(slope) / np.pi * 180



# slope(polygons)

# for i, poly in enumerate(polygons):

#     print(i, calc_area(poly[0]), poly[1])

def denoise(polygons):

    denoised = copy.deepcopy(polygons)

    noise_indexs = []

    for i, poly in enumerate(denoised):

        text = poly[1]

        contain_number = any(char.isdigit() for char in text)

        contain_alpha = any(char.isalpha() for char in text)

        if (not contain_number) and (not contain_alpha):

            noise_indexs.append(i)

    for index in sorted(noise_indexs, reverse=True):

        del denoised[index]

    return denoised



def remove_nonalpha(s):

    return ''.join([char for char in list(s) if char.isalpha() or char==' ']).strip()



def extract_head(polygons):

    '''

    returns list e.g. [denied, manhattan, polys_nohead]

    '''

    ret = [None, None, None]

    nohead = copy.deepcopy(polygons)

    head_indexs = []

    bor_y = 0

    for i, poly in enumerate(nohead):

        box, text = poly[0], poly[1]

        if 'borough' in text.lower():

            head_indexs.append(i)

            text = remove_nonalpha(text)

            ret[1] = text.split()[-1]

            bor_y = box[1]

        if any(_ in text.lower() for _ in ['denied', 'revoked', 'granted']):

            head_indexs.append(i)

            ret[0] = text

    for i, poly in enumerate(nohead):

        box = poly[0]

        if box[1] < bor_y:

            head_indexs.append(i)

    for index in sorted(head_indexs, reverse=True):

        del nohead[index]

    ret[2] = nohead

    return ret
# extractor = extract_head(denoise(polygons))

# result, borough, polygons_nohead = extractor[0], extractor[1], extractor[2]

# result, borough

polygons = denoise(polygons)
def get_ncols(polygons, candidates=[1,2,3,4]):

    for c in reversed(sorted(candidates)):

        X = np.array([calc_center(poly[0])[0] for poly in polygons])

        X = X.reshape(-1,1)

        column_cluster = KMeans(n_clusters=c, random_state=0)

        column_cluster.fit(X)

        centers = column_cluster.cluster_centers_

        centers = centers.reshape(-1)

#         print(centers)

        centers.sort()

        gaps = np.array([centers[i+1]-centers[i] for i in range(len(centers)-1)])

        if np.var(gaps) < 1e5:

            break

    return c

ncols = get_ncols(polygons)

ncols
X = np.array([calc_center(poly[0])[0] for poly in polygons])

X = X.reshape(-1,1)

column_cluster = KMeans(n_clusters=ncols, random_state=0)

column_cluster.fit(X)

# for i, p in enumerate(polygons):

#     print(kmeans.labels_[i], p[1])
color_dict = {

    0: 'y',

    1: 'r',

    2: 'b',

    3: 'g'

    }

label_count = {col:0 for col in range(ncols)}

gt_names = ['number', 'name', 'nature', 'address']

column_names = {}



column_cluster_centers = column_cluster.cluster_centers_.reshape(ncols).tolist()

sorted_column_cluster_centers = list(sorted(column_cluster_centers))

for i, col in enumerate(column_cluster_centers):

    column_names[i] = gt_names[sorted_column_cluster_centers.index(col)]

image = Image.open(path)

plt.rcParams['figure.figsize'] = [15, 15]

ax = plt.imshow(image, cmap='Greys_r')

for polygon in polygons:

    vertices = [(polygon[0][i], polygon[0][i+1])

                for i in range(0, len(polygon[0]), 2)]

    # text = polygon[1]



    box = polygon[0]

    center = calc_center(box)

    

    x = center[0]

    label = column_cluster.predict([[x]])[0]

    label_count[label] += 1



    patch = Polygon(vertices, closed=True, fill=False, linewidth=2, color=color_dict[label])

    ax.axes.add_patch(patch)

    plt.text(vertices[1][0], vertices[1][1], column_names[label], fontsize=10, va="top", color=color_dict[label])

plt.axis('off')

plt.show()

print(label_count)

name_count = {column_names[k]: v for k, v in label_count.items()}

print(name_count)
def clsfy_sort(polygons, kmeans, ncols):

    gt_names = ['number', 'name', 'nature', 'address']

    column_names = {}

    column_cluster_centers = kmeans.cluster_centers_.reshape(ncols).tolist()

    sorted_column_cluster_centers = list(sorted(column_cluster_centers))

    for i, col in enumerate(column_cluster_centers):

        column_names[i] = gt_names[sorted_column_cluster_centers.index(col)]

    polygons_col_dict = {k:[] for k in gt_names}

    for i, poly in enumerate(polygons):

        box, text = poly[0], poly[1]

        center = calc_center(box)

        x = center[0]

        label = kmeans.predict([[x]])[0]

        polygons_col_dict[column_names[label]].append(poly)

    for name, poly in polygons_col_dict.items():

        gety = lambda poly: calc_center(poly[0])[1]

        poly.sort(key=gety)

    return polygons_col_dict



def poly_union(poly1, poly2):

    '''

    polygon -- a tuple of format: ([

        upper_left_x, 

        upper_left_y, 

        upper_right_x,

        upper_right_y,

        lower_right_x,

        lower_right_y

        lower_left_x,

        lower_left_y], text)

    '''

    a1 = min(poly1[0][0], poly2[0][0])

    a2 = min(poly1[0][1], poly2[0][1])

    a3 = max(poly1[0][2], poly2[0][2])

    a4 = min(poly1[0][3], poly2[0][3])

    a5 = max(poly1[0][4], poly2[0][4])

    a6 = max(poly1[0][5], poly2[0][5])

    a7 = min(poly1[0][6], poly2[0][6])

    a8 = max(poly1[0][7], poly2[0][7])

    text = poly1[1]+' '+poly2[1]

    return ([a1, a2, a3, a4, a5, a6, a7, a8], text)



def combine_same_row(polygons_col_dict, thr=0.5):

    from copy import deepcopy

    combined = deepcopy(polygons_col_dict)

    for col_name, polys in polygons_col_dict.items():

        if len(polys) <= 1:

            continue

        comb_is = []

        for i in range(len(polys)-1):

            iou = calc_iou(polys[i][0], polys[i+1][0])

            if iou > thr:

                comb_is.append(i)

        for comb_i in comb_is:

            combined[col_name][comb_i] = poly_union(combined[col_name][comb_i],combined[col_name][comb_i+1])

        comb_is.sort()

        for comb_i in reversed(comb_is):

            del combined[col_name][comb_i+1]

    return combined

polygons_col_dict = clsfy_sort(polygons, column_cluster, ncols)

polygons_col_dict=combine_same_row(polygons_col_dict)

polygons_col_dict.keys()
color_dict = {

    0: 'y',

    1: 'r',

    2: 'b',

    3: 'g'

    }

image = Image.open(path)

plt.rcParams['figure.figsize'] = [15, 15]

ax = plt.imshow(image, cmap='Greys_r')

i = 0

for col_name, polys in polygons_col_dict.items():

    for polygon in polys:

        vertices = [(polygon[0][i], polygon[0][i+1])

                    for i in range(0, len(polygon[0]), 2)]

        box = polygon[0]

        patch = Polygon(vertices, closed=True, fill=False, linewidth=2, color=color_dict[i])

        ax.axes.add_patch(patch)

        plt.text(vertices[1][0], vertices[1][1], col_name, fontsize=10, va="top", color=color_dict[i])

    i += 1

plt.axis('off')

plt.show()
def remove_non_digits(lst):

    for i in range(len(lst)):

        lst[i] = re.sub("[^0-9]", "", lst[i])



def is_all_digit(s):

    return all([c.isdigit() for c in s])



def autofill_address(lst):

    if len(lst) < 2:

        return

    for i in range(len(lst)-1):

        former, latter = lst[i], lst[i+1]

        if is_all_digit(latter) and (not is_all_digit(former)):

            omitted = ' '.join([word for word in former.split() if not is_all_digit(word)])

            lst[i+1] = f'{latter} {omitted}'



def seg_row(polygons_col_dict):

    polygons = []

    for v in polygons_col_dict.values():

        polygons.extend(v)

    avgh = avg_h(polygons)

    gt_names = ['number', 'name', 'nature', 'address']

    rows_dict = {k:[] for k in gt_names}

    for name in gt_names:

        lasty = None

        for poly in polygons_col_dict[name]:

            box, text = poly[0], poly[1]

            y = calc_center(box)[1]

            if lasty is None:

                rows_dict[name].append(text)

            else:

                # print((y-lasty)/avgh)

                gaps = int(((y-lasty)/avgh)+0.5)

                for _ in range(gaps-1):

                    rows_dict[name].append('')

                rows_dict[name].append(text)

            lasty = y

    # add empty strings to end of list

    max_len = max([len(v) for v in rows_dict.values()])

    for v in rows_dict.values():

        for _ in range(max_len-len(v)):

            v.append('')

    #remove non-digits in 'number' column

    remove_non_digits(rows_dict['number'])



    #autofill omitted address

    autofill_address(rows_dict['address'])

    

    # return rows_dict

    return pd.DataFrame.from_dict(rows_dict)

rows_df = seg_row(polygons_col_dict)

print(rows_df)
plt.rcParams['figure.figsize'] = [15, 15]

plt.imshow(Image.open(path), cmap='Greys_r')

plt.axis('off')

plt.show()