import numpy as np

import matplotlib.pyplot as plt

from glob import glob

from skimage.io import imread

from skimage.color import rgb2grey

from sklearn.feature_extraction import image

from sklearn.cluster import KMeans



from skimage.filters import rank, threshold_otsu

from skimage.morphology import closing, square, disk

from skimage import exposure as hist, data, img_as_float

from skimage.segmentation import chan_vese

from skimage.feature import canny

from skimage.color import rgb2gray

from scipy import ndimage as ndi 
mal_images = glob('../input/skin-cancer-malignant-vs-benign/data/train/malignant/*')[:5]

ben_images = glob('../input/skin-cancer-malignant-vs-benign/data/train/benign/*')[:5]
def binary(image):

    return image > threshold_otsu(image)



def equalize(image):

    return hist.equalize_hist(image)



def mean_filter(image, raio_disk):

    return rank.mean_percentile(image, selem = disk(raio_disk))



def preenche_bords(image):

    return ndi.binary_fill_holes(image)



def load_images(paths):

    tmp = []

    for path in paths:

        tmp.append(imread(path))

    return tmp

    

def plot_any(arr, title = ''):

    plt.figure(figsize = (15, 25))

    for i in range(len(arr)):

        plt.subplot(1,len(arr),i + 1)

        plt.title(title)

        plt.imshow(arr[i]);

        

def plot_camadas(img):

    plt.figure(figsize = (15, 25))

    for i in range(3):

        plt.subplot(1, 3, i + 1)

        plt.imshow(img[:,:,i], cmap = 'gray');

        

def d2Kmeans(img, k):

    return KMeans(n_jobs=-1, 

                  random_state=1, 

                  n_clusters = k, 

                  init='k-means++'

    ).fit(img.reshape((-1,1))).labels_.reshape(img.shape)



def merge_segmented_mask_ROI(uri_img, img_kluster):

    new_img = uri_img.copy()

    for ch in range(3):

        new_img[:,:, ch] *= img_kluster

    return new_img



def elbow(img, k):

    hist = []

    for kclusters in  range(1, k):

        Km = KMeans(n_jobs=-1, random_state=1, n_clusters = kclusters, init='k-means++').fit(img.reshape((-1,1)))  

        hist.append(Km.inertia_)

        

    plt.figure(figsize = (15, 8))

    plt.grid()

    plt.plot(range(1, k), hist, 'o-')

    plt.ylabel('Soma das distâncias quadradas')

    plt.xlabel('k clusters')

    plt.title('Elbow')

    plt.show();
mal = load_images(mal_images)

ben = load_images(ben_images)
plot_any(ben, 'Benigma')

plot_any(mal, 'Maligna')
img_selected = mal[1]
elbow(img_selected, 6)
k_klusters = 2
result_gray = d2Kmeans(rgb2grey(img_selected), k_klusters)

result_img = d2Kmeans(img_selected, k_klusters)
klusters_gray = [result_gray == i for i in range(k_klusters)]

plot_any(klusters_gray)
def select_cluster_index(clusters):

    minx = clusters[0].mean()

    index = 0

    for i in clusters:

        if i.mean() < minx:

            minx = i.mean()

            index += 1

    return index
index_kluster = select_cluster_index(klusters_gray)

print(index_kluster)

selecionado = klusters_gray[index_kluster]
# for ch in range(3):

#     img_k = []

#     for K in range(k_klusters):

#         img_k.append(result_img[:, :, ch] == K)

#     plot_any(img_k)
# clusters = [(result_img[:,:,1] == K) for K in range(k_klusters)]
new_img = merge_segmented_mask_ROI(img_selected, selecionado)
plot_any([new_img])
image_mean_filter = mean_filter(selecionado, 20)

test_binary = binary(image_mean_filter)
plot_any([selecionado, image_mean_filter, test_binary])
final_result = merge_segmented_mask_ROI(img_selected ,test_binary)
plot_any([test_binary, new_img, final_result])
# max_mean = 0

# img_gray = rgb2gray(final_result)

# img_bin  = binary(img_gray)

# x, y = img_bin.shape



# limits_before = []

# for i in range(x):

#     for j in range(y):

#         if  img_bin[i, j]:

#             limits_before.append(j)

            

# stop_before = ( sum(limits_before) // len(limits_before) ) // 2

# img_copy = img_bin.copy()

# for i in range(x):

#     for j in range(stop_before):

#         img_copy[i, j] = 0

        

# limits_after = []

# for i in range(x):

#     for j in range(y - 1, 0, -1):

#         if  img_copy[i, j]:

#             limits_after.append(j)

            

# stop_after = sum(limits_after) // len(limits_after) + min(limits_after)

# for i in range(x):

#     for j in range(stop_after, y):

#         img_copy[i, j] = 0



# mean_result = mean_filter(img_copy, 15)

# mean_result = binary(mean_result)

# final_result = merge_segmented_mask_ROI(img_selected , mean_result)





# plot_any([mean_result, final_result])