import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)





import matplotlib.pyplot as plt

import seaborn as sns 





from time import time



plt.rcParams['figure.figsize'] = 10,8 # setting default figure size for the kernel





from sklearn.cluster import KMeans 

from skimage import io
url = 'https://i.imgur.com/KLwsDXD.jpg'

img_original = io.imread(url)

plt.axis('off')

plt.imshow(img_original)

plt.title('CodeCamp Image--')

plt.show()


img = np.array(img_original,dtype=float) / 255





w, h, d = original_shape = img.shape

print('Original Shape'.center(20,'='))

print(img.shape)



image_array = img.reshape(-1,d)

print('ReShaped'.center(20,'='))

print(image_array.shape)
n_colours = [5,10]



# 64 colour image

t0 = time()

kmeans64 = KMeans(n_clusters = n_colours[0],random_state=42,verbose=2,n_jobs=-1).fit(image_array)



print(f'Completed 64 clusters in {round(time()-t0,2)} seconds.')



# 32 colour image

t0 = time()

kmeans32 = KMeans(n_clusters = n_colours[1],random_state=42,verbose=2,n_jobs=-1)

kmeans32.fit(image_array)



print(f'Completed 32 clusters in {round(time()-t0,2)} seconds.')



labels64 = kmeans64.labels_

labels32 = kmeans32.labels_
print(f'Within cluster sum of square error for {n_colours[0]} clusters = {round(kmeans64.inertia_,2)}')

print(f'Within cluster sum of square error for {n_colours[1]} clusters = {round(kmeans32.inertia_,2)}')


compressed = pd.DataFrame(image_array,columns=['Red','Green','Blue'])

compressed['labels'] = kmeans64.labels_

print (compressed)
def recreate_image(centroids, labels, w, h):

    # centroids variable are calculated from the flattened image

    # centroids: w*h, d 

    d = centroids.shape[1]

    image = np.zeros((w, h, d))

    label_idx = 0

    for i in range(w):

        for j in range(h):

            

            image[i][j] = centroids[labels[label_idx]]

            label_idx += 1

    return image
plt.figure(figsize=(20,10))

plt.subplot(132)

plt.axis('off')

plt.title('CodeCamp Image --')

#plt.imshow(img)



plt.subplot(131)

plt.axis('off')

plt.title('Compressed image (5 Colors)')

plt.imshow(recreate_image(kmeans64.cluster_centers_, labels64, w, h))



plt.subplot(133)

plt.axis('off')

plt.title('Compressed image (10 Colors)')

#plt.imshow(recreate_image(kmeans32.cluster_centers_, labels32, w, h))



plt.show()