import random

import numpy as np

import matplotlib.pyplot as plt



from sklearn.cluster import MiniBatchKMeans

from sklearn.linear_model import Perceptron
#import the dataset

print("imposting dataset...")

f = open("../input/train.csv")

L = f.read().split('\n')[1:-1]

f.close()



L = list(map(lambda txt: txt.split(','), L))

L = [list(map(int, lst)) for lst in L]



#shuffle the dataset

print("shuffling dataset...")

random.shuffle(L)



print("splitting dataset...")



#spliting the dataset into a training and testing set

training_set = L[:int(.8 * len(L))]

testing_set  = L[int(.8 * len(L)):]



#separing data and labels...

training_set_data   = [L[1:] for L in training_set]

training_set_labels = [L[0]  for L in training_set]



testing_set_data   = [L[1:] for L in testing_set]

testing_set_labels = [L[0]  for L in testing_set]



#make the data explotable by reformating them

training_set_data = map(lambda lst: np.array(lst).reshape((28, 28)), training_set_data)

testing_set_data  = map(lambda lst: np.array(lst).reshape((28, 28)), testing_set_data)



testing_set_data  = list(testing_set_data)

training_set_data = list(training_set_data)



print("done.")
def extract_patch(patch_size, image):

    L = []

    for i in range(len(image) - patch_size):

        for j in range(len(image[0]) - patch_size):

            L.append(image[i:i+patch_size,j:j+patch_size])

    return L
training_patches_8x8 = sum(map(lambda img: extract_patch(8, img), training_set_data[:1000]), [])

training_patches_vector_64 = list(map(lambda img: img.reshape(1, 64)[0], training_patches_8x8))
#let clusturing the patches 

mb_kmeans = MiniBatchKMeans(n_clusters=64, max_iter=100) #64 clusters 

mb_kmeans.fit(training_patches_vector_64)
plt.figure(figsize=(4.2, 4))

for i, patch in enumerate(mb_kmeans.cluster_centers_):

    plt.subplot(8, 8, i + 1)

    plt.imshow(patch.reshape((8, 8)), cmap=plt.cm.gray, interpolation='nearest')

    plt.xticks(())

    plt.yticks(())





plt.suptitle('Cluster centers of the patches\n')

plt.subplots_adjust(0.08, 0.02, 0.92, 0.85, 0.08, 0.23)



plt.show()
def one_hot(v, max_value):

    L = [0] * max_value

    L[v] = 1

    return L
def transform_img(img):

    #make the image a 24x24 pixels image (to be divisible by 8)

    img = img[2:-2,2:-2]

    

    #extract patches from image

    patches = list(map(lambda img: img.reshape(1, 64)[0], extract_patch(8, img)))

    clusters_assigments = mb_kmeans.predict(patches)

    

    #return the one hot represenation of the image

    return  sum(map(lambda v: one_hot(v, 64), clusters_assigments), [])
#transform only 4000 images the training set...

training_set_data_one_hot = map(transform_img, training_set_data[1001:5001])

training_set_data_one_hot = list(training_set_data_one_hot)
#transform only 1000 images the testing set...

testing_set_data_one_hot = map(transform_img, testing_set_data[:1000])

testing_set_data_one_hot = list(testing_set_data_one_hot)
#train with only 4000 images

training_set_data_flattern = map(lambda img: img.reshape(1, 28 ** 2)[0], training_set_data[1001:5001])

training_set_data_flattern = list(training_set_data_flattern)
testing_set_data_flattern = map(lambda img: img.reshape(1, 28 ** 2)[0], testing_set_data[:1000])

testing_set_data_flattern = list(testing_set_data_flattern)
original_perceptron = Perceptron()

original_perceptron.fit(training_set_data_flattern, training_set_labels[1001:5001])
print(

    "original perceptron accuracy : ", 

    original_perceptron.score(testing_set_data_flattern, testing_set_labels[:1000])

)
perceptron = Perceptron()

perceptron.fit(training_set_data_one_hot, training_set_labels[1001:5001])
print(

    "new perceptron accuracy : ", 

    perceptron.score(testing_set_data_one_hot, testing_set_labels[:1000])

)
training_patches_6x6 = sum(map(lambda img: extract_patch(6, img), training_set_data[:1000]), [])

training_patches_vector_36 = list(map(lambda img: img.reshape(1, 36)[0], training_patches_6x6))
#let clusturing the patches 

mb_kmeans_6x6_patches = MiniBatchKMeans(n_clusters=36, max_iter=100) #36 clusters 

mb_kmeans_6x6_patches.fit(training_patches_vector_36)
plt.figure(figsize=(4.2, 4))

for i, patch in enumerate(mb_kmeans_6x6_patches.cluster_centers_):

    plt.subplot(6, 6, i + 1)

    plt.imshow(patch.reshape((6, 6)), cmap=plt.cm.gray, interpolation='nearest')

    plt.xticks(())

    plt.yticks(())





plt.suptitle('Cluster centers of the 6x6 patches\n')

plt.subplots_adjust(0.08, 0.02, 0.92, 0.85, 0.08, 0.23)



plt.show()
def get_4x4_patches_clusters_of_image(img):

    #make the image a 24x24 pixels image (to be divisible by 8)

    img = img[2:-2,2:-2]



    #extract 4x4 patches of size 8x8

    M = [[None] * 4 for _ in range(4)]

    for i in range(4):

        for j in range(4):

            M[i][j] = mb_kmeans_6x6_patches.predict([

                img[i*6:(i+1)*6,j*6:(j+1)*6].reshape((1, 36))[0]

            ])[0]

    return M

        
#get the patches cluster numbers

training_4x4_patches_6x6_set_data = list(map(

    lambda img: sum(get_4x4_patches_clusters_of_image(img), []),

    training_set_data[1001:5001]

))

testing_4x4_patches_6x6_set_data = list(map(

    lambda img: sum(get_4x4_patches_clusters_of_image(img), []),

    testing_set_data[:1000]

))
#apply one hot on the previous data

training_4x4_6x6_oh_set_data = map(

    lambda lst: sum(map(lambda n: one_hot(n, 36), lst), []),

    training_4x4_patches_6x6_set_data

)

testing_4x4_6x6_oh_set_data = map(

    lambda lst: sum(map(lambda n: one_hot(n, 36), lst), []),

    testing_4x4_patches_6x6_set_data

)
training_4x4_6x6_set_data = list(training_4x4_6x6_oh_set_data)

testing_4x4_6x6_set_data = list(testing_4x4_6x6_oh_set_data)
perceptron_4x4_6x6_no_pooling = Perceptron()

perceptron_4x4_6x6_no_pooling.fit(training_4x4_6x6_set_data, training_set_labels[1001:5001])
print(

    "perceptron 4x4 6x6 no pooling accuracy : ", 

    perceptron_4x4_6x6_no_pooling.score(testing_4x4_6x6_set_data, testing_set_labels[:1000])

)
print(training_4x4_patches_6x6_set_data[0])
def pool_4x4_patches_into_2x2(L):

    #make 4 pooling groups

    G1 = [ L[0],  L[1],  L[4],  L[5]] #corner left  top

    G2 = [ L[2],  L[3],  L[6],  L[7]] #corner right top

    G3 = [ L[8],  L[9], L[12], L[13]] #corner left  bottom

    G4 = [L[10], L[11], L[14], L[15]] #corner right bottom

    #one hot them and regroup by pool

    out = [1 if i in G1 else 0 for i in range(36)] + [1 if i in G2 else 0 for i in range(36)] + [1 if i in G3 else 0 for i in range(36)] + [1 if i in G4 else 0 for i in range(36)]

    return out



print(pool_4x4_patches_into_2x2(training_4x4_patches_6x6_set_data[1]))
#make the pooled ttrainingg and testing sets

training_pooled_set_data = list(map(pool_4x4_patches_into_2x2, training_4x4_patches_6x6_set_data))

testing_pooled_set_data = list(map(pool_4x4_patches_into_2x2,  testing_4x4_patches_6x6_set_data))
#train the pooled percpetron

perceptron_pooled = Perceptron()

perceptron_pooled.fit(training_pooled_set_data, training_set_labels[1001:5001])
print(

    "pooled perceptron accuracy : ", 

    perceptron_pooled.score(testing_pooled_set_data, testing_set_labels[:1000])

) #hummm mutch lower.... hummm... I missed something... hummm... TODO: make it better... :p
def words_extractor_8x8(img):

    #make the image a 24x24 pixels image (to be divisible by 8)

    img = img[2:-2,2:-2]



    #extract patches from image

    patches = list(map(lambda img: img.reshape(1, 64)[0], extract_patch(8, img)))

    clusters_assigments = mb_kmeans.predict(patches)



    #make the BoW

    L = [0] * 64

    for c in clusters_assigments:

        L[c] += 1



    return L



print(words_extractor_8x8(training_set_data[0]))
#extract BoW for makint the trainnig and testing set

training_bow_8x8_set_data = list(map(words_extractor_8x8, training_set_data[1001:5001]))

testing_bow_8x8_set_data = list(map(words_extractor_8x8, testing_set_data[:1000]))
#train a perceptron

perceprton_bow_8x8 = Perceptron()

perceprton_bow_8x8.fit(training_bow_8x8_set_data, training_set_labels[1001:5001])
#test the perceptron

print(

    "BoW perceptron accuracy : ", 

    perceprton_bow_8x8.score(testing_bow_8x8_set_data, testing_set_labels[:1000])

)