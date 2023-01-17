import os
import matplotlib.pyplot as plt
import PIL.Image as Image
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import numpy as np
file_names = os.listdir("../input/fruit")
file_names.sort()
print(file_names)
print('The number of fruit images: ', len(file_names))
c = 8
r = len(file_names) // c + 1
plt.figure(figsize=(15,15))
for i, file_name in enumerate(file_names):
    abs_file_path = '../input/fruit/' + file_name
    img = image.load_img(abs_file_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = x.astype('float32') / 255
    plt.subplot(r, c, i+1)
    plt.title(i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(x)
# consine similarity
def cosine_similarity(ratings):
    sim = ratings.dot(ratings.T)
    norms = np.array([np.sqrt(np.diagonal(sim))])
    return (sim / norms / norms.T)
y_test = []
x_test = []
for file_name in file_names:
    abs_file_path = '../input/fruit/' + file_name
    img = image.load_img(abs_file_path, target_size=(224, 224))
    y_test.append(int(file_name[0:2]))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    if len(x_test) > 0:
        x_test = np.concatenate((x_test, x))
    else:
        x_test = x
x_test.shape
print(y_test)
# convert input to VGG format
x_test = preprocess_input(x_test)

# include_top=False: exclude top(last) 3 fully-connected layers. get features dim=(1,7,7,512)
model = VGG16(weights='imagenet', include_top=False)

# use VGG to extract features
features = model.predict(x_test)
# flatten as one dimension
features_compress = features.reshape(len(y_test), 7 * 7 * 512)

# compute consine similarity
cos_sim = cosine_similarity(features_compress)
# random choose 5 samples to test
inputNos = np.random.choice(len(y_test), 5, replace=False)

for inputNo in inputNos:
    # select two best similar images 
    top = np.argsort(-cos_sim[inputNo], axis=0)[1:3]
    recommend = [y_test[i] for i in top]
    output = 'input: \'{}\', recommend: {}'.format(inputNo + 1, recommend)
    print(output)
