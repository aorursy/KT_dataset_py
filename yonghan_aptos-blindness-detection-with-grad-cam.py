import os



import numpy as np

import pandas as pd

import cv2



from PIL import Image

from matplotlib import pyplot as plt



from keras.applications.inception_resnet_v2 import preprocess_input, decode_predictions

from keras.preprocessing.image import load_img, img_to_array

from keras.models import Model, load_model

from keras import backend as K
DATA_PATH = '../input/aptos2019-blindness-detection'

TRAIN_IMG_PATH = os.path.join(DATA_PATH, 'train_images')

TEST_IMG_PATH = os.path.join(DATA_PATH, 'test_images')

TRAIN_LABEL_PATH = os.path.join(DATA_PATH, 'train.csv')

TEST_LABEL_PATH = os.path.join(DATA_PATH, 'test.csv')



train_df = pd.read_csv(TRAIN_LABEL_PATH)

test_df = pd.read_csv(TEST_LABEL_PATH)



train_df.head()
model_path = '../input/aptos-2019-pretrained-models/'

weight_file = 'weights-InceptionResNetV2.hdf5'

model = load_model(os.path.join(model_path, weight_file))

# model.summary()
fig, ax = plt.subplots(3, 5, figsize=(20,10))



image_size = (299, 299)

start_index = 0

num_output = 5



for idx in range(num_output):

    index = idx

    index += start_index

    img_path = os.path.join(TRAIN_IMG_PATH, train_df['id_code'][index]+'.png')

    

    # ==================================

    #   1. Test images visualization

    # ==================================

    img = load_img(img_path, target_size=image_size)

    img = np.expand_dims(img, axis=0)

    pred_img = preprocess_input(img)

    pred = model.predict(pred_img)

    ax[0][idx].imshow(img[0])

    ax[0][idx].set_title('ID: {}, Predict: {}'.format(train_df['id_code'][index], np.argmax(pred)))

    

    # ==============================

    #   2. Heatmap visualization 

    # ==============================

    # Item of prediction vector

    pred_output = model.output[:, np.argmax(pred)]

    

    # Feature map of 'conv_7b_ac' layer, which is the last convolution layer

    last_conv_layer = model.get_layer('conv_7b_ac')

    

    # Gradient of class for feature map output of 'conv_7b_ac'

    grads = K.gradients(pred_output, last_conv_layer.output)[0]

    

    # Feature map vector with gradient average value per channel

    pooled_grads = K.mean(grads, axis=(0, 1, 2))

    

    # Given a test image, get the feature map output of the previously defined 'pooled_grads' and 'conv_7b_ac'

    iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])

    

    # Put a test image and get two numpy arrays

    pooled_grads_value, conv_layer_output_value = iterate([pred_img])

    

    # Multiply the importance of a channel for a class by the channels in a feature map array

    for i in range(int(pooled_grads.shape[0])):

        conv_layer_output_value[:, :, i] *= pooled_grads_value[i]

        

    # The averaged value along the channel axis in the created feature map is the heatmap of the class activation

    heatmap = np.mean(conv_layer_output_value, axis=-1)

    

    # Normalize the heatmap between 0 and 1 for visualization

    heatmap = np.maximum(heatmap, 0)

    heatmap /= np.max(heatmap)

    ax[1][idx].imshow(heatmap)

    

    # =======================

    #   3. Apply Grad-CAM

    # =======================

    ori_img = load_img(img_path, target_size=image_size)

    

    heatmap = cv2.resize(heatmap, image_size)

    heatmap = np.uint8(255 * heatmap)

    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)



    superimposed_img = heatmap * 0.5 + ori_img

    cv2.imwrite('./grad_cam_result{}.jpg'.format(idx), superimposed_img)

    grad_img = cv2.imread('./grad_cam_result{}.jpg'.format(idx))

    

    ax[2][idx].imshow(grad_img)

    

plt.show()
! rm -rf *.jpg