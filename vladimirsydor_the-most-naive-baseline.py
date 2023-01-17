from os import path

from glob import glob



import pandas as pd

import numpy as np

import cv2



from tqdm.notebook import tqdm

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import cross_val_score
ROOT_DIR_TRAIN = '/kaggle/input/evohackaton/train/train'
def extract_flatten_image_from_image_name(img_name, root=None):

    if root is not None:

        im_path = path.join(root, img_name)

    else:

        im_path = img_name

    im = cv2.imread(im_path, 0)

    im = cv2.resize(im, (32,32))

    return im.flatten()
train_df = pd.read_csv('/kaggle/input/evohackaton/train.csv')

train_images = list(map(lambda x: extract_flatten_image_from_image_name(x, ROOT_DIR_TRAIN), train_df['name']))
train_images = np.stack(train_images).astype(float)

train_labels = np.array(train_df['category'])
model = LogisticRegression()
val_results = cross_val_score(X=train_images, y=train_labels, estimator=model, cv=5)
full_image_pathes = glob('/kaggle/input/evohackaton/test/test/*.jpg')

test_image_names = list(map(path.basename, full_image_pathes))



test_images = list(map(extract_flatten_image_from_image_name, full_image_pathes))
test_images = np.stack(test_images).astype(float)
model.fit(train_images, train_labels)
y_test_hat = model.predict(test_images)
submission_df = pd.DataFrame({

    'name':test_image_names,

    'category':y_test_hat

})
submission_df.to_csv('submission.csv',index=False)