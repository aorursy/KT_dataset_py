import os



import numpy as np

import pandas as pd



from sklearn.metrics import roc_auc_score, roc_curve

from sklearn.model_selection import train_test_split



import matplotlib.pyplot as plt



from skimage import io



%matplotlib inline
font = {

    'family': 'serif',

    'color':  'darkred',

    'weight': 'bold',

    'size': 22,

}
SEED = 257



TRAIN_DIR = '../input/train/train/'

TEST_DIR = '../input/test/test/'
categories = ['hot dog', 'not hot dog']
X, y = [], []



for category in categories:

    category_dir = os.path.join(TRAIN_DIR, category)

    for image_path in os.listdir(category_dir):

        X.append(io.imread(os.path.join(category_dir, image_path)))

        y.append(category)
y = [1 if x == 'hot dog' else 0 for x in y]
X_train, X_test, y_train, y_test = train_test_split(np.array(X), np.array(y), test_size=0.25, random_state=SEED)
X_train.shape, X_test.shape, y_train.shape, y_test.shape
import tensorflow as tf

from tensorflow.keras import backend as K



def auc_neural(y_true, y_pred):

    auc = tf.metrics.auc(y_true, y_pred)[1]

    K.get_session().run(tf.local_variables_initializer())

    return auc
from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Convolution2D, MaxPooling2D, BatchNormalization, Flatten, Dropout, Dense



from tensorflow.keras import optimizers



model = Sequential()

model.add(BatchNormalization(axis=1))

model.add(Convolution2D(128, kernel_size=3, activation='relu'))

model.add(BatchNormalization(axis=1))

model.add(MaxPooling2D(pool_size=2))

model.add(BatchNormalization(axis=1))

model.add(Convolution2D(64, kernel_size=3, activation='relu'))

model.add(BatchNormalization(axis=1))

model.add(MaxPooling2D(pool_size=2))

model.add(Flatten())

model.add(Dropout(0.7))

model.add(BatchNormalization(axis=1))

model.add(Dense(1, activation='sigmoid'))



optimizer = optimizers.Adam(lr=3e-5)

model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy', auc_neural])
model.fit(X_train, y_train, 

          batch_size=128,

          epochs=300,

          validation_data=(X_test, y_test))
prediction = model.predict(X_test)

auc_score = roc_auc_score(y_test, prediction)

print(auc_score)
fpr_color, tpr_color, threshold_color = roc_curve(y_test, prediction)



fpr_color, tpr_color, threshold_color = roc_curve(y_test, prediction)

plt.plot(fpr_color, tpr_color, color='darkorange',

         lw='3', label='ROC curve (area = %0.3f)' % auc_score)

plt.legend(loc="lower right")
leaderboard_X = []

leaderboard_filenames = []
for image_path in os.listdir(TEST_DIR):

    leaderboard_X.append(io.imread(os.path.join(TEST_DIR, image_path)))

    leaderboard_filenames.append(image_path)
leaderboard_X = np.array(leaderboard_X)



leadeboard_predictions = model.predict(leaderboard_X)



leadeboard_predictions = leadeboard_predictions.reshape(len(leadeboard_predictions))
idx = 10



plt.axis("off");

if leadeboard_predictions[idx] > 0.5:

    plt.text(20, -5, 'HOT DOG!!!', fontdict=font)

else:

    plt.text(15, -5,'not hot dog...', fontdict=font)

plt.imshow(leaderboard_X[idx], cmap='gray');
submission = pd.DataFrame(

    {

        'image_id': leaderboard_filenames, 

        'image_hot_dog_probability': leadeboard_predictions

    }

)
submission.head()
submission.to_csv('submit.csv', index=False)