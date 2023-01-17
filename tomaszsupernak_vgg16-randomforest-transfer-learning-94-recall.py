#importing all the neccesary packages
import numpy as np
import pandas as pd
import cv2
import seaborn as sns
import os
from PIL import Image

import matplotlib.pyplot as plt
import glob

from keras.models import Sequential
from keras.layers import Conv2D, Dense, Flatten, BatchNormalization, MaxPooling2D

from keras.applications import VGG16

from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from keras.utils import  to_categorical
from sklearn import metrics
from sklearn.metrics import confusion_matrix
SIZE = 120
def datafunc(datadir): 
    images = []
    labels = []
    for dir_path in glob.glob(datadir):
        label = dir_path.split('/')[-1] 
        print(label)
        for img_path in glob.glob(os.path.join(dir_path, '*.jpeg')):
            try:
                img = cv2.imread(img_path, cv2.IMREAD_COLOR)
                img = cv2.resize(img, (SIZE, SIZE))
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                images.append(img)
                labels.append(label)
            except Exception as e:
                print(e)
      
    return np.array(images), np.array(labels)
train_images, train_labels = datafunc('../input/chest-xray-pneumonia/chest_xray/train/*')
test_images, test_labels = datafunc('../input/chest-xray-pneumonia/chest_xray/test/*')
val_images, val_labels = datafunc('../input/chest-xray-pneumonia/chest_xray/val/*')
plt.figure(figsize=(20,5))

plt.subplot(1,3,1)
sns.countplot(train_labels)
plt.title('Train data')

plt.subplot(1,3,2)
sns.countplot(test_labels)
plt.title('Test data')

plt.subplot(1,3,3)
sns.countplot(val_labels)
plt.title('Validation data')

plt.show()
df = pd.DataFrame({'label': train_labels, 'images': list(train_images)}, columns=['label', 'images'])

count_class_0, count_class_1 = df.label.value_counts()

df_class_0 = df[df['label'] == 'PNEUMONIA']
df_class_1 = df[df['label'] == 'NORMAL']

print('Pneumonia cases =', count_class_0)
print('Normal =', count_class_1)
df_class_0_under = df_class_0.sample(count_class_1)
df_test_under = pd.concat([df_class_0_under, df_class_1], axis=0)

print('Random under-sampling:')
print(df_test_under.label.value_counts())

df_test_under.label.value_counts().plot(kind='bar', title='Count (label)');
t_images = []
for img in df_test_under['images']:
    img = Image.fromarray(img)
    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    t_images.append(img)
t_images = np.array(t_images)
train_labels = np.array(df_test_under['label'])
le = preprocessing.LabelEncoder()
le.fit(test_labels)
test_labels_encoded = le.transform(test_labels)
le.fit(train_labels)
train_labels_encoded = le.transform(train_labels)

x_train, y_train, x_test, y_test = t_images, train_labels_encoded, test_images, test_labels_encoded
x_train, x_test = x_train / 255.0, x_test / 255.0 #scaling the data
VGG_model = VGG16(weights = 'imagenet', include_top = False, input_shape = (SIZE, SIZE, 3))  

for layer in VGG_model.layers:
    layer.trainable = False

VGG_model.summary()
feature_extractor = VGG_model.predict(x_train)
features = feature_extractor.reshape(feature_extractor.shape[0], -1)

X_rf = features 
RF_model = RandomForestClassifier(n_estimators=100, random_state=42)

RF_model.fit(X_rf, y_train)

X_test_feature = VGG_model.predict(x_test)
X_test_features = X_test_feature.reshape(X_test_feature.shape[0], -1)

prediction_RF = RF_model.predict(X_test_features)
cm = confusion_matrix(y_test, prediction_RF)
sns.heatmap(cm, annot = True) 

tp, fp, fn, tn = cm.ravel()

recall = tp/(tp+fn)
precision = tp/(tp+fp)

print('Accuracy =', metrics.accuracy_score(y_test, prediction_RF))
print("Recall =", recall)
print("Precision =", precision)