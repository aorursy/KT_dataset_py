import numpy as np
import pandas as pd
import pylab as plt
import seaborn as sns
import cv2
import os
train = pd.read_csv('../input/train.csv')
train.head()
img_path='../input/train/'+train.id[5]+".png"
image=cv2.imread(img_path)
image_rgb=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
h,s,v=np.average(hsv_image,axis=(0,1))

plt.subplot(131),plt.imshow(image),plt.title('BGR')
plt.subplot(132),plt.imshow(image_rgb),plt.title('RBG')
plt.subplot(133),plt.imshow(hsv_image),plt.title('HSV')
image=cv2.imread(img_path)
img=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# Convert BGR to HSV
hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
# define range of blue color in HSV
lower_blue=np.array([20,25,15])
upper_blue=np.array([130,255,255])
#Threshold the HSV impage to get only blue colors
mask=cv2.inRange(hsv_image,lower_blue,upper_blue)
# Bitwise-And mask and original image
res = cv2.bitwise_and(img,img,mask=mask)
plt.subplot(131),plt.imshow(img),plt.title('ORIGINAL')
plt.subplot(132),plt.imshow(mask),plt.title('Mask')
plt.subplot(133),plt.imshow(res),plt.title('Res')
def mask(img_path):
    image=cv2.imread(img_path)
    img=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_blue=np.array([20,25,15])
    upper_blue=np.array([130,255,255])
    mask=cv2.inRange(hsv_image,lower_blue,upper_blue)
    res = cv2.bitwise_and(img,img,mask=mask)
    plt.subplot(131),plt.imshow(img),plt.title('ORIGINAL')
    plt.subplot(132),plt.imshow(mask),plt.title('Mask')
    plt.subplot(133),plt.imshow(res),plt.title('Res')
    plt.show()
    h,s,v=np.average(res,axis=(0,1))
    print(h, s, v)

for i in range(5):
    img_path='../input/train/'+train.id[i]+".png"
    mask(img_path)
def image_feature_extracion(img_path):
    image=cv2.imread(img_path)
    img=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_blue=np.array([20,25,15])
    upper_blue=np.array([130,255,255])
    mask=cv2.inRange(hsv_image,lower_blue,upper_blue)
    res = cv2.bitwise_and(img,img,mask=mask)
    h,s,v=np.average(hsv_image,axis=(0,1))
    return h,s,v

read_len=1000
hsv_list=[]
for i in range(read_len):    
    img_path='../input/train/'+train.id[i]+".png"    
    hsv_list.append(image_feature_extracion(img_path))
    
import seaborn as sns
df = pd.DataFrame(hsv_list, columns=["Hue", "y",'Brightness(Values)'])
sns.jointplot(x="Hue", y="Brightness(Values)", data=df)
df.head()
def image_feature_extracion(img_path,ID):
    image=cv2.imread(img_path)
    img=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_blue=np.array([20,25,15])
    upper_blue=np.array([130,255,255])
    mask=cv2.inRange(hsv_image,lower_blue,upper_blue)
    res = cv2.bitwise_and(img,img,mask=mask)
    h,s,v=np.average(hsv_image,axis=(0,1))
    return ID,h,s,v

read_len=10000 #109237
hsv_list=[]
for i in range(read_len):    
    img_path='../input/train/'+train.id[i]+".png"
    ID=train.id[i]
    hsv_list.append(image_feature_extracion(img_path,ID))

df = pd.DataFrame(hsv_list, columns=["ID","Hue", "y",'Brightness(Values)'])
df.head()
df.shape
train['attribute_ids'].head()
train["attribute_ids"] = train["attribute_ids"].apply(lambda x:x.split(" "))
train['attribute_ids'].head()
labels = pd.read_csv('../input/labels.csv')
labels.shape
train_labels = []
for label in train['attribute_ids'][:10000].values:
    zeros = np.zeros(labels.shape[0])
    for label_i in label:
        zeros[int(label_i)] = 1
    train_labels.append(zeros)
    
train_labels = np.asarray(train_labels)
train_labels
train_labels.shape
Y = train_labels
features = ['Hue','y','Brightness(Values)']
X = df[features]
print(Y.shape,X.shape)
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
model.fit(X, Y)
sns.set(style="darkgrid")
fig, ax = plt.subplots(figsize=(6,6))
y_pos = np.arange(len(features))
plt.barh(y_pos, model.feature_importances_, align='center', alpha=0.4)
plt.yticks(y_pos, features)
plt.xlabel('features')
plt.title('feature_importances')
plt.show()