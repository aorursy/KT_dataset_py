# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator, load_img
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import random
import os
import zipfile
import tensorflow as tf
from keras.models import load_model

print(os.listdir("../input"))
with zipfile.ZipFile("/kaggle/input/dogs-vs-cats/train.zip","r") as z:
    z.extractall(".")
    
with zipfile.ZipFile("/kaggle/input/dogs-vs-cats/test1.zip","r") as z:
    z.extractall(".")

print(os.listdir('../working/train')) # os.listdir 로 working
filenames = os.listdir('../working/train')
categories = []
print(len(os.listdir('../working/test1')))
for filename in filenames:
    category = filename.split('.')[0]
    if category == 'dog':
        categories.append(1)
    else:
        categories.append(0)
len(categories)
df1 = pd.DataFrame({
    'name' : filenames,
    'category' : categories
})
df1
df1['category'].value_counts().plot.bar()
import collections
collections.Counter(df1['category']) 
#각각 12500 , 12500개가 있음을 알 수 있다. 
sample = random.choice(filenames)
sample
img = load_img('../working/train/'+sample)
plt.imshow(img)
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Activation, BatchNormalization
# BatchNormalization 을 통해 Vanishing 현상을 방지해준다고함. 평균과 분산을 0 , 1에 맞춰 준다고 함...  ? 
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape =(128, 128, 3)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

# 두번재 layer 층 생성
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, (3,3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
model.summary()
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
# ReduceLROnPlateau 는 learning rate를 줄여주거나 높여주어 local minima 에서 빠져 나오도록 도와주는 callback함수이다. 

MODEL_DIR = './model/'
if not os.path.exists(MODEL_DIR) :
    os.mkdir(MODEL_DIR)
modelpath = './model/{epoch:02d}-{val_loss:.4f}.hdf5'
CheckPointer = ModelCheckpoint(filepath = modelpath, moniter='val_loss', verbose1=1, save_best_only=True)
earlystop = EarlyStopping(patience=10 )
learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', # 검증 accuracy를 모니터링함.
                                            patience=2, # 2번동안 개선되지 않으면 callback을 호출한다 
                                            verbose=1, # 과정을 보여줌
                                            factor=0.5, # 개선이 없어 callback 호출시 학습률을 1/2 로 줄임
                                            min_lr=0.00001) # 학습률의 하한값을 설정 
callbacks=[earlystop, learning_rate_reduction, CheckPointer]

df1
df1['category'] = df1['category'].replace({0: 'cat', 1: 'dog'})
df1
train_df, test_df = train_test_split(df1, test_size=0.3, random_state=30)
 
train_df = train_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)
# 인덱스를 초기화 시켜준다.
print(train_df['category'].value_counts())
print(test_df['category'].value_counts())
print(train_df.shape[0])
print(test_df.shape[0])
# ImageDataGenerator 를 사용하여 이미지 데이터를 부풀려준다.

train_datagen = ImageDataGenerator(
    rotation_range = 30, # 지정학 각도 내에서 이미지를 회전
    rescale= 1./255, # ?? 이미지 픽셀값을 0 ~ 1 사이로 맞춰주기 위해서 1./255로 설정해주었다. 
    shear_range=0.1, # 시계 반대방향으로 밀림강도? 를 나타낸다. 
    zoom_range = 0.3, # 원본이미지를 확대/ 축소한다.
    horizontal_flip=True, # 수평방향으로 좌우 반전 한다.
    width_shift_range=0.1,# 수평방향 이동범위 내에서 이동시킨다.
    height_shift_range=0.1 # 지정된 수직방향 범위 내에서 임의로 이동시킨다.  
)
# 위에서 설정한 객체 train_datagen을 이용.
# ImageDataGenerator 의 메소드 flow_from_dataframe 을 사용한다. 
train_generator = train_datagen.flow_from_dataframe(
    train_df, # 사용할 데이터 프레임
    '../working/train/',
    x_col='name',
    y_col='category',
    target_size=(128, 128),
    class_mode='categorical',
    batch_size=15
)
test_datagen = ImageDataGenerator(rescale= 1./255)
test_generator = test_datagen.flow_from_dataframe(
    test_df, # 사용할 데이터 프레임
    '../working/train/', # 데이터의 위치
    x_col='name', # 파일위치 열이름
    y_col='category', # 클래스 열이름
    target_size=(128, 128), # 이미지 사이즈
    class_mode='categorical', # y값 변화방법
    batch_size=15 # 배치 사이즈 
)
# flow_from_directory를 사용하여 directory를 이용하여 데이터를 생성할 수 잇다.
# 시험해보기 
# example_df = train_df.sample(n=1) # 샘플 한개를 뽑는다
example_df = train_df.sample(n=1).reset_index(drop=True) # 인덱스를 리셋해준다. 

example_generater = train_datagen.flow_from_dataframe(
    example_df,
    '../working/train/', # 데이터의 위치
    x_col='name', # 파일위치 열이름
    y_col='category', # 클래스 열이름 여기선 dog, cat  을 저장한 열의 이름
    target_size=(128, 128), # 이미지 사이즈
    class_mode='categorical', # y값 변화방법
    batch_size=8 # 배치 사이즈 
)

# for x_batch, y_batch in example_generater:
#     print(y_batch[0])
# example_generater 확인결과 # 뭔가 데이터가 많이 생성됨을 알 수 있었음 . 
# x_batch , y_batch 를 돌려본결과 뭔가 임의의 데이터를 생성해 주는 것 같았음 .
# 몇개인지는 자세히 모르겠음 . 질문해야함.
plt.figure(figsize=(12, 12))
for i in range(0, 15) :
    plt.subplot(5, 3, i+1)
    # 서브플롯을 생성해준다. 
    for x_batch, y_batch in example_generater:
        image = x_batch[0]
        plt.imshow(image)
        break
    # 실행을 해줄때 마다 이미지 데이터가 바뀌는 걸로 보아임읠 생성된 이미지 데이터를 그려주는 듯함. 
plt.tight_layout()
plt.show()
# https://medium.com/@vijayabhaskar96/tutorial-on-keras-imagedatagenerator-with-flow-from-dataframe-8bd5776e45c1 
# 위에 사이트를 참조한다. 

epochs=3 
history = model.fit_generator(
    train_generator, # 훈련데이터셋을 제공할 제네레이터를 지정합니다. train_generator 
    epochs=epochs, # 에폭스는 앞에 지정한 3을 
    validation_data=test_generator, # 검증데이터셋을 제공할 제네레이터를 지정합니다. 앞서 지정한 test_generator를 사용하여 검증한다.
    validation_steps=train_df.shape[0]//15, # 한 epoch 종료 시 마다 검증할 때 사용되는 검증 스텝 수를 지정합니다.
    steps_per_epoch=test_df.shape[0]//15, # 한 epoch에 사용한 스텝 수를 지정합니다. 훈련샘플수/batch_size 로 스텝수를 지정.
    callbacks=callbacks # 앞에서 지정한 callbacks 를 이용하여 모델에 추가.

) # 제너레이터로 생성된 배치로 학습을 시키는 경우에는 fit_generator 로 학습을 시킨다. 
model.save("model.h5")
# 모델을 저장해준다.

model.load_model('./model.h5') # 어제 학습시킨 모델의 가중치들을 가져온다 .
model.weights
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))

# loss[손실값], val_loss[검증손실값]을 순서대로 그려준다. 
ax1.plot(history.history['loss'], color='b', label='training loss')
ax1.plot(history.history['val_loss'], color='r', label='validation loss')
# 모델을 load 하는 과정에서 어제 그린 그래프가 사라졌다. 대충 학습이 진행될 수록 (convex) loss 값과 val_loss 값이 줄어드는 그래프가 그려짐.
ax2.set_xticks(np.arange(1, epochs, 1)) # set_xticks x축에 표시하고 싶은 값들을 설정한다. 

legend = plt.legend(loc='best', shadow=True) # 구분자의 위치를 설정해준다. 
plt.tight_layout()
plt.show()
test_filenames = os.listdir("../working/test1/")
test_df = pd.DataFrame({
 'filename': test_filenames
 })
nb_samples = test_df.shape[0]
print(len(test_df))
test_df['filename']
test_gen = ImageDataGenerator(rescale=1./255)

test_generator = test_gen.flow_from_dataframe(
    test_df, 
    "../working/test1/", 
    x_col='filename',
    y_col=None,
    class_mode=None,
    target_size=(128, 128),
    batch_size=15,
    shuffle=False
)
print(test_generator)
print(test_df.shape[0])
print(np.ceil(test_df.shape[0]/15))
predict = model.predict_generator(test_generator,
                                  steps=np.ceil(test_df.shape[0]/15)) 
# fit_generator 와 같이 predict_generator 를 사용하여 예측해준다. 
print(predict.shape)
print(predict[:,-1])
np.argmax(predict, axis=-1)
print(np.argmax(predict, axis=-1).shape)
print(test_df)
np.argmax(predict, axis=1)
test_df['category'] = np.argmax(predict, axis=-1)
# predict 값의 argmax를 이용해 0,1 인지 판단하여 test category 값에 넣어준다.
label_map = dict((v,k) for k,v in train_generator.class_indices.items())
# 이것도 찍어보고 이해할것 . 
test_df['category'] = test_df['category'].replace(label_map)
# 위에서 설정한 label_map 으로 replace 하는 것 같음. 
test_df['category'] = test_df['category'].replace({ 'dog': 1, 'cat': 0 })
# label_map으로 replace 한값을 다시 1 , 0 으로 replace 함
test_df['category'].value_counts().plot.bar()
# barplot을 그려서 데이터 값을 확인해 준다. 
# image와 함께 predict를 확인한다. 

sample_test = test_df.head(18)
sample_test.head()
plt.figure(figsize=(12, 24))

for index, row in sample_test.iterrows():
    filename = row['filename']
    category = row['category']
    img = load_img("../working/test1/"+filename, target_size=(128, 128))
    plt.subplot(6, 3, index+1)
    plt.imshow(img)
    plt.xlabel(filename + '(' + "{}".format(category) + ')' )
plt.tight_layout()
plt.show()
# 제출한다. 

submission_df = test_df.copy()
submission_df['id'] = submission_df['filename'].str.split('.').str[0]
submission_df['label'] = submission_df['category']
submission_df.drop(['filename', 'category'], axis=1, inplace=True)
submission_df.to_csv('submission.csv', index=False)

history.history
