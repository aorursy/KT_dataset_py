import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import glob
import seaborn as sns
import random
from sklearn.preprocessing import LabelBinarizer
import os 
os.listdir('../input/dogcat-release/')
train = glob.glob('../input/dogcat-release/train/*.jpg')
test =glob.glob('../input/dogcat-release/test/*.jpg')
train = np.random.permutation(train)
train[:5]
label_names = set([p.split('/')[-1].split('.')[0] for p in train])
label_names 
labels = [p.split('/')[-1].split('.')[0] for p in train]
labels[:5]
labels = pd.DataFrame(labels,columns=['Type'])
Class = labels['Type'].unique()
Class_dict = dict(zip(Class, range(1,len(Class)+1)))

labels['str'] = labels['Type'].apply(lambda x: Class_dict[x])
lb = LabelBinarizer()
lb.fit(list(Class_dict.values()))
transformed_labels = lb.transform(labels['str'])
y_bin_labels = []  

for i in range(transformed_labels.shape[1]):
    y_bin_labels.append('str' + str(i))
    labels['str' + str(i)] = transformed_labels[:, i]
Class_dict
labels.drop('str',axis=1,inplace=True)
labels.drop('Type',axis=1,inplace=True)
labels = labels.str0.values
labels[:5]
#预处理函数
def preprocess_image(path,label):
    image = tf.io.read_file(path)                           
    image = tf.image.decode_jpeg(image,3)               
    image = tf.image.resize(image,[224,224])       
    image = tf.cast(image/127.5 -1,tf.float32)     

    return image,label       
dataset = tf.data.Dataset.from_tensor_slices((train, labels)) 
dataset = dataset.shuffle(len(train))
dataset
#创建数据集
AUTO = tf.data.experimental.AUTOTUNE
dataset = dataset.map(preprocess_image, num_parallel_calls = AUTO)
#切分数据集
test_count = int(len(train)*0.2)
train_count = len(train) - test_count
train_count,test_count
train_dataset = dataset.skip(test_count) 
test_dataset = dataset.take(test_count)
batch_size = 128
train_dataset = train_dataset.repeat().shuffle(800).batch(batch_size)
train_dataset = train_dataset.prefetch(AUTO)
test_dataset = test_dataset.batch(batch_size)
train_dataset
#Xception
conv1 = keras.applications.xception.Xception(weights='imagenet',
                                            include_top=False,
                                            input_shape=(224,224,3),
                                            pooling='avg')
#ResNet152
conv2 = keras.applications.resnet.ResNet152(weights='imagenet',
                                            include_top=False,
                                            input_shape=(224,224,3),
                                            pooling='avg')
#InceptionResNetV2
conv3 = keras.applications.inception_v3.InceptionV3(weights='imagenet',
                                                    include_top=False,
                                                    input_shape=(224,224,3),
                                                    pooling='avg')
conv1.trainable = False
conv2.trainable = False
conv3.trainable = False
conv1.inputs,conv2.inputs,conv3.inputs
conv1.outputs,conv2.outputs,conv3.outputs
def model():
    inputs = tf.keras.layers.Input(shape=(224, 224, 3))
    out1 = conv1(inputs)
    out2 = conv2(inputs)
    out3 = conv3(inputs)
    out = tf.keras.layers.concatenate([out1,out2,out3],axis=1)
    out = tf.keras.layers.Dropout(0.5)(out)
    output = tf.keras.layers.Dense(1, activation='sigmoid')(out)
    
    return tf.keras.Model(inputs=inputs, outputs=output)
model = model()
model.summary()
tf.keras.utils.plot_model(model,show_shapes=True,dpi=300)
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['acc'])
EPOCHS = 10
history = model.fit(train_dataset,
                   steps_per_epoch=train_count//batch_size,
                   epochs=EPOCHS,
                   validation_data=test_dataset,
                   validation_steps=test_count//batch_size,
                   )
def plot_history(history):                
    hist = pd.DataFrame(history.history)           
    hist['epoch']=history.epoch
    
    plt.figure()                                     
    plt.xlabel('Epoch')
    plt.ylabel('Binary_crossentropy')               
    plt.plot(hist['epoch'],hist['loss'],
            label='Train Loss')
    plt.plot(hist['epoch'],hist['val_loss'],
            label='Val Loss')                           
    plt.legend()
    
    plt.figure()                                      
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')               
    plt.plot(hist['epoch'],hist['acc'],
            label='Train Acc')
    plt.plot(hist['epoch'],hist['val_acc'],
            label='Val Acc')
    plt.legend()      
    
    plt.show()
    
plot_history(history)          
y_pred = model.predict(test_dataset, verbose=1)
y_pred = y_pred.clip(min=0.005, max=0.995)
y_pred
def preprocess_test(path):
    image = tf.io.read_file(path)                           
    image = tf.image.decode_jpeg(image,3)               
    image = tf.image.resize(image,[224,224])       
    image = tf.cast(image/127.5 -1,tf.float32)     

    return image  
Dir = "../input/dogcat-release/test/"
imgList = os.listdir(Dir)
imgList.sort(key=lambda x: int(x.split('/')[-1].split('.')[0]))
imgList[:5]
test_path = []
for count in range(0, len(imgList)):
    im_name = imgList[count]
    im_path = os.path.join(Dir,im_name)
    test_path.append(im_path)
    print(im_path)
test_path[:5]
val_dataset = tf.data.Dataset.from_tensor_slices(test_path) 
val_dataset = val_dataset.map(preprocess_test, num_parallel_calls = AUTO)
val_dataset = val_dataset.batch(batch_size)
y_pred = model.predict(val_dataset, verbose=1)
y_pred[:10]
pred = pd.DataFrame(y_pred).iloc[:,0].values
pred[:5]
def type_change(data):
    for i in range(data.shape[0]):
        if data[i] > 0.5:
            data[i] = 0.005
        else: data[i] = 0.995
    return data

predict_labels = type_change(pred)
predict_labels[:10]
prediction = pd.DataFrame({"label":predict_labels})
prediction.index += 1 
prediction.to_csv('pred.csv',
                  index_label='id')
