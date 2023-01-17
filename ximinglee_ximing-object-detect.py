import tensorflow as tf

import matplotlib.pyplot as plt

%matplotlib inline

import numpy as np

import glob

from lxml import etree

from matplotlib.patches import Rectangle#画矩形框的方法

import os
gpu = tf.config.experimental.list_physical_devices(device_type='GPU')

tf.config.experimental.set_memory_growth(gpu[0],True)
print(os.listdir('../input/the-oxfordiiit-pet-dataset/images/images')[:3])
image = tf.io.read_file('../input/the-oxfordiiit-pet-dataset/images/images/Abyssinian_1.jpg')

image = tf.image.decode_jpeg(image)

plt.imshow(image)
xml = open('../input/the-oxfordiiit-pet-dataset/annotations/annotations/xmls/Abyssinian_1.xml').read()

sel = etree.HTML(xml)
int(sel.xpath('//name/text()')[0] == 'cat')  #cat=1.dog=0
label = int(sel.xpath('//name/text()')[0] == 'cat')

width = int(sel.xpath('//width/text()')[0])

height = int(sel.xpath('//height/text()')[0])

xmin = int(sel.xpath('//xmin/text()')[0])

ymin = int(sel.xpath('//ymin/text()')[0])

xmax = int(sel.xpath('//xmax/text()')[0])

ymax = int(sel.xpath('//ymax/text()')[0])
plt.imshow(image)

rect = Rectangle((xmin,ymin),(xmax-xmin),(ymax-ymin),fill=False,color='red')

ax = plt.gca()

ax.axes.add_patch(rect)
b1 = xmin/width

b2 = xmax/width

b3 = ymin/height

b4 = ymax/height
b1,b2,b3,b4
image.shape
image = tf.image.resize(image,(256,256))

image = image/255

xmin = b1*256

xmax = b2*256

ymin = b3*256

ymax = b4*256

plt.imshow(image)

rect = Rectangle((xmin,ymin),(xmax-xmin),(ymax-ymin),fill=False,color='red')

ax = plt.gca()

ax.axes.add_patch(rect)
print(image)
image.shape
images = glob.glob('../input/the-oxfordiiit-pet-dataset/images/images/*.jpg')

xmls = glob.glob('../input/the-oxfordiiit-pet-dataset/annotations/annotations/xmls/*.xml')
images[:3]
xmls[:3]
names = [x.split('/')[-1].split('.')[0] for x in xmls]
names[:3]
train_images = [image for image in images if (image.split('/')[-1].split('.')[0]) in names]

test_images = [image for image in images if (image.split('/')[-1].split('.')[0]) not in names]
len(train_images),len(test_images)
train_images.sort(key=lambda x:x.split('/')[-1].split('.')[0])

xmls.sort(key=lambda x:x.split('/')[-1].split('.')[0])
t=0

for i in range(3686):

    if train_images[i].split('/')[-1].split('.')[0] ==xmls[i].split('/')[-1].split('.')[0]:

        t+=1

print(t)
def load_label(path):

    xml = open(path).read()

    sel = etree.HTML(xml)

    pet_label = int(sel.xpath('//name/text()')[0] == 'cat')

    width = int(sel.xpath('//width/text()')[0])

    height = int(sel.xpath('//height/text()')[0])

    xmin = int(sel.xpath('//xmin/text()')[0])

    ymin = int(sel.xpath('//ymin/text()')[0])

    xmax = int(sel.xpath('//xmax/text()')[0])

    ymax = int(sel.xpath('//ymax/text()')[0])

    return [pet_label,xmin/width,ymin/height,xmax/width,ymax/height]
labels = [load_label(path) for path in xmls]
pet_label,out1,out2,out3,out4 = list(zip(*labels))
pet_label = np.array(pet_label)

out1 = np.array(out1)

out2 = np.array(out2)

out3 = np.array(out3)

out4 = np.array(out4)

labels_ds = tf.data.Dataset.from_tensor_slices((pet_label,out1,out2,out3,out4))
labels_ds
def load_image(path):

    image = tf.io.read_file(path)

    image = tf.image.decode_jpeg(image,channels=3)

    image = tf.image.resize(image,[224,224])

    image = image/127.5-1 

    return image
image_ds = tf.data.Dataset.from_tensor_slices(train_images)
image_test_ds = tf.data.Dataset.from_tensor_slices(test_images)
image_ds = image_ds.map(load_image)
image_test_ds = image_test_ds.map(load_image)
image_ds
image_test_ds#测试集数据
dataset = tf.data.Dataset.zip((image_ds,labels_ds))
dataset
BATCH_SIZE = 32
dataset = dataset.shuffle(len(train_images)).batch(BATCH_SIZE).repeat()
test_dataset = image_test_ds.batch(BATCH_SIZE)
dataset
test_dataset
test_count = (int)(0.2*len(xmls))#验证集

train_count = len(xmls)-test_count
test_count,train_count
test_ds = dataset.skip(test_count)
test_ds
train_ds = dataset.take(train_count)
train_ds
#检查一下

for img,label in train_ds.take(1):

    plt.imshow(tf.keras.preprocessing.image.array_to_img(img[0]))

    pet_label,out1,out2,out3,out4 = label

    plt.title('cat' if pet_label[0]==1 else 'dog')

    xmin = out1[0].numpy()*224

    ymin = out2[0].numpy()*224

    xmax= out3[0].numpy()*224

    ymax = out4[0].numpy()*224

    rect = Rectangle((xmin,ymin),(xmax-xmin),(ymax-ymin),fill=False,color='red')

    ax = plt.gca()

    ax.axes.add_patch(rect)
xception = tf.keras.applications.Xception(weights='imagenet',

                                         include_top = False,

                                         input_shape=(224,224,3))
xception.summary()
inputs = tf.keras.layers.Input(shape=(224,224,3))
x = xception(inputs)

x = tf.keras.layers.GlobalAveragePooling2D()(x)
x.get_shape()
x = tf.keras.layers.Dense(2048,activation='relu')(x)

x = tf.keras.layers.Dense(256,activation='relu')(x)
#输出宠物类别

out_pet = tf.keras.layers.Dense(1,activation='sigmoid',name='out_pet')(x)
out_pet.get_shape()
#输出四个值，不需要激活了

out1 = tf.keras.layers.Dense(1,name='out1')(x)

out2 = tf.keras.layers.Dense(1,name='out2')(x)

out3 = tf.keras.layers.Dense(1,name='out3')(x)

out4 = tf.keras.layers.Dense(1,name='out4')(x)



prediction = [out_pet,out1,out2,out3,out4]



model = tf.keras.models.Model(inputs=inputs,outputs=prediction)
model.summary()
tf.keras.utils.plot_model(model,show_shapes=True)
#编译

model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.0001),

             loss = {'out_pet':'binary_crossentropy','out1':'mse'

                     ,'out2':'mse','out3':'mse','out4':'mse'},

             metrics = ['acc', 

                        ['mse'],

                        [ 'mse'],

                        [ 'mse'],

                        [ 'mse']

                     ]#mae平均绝对误差

             )
EPOCH = 15
history = model.fit(dataset,

                    steps_per_epoch=train_count//BATCH_SIZE,

                    epochs=EPOCH,

                    validation_data=test_ds,

                    validation_steps=test_count//BATCH_SIZE

                   )
loss = history.history['loss']

val_loss = history.history['val_loss']



epochs = range(EPOCH)



plt.figure()

plt.plot(epochs,loss,'r',label='Training loss')

plt.plot(epochs,val_loss,'bo',label='Validation loss')

plt.title('Training and Validation Loss')

plt.xlabel('Epoch')

plt.ylabel('Loss Value')

plt.legend()

plt.show()
model.save('detect_v2.h5')
new_model = tf.keras.models.load_model('../input/ximing-object-detect/detect_v2.h5')
#创建一个画布 放三个图片

plt.figure(figsize=(8,48))

for img in test_dataset.take(1):

    pet,out1,out2,out3,out4 = new_model.predict(img)

    #画三个

    for i in range(6):

        plt.subplot(6,1,i+1)

        plt.imshow(tf.keras.preprocessing.image.array_to_img(img[i]))

        plt.title('cat' if pet[0][0] >= 0.5 else 'dog')

        xmin,ymin,xmax,ymax = out1[i]*224,out2[0]*224,out3[0]*224,out4[0]*224

        rect = Rectangle((xmin,ymin),(xmax-xmin),(ymax-ymin),fill=False,color='red')

        ax = plt.gca()

        ax.axes.add_patch(rect)