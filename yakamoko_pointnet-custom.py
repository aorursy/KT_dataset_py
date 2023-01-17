!pip install h5py

!pip install keras

#!pip install tensorflow==2.0.0
!pip install pyntcloud

!pip install pythreejs

#pip install pyvista
import tensorflow as tf

import os

import h5py

import pandas as pd

import numpy as np

tf.__version__
#2048 points per point cloud

#load h5 format input

#Input is the path to h5 file and 

#output is the tuple (data,label) np.array



#file=h5py.File('/kaggle/input/pointnetcustom/modelnet40_ply_hdf5_2048/ply_data_train0.h5','r')

#list(file.keys()) like python dictionary



def load_h5(filename):

  file = h5py.File(filename)#h5 file ['data','faceID'.'label','normal'] format

  data = file['data']

  label = file['label']

  return (data,label)



modelnet = '/kaggle/input/pointnetcustom/modelnet40_ply_hdf5_2048/'

train_path = ['ply_data_train0.h5','ply_data_train1.h5','ply_data_train2.h5','ply_data_train3.h5','ply_data_train4.h5']

test_path = ['ply_data_test0.h5','ply_data_test1.h5']

test_path
#Read/Build Training Dataset



data_train0, label_train0 = load_h5(os.path.join(modelnet,train_path[0]))

data_train1, label_train1 = load_h5('/kaggle/input/pointnetcustom/modelnet40_ply_hdf5_2048/ply_data_train1.h5')

data_train2, label_train2 = load_h5(os.path.join(modelnet,train_path[2]))

data_train3 ,label_train3 = load_h5(os.path.join(modelnet,train_path[3]))

data_train4 ,label_train4 = load_h5(os.path.join(modelnet,train_path[4]))



train_data = np.concatenate((data_train0,data_train1), axis=0)

train_data = np.concatenate((train_data,data_train2),axis=0)

train_data = np.concatenate((train_data,data_train3),axis=0)

train_data = np.concatenate((train_data,data_train4),axis=0)#shape (9840,2048,3)



train_labels = np.concatenate((label_train0,label_train1), axis=0)

train_labels = np.concatenate((train_labels,label_train2), axis=0)

train_labels = np.concatenate((train_labels,label_train3),axis=0)

train_labels = np.concatenate((train_labels,label_train4),axis=0)#shape (9840,1)

train_labels = np.reshape(train_labels,[-1])#shape (9840,)



#train_data = train_data[:,:1024,:]#train_data shape :(9840,1024,3) 1024 points per cloud gives approx same accuracy 



train_data.shape

#train_labels.shape



#Read/Build Test Dataset



data_test0, label_test0 = load_h5(os.path.join(modelnet,test_path[0]))#data_test0 = (2048,2048,3)||label_test = (2048,1)

data_test1, label_test1 = load_h5(os.path.join(modelnet, test_path[1]))#data_test1 = (420,2048,3)

test_data = np.concatenate((data_test0,data_test1), axis=0)#test_data = (2468,2048,3)

test_labels = np.concatenate((label_test0, label_test1), axis=0)

test_labels = np.reshape(test_labels,[-1])

#test_data = test_data[:,:1024,:]

test_data.shape

#split  = int(round(len(test_data)*0.5))#splitting test_data into val_data + test_data 





#test_labels.shape

#test_labels[0]
#Files for all shape labels

shape_names_file = os.path.join(modelnet, 'shape_names.txt')

shape_names = [line.rstrip('\n,') for line in open((shape_names_file),encoding="utf-8")]



shape_names
#Random training obj visualised in point cloud format#



from pyntcloud import PyntCloud

import random



random_sample_idx = random.randrange(test_data.shape[0])

sample = train_data[random_sample_idx]

sample

df = pd.DataFrame(data=sample, columns = ['x', 'y', 'z'])



cloud = PyntCloud(df)

cloud

cloud.plot( initial_point_size=0.01,backend='pythreejs')

#cloud.plot(backend='threejs')
#Random visualised sample label

shape_names[train_labels[random_sample_idx]]
#Exploring Training Dataset we can see there is a descent level of class imbalance



unique,counts = np.unique(train_labels,return_counts=True)



a=dict(zip(unique, counts))

#unique

counts

#Exploring Test Dataset we can see there is a descent level of class imbalance



uniquet,countst = np.unique(test_labels,return_counts=True)

dict(zip(uniquet, countst))

countst

#Data imbalance pre-processing step.We define different weights for the Training classes depending on how many obj's they have#



weightsforlabels = []



for i in range(40):

    weightsforlabels.append(66)



for i in range(40):

    

    if weightsforlabels[i]== 66:

        weightsforlabels[i] = np.max(counts)/counts[unique[i]]

    else:

        pass

    

weightsforlabels

b = []

a = np.array(weightsforlabels)

for i in range (40):

  

        b.append(a)

b = np.array(b)#SAMPLE WEIGHTS FOR 40 batch





b

weightsforlabels
#Due to data imbalance we set different weights for the Test categories based on object's population in each one# 



weightsforlabelstest = []

for i in range(40):

    weightsforlabelstest.append(66)



for i in range(40):

    

    if weightsforlabelstest[i]== 66:

        weightsforlabelstest[i] = np.max(countst)/countst[uniquet[i]]

    else:

        pass

    

weightsforlabels

b = []

a = np.array(weightsforlabelstest)

for i in range (40):

  

        b.append(a)

c = np.array(b)#SAMPLE WEIGHTS FOR 40 batch





c.shape

weightsforlabelstest

classweights =  dict.fromkeys(range(0, 40))

classweights = dict(zip(range(0, 40),weightsforlabels))

classweights
#T-Net#

from tensorflow.python.keras.models import Model

from tensorflow.python.keras.layers import Input, Conv2D, Dense, MaxPool2D, Dropout ,BatchNormalization, Layer, Reshape, Flatten



n_points = train_data.shape[1]

batch_size = train_data.shape[0]#batch_size = 9840 samples from training data

pt_cloud = Input(shape=(batch_size,n_points,3))



###Transform Nets###

#Transformation Nets(Joint alignment nets)map inputs into canonicalized space





""" Input (x,y,z) Transform Net, input is BxNx3 gray image

        Return: Transformation matrix of size 3xK """#K=3 for input transform and K=64 for feature transform



#Write extending the tf.keras.Layers class implementing: * __init__ , where you can do all input-independent initialization 

#* build, where you know the shapes of the input tensors and can do the rest of the initialization * call, where you do the forward computation



class TNet (Layer):

    def __init__(self,**kwargs):

    

        

        super(TNet,self).__init__(**kwargs)

       

        self.conv1 = Conv2D(64,(1,1), activation='relu',padding='same')#Shared MLP(64)

        self.bn1 = BatchNormalization(momentum=0.99,axis=-1)



        self.conv2 = Conv2D(128, (1,1), activation='relu')#Shared MLP(128)

        self.bn2 = BatchNormalization(momentum=0.99,axis=-1)

        

        #self.norm2 = BatchNormalization(momentum=0.99)

        self.conv3 = Conv2D(1024, (1,1), activation='relu')#(1024)

        self.bn3 = BatchNormalization(momentum=0.99,axis=-1)

        

        #self.norm3 = BatchNormalization(momentum=0.99)

        self.fc1 = Dense(512, activation='relu')

        self.bn4 = BatchNormalization(momentum=0.99,axis=-1)

        

        self.fc2 = Dense(256, activation='relu')#last layer no bn

       

        

    def build(self, input_cloud):#Creates variables for the layer

        

        self.K = input_cloud[-1]#We want this to  be the number of dimensions,K=3 for input transformation and 64 for feature extraction

        

        self.w = self.add_weight(shape=(256,self.K**2), dtype=tf.float32, initializer=tf.zeros_initializer, trainable=True, name='w')#weights of 3x3 Tnet are Bx256x9(K=3)

        

        self.b = self.add_weight(shape=(self.K,self.K),dtype=tf.float32, initializer=tf.zeros_initializer, trainable=True, name='b')

        

        self.b = self.b + np.eye(self.K)#.flatten()#bias matrix initialized as identity matrix shape(1,9)

       

       

        

    def call(self,input_tensor):

        x = input_tensor

        #input_cloud = x #BxNxK(for input trans is BxNx3)

        #need to expand dims of array for convolution operation

        x = tf.expand_dims(x,axis=2)#BxNx1xK

        #MLP64

        x = self.conv1(x)

        x = self.bn1(x)

        #MLP128

        x = self.conv2(x)

        x = self.bn2(x)

        #MLP1024

        x = self.conv3(x)

        x = self.bn3(x)

        #Global features

        x = tf.squeeze(x, axis=2)        

        x = tf.reduce_max(x,axis=1)#input_cloud[1]-->1024 points of the point cloud-->There we have shape of [None,1,1,1024]

        

        #FC

        x = self.fc1(x)#Bx512

        x = self.bn4(x)

        x = self.fc2(x)#all layers with relu+bn except last one||Bx256

        #Now expand dims to mul with weights

        x = tf.expand_dims(x,1)#Bx1X256

        x = tf.matmul(x, self.w)#(Bx1x256)*(Bx256x9)--->(Bx1x9)

        x = tf.squeeze(x, axis=1)

        x = tf.reshape(x,[-1,self.K,self.K])

        x = x + self.b

        

        return x#returns BxKxK matrix for input-trans transformation

                #i want the shape of input trans Bx3x3||shape of output trans Bx64x64

        

      

        

        

      
#Cls model#



    input_cloud = Input(shape=[n_points,3])

            #Iput Transform

        

    transform = TNet(name='input_transform')(input_cloud)#Tnet input shape BxNxK

    cloud_trans = tf.matmul(input_cloud,transform)#BxNx3



                #BxNx64

    cloud_img = tf.expand_dims(cloud_trans,axis=2)#BxNx1x3 for weight sharing

    mlp64 = Conv2D(64, (1,1), activation='relu', padding='same')(cloud_img)

    x = BatchNormalization(momentum=0.99)(mlp64)

    cloud64 = Conv2D(64, (1,1), activation='relu')(x)

    x64 = BatchNormalization(momentum=0.99)(cloud64)#BxNx1x64

    x64 = tf.squeeze(x64,axis=2)#BxNx64

           #Feature Transform

        

    feat_trans = TNet(name='feature_transform')(x64)#K=64

    embed_trans64 = tf.matmul(x64,feat_trans)#BxNx64

    trans_cloud64 = tf.expand_dims(embed_trans64,axis=2)#expand dims BxNx1x64 for conv2d

    net64 = Conv2D(64,(1,1),activation='relu')(trans_cloud64)#Shared MLP64 after feature trans

    x = BatchNormalization(momentum=0.99)(net64)

    net128 = Conv2D(128,(1,1), activation='relu')(x)

    x = BatchNormalization(momentum=0.99)(net128)

    net1024 = Conv2D(1024,(1,1),activation='relu')(x)

    x = BatchNormalization(momentum=0.99)(net1024)#Bx1024x1x1024

    cpts = MaxPool2D((n_points,1))(x)#Evala auta

    glfeats = tf.reshape(cpts,[-1,1024])#Bx1024--->These are the global features



    fc512 = Dense(512,activation='relu')(glfeats)



    x = BatchNormalization(momentum=0.99)(fc512)

    drop = Dropout(rate=0.3)(x)#dropout rate 0.3 meaning keeping prob=0.7

    fc256 = Dense(256,activation='relu')(drop)

    x = BatchNormalization(momentum=0.99)(fc256)

    drop = Dropout(rate=0.3)(x)

    out = Dense(40)(drop)#Output scores



    #return Model(inputs=input_cloud, outputs=out, name='Wombo')



    

# Model creatino and summary for evaluation#

model = Model(inputs=input_cloud,outputs=out)

model.summary()
# L2 reg loss added to model's losses after feature transform  T-net(64)#



model.add_loss(1e-3*tf.nn.l2_loss(tf.constant(np.eye(64), dtype=tf.float32)-tf.matmul(feat_trans, tf.transpose(feat_trans, perm=[0, 2, 1]))))
# Prepare the metrics.

from tensorflow import keras



#define learning rate with schedule#



def learning_rate():

    learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(

        0.001,  # Initial learning rate

        200000,          # Decay step.

        0.7,          # Decay rate.

        staircase=True)

    return learning_rate   

lr=learning_rate()





# Instantiate metric objects #

train_acc = tf.keras.metrics.CategoricalAccuracy()

train_prec = tf.keras.metrics.Precision()

train_recall = tf.keras.metrics.Recall()

val_acc = tf.keras.metrics.CategoricalAccuracy()

val_prec = tf.keras.metrics.Precision()

val_recall = tf.keras.metrics.Recall()



#Define loss for our model#

def c_loss(label,pred):

   

    loss = tf.nn.softmax_cross_entropy_with_logits(label,pred)

    cls_loss = tf.math.reduce_mean(loss)

    return cls_loss



#Set the batch size for training#

batch_size = 40



#Define Adam optimizer with custom learning rate decay#

optimizer = tf.keras.optimizers.Adam(learning_rate=lr)



#Data augmentation for training empowering results#

#def augmentation(input_cl,label):#one hot encode label data

#    hot_label = tf.one_hot(label, depth=40, dtype=float32)

#    pt_cl = random_rotate(input_cl)

#    pt_cl = jitter(input_cl)

#    return pt_cl, hot_label



#Data augmentation for training empowering results#

#def augment(points, label):

    # jitter points

 #   points += tf.random.uniform(points.shape, -0.005, 0.005, dtype=tf.float32)

    # shuffle points

 #   points = tf.random.shuffle(points)

 #   return points, label





    



#Create train dataset with shuffle augmented data of batch size = 40# 

train_dataset = tf.data.Dataset.from_tensor_slices((train_data,train_labels))#Combines consecutive elements of this dataset into batches.

#train_dataset = train_dataset.shuffle(buffer_size=len(train_data)).map(augment).batch(batch_size) #This dataset fills a buffer with buffer_size elements, then randomly samples elements from this buffer, 

                                                                                                  #replacing the selected elements with new elements

train_dataset = train_dataset.shuffle(buffer_size=len(train_data)).batch(batch_size)

                                       

train_dataset = train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)#Whether to automatically tune performance knobs. If None, defaults to True. 
#Test data initialization#

batch_size = 40

test_dataset = tf.data.Dataset.from_tensor_slices((test_data,test_labels))

test_dataset = test_dataset.shuffle(buffer_size=len(test_data)).batch(batch_size)



#Instantiate metric obj for test data

test_acc = tf.keras.metrics.CategoricalAccuracy()

test_prec = tf.keras.metrics.Precision()


step = 0

for epoch in range(80):

    print('\nEpoch', epoch)



    # Reset metrics

    train_acc.reset_states()

    train_prec.reset_states()

    

    test_acc.reset_states()

    

    for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):

        #print(x_batch_train.shape)

         # Forward pass with gradient tape and loss calc

        onehotlab = tf.one_hot(y_batch_train,depth=40,dtype=tf.float32)

        with tf.GradientTape() as tape:

            logits = model(x_batch_train, training=True)

           

            loss = c_loss(onehotlab, logits) + sum(model.losses)#AYTO SWSTO

           

        #print(logits.shape)

        train_probs = tf.nn.softmax(logits)

        

        max_idxs = tf.math.argmax(train_probs, axis=1)#0 h 1 thelw na pairnei to max kathe minibatch

        train_acc.update_state(onehotlab,train_probs)#SAMPLE WEIGTHS

        #print(max_idxs)

        train_one_hot = tf.one_hot(max_idxs, depth=40, dtype=tf.float32)

        gradients = tape.gradient(loss, model.trainable_weights)

        optimizer.apply_gradients(zip(gradients, model.trainable_weights))

          

        train_prec.update_state(onehotlab,train_one_hot, sample_weight=b)

        

        if step % 100 == 0:

            print('Training loss (for one batch) at step %s: %s' % (step, float(loss)))

            print('Training accuracy(for one batch at step %s : %s'% (step, float( train_acc.result())))

            print('Training precision(for one batch) at step %s : %s'%(step,float(train_prec.result())))

            print('Seen so far: %s samples' % ((step + 1) * 40))

   

        

for step, (x_batch_test, y_batch_test) in enumerate(test_dataset):

         # Forward pass with gradient tape and loss calc

    onehotlab = tf.one_hot(y_batch_test,depth=40,dtype=tf.float32)

    with tf.GradientTape() as tape:

        logits = model(x_batch_test, training=False)

        

            

    test_probs = tf.nn.softmax(logits)

    max_idxs = tf.math.argmax(test_probs, axis=1)

    test_one_hot = tf.one_hot(max_idxs, depth=40, dtype=tf.float32)

    test_acc.update_state(onehotlab, test_one_hot)

   # train_prec.update_state(onehotlab, test_one_hot, sample_weight = c)

        

  

            #print('Training loss (for one batch) at step %s: %s' % (step, float(loss)))

print('Test accuracy(for one batch at step %s : %s'% (step, float( test_acc.result())))

            

    #        print('Τest precision(for one batch) at step %s : %s'%(step,float(train_prec.result())))

print('Seen so far: %s samples' % ((step + 1) * 40))

        

for step, (x_batch_test, y_batch_test) in enumerate(test_dataset):

         # Forward pass with gradient tape and loss calc

    onehotlab = tf.one_hot(y_batch_test,depth=40,dtype=tf.float32)

    with tf.GradientTape() as tape:

        logits = model(x_batch_test, training=False)

        

            

    test_probs = tf.nn.softmax(logits)

    max_idxs = tf.math.argmax(test_probs, axis=1)

    test_one_hot = tf.one_hot(max_idxs, depth=40, dtype=tf.float32)

    #print(onehotlab)

    test_acc.update_state(onehotlab, test_one_hot)

    #train_prec.update_state(onehotlab, test_one_hot, sample_weight = c)

        

  

            #print('Training loss (for one batch) at step %s: %s' % (step, float(loss)))

print('Test accuracy(for one batch at step %s : %s'% (step, float( test_acc.result())))

            

    #        print('Τest precision(for one batch) at step %s : %s'%(step,float(train_prec.result())))

print('Seen so far: %s samples' % ((step + 1) * 40))

        
#Test dataset evaluation for our model and final accuracy result#



for step, (x_batch_test, y_batch_test) in enumerate(test_dataset):

         # Forward pass with gradient tape and loss calc

    onehotlab = tf.one_hot(y_batch_test,depth=40,dtype=tf.float32)

    with tf.GradientTape() as tape:

        logits = model(x_batch_test, training=False)

        

           

    

    test_probs = tf.math.sigmoid(logits)

    

    max_idxs = tf.math.argmax(test_probs, axis=1)

    test_one_hot = tf.one_hot(max_idxs, depth=40, dtype=tf.float32)

    #print(test_one_hot.shape)

    test_acc.update_state(onehotlab, test_probs)

    #train_prec.update_state(onehotlab, test_one_hot, sample_weight = c)

        



print('Test accuracy(for one batch at step %s : %s'% (step, float( test_acc.result())))

            

#print('Τest precision(for one batch) at step %s : %s'%(step,float(train_prec.result())))

print('Seen so far: %s samples' % ((step + 1) * 40))
#Train fast the model for matplotlib vizualisations#



model.compile(loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True), optimizer=keras.optimizers.Adam(learning_rate=0.001),

              metrics=["sparse_categorical_accuracy"],)

model.fit(train_dataset, epochs=20, validation_data=test_dataset)
from matplotlib import pyplot as plt

#Plot predicted values and true labels for 8 random objects from test dataset#



data = test_dataset.take(1)



points, labels = list(data)[0]

points = points[:8, ...]

labels = labels[:8, ...]



# run test data through model

preds = model.predict(points)

preds = tf.math.argmax(preds, -1)



points = points.numpy()



# plot points with predicted class and label

fig = plt.figure(figsize=(15, 10))

for i in range(8):

    ax = fig.add_subplot(2, 4, i + 1, projection="3d")

    ax.scatter(points[i, :, 0], points[i, :, 1], points[i, :, 2])

    ax.set_title(

      

        "pred: {:}, label: {:}".format(

             shape_names[preds[i]], shape_names[labels[i]]

    

        )

    )

    ax.set_axis_off()

plt.show()
