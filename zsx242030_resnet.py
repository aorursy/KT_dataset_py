# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import tensorflow as tf
import os
import  numpy as np
import pickle
CIFAR_DIR="../input"
print(os.listdir(CIFAR_DIR))
def load_data(filename):
    """read data from data file."""
    with open(filename,"rb") as f:
        data=pickle.load(f,encoding='latin1')
        return data["data"],data["labels"]
    
class CifarData:
    def __init__(self,filenames,need_shuffle):
        all_data=[]
        all_labels=[]
        for filename in filenames:
            data,labels=load_data(filename)
            all_data.append(data)
            all_labels.append(labels)
        self._data=np.vstack(all_data)
        self._data=self._data/127.5-1
        self._labels=np.hstack(all_labels)
        print(self._data.shape,self._labels.shape)
        self._num_examples=self._data.shape[0]
        self._need_shuffle=need_shuffle
        self._indicator=0
        if self._need_shuffle:
            self._shuffle_data()
            
    def _shuffle_data(self):
        #得到一个混乱的排列
        p=np.random.permutation(self._num_examples)
        self._data=self._data[p]
        self._labels=self._labels[p]
        
    def next_batch(self,batch_size):
        end_indicator=self._indicator + batch_size
        if end_indicator > self._num_examples:
            if self._need_shuffle:
                #重新使数据乱排
                self._shuffle_data()
                #设置参数从头开始取出数据
                self._indicator=0
                end_indicator=batch_size
            else:
                raise Exception("have no more examples")
        if end_indicator > self._num_examples:
            raise Exception("batch size is larger than all examples")
        batch_data=self._data[self._indicator:end_indicator]
        batch_labels=self._labels[self._indicator:end_indicator]
        self._indicator=end_indicator
        return batch_data,batch_labels
train_filenames=[os.path.join(CIFAR_DIR,"data_batch_%d"%(i)) for i in range(1,6)]
test_filenames=[os.path.join(CIFAR_DIR,"test_batch")]
print(train_filenames)
print(test_filenames)

train_data=CifarData(train_filenames,True)    
test_data=CifarData(test_filenames,True)  
print(train_data)      
print(test_data)
#通道扩大两倍，图像缩小两倍
#[None,32,32,3]
#[None,32,32,32]
#16
def residual_block(x,output_channel):
    input_channel=x.get_shape().as_list()[-1]
    #输出通道是输入通道的2倍
    #图像的宽和高要缩小两倍
    if input_channel*2==output_channel:
        #是否对图像缩小2被
        increase_dim=True
        strides=(2,2)
    #如果输入通道和输出通道一样
    #图像的宽和高不变
    elif input_channel==output_channel:
        increase_dim=False
        strides=(1,1)
    else:
        raise Exception("input channel cannot match output channel")
    conv1=tf.layers.conv2d(x,output_channel,(3,3),strides=strides,padding="SAME",
                           activation=tf.nn.relu,name="conv1")
    #第二个卷积保持图像的大小不变，通道保持和上面一样
    conv2=tf.layers.conv2d(conv1,output_channel,(3,3),strides=(1,1),padding="SAME",
                          activation=tf.nn.relu,name="conv2")
    if increase_dim:
        pooled_x=tf.layers.average_pooling2d(x,(2,2),(2,2),padding="valid")
        padded_x=tf.pad(pooled_x,[[0,0],[0,0],[0,0],[input_channel//2,input_channel//2]])
    else:
        padded_x=x;
    output_x=conv2+padded_x
    print(output_x.shape)
    return output_x
#res_net(x_image,[2,3,2],32,10)
#x_image=[None,32,32,3]
#num_residual_blocks=[2,3,2]
#num_filter_base=32
#class_num=10
def res_net(x,num_residual_blocks,num_filter_base,class_num):
    #num_subsampling=3
    num_subsampling=len(num_residual_blocks)
    layers=[]
    #x:[None,width,height,channel]->[width,height,channel]
    #[32,32,3]
    input_size=x.get_shape().as_list()[1:]
    with tf.variable_scope("conv0"):
        #[None,32,32,32]
        conv0=tf.layers.conv2d(x,num_filter_base,(3,3),strides=(1,1),padding="SAME",activation=tf.nn.relu,name="conv0")
        #layers=[[None,32,32,32]]
        layers.append(conv0)
    #num_subsampling=4,sample_id=[0,1,2,3]
    for sample_id in range(num_subsampling):
        #i=0,num_residual_blocks[0]=2
        #i=1,num_residual_blocks[1]=3
        #i=2,num_residual_blocks[2]=2
        for i in range(num_residual_blocks[sample_id]):
            #i=0,1
            #i=0,1,2
            #i=0,1
            with tf.variable_scope("conv%d_%d"%(sample_id,i)):
                #residual_block([None,32,32,32],32*1)
                #residual_block()
                #residual_block()
                conv=residual_block(layers[-1],num_filter_base*(2**sample_id))
                layers.append(conv)
    multiplier=2**(num_subsampling-1)
    assert layers[-1].get_shape().as_list()[1:]==[input_size[0]/multiplier,
                                                  input_size[1]/multiplier,num_filter_base*multiplier]
    with tf.variable_scope("fc") :
        #layers[-1].shape:[None,width,height,channel]
        global_pool=tf.reduce_mean(layers[-1],[1,2])
        logits=tf.layers.dense(global_pool,class_num)
        layers.append(logits)
    return layers[-1]
x=tf.placeholder(tf.float32,[None,3072])
y=tf.placeholder(tf.int64,[None])
x_image=tf.reshape(x,[-1,3,32,32])
x_image=tf.transpose(x_image,perm=[0,2,3,1])
y_=res_net(x_image,[2,3,2],32,10)
loss=tf.losses.sparse_softmax_cross_entropy(labels=y,logits=y_)
predict=tf.argmax(y_,1)
correct_prediction=tf.equal(predict,y)
accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
with tf.name_scope("train_op") :
    train_op=tf.train.AdamOptimizer(1e-3).minimize(loss)
init=tf.global_variables_initializer()
batch_size=20
train_step=10000
test_step=100
with tf.Session() as sess:
    sess.run(init)
    for i in range(train_step):
        batch_data,batch_label=train_data.next_batch(batch_size)
        loss_val,acc_val,_=sess.run([loss,accuracy,train_op],feed_dict={x:batch_data,y:batch_label})
        if (i+1) % 500 ==0:
            print("[Train] Step: %d ,loss:%4.5f ,acc: %4.5f"%((i+1),loss_val,acc_val))
        if (i+1) % 5000==0:
            all_test_acc_val=[]
            for j in range(test_step):
                test_batch_data,test_batch_label=test_data.next_batch(batch_size)
                test_acc_val=sess.run([accuracy],feed_dict={x:test_batch_data,y:test_batch_label})
                all_test_acc_val.append(test_acc_val)
            test_acc=np.mean(all_test_acc_val)
            print("[Test] Step:%d ,acc: %4.5f"%((i+1),test_acc))
