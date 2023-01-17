#!/usr/bin/env python3

# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



# my DNN Approach (using chainer)

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)





#import cupy as xp

import scipy

import time

import pickle 

import chainer

import chainer.functions as F

import chainer.initializers as I

import chainer.links as L

import chainer.optimizers as O

from chainer import reporter

from chainer import training

from chainer.training import extensions

from chainer import cuda, Function, gradient_check, report, training, utils, Variable

from chainer import datasets, iterators, optimizers, serializers

from chainer import Link, Chain, ChainList

import matplotlib.pyplot as plt

import csv

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
gpu = -1 # use GPU(set 0)
#model define

class standar_model(Chain):

    """Definition of standar Model""" 

    def __init__(self, X_len,Y_len):

        super(standar_model, self).__init__()



        with self.init_scope():

            self.L1 = L.Linear(X_len, X_len *4 )

            self.L2 = L.Linear(X_len *4 , X_len)

            self.L3 = L.Linear(X_len , Y_len) 

            self.bn = L.BatchNormalization(X_len *4)

    def forward(self, x,ys):

        if gpu >= 0:

            x = xp.array(x, dtype=xp.float32)

        else:

            x = np.array(x, dtype=np.float32)

        h1 =self.L1(x)

        h1 =self.bn(h1)

        h2 = self.L2(h1)

        y = self.L3(h2)

        loss = F.softmax_cross_entropy(y, ys) 

        acc = F.accuracy(y, ys)

        report({'accuracy': acc.data}, self)

        report({'loss': loss.data}, self)

        return loss  

    def predict(self, x):

        with chainer.no_backprop_mode(), chainer.using_config('train', False):

            if gpu >=0:

                x = xp.array(x, dtype=xp.float32)

            else:

                x = np.array(x, dtype=np.float32)  

            h1 =self.L1(x)

            h1 =self.bn(h1)

            h2 =self.L2(h1)

            y = self.L3(h2)

            return y
def create_train_data(proc_data):

    train = pd.DataFrame(proc_data, columns=["var_" + str(i) for i in range(200)]) 

    x = train.values.tolist() 

    y = np.round(proc_data['target'])

    x_y= list(zip(x,y))

    x_y2 = []

    for _ in range(5):

        for item in x_y:

            x_y2.append(item)

    x = [item[0] for item in x_y2]

    y = [item[1] for item in x_y2]

    return  x ,y



def create_visualize_data(x,y):

    x_y = []

    x_y_1 = list(zip(x,y))

    x0 =[]

    x1 =[]

    for item in x_y_1[:200000]:

        if item[1] == 0:

                x0.append(item[0]) #Label =0 data

        else:

                x1.append(item[0]) #Label =1 data

    return x0,x1

if __name__ =='__main__':

    print("start training")

    proc_data = pd.read_csv(r"../input/train.csv") 

    x,y = create_train_data(proc_data) 

    print(len(x))

    model =standar_model(200,2)

    model.compute_accuracy = True

    dir1 = 'standar/'

    project_name = 'standar'

    if gpu != -1:

        model.to_gpu(gpu)

    optimizer = optimizers.Adam() 

    optimizer.setup(model)

    # Setup optimizer

    train, test = datasets.split_dataset_random(datasets.TupleDataset(x, y),int(len(x) * 0.90))

    train_iter = iterators.SerialIterator(train, batch_size=2048, shuffle=True)

    test_iter = iterators.SerialIterator(test, batch_size=2048, repeat=False, shuffle=True) 

    updater = training.StandardUpdater(train_iter, optimizer, device=gpu)

    trainer = training.Trainer(updater, (30, 'epoch'), out=dir1 +"result")

    trainer.extend(extensions.Evaluator(test_iter, model, device=gpu))

    trainer.extend(extensions.LogReport(log_name= project_name + 'log.txt'))

    trainer.extend(extensions.PrintReport( ['epoch', 'main/loss', 'validation/main/loss', 'main/accuracy', 'validation/main/accuracy']))

    trainer.extend(extensions.ProgressBar()) 

   



    # trainer.extend(extensions.snapshot(filename=project_name + 'snapshot_epoch-{.updater.epoch}'))

    # trainer.extend(extensions.snapshot_object(model.predictor, filename='model_epoch-{.updater.epoch}'))

    trainer.run() #Start Learning



    model.to_cpu()

    serializers.save_npz(dir1 +project_name + '.model', model) #Save Model

    serializers.save_npz(dir1 +project_name + '.state', optimizer) #Save Optimizer

    result1 = []

    x0,x1 = create_visualize_data(x,y)

    for i,item in enumerate(x0):



        item = [item,item]

        pred1 = model.predict(item).data[0]

        result1.append(pred1) 

    for i,item in enumerate(x1): 

        item = [item,item] 

        pred2 = model.predict(item).data[0]

        result1.append(pred2)

    plt.title('distribution of training data')

    plt.scatter([item[0] for item in result1[:len(x0)]],[item[1] for item in result1[:len(x0)]],c='red',Label="Label = 0",alpha=0.7) #Red color is  Label 0

    plt.scatter([item[0] for item in result1[len(x0):]],[item[1] for item in result1[len(x0):]],c='blue',Label="Label = 1",alpha=0.7) #Blue color is Label 1

    plt.show() #Show plot

    print("predict test Data.")

    test_data = pd.read_csv(r"../input/test.csv") 

    x = []

    y = [] 

    id1 = test_data['ID_code'].values.tolist()

    test_x = pd.DataFrame(test_data, columns=["var_" + str(i) for i in range(200)])

    test_x = test_x.values.tolist()

    



    pred = model.predict(test_x).data



    res = F.softmax(np.array(pred)).data.tolist()

    result = []

    result.append(["ID_code","target"])

    for i,item in enumerate(res): 

        result.append([id1[i],np.round(item[1],3)])

         

    import csv



    with open('sample_submission.csv', 'w') as f:

        writer = csv.writer(f, lineterminator='\n') 

        writer.writerows(result) 

    print("saved csv.")