import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.



import os

#print(os.listdir("../input"))





# SciKitLearn is a machine learning utilities library

import sklearn



# The sklearn dataset module helps generating datasets

import sklearn.datasets

from sklearn.datasets import load_iris



np.random.seed(0)
def load_data(show = False):

    iris = load_iris()

    X = iris.data

    Y = iris.target

    Y = Y.reshape(-1,1)

    if show : 

        print("X contains input and Y contains output ,each row is for one datapoint \ntotal datapoints = ",X.shape[0])

        print("Shape of X = ",X.shape)

        print("Shape of Y = ",Y.shape)

        print("OneHotEncoding started ")

    Y1 = np.zeros((Y.shape[0],3))

    for i in range(Y.shape[0]):

        Y1[i][Y[i][0]] = 1

    Y = Y1

    if show : print("final shape of Y = ",Y.shape)

    return X,Y 

# Loading Data into X and Y

X,Y = load_data(True)
# This block is used as config file containing all global variables for model

# no of layers is all hidden + output layer 



no_of_layers = 2



# no_of_features is input dimension 



no_of_features = X.shape[1]



no_of_data_points = X.shape[0]



# output_nodes is a list of nodes in each hidden and output layer 1-by-1



output_nodes =[5,3]
def weight_initi(show = False):

    if show : print("Weight Initialization started ")

    inp_size = no_of_features

    param_dict = {}

    for index,cur_op in enumerate(output_nodes):

        W_val = np.random.rand(inp_size,cur_op)

        W_key = "W"+str(index)

        if show : print("shape of ",W_key," is " ,W_val.shape)

        B_val = np.random.rand(1,cur_op)

        B_key = "b"+str(index)

        if show : print("shape of ",B_key," is " ,B_val.shape)

        inp_size = cur_op

        param_dict[W_key] = W_val

        param_dict[B_key] = B_val

    if show : print("Weight Initialization Finished ")

    return param_dict
# Testing Weight_initialization function 

my_params = weight_initi(True)

print(my_params)
# softmax non-linearity , to be used in last layer of model 

def softmax(Arr,axis,show = False ):

    arr = np.exp(Arr)

    dir_one = 1

    if axis == 1 : dir_one = arr.shape[0]

    arr_sum = np.array(np.sum(arr,axis = axis)).reshape(dir_one,-1)

    if show : print(arr_sum.shape)

    if show : print(arr_sum)

    arr = arr/arr_sum

    return arr

    
# testing Softmax function 

arr = np.array([[1,2,3],[1,2,3]])

softmax(arr,1)
# Feed-Forward function 

def feed_forward(X,params,show = False):

    A = X

    params["A0"] = A

    if show : print("A0 shape = ",A.shape)

    for i in range (no_of_layers):

        wt_name = "W"+str(i)

        bias_name = "b"+str(i)

        wt = params[wt_name]

        b = params[bias_name]

        if show : print(wt_name," shape = ",wt.shape)

        if show : print(bias_name," shape = ",b.shape)

        Z = np.dot(A,wt)

        Z_name = "Z"+str(i+1)

        if show : print(Z_name," shape = ",Z.shape)

        Z = Z + b

        if show : print(Z_name," shape = ",Z.shape)

        params[Z_name] = Z

        A_name = "A"+str(i+1)

        if i < no_of_layers -1 : 

            A = np.tanh(Z)

        else :

            A = softmax(Z,1)

        if show : print(A_name," shape = ",A.shape)

        params[A_name] = A

    

    return A,params
# Testing Feed forward function 

A,params = feed_forward(X,my_params,True)
params.keys()
# Fuction to update weights and biases 

def update_weights(params,error,learning_rate,show = False):

    #print(params.keys())

    DA = error 

    loop = no_of_layers

    #print("Learning Rate = ",learning_rate)

    for i in range(loop):

        cur = loop - i -1 

        wt_name = "W"+str(cur)

        bias_name = "b"+str(cur)

        z_name = "Z"+str(cur+1)

        a_name = "A"+str(cur)

        a_next_name = "A"+str(cur+1)

        if show :

            print(wt_name," ",bias_name," ",z_name," ",a_name)

            print("DA shape = ",DA.shape)

            print(z_name," shape = ",params[z_name].shape)

            print(wt_name," shape = ",params[wt_name].shape)

            print(bias_name," shape = ",params[bias_name].shape)

            print(a_name," shape = ",params[a_name].shape)

            print(a_next_name," shape = ",params[a_next_name].shape)

        a = params[a_name]

        a_next = params[a_next_name]

        wt = params[wt_name]

        #print(wt[0])

        bias = params[bias_name]

        #DZ = np.multiply(DA,der_sigmoid(params[z_name]))

        if i == 0 : 

            DZ = DA

        else:

            D_tmp = 1 - np.multiply(a_next,a_next)

            if show : print("A^2 shape = ",D_tmp.shape)

            DZ = np.multiply(DA,D_tmp)

        #print("DZ shape = ",DZ.shape)

        #DW = 1/no_of_data_points * np.dot(DZ,a.T)

        DW =  1/no_of_data_points * np.dot(a.T,DZ)

        if show : print("DW shape = ",DW.shape)

        DB = 1/no_of_data_points * np.sum(DZ,axis = 0).reshape(1,-1)

        if show : print("DB shape = ",DB.shape)

        #DA = 1/no_of_data_points * np.dot(wt.T,DZ)

        DA =  np.dot(DZ,wt.T)

        if show : print("DA shape = ",DA.shape)

        wt = wt - learning_rate*DW

        bias = bias - learning_rate*DB

        params[wt_name] = wt

        params[bias_name] = bias

    return params
# Testing Update Function 

updated_params = update_weights(params,A-Y,0.01,True)
# this will calculate accuracy and print that along with correct prediction 

def calculate_acc(A,Y):

    no_data_points = A.shape[0]

    no_of_class = A.shape[1]

    totla_correct = 0

    for j in range(no_of_data_points):

        maxv = A[j][0]

        ind = 0

        for i in range(no_of_class):

            if maxv < A[j][i] :

                maxv = A[j][i]

                ind = i

        if Y[j][ind] == 1:

            totla_correct = totla_correct + 1

    acc =  ( totla_correct / no_of_data_points ) * 100

    print("correct pre = ",totla_correct)

    print("Accuracy = ",acc,"%")
# This will return final trained model 

def train_model(X,Y,iterations,learning_rate = 0.01):

    cur_params = weight_initi()

    for i in range(iterations):

        print("Iteration ",i+1)

        a_out,cur_params = feed_forward(X,cur_params)

        #print(a_out)

        #error = calculate_mse(a_out,Y)

        calculate_acc(a_out,Y)

        #DA = np.multiply(a_out-Y,a_out-Y)

        DA = a_out - Y

        cur_params = update_weights(cur_params,DA,learning_rate)

    return cur_params
print("Using only one layer with 3 nodes i.e no Hidden layer ")

#my_final_params = train_model(X,Y,2000,0.01)

#print(my_final_params)

print("145/150 correct ")
print("Using two layers [5,3] i.e one Hidden layer ")

my_final_params = train_model(X,Y,500,0.1)

print(my_final_params)
# more than 95% Accuracy 