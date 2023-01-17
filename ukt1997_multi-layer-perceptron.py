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

np.random.seed(0)
def create_input_output(path_to_csv_file,show = False):

    df = pd.read_csv(path_to_csv_file)

    inp = np.array(df)

    # inputs are stacked column wise i.e in each col a new data point is stored and no of rows = no of features 

    X = inp[:,1:]

    Y = inp[:,:1]

    Y1 = np.zeros((Y.shape[0],10))

    for i in range(Y.shape[0]):

        Y1[i][Y[i][0]] = 1

    Y = Y1

    if show :

        print("Input.Shape  = ",X.shape)

        print("Output.shape = ",Y.shape)

    return X,Y
import random
X,Y = create_input_output("../input/train.csv",True)

X = X / (255) 
def train_test_split(X,Y,show = False):

    total_data_points = X.shape[0]

    train_size = int(total_data_points *0.8)

    index_list = [] 

    for i in range(total_data_points):

        index_list.append(i)

    train_index = random.sample(index_list,train_size)

    X_train = X[train_index]

    Y_train = Y[train_index]

    test_index = list(set(index_list) - set(train_index))

    X_test = X[test_index]

    Y_test = Y[test_index]

    if show :

        print("X_train shape = ",X_train.shape)

        print("Y_train shape = ",Y_train.shape)

        print("X_test shape = ",X_test.shape)

        print("Y_test shape = ",Y_test.shape)

    if X_train.shape[0] + X_test.shape[0] != total_data_points:

        print("Size Assertion Error")

    return X_train,Y_train,X_test,Y_test

    

    
X_train,Y_train,X_test,Y_test = train_test_split(X,Y,True)

# This block is used as config file containing all global variables for model

# no of layers is all hidden + output layer 



no_of_layers = 3



# no_of_features is input dimension 



no_of_features = X_train.shape[1]



#no_of_data_points = X_train.shape[0]



# output_nodes is a list of nodes in each hidden and output layer 1-by-1



output_nodes =[32,16,10]
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
my_params = weight_initi(True)
my_params.keys()
def sigmoid(arr):

    arr = -1*arr

    arr = np.exp(arr)

    arr = arr + 1

    arr = np.power(arr,-1)

    return arr
def relu(arr):

    arr1 = np.multiply(arr,arr)

    arr2 = np.sqrt(arr1)

    arr3 = arr + arr2

    arr = 1/2*arr3

    return arr

    
def d_relu(arr):

    for i in range(arr.shape[0]):

        for j in range(arr.shape[1]):

            if arr[i][j] < 0:

                arr[i][j] = 0

            else:

                arr[i][j] = 1

    return arr
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
def der_sigmoid(arr):

    return np.multiply(sigmoid(arr),1-sigmoid(arr))
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

        Z = 1/A.shape[1]*np.dot(A,wt)

        Z_name = "Z"+str(i+1)

        if show : print(Z_name," shape = ",Z.shape)

        Z = Z + b

        if show : print(Z_name," shape = ",Z.shape)

        params[Z_name] = Z

        #print(np.sum(Z,axis = 1))

        if show : print(Z)

        A_name = "A"+str(i+1)

        if i < no_of_layers -1 : 

            A = np.tanh(Z) #A = relu(Z)

        else :

            A = softmax(Z,1)

        if show : print(A_name," shape = ",A.shape)

        #print(np.sum(A,axis = 1))

        if show : print(A)

        params[A_name] = A

    

    return A,params
A_out,new_p = feed_forward(X_train,my_params,True)
for i in range (no_of_data_points):

    print(A_out[i])

    print(Y_train[i])
print(A_out[0])

print(A_out[1086])

print(A_out[1240])

print(A_out[1860])

print(A_out[1390])

print(A_out[18976])

print(A_out[9000])

print(A_out[18006])

print(A_out[12000])

print(A_out[186])

print(A_out[16000])

print(A_out[28600])
print(my_params["b1"])
new_p.keys()
# Fuction to update weights and biases 

def update_weights(params,error,batch_size,learning_rate,show = False):

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

            D_tmp = 1 - np.multiply(a_next,a_next) #d_relu(a_next)

            if show : print("A^2 shape = ",D_tmp.shape)

            DZ = np.multiply(DA,D_tmp)

        #print("DZ shape = ",DZ.shape)

        #DW = 1/no_of_data_points * np.dot(DZ,a.T)

        DW =  1/batch_size * np.dot(a.T,DZ)

        if show : print("DW shape = ",DW.shape)

        DB = 1/batch_size * np.sum(DZ,axis = 0).reshape(1,-1)

        if show : print("DB shape = ",DB.shape)

        #DA = 1/no_of_data_points * np.dot(wt.T,DZ)

        DA =  np.dot(DZ,wt.T)

        if show : print("DA shape = ",DA.shape)

        wt = wt - learning_rate*DW

        bias = bias - learning_rate*DB

        params[wt_name] = wt

        params[bias_name] = bias

    return params
updated_params = update_weights(new_p,A_out - Y_train,A_out.shape[0],0.01,True)
# this will calculate accuracy and print that along with correct prediction 

def calculate_acc(A,Y):

    #print(A.shape)

    #print(Y.shape)

    no_data_points = A.shape[0]

    #print(no_data_points)

    no_of_class = A.shape[1]

    totla_correct = 0

    for j in range(no_data_points):

        maxv = A[j][0]

        ind = 0

        for i in range(no_of_class):

            if maxv < A[j][i] :

                maxv = A[j][i]

                ind = i

        if Y[j][ind] == 1:

            totla_correct = totla_correct + 1

    acc =  ( totla_correct / no_data_points ) * 100

    print("correct pre = ",totla_correct)

    print("Accuracy = ",acc,"%")
# This will return final trained model 

def train_model(X,Y,iterations,batch_size=1000,learning_rate = 0.01):

    cur_params = weight_initi()

    no_of_data_points = X.shape[0]

    index_list = [] 

    for i in range(no_of_data_points):

        index_list.append(i)

    for i in range(iterations):

        for j in range(int(no_of_data_points / batch_size)):

            print("Iteration ",i+1," Batch ",j+1)

            random_index = random.sample(index_list,batch_size)

            X_sample = X[random_index]

            Y_sample = Y[random_index]

            a_out,cur_params = feed_forward(X_sample,cur_params)

            #print(a_out)

            #error = calculate_mse(a_out,Y)

            calculate_acc(a_out,Y_sample)

            #DA = np.multiply(a_out-Y,a_out-Y)

            DA = a_out - Y_sample

            cur_params = update_weights(cur_params,DA,batch_size,learning_rate)

    return cur_params
print("Using 3 layers [32,16,10] i.e 2 Hidden layer ")

#my_final_params = train_model(X_train,Y_train,150,1000,0.1)

#print(my_final_params)
def predict(X_test,trained_params,show = False):

    prediction,_ = feed_forward(X_test,trained_params)

    return prediction
#Y_test_output = predict(X_test,my_final_params)
#calculate_acc(Y_test_output,Y_test)