# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
Raw_data = pd.read_csv('../input/Iris.csv')


# coding: utf-8

#TrainSet = Raw_data[Raw_data['TRAIN']==1 ]

TrainSet = Raw_data

SETOSA_set = TrainSet[TrainSet['Species']=='Iris-setosa']

VIRGINIC_set = TrainSet[TrainSet['Species']=='Iris-virginica']

VERSICOL_set = TrainSet[TrainSet['Species']=='Iris-versicolor']





# In[2]:



title = Raw_data.columns



SETOSA_dataArr = pd.Series.as_matrix(SETOSA_set[title[:4]])

VIRGINIC_dataArr = pd.Series.as_matrix(VIRGINIC_set[title[:4]])

VERSICOL_dataArr = pd.Series.as_matrix(VERSICOL_set[title[:4]])



num_SET = SETOSA_dataArr.shape[0]

num_VER = VERSICOL_dataArr.shape[0]

num_VIR = VIRGINIC_dataArr.shape[0]

num_of_train = TrainSet.shape[0]

DataArr_Dict = {'SETOSA':SETOSA_dataArr,'VIRGINIC':VIRGINIC_dataArr, 'VERSICOL':VERSICOL_dataArr}













# In[3]:



Num_of_Prototype = 5

Num_of_Variable = SETOSA_dataArr.shape[1]





# In[4]:



Flower_name = ['SETOSA','VIRGINIC','VERSICOL']

num_SET = SETOSA_dataArr.shape[0]

num_VIR = VIRGINIC_dataArr.shape[0]

num_VER = VERSICOL_dataArr.shape[0]



Init_SET = np.random.randint(0,SETOSA_dataArr.shape[0],Num_of_Prototype)

Init_VIR = np.random.randint(0,VIRGINIC_dataArr.shape[0],Num_of_Prototype)

Init_VER = np.random.randint(0,VERSICOL_dataArr.shape[0],Num_of_Prototype)

Init = {'SETOSA':Init_SET,'VIRGINIC':Init_VIR, 'VERSICOL':Init_VER}





# In[5]:



#Initialize the Prototpy

Prototype_Dict = dict()

for flower in Flower_name:

    tmpArray = np.zeros( (Num_of_Prototype,Num_of_Variable))

    tmpArray = DataArr_Dict[flower][Init[flower]]

    Prototype_Dict[flower]=tmpArray





# In[6]:



def K_Mean_Proto_GD(Data,PrototypeInit):

    learning_rate = 0.01

    DataSize = Data.shape[0]

    #decrease_step = 0.01/DataSize           # for some learning policy 

    Prototype = PrototypeInit.copy()

    for epoch in range(10):

        for i in range(DataSize):

            testData = Data[i]

            norm_result = np.linalg.norm(Prototype-testData,axis=1)

            update_ind = np.argmin(norm_result) # the update index of the prototype

            Prototype[update_ind] = Prototype[update_ind] - learning_rate*norm_result[update_ind]

            # This place could add some learning policy there to decreasing the learning rate

            #learning_rate = learning_rate*np.exp(-0.01*i)

    return Prototype





# In[7]:



K_Mean_Dict = dict()

for i,flw in enumerate(Flower_name):

    K_Mean_Dict[flw] = K_Mean_Proto_GD(DataArr_Dict[flw],Prototype_Dict[flw])





# In[8]:



TmpTest = DataArr_Dict['SETOSA']

test = TmpTest[0]

def K_Mean_Classifier(query,Prototype_Dict):

    index = 0

    minimum = 100000

    for i,flw in enumerate(Flower_name):

        length = np.linalg.norm(K_Mean_Dict[flw]-query,axis=1).min()

        if length < minimum:

            minimum = length

            index = i

    return index#Flower_name[index]





# In[9]:



# Testing of the Array

K_Mean_Classifier(TmpTest[20],Prototype_Dict)



Total = DataArr_Dict[Flower_name[0]]

Response = 0*np.ones((Total.shape[0],1)) # Initialize the Response SETOSA 0

for i in range(1,len(Flower_name)):

    stacktmp = DataArr_Dict[Flower_name[i]]

    Total = np.vstack((Total,stacktmp))

    Response = np.vstack((Response,i*np.ones((stacktmp.shape[0],1))))

#Prototype 0~4 for SETOSA 5~9 for VIRGINIC 10~14 for VERSICOL

TotalPrototype = Prototype_Dict[Flower_name[0]]

PrototypeClass = np.zeros((Prototype_Dict[Flower_name[0]].shape[0],1))

for i in range(1,len(Flower_name)):

    stacktmp = Prototype_Dict[Flower_name[i]]

    TotalPrototype = np.vstack((TotalPrototype,stacktmp))

    PrototypeClass = np.vstack((PrototypeClass,i*np.ones((stacktmp.shape[0],1))))

#Prototype 0~4 for SETOSA 5~9 for VIRGINIC 10~14 for VERSICOL





# In[10]:



# Learning Vector Qunatization (LVQ)

def Learning_Vector_Quantization_GD(Total,Response,TotalPrototype):

    Prototype = TotalPrototype.copy()

    learning_rate = 0.05

    for query,r in zip(Total,Response):

        diff = Prototype - query

        norm_result = np.linalg.norm(diff,axis=1)

        update_ind = np.argmin(norm_result)

        delta = learning_rate*norm_result[update_ind]

        if r == update_ind//3:

            Prototype[update_ind] = Prototype[update_ind] - learning_rate*delta

        else:

            Prototype[update_ind] = Prototype[update_ind] + learning_rate*delta

        # This place could add some learning policy there to decreasing the learning rate

        #learning_rate = learning_rate*np.exp(-0.01*i)

    return Prototype





# In[11]:



LVQ_Parameter = Learning_Vector_Quantization_GD(Total,Response,TotalPrototype)

def LVQ_Classifier(query,Prototype,PrototypeClass):

    diff = Prototype - query

    norm_result = np.linalg.norm(diff,axis=1)

    update_ind = np.argmin(norm_result)

    return PrototypeClass[update_ind,-1]





# In[12]:



#Kernel Density Classification

Prior = {'SETOSA':num_SET/num_of_train,'VIRGINIC':num_VIR/num_of_train,'VERSICOL':num_VER/num_of_train}







def Kernel_Estimator(query,Total,h):

    num_of_train = Total.shape[0]

    dist = np.linalg.norm(query-Total,axis=1)

    mask = dist[:] < h/2

    dim = Total.shape[1]

    Summation =  (1/ np.sqrt(2*np.pi)**dim)*np.exp(-dist[mask]**2/2).sum()

    return Summation/num_of_train/h**dim





# In[13]:



# Usage Gaussian_Kernel_Classifier(query,DataArr_Dict,1)

def Gaussian_Kernel_Classifier(query,DataArr_Dict,h,Prior):

    maximum = -1

    maxindex = str()

    for i,(key,value) in enumerate(DataArr_Dict.items() ):

        est = Kernel_Estimator(query,value,h)*Prior[key]

        if est > maximum:

            maximum = est

            maxindex = key

    return Flower_name.index(maxindex)

          





# In[14]:



# Distance Weighted KNN Classifier



def Distance_Weighted_KNN_Classifier(query,Total,Response,k):

    dim = query.shape[0]

    norm_of_query = np.linalg.norm(query - Total,axis=1)

    mask = np.argsort(norm_of_query[:],axis=0)

    SortedResponse = Response[mask][:k] # All the k candidate response

    Sorted_Norm = norm_of_query[mask][:k]

    h = Sorted_Norm[-1]

    Sorted_Norm = norm_of_query[mask][:k]/h

    mask0 = (SortedResponse == 0.0).reshape(-1)

    mask1 = (SortedResponse == 1.0).reshape(-1)

    mask2 = (SortedResponse == 2.0).reshape(-1)

    masklist = [mask0,mask1,mask2]

    maxind = 0

    maximum = 0

    for i,submask in enumerate(masklist):

        Summation =  (1/ np.sqrt(2*np.pi)**dim)*np.exp(-Sorted_Norm[submask]**2/2).sum()

        if Summation > maximum:

            maximum = Summation

            maxind = i

    

    return maxind

        





# In[15]:



Validation_Set = Raw_data

Val_Total = pd.Series.as_matrix(Validation_Set[title[0]] ).reshape(-1,1)

for i in range(4):

    if i>0:

        tmp = pd.Series.as_matrix(Validation_Set[title[i]]).reshape(-1,1)

        Val_Total = np.hstack((Val_Total,tmp))





# In[16]:







# In[17]:



# Create Validation Response

num_of_Val = len(Response)

Val_Response = Response





# In[18]:



# K Mean Classifier

Answer_Sheet = np.zeros_like(Val_Response)

for i,data in enumerate(Val_Total):

    Answer_Sheet[i] = K_Mean_Classifier(data,Prototype_Dict)

    

ConfusionMatrix = np.zeros((4,4))

for i in range(Val_Response.size) :

    row = int(Answer_Sheet[i])

    col = int(Val_Response[i])

    ConfusionMatrix[row,col] = ConfusionMatrix[row,col] + 1

    

for j,i in enumerate(ConfusionMatrix.sum(axis=1)):

    ConfusionMatrix[j,-1] = i

    

for j,i in enumerate(ConfusionMatrix.sum(axis=0)):

    ConfusionMatrix[-1,j] = i

    

accuracy = np.diag(ConfusionMatrix[:3,:3]).sum()/ num_of_Val

print('K-Mean Classifier Confusion Matrix')

for i in ConfusionMatrix:

    print(i)

print('Accuracy:',accuracy)





# In[19]:



#LVQ Classifier

Answer_Sheet = np.zeros_like(Val_Response)

for i,data in enumerate(Val_Total):

    Answer_Sheet[i] = LVQ_Classifier(data,LVQ_Parameter,PrototypeClass)

    

ConfusionMatrix = np.zeros((4,4))

for i in range(Val_Response.size) :

    row = int(Answer_Sheet[i])

    col = int(Val_Response[i])

    ConfusionMatrix[row,col] = ConfusionMatrix[row,col] + 1

    

for j,i in enumerate(ConfusionMatrix.sum(axis=1)):

    ConfusionMatrix[j,-1] = i

    

for j,i in enumerate(ConfusionMatrix.sum(axis=0)):

    ConfusionMatrix[-1,j] = i

    

accuracy = np.diag(ConfusionMatrix[:3,:3]).sum()/ num_of_Val

print('LVQ Classifier Confusion Matrix')

for i in ConfusionMatrix:

    print(i)

print('Accuracy:',accuracy)





# In[20]:



#Gaussian Kernel Classifier

for h in [0.1,0.5,1.0]:

    Answer_Sheet = np.zeros_like(Val_Response)

    for i,data in enumerate(Val_Total):

        Answer_Sheet[i] = Gaussian_Kernel_Classifier(data,DataArr_Dict,h,Prior)

        

    ConfusionMatrix = np.zeros((4,4))

    for i in range(Val_Response.size) :

        row = int(Answer_Sheet[i])

        col = int(Val_Response[i])

        ConfusionMatrix[row,col] = ConfusionMatrix[row,col] + 1

        

    for j,i in enumerate(ConfusionMatrix.sum(axis=1)):

        ConfusionMatrix[j,-1] = i

        

    for j,i in enumerate(ConfusionMatrix.sum(axis=0)):

        ConfusionMatrix[-1,j] = i

        

    accuracy = np.diag(ConfusionMatrix[:3,:3]).sum()/ num_of_Val

    print('Gaussian Kernel Classifier Confusion Matrix with h =',h)

    for i in ConfusionMatrix:

        print(i)

    print('Accuracy:',accuracy)





# In[21]:



#Gaussian Kernel Classifier

for k in [1,3,5]:

    Answer_Sheet = np.zeros_like(Val_Response)

    for i,data in enumerate(Val_Total):

        Answer_Sheet[i] = Distance_Weighted_KNN_Classifier(data,Total,Response,k)

        

    ConfusionMatrix = np.zeros((4,4))

    for i in range(Val_Response.size) :

        row = int(Answer_Sheet[i])

        col = int(Val_Response[i])

        ConfusionMatrix[row,col] = ConfusionMatrix[row,col] + 1

        #if(row != col):

        #    print(i)

        

    for j,i in enumerate(ConfusionMatrix.sum(axis=1)):

        ConfusionMatrix[j,-1] = i

        

    for j,i in enumerate(ConfusionMatrix.sum(axis=0)):

        ConfusionMatrix[-1,j] = i

        

    accuracy = np.diag(ConfusionMatrix[:3,:3]).sum()/ num_of_Val

    print('Distance_Weighted_KNN_Classifier Confusion Matrix with k =',k)

    for i in ConfusionMatrix:

        print(i)

    print('Accuracy:',accuracy)
