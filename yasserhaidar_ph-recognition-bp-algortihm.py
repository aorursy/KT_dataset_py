import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import numpy as np 
import pandas as pd #CSV file I/O (e.g. pd.read_csv)

#Read the csv file and create a data frame
train_df = pd.read_csv('/kaggle/input/ph-recognition/ph-data.csv',dtype=int)
train_df.head()
import csv
with open('/kaggle/input/ph-recognition/ph-data.csv', newline='') as csvfile:
     d = list(csv.reader(csvfile))
data = np.array(d)

rows=len(d) 
cols=(len(d[0]))
filtered_data=np.zeros((rows,cols-1))
for i in range(rows):
    if(i==0):
        continue
    else:
        for j in range(cols-1):
            filtered_data[i][j]=data[i][j]
filtered_data=filtered_data[1:154] # this is our training dataset but we limit it for the first 153 element for calculation purposes only
print("An Example for filtered_data will be index 100:",filtered_data[100])
numOutput=4
numHidden=45
numInputs=24
numEpochs=1000
numTraining=153
learnrate=0.05
def decimal2Binary(n):  
    b=bin(int(n)).replace("0b", "")
    #print(len(b))
    if(len(b)==1):
        b='0000000'+b
    elif(len(b)==2):
        b='000000'+b    
    elif(len(b)==3):
        b='00000'+b 
    elif(len(b)==4):
        b='0000'+b 
    elif(len(b)==5):
        b='000'+b
    elif(len(b)==6):
        b='00'+b 
    elif(len(b)==7):
        b='0'+b 
    elif(len(b)==8):
        b=b 
    #print(b)
    return b
def dec2Bipolar(index):   
    #print(filtered_data[index])
    intArray=np.zeros(24)#Binary Form
    output=np.zeros(24)#Bipolar Form
    stringForm=""
    if(index>652):
        return "Error in index Bounds"
    else:
        stringForm=decimal2Binary(filtered_data[index][0])+decimal2Binary(filtered_data[index][1])+decimal2Binary(filtered_data[index][2])
        
        array=list(stringForm)
        for p in range(0, len(array)): 
                intArray[p] = int(array[p])
        k = 0
        for h,v in enumerate (intArray):

            if int(intArray[h]) == 0 :
                output [k] = -1
            elif int(intArray[h]) == 1 :
                output [k] = 1
            k = k+1
        return output
def getBipolar():
    final=np.zeros((numTraining,24))
    for i in range(0,numTraining):
        final[i]=dec2Bipolar(i)
    return final
training_input=getBipolar()
print("Our Training Set becomes: \n",training_set)
inputB = np.insert(training_input, 0,np.ones(numTraining),1)
print("Our Training Set with bias becomes: \n",inputB)
label=np.zeros((rows,4))
desired = np.zeros((654,4))
for i in range(rows):
    if(i==0):#First row contains string values(Blue,Green,....)
        continue
    else:    
        
        label[i][0]=data[i][3]
        
        if(label[i][0]==0.0):
            desired[i]=[-1,-1,-1,-1]
        elif(label[i][0]==1.0):
            desired[i]=[-1,-1,-1,+1]
        elif(label[i][0]==2.0):
            desired[i]=[-1,-1,+1,-1]
        elif(label[i][0]==3.0):
            desired[i]=[-1,-1,+1,+1]
        elif(label[i][0]==4.0):
            desired[i]=[-1,+1,-1,-1]
        elif(label[i][0]==5.0):
            desired[i]=[-1,+1,-1,+1]
        elif(label[i][0]==6.0):
            desired[i]=[-1,+1,+1,-1]
        elif(label[i][0]==7.0):
            desired[i]=[-1,+1,+1,+1]
        elif(label[i][0]==8.0):
            desired[i]=[+1,-1,-1,-1]
        elif(label[i][0]==9.0):
            desired[i]=[+1,-1,-1,+1]
        elif(label[i][0]==10.0):
            desired[i]=[+1,-1,+1,-1]
        elif(label[i][0]==11.0):
            desired[i]=[+1,-1,+1,+1]
        elif(label[i][0]==12.0):
            desired[i]=[+1,+1,-1,-1]
        elif(label[i][0]==13.0):
            desired[i]=[+1,+1,-1,+1]
        elif(label[i][0]==14.0):
            desired[i]=[+1,+1,+1,-1]
    
#Use only the first 153 element of the desired array
desired=desired[1:numTraining+1]
w1=np.zeros((numEpochs*numTraining,numHidden,numInputs+1))
w2=np.zeros((numEpochs*numTraining,numOutput,numHidden+1))

w1[0,:,:] = np.random.uniform(-0.1, 0.1,size=[numHidden,numInputs+1])
w2[0,:,:] = np.random.uniform(-0.1, 0.1,size=[numOutput,numHidden+1])
V1=np.zeros((numEpochs,numTraining,numHidden+1))
Y1=np.zeros((numEpochs,numTraining,numHidden+1))
V2=np.zeros((numEpochs,numTraining,numOutput))
Y2=np.zeros((numEpochs,numTraining,numOutput))

V1Test=np.zeros((numEpochs,numTraining,numHidden+1))
Y1Test=np.zeros((numEpochs,numTraining,numHidden+1))
V2Test=np.zeros((numEpochs,numTraining,numOutput))
Y2Test=np.zeros((numEpochs,numTraining,numOutput))
e=np.zeros((numEpochs,numTraining,numOutput))
E=np.zeros((numEpochs,numTraining,numOutput))
derActiv1=np.zeros((numEpochs,numTraining,numOutput))
delt1=np.zeros((numEpochs,numTraining,numOutput))
derActiv2=np.zeros((numEpochs,numTraining,numHidden))
delt2=np.zeros((numEpochs,numTraining,numHidden))
a=1
b=1
kk=0
for k in range(numEpochs-1):
        for l in range(numTraining):
            for j in range(numHidden):
                for i in range(numInputs+1):
                    V1[k,l,j+1] = V1[k,l,j+1]+w1[kk,j,i]*inputB[l,i]
                    Y1[k,l,j+1]=a*np.tanh(b*V1[k,l,j+1])
                Y1[k,l,0]=1 # add bias
             # output Layer
            for j in range(numOutput):
                for i in range(numHidden+1):
                    V2[k,l,j] = V2[k,l,j]+ w2[kk,j,i]*Y1[k,l,i]
                    Y2[k,l,j]=a*np.tanh(b*V2[k,l,j])
            # Calculate error
            for j in range (numOutput):
                e[k,l,j]=desired[l,j]-Y2[k,l,j]
                E[k,l,j]= 1/2*((e[k,l,j])**2)
            # output layer
            for j in range (numOutput):
                derActiv1[k,l,j] = (b/a)*(a+Y2[k,l,j])*(a-Y2[k,l,j])
                delt1[k,l,j]=e[k,l,j]*derActiv1[k,l,j]
            #update weights between hidden and output
            for j in range(numOutput):
                for i in range(numHidden+1):
                    w2[kk+1,j,i]=w2[kk,j,i]+learnrate*delt1[k,l,j]*Y1[k,l,i]
            # hidden layer
            for i in range (numHidden):
                derActiv2[k,l,i] = (b/a)*(a+Y1[k,l,i+1])*(a-Y1[k,l,i+1])# start from 1 dont take bias input
            tempdel=0;
            for j in range(numOutput):
                tempdel=tempdel+delt1[k,l,j]*w2[kk,j,i]
                delt2[k,l,i]=derActiv2[k,l,i]*tempdel
            # update weights
            for j in range(numHidden):
                for i in range(numInputs+1):
                    w1[kk+1,j,i]=w1[kk,j,i]+learnrate*delt2[k,l,j]*inputB[l,i]
            kk=kk+1
#print(e)
V1Test=np.zeros((numTraining,numHidden+1))
Y1Test=np.zeros((numTraining,numHidden+1))
V2Test=np.zeros((numTraining,numOutput))
Y2Test=np.zeros((numTraining,numOutput))
eTest=np.zeros((numTraining,numOutput))
ETest=np.zeros((numTraining,numOutput))

totalError=0
totalMSE=0

for l in range(numTraining):
    for j in range(numHidden):
        for i in range(numInputs+1):
            V1Test[l,j+1] = V1Test[l,j+1]+w1[kk-1,j,i]*inputB[l,i]
            Y1Test[l,j+1]=a*np.tanh(b*V1Test[l,j+1])
            Y1Test[l,0]=1 # add bias
            
    # output Layer
    for j in range(numOutput):
        for i in range(numHidden+1):
            V2Test[l,j] = V2Test[l,j]+ w2[kk,j,i]*Y1Test[l,i]
            Y2Test[l,j]=a*np.tanh(b*V2Test[l,j])
    # Calculate error
    for j in range (numOutput):
        eTest[l,j]=desired[l,j]-Y2Test[l,j]
        ETest[l,j]= 1/2*((eTest[l,j])**2)
        totalMSE=totalMSE+ETest[l,j]
    print("testing Sample ", l, " Output = ", desired[l,:], " Predicted =", Y2Test[l,:])
    
AverageMSE=totalMSE/(numTraining*numOutput)
print("Average Mean sqaured Error= ", AverageMSE)
numTesting=3 
inp=24
testing=np.zeros((3,3))
target=np.zeros((3,4))
#Green (7): 
testing[0]=np.array([[0,153,0]])#B,G,R
target[0]=np.array([[-1,1,1,1]])#Green is ph=7
#Red (0): 
testing[1]=np.array([[36,84,250]])
target[1]=np.array([[-1,-1,-1,-1]])#Red is ph=0
#Dark Violet(14)
testing[2]=np.array([[131,47,78]])
target[2]=np.array([[1,1,1,-1]])

print("Testing DataSet is: \n",testing)
def convertTesting2Bipolar(index):   
    #print(filtered_data[index])
    intArray=np.zeros(24)#Binary Form
    output=np.zeros(24)#Bipolar Form
    stringForm=""
    if(index>652):
        return "Error in index Bounds"
    else:
        stringForm=decimal2Binary(testing[index][0])+decimal2Binary(testing[index][1])+decimal2Binary(testing[index][2])        
        array=list(stringForm)
        for p in range(0, len(array)): 
                intArray[p] = int(array[p])
        k = 0
        for h,v in enumerate (intArray):

            if int(intArray[h]) == 0 :
                output [k] = -1
            elif int(intArray[h]) == 1 :
                output [k] = 1
            k = k+1
        return output
test=np.zeros((len(testing),24))
#Create 2D array of all testing values
for i in range(len(testing)):
        test[i]=convertTesting2Bipolar(i)
testing_data = np.insert(test, 0,np.ones(1),1)
testing_data
V1Test=np.zeros((numTesting,numHidden+1))
Y1Test=np.zeros((numTesting,numHidden+1))
V2Test=np.zeros((numTesting,numOutput))
Y2Test=np.zeros((numTesting,numOutput))
eTest=np.zeros((numTesting,numOutput))
ETest=np.zeros((numTesting,numOutput))

totalError=0
totalMSE=0

for l in range(numTesting):
    for j in range(numHidden):
        for i in range(inp+1):
            V1Test[l,j+1] = V1Test[l,j+1]+w1[kk-1,j,i]*testing_data[l,i]
            Y1Test[l,j+1]=a*np.tanh(b*V1Test[l,j+1])
        Y1Test[l,0]=1 # add bias
            
    # output Layer
    for j in range(numOutput):
        for i in range(numHidden+1):
            V2Test[l,j] = V2Test[l,j]+ w2[kk,j,i]*Y1Test[l,i]
            Y2Test[l,j]=a*np.tanh(b*V2Test[l,j])
    # Calculate error
    for j in range (numOutput):
        eTest[l,j]=target[l,j]-Y2Test[l,j]
        ETest[l,j]= 1/2*((eTest[l,j])**2)
        totalMSE=totalMSE+ETest[l,j]
    print("Actual = ",target[l,:]," Predicted =", Y2Test[l,:])
    
AverageMSE=totalMSE/(numTesting*numOutput)
print("Average Mean sqaured Error in Testing Phase= ", AverageMSE)