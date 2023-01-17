import pandas as pd

import numpy as np

!pip install -U scikit-fuzzy

import skfuzzy as fuzz

from sklearn.metrics import mean_squared_error

stri=input('Enter Valid address to your Yacht_hydrodynamics csv file:  ')

frame=pd.read_csv(stri)
length=(frame.count(axis=0)[0])

width=((frame.count(axis=1)[0])-1)

no_of_rules=2

print('no_of_membership_functions=2')



relation_count=(no_of_rules**width)*(width+1)
print(length)

print(width)

print(no_of_rules)

print(relation_count)
mat=np.zeros((length,relation_count))
epoch=input('Enter no of epochs : ')

epoch=int(epoch)
param1=[]

param2=[]

param1=np.zeros((width,3))

param2=np.zeros((width,3))

param1
final_error=50

final_param1=np.zeros((width,3))

final_param2=np.zeros((width,3))

#final_object=np.matrix()

for z in range(epoch):

    for i in range(width):

        param1[i,0]=np.random.uniform(min(frame.iloc[:,i]),max(frame.iloc[:,i])) #center

        param1[i,1]=np.random.uniform(0,(max(frame.iloc[:,i])-min(frame.iloc[:,i]))) #spread

        param1[i,2]=np.random.uniform(10)  #slope

        param2[i,0]=np.random.uniform(min(frame.iloc[:,i]),max(frame.iloc[:,i])) #center

        param2[i,1]=np.random.uniform(0,(max(frame.iloc[:,i])-min(frame.iloc[:,i]))) #spread

        param2[i,2]=np.random.uniform(10)  #slope

    i=0    

    for i in range(length):

        for j in range(no_of_rules**width):

            temp_arr=[]

            temp_w=1

            if(j&1):

                temp_w=temp_w*(fuzz.gbellmf(frame.iloc[i,0],param1[0,1],param1[0,2],param1[0,0]))

            else:

                temp_w=temp_w*(fuzz.gbellmf(frame.iloc[i,0],param2[0,1],param2[0,2],param2[0,0]))

            if(j&2):

                temp_w=temp_w*(fuzz.gbellmf(frame.iloc[i,1],param1[1,1],param1[1,2],param1[1,0]))

            else:

                temp_w=temp_w*(fuzz.gbellmf(frame.iloc[i,1],param2[1,1],param2[1,2],param2[1,0]))

            if(j&4):

                temp_w=temp_w*(fuzz.gbellmf(frame.iloc[i,2],param1[2,1],param1[2,2],param1[2,0]))

            else:

                temp_w=temp_w*(fuzz.gbellmf(frame.iloc[i,2],param2[2,1],param2[2,2],param2[2,0]))

            if(j&8):

                temp_w=temp_w*(fuzz.gbellmf(frame.iloc[i,3],param1[3,1],param1[3,2],param1[3,0]))

            else:

                temp_w=temp_w*(fuzz.gbellmf(frame.iloc[i,3],param2[3,1],param2[3,2],param2[3,0]))

            if(j&16):

                temp_w=temp_w*(fuzz.gbellmf(frame.iloc[i,4],param1[4,1],param1[4,2],param1[4,0]))

            else:

                temp_w=temp_w*(fuzz.gbellmf(frame.iloc[i,4],param2[4,1],param2[4,2],param2[4,0]))

            if(j&32):

                temp_w=temp_w*(fuzz.gbellmf(frame.iloc[i,5],param1[5,1],param1[5,2],param1[5,0]))

            else:

                temp_w=temp_w*(fuzz.gbellmf(frame.iloc[i,5],param2[5,1],param2[5,2],param2[5,0]))

            temp_arr.append(temp_w)

            for l in range(width+1):

                if l==width:

                    mat[i,7*j+width]=temp_w

                else:

                    mat[i,7*j+l]=temp_w*frame.iloc[i,l]

        final=np.sum(temp_arr)

        for j in range(relation_count):

            mat[i,j]=mat[i,j]/final

    tar=frame.iloc[:,width]  

    tar=np.matrix(tar)

    obj=np.matrix

    obj=np.matmul(np.linalg.pinv(mat),np.transpose(tar))

    #testing

    mat1=np.zeros((length,relation_count))

    for i in range(length):

        for j in range(no_of_rules**width):

            temp_arr=[]

            temp_w=1

            if(j&1):

                temp_w=temp_w*(fuzz.gbellmf(frame.iloc[i,0],param1[0,1],param1[0,2],param1[0,0]))

            else:

                temp_w=temp_w*(fuzz.gbellmf(frame.iloc[i,0],param2[0,1],param2[0,2],param2[0,0]))

            if(j&2):

                temp_w=temp_w*(fuzz.gbellmf(frame.iloc[i,1],param1[1,1],param1[1,2],param1[1,0]))

            else:

                temp_w=temp_w*(fuzz.gbellmf(frame.iloc[i,1],param2[1,1],param2[1,2],param2[1,0]))

            if(j&4):

                temp_w=temp_w*(fuzz.gbellmf(frame.iloc[i,2],param1[2,1],param1[2,2],param1[2,0]))

            else:

                temp_w=temp_w*(fuzz.gbellmf(frame.iloc[i,2],param2[2,1],param2[2,2],param2[2,0]))

            if(j&8):

                temp_w=temp_w*(fuzz.gbellmf(frame.iloc[i,3],param1[3,1],param1[3,2],param1[3,0]))

            else:

                temp_w=temp_w*(fuzz.gbellmf(frame.iloc[i,3],param2[3,1],param2[3,2],param2[3,0]))

            if(j&16):

                temp_w=temp_w*(fuzz.gbellmf(frame.iloc[i,4],param1[4,1],param1[4,2],param1[4,0]))

            else:

                temp_w=temp_w*(fuzz.gbellmf(frame.iloc[i,4],param2[4,1],param2[4,2],param2[4,0]))

            if(j&32):

                temp_w=temp_w*(fuzz.gbellmf(frame.iloc[i,5],param1[5,1],param1[5,2],param1[5,0]))

            else:

                temp_w=temp_w*(fuzz.gbellmf(frame.iloc[i,5],param2[5,1],param2[5,2],param2[5,0]))

            temp_arr.append(temp_w)

            for l in range(width+1):

                if l==width:

                    mat[i,7*j+width]=temp_w

                else:

                    mat[i,7*j+l]=temp_w*frame.iloc[i,l]

        final=np.sum(temp_arr)

        for j in range(relation_count):

            mat1[i,j]=mat1[i,j]/final

    output=np.matmul(mat1,obj)

    error=mean_squared_error(output,frame.iloc[:,width].values)

    if(error<final_error):

        final_error=error

        final_param1=param1

        final_param2=param2

        final_object=obj

        

        
print('final_error= least final RMSE after all apochs')

print('final_param1=parameters of set-1 bell functions corresponding to least RMSE')

print('final_param2=parameters of set-2 bell functions corresponding to least RMSE')

print('final_object=final output layer weights corresponding to least RMSE')
print('final_error =', final_error) 

print('final_param1=', final_param1)

print('final_param2=', final_param2)

print('final_object=', final_object)