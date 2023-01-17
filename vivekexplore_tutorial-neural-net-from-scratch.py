#Features: X1, X2 Label:Y

X1=[1.0,2.0,3.0,4.0] #Input 1

X2=[3.0,6.0,9.0,12.0] #Input 2

Y=[5.0,10.0,15.0,20.0] #Derived from 2*(x1) + 1*(x2) [2-Dimentional]

#Initilizing Parameters

w1,w2,b=0,0,0 #We want the Nural Net. to figure this out, so it is assigned as 0.

alpha=0.01 #Learning Rate

iteration=1

x=0

costDV=1

while costDV!=0: 

        print("Iteration:",iteration)

        iteration+=1

        #Forward Propagation

        pred=(w1*X1[x%4])+(w2*X2[x%4]) #Hidden Layer

        #Back Propogation: Using Chain rule on Cost

        costDV=(pred-Y[x%4]) #It is actually 2*(pred-y1[x%4]), but it is optimized

        costDV_w1=costDV*(X1[x%4]) #Partial Derivative of w1

        costDV_w2=costDV*(X2[x%4]) #Partial Derivative of w2

        b=Y[x%4]-pred

        print("Prediction",round(pred,3),"Actual",Y[x%4],"Cost Derivative:","//",round(costDV,3),"//",round(costDV_w1,3),round(costDV_w2,2),round(b,2))

        print("w1:",w1,"\nw2:",w2,"\nbias:",round(b,3),"\n") #This is derived from 3-D plain.

        #Gradient Descent: Changing the weight besed on Derivative until it becomes 0.

        w1-=alpha*(costDV_w1)

        w2-=alpha*(costDV_w2)

        x+=1
#Initilizing the data into a Pandas Dataframe using Numpy

import pandas as pd

import numpy as np

x1=[1.0,2.0,3.0,4.0]

x2=[3.0,6.0,9.0,12.0]

y1=[5.0,10.0,15.0,20.0]

df=pd.DataFrame(x1,columns=["x1"])

df["x2"]=np.array(x2)

df["y1"]=y1

df.describe()
from sklearn import linear_model

model=linear_model.LinearRegression()

model.fit(df[["x1","x2"]],df["y1"])

print("Coefficients (w1 and w2) from SKlearn Library:",model.coef_)

print("Coefficients (w1 and w2) from Nural Net from Scratch:",round(w1,2),round(w2,2))
