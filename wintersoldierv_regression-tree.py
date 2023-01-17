import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
df=pd.DataFrame()
df['hrs']=np.array([1,2.5,1.5,2.1,5,4,6.5,3.5,8,7.5])
df['marks']=np.array([37,50,43,60,27,80,30,75,64,56])
df.sort_values(by='hrs',inplace=True)
df #toy_data
plt.scatter(x=df['hrs'],y=df['marks'])
plt.xlabel("hours")
plt.ylabel("marks")
plt.grid()
plt.show()
calculated_sum_of_error=[]
spliting_criterias=[]

for i in range(0,df.shape[0]-1):
    
    point2=i+1
    point1=i
       
    spliting_criteria=df['hrs'].iloc[point1:(point1+2)].mean()
    
    df1=df[df['hrs']<spliting_criteria]
    df2=df[df['hrs']>spliting_criteria]
    
    mean_of_rest_of_the_marks_df1=df1['marks'].mean()
    mean_of_rest_of_the_marks_df2=df2['marks'].mean()
    
    error_df1=[]
    for j in range(df1.shape[0]):
        d1=((mean_of_rest_of_the_marks_df1-df1['marks'].iloc[j])**2)
        error_df1.append(d1)
        
    error_df2=[]
    for k in range(df2.shape[0]):
        d2=((mean_of_rest_of_the_marks_df2-df2['marks'].iloc[k])**2)
        error_df2.append(d2)
        
    sum_of_all_errors=np.sum(error_df1)+np.sum(error_df2)
    
    spliting_criterias.append(spliting_criteria)
    calculated_sum_of_error.append(sum_of_all_errors)
plt.scatter(x=spliting_criterias,y=calculated_sum_of_error)
plt.xlabel("spliting criterias")
plt.ylabel("errors")
plt.show()
data=pd.DataFrame()
data['Spliting_criterias']=spliting_criterias
data['errors']=calculated_sum_of_error
data
data['errors'].min()
#4.50 hours has the least error
plt.scatter(df['hrs'],df['marks'])
plt.axvline(x = 4.50, ymax = 1, 
            color ='red',label='First Split')
plt.xlabel("Hours Studied")
plt.ylabel("Marks Obtained")
plt.title("Hours Studied VS Marks Obtained")
plt.legend()
plt.show()
# and soon...