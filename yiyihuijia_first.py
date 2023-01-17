import numpy as np
import pandas as pd
import math
colum_target=pd.read_csv("../input/train.csv",usecols=['Demanda_uni_equil'])
m=colum_target['Demanda_uni_equil'].tolist()
#print(m)
#m=np.mean(np.linalg.logm(colum_target['Demanda_uni_equil'].value+1))
#m=np.exp(math.log(+1).mean())
#mm=colum_target['m'].mean()
#print(mm)
#result_mean=pd.read_csv("../input/test.csv",usecols=['id'])
#result_mean['Demanda_uni_equil']=exp(mean)
#result_mean.to_csv('result_mean.csv',index=False)
x=np.exp(np.mean(np.log(np.array(m)+1)))-1
print (x)
result_logmean=pd.read_csv("../input/test.csv",usecols=['id'])
result_logmean['Demanda_uni_equil']=x
result_logmean.to_csv('result_logmean.csv',index=False)