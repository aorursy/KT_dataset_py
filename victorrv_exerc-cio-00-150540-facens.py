# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
import random
# Any results you write to the current directory are saved as output.
dia = []
for i in range(24):
    x=random.randint(1,101) #Gera valores int aleatórios de 0 a 100 
    dia.append(x)
print(dia)
    

mediadia=[]
for j in range (23):
    if ((dia[j]-dia[j+1])<0):
        mediadia.append(-(dia[j]-dia[j+1]))
    else:    
        mediadia.append(dia[j]-dia[j+1])
print(mediadia)    
mediadiaabsol = 0
for k in range (23):
    mediadiaabsol+=mediadia[k-1]
mediadiaabsol/=23
print(mediadiaabsol)
print ("Há uma mudança acima da média nos seguintes horários:")
for l in range (23):
    if(mediadia[l]>mediadiaabsol):
        print(l," -> ",l+1,"(",mediadia[l],")")