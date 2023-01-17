# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import torch 

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
admissiondata=pd.read_csv('/kaggle/input/graduate-admissions/Admission_Predict.csv')
admissiondata
del admissiondata['Serial No.']
admissiondata.dtypes
admissiondata.isna().sum()
g = sns.PairGrid(admissiondata)

g.map_upper(plt.scatter)

g.map_lower(sns.kdeplot)

g.map_diag(sns.kdeplot, lw=3, legend=False);
#GRE score,TOEFl score,University rating,SOP,LOR,CGPA

input_arrays=admissiondata[['GRE Score','TOEFL Score','University Rating','SOP','LOR ','CGPA']].to_numpy()

target_array=admissiondata[['Chance of Admit ']].to_numpy()
input_arrays=torch.tensor(input_arrays)

target_arrays=torch.tensor(target_array)
import torch.nn.functional as F

import torch.nn as nn



class Model(nn.Module): 

  

    def __init__(self): 

        super(Model, self).__init__() 

        self.linear = torch.nn.Linear(input_arrays.shape[1], target_arrays.shape[1]) 

  

    def forward(self, x): 

        y_pred = self.linear(x) 

        return y_pred

    

    

model = Model()

criterion = nn.MSELoss(size_average = False) 

optimizer = torch.optim.Adam(model.parameters(), lr = 0.01) 
for epoch in range(5000): 



    pred_y = model(input_arrays.float()) 

   

    loss = criterion(pred_y, target_arrays.float()) 

 

    optimizer.zero_grad() 

    loss.backward() 

    optimizer.step() 

    print('epoch {}, loss {}'.format(epoch, loss.item())) 
preds=model(input_arrays.float())

preds
plt.scatter(target_arrays.detach().numpy(),preds.detach().numpy())