import pandas as pd

import seaborn as sns
data = pd.read_csv("../input/CompleteDataset.csv")



data.head()
data['Value']=data['Value'].str.replace(r'[a-zA-Z]','')

data['Value'] = data['Value'].astype(float)



type(data['Value'][2])
data.head()