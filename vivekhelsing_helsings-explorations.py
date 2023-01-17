import pandas as pd
import numpy as np

readdata=pd.read_csv('../input/train.csv',nrows=1000000)
#head(readdata)
#readdata.shape

grouped = readdata.groupby('Semana')
#grouped['Venta_hoy'].aggregate(np.max)
len(grouped.groups)
#nationWins=grouped.size()
nationWins.head()