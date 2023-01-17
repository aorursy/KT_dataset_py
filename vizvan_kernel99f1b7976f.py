# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
#Import data and save it as MPData
MPIData = pd.read_csv("../input/MPIData_augmented.csv")
#(rows,columns)
MPIData.shape
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls

import numpy as np


trace0 = go.Box(
    y=MPIData["precipitation"].values.tolist()
)
data = [trace0]
py.iplot(data)
corr = MPIData[["precipitation",'PercPoverty', 'DepvrIntensity', 
                'popDensity',  'TimeToCity',"MPI_Region",
       'AvgNightLight', 'LandClassification', 'Elevation', 'Temperature',
       'Evaporation', 'Modis_LAI', 'Modis_EVI', 'Conflicts_total']].corr()

trace = go.Heatmap(z=corr.values.tolist(),
                   x=corr.columns,
                   y=corr.columns)
data=[trace]
py.iplot(data, filename='labelled-heatmap')
trace0 = go.Histogram(
    x=MPIData["popDensity"].values.tolist(),name = "popDensity"
)
trace1 = go.Histogram(
    x=MPIData["MPI_Region"].values.tolist(),name = "MPI"
)

from numpy.linalg import inv

trace2 = go.Histogram(
    x=np.log(MPIData["MPI_Region"]+0.0001).values.tolist(),name = "MPI (Log)"
)

trace2 = go.Histogram(
    x=(1/(MPIData["MPI_Region"]+0.0001).values).tolist(), name = "MPI (Log)"
)
data = [trace0]
py.iplot(data)
data = [trace1]
py.iplot(data)
data = [trace2]
py.iplot(data)
# Create a trace
trace = go.Scatter(
    x = MPIData["PercPoverty"].values.tolist(),
    y = MPIData["MPI_Region"].values.tolist(),
    mode = 'markers'
)

data = [trace]

# Plot and embed in ipython notebook!
py.iplot(data, filename='basic-scatter')
import statsmodels.api as sm

MPIData_no_na = MPIData[["MPI_Region","precipitation","Modis_EVI"]].dropna()
y = MPIData_no_na["MPI_Region"]
X = MPIData_no_na[["precipitation","Modis_EVI"]]/1000

# Note the difference in argument order
model = sm.OLS(y, X).fit()
predictions = model.predict(X) # make the predictions by the model

# Print out the statistics
model.summary()
import statsmodels.api as sm

MPIData_no_na = MPIData[["MPI_Region","Evaporation","Modis_EVI","AvgNightLight"]].dropna()
#y = MPIData_no_na["MPI_Region"]
y = np.log(MPIData_no_na["MPI_Region"]+.0001)
X = MPIData_no_na[["Evaporation","Modis_EVI","AvgNightLight"]]/1000

# Note the difference in argument order
model = sm.OLS(y, X).fit()
predictions = model.predict(X) # make the predictions by the model

# Print out the statistics
model.summary()