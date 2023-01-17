# Importing Pandas and Reading the Data:
import pandas as pd
data = pd.read_csv("../input/building-permit-applications-data/Building_Permits.csv")
data = data.iloc[0:10000,:]
data.head(5)
# Lets take a look at the null values in the data:
null = pd.DataFrame(pd.isnull(data).sum())
null
# Instead of coursing throught the 43 columns, lets make a list of the features which have more than 10K null values:
droplist = []
for i in null.index:
    if int(null.loc[i].values) > 2000:
        droplist.append(i)
# Here is how it looks:
droplist.append('Description')
droplist
# Finally dropping the useless columns:
newdata = data.drop(droplist, axis = 1)
# Defining the Cateogircal Columns:
c = (newdata.dtypes == 'object')
catcol = list(c[c].index)
catcol
# Defining the Numerical Columns:
n = (newdata.dtypes != 'object')
numcol = list(n[n].index)
numcol
# Now we must deal with the features who have null values but less than 10K in count:
from sklearn.impute import SimpleImputer
CatSimp = SimpleImputer(strategy = 'most_frequent')

CatSimp.fit(newdata[catcol])
newdata[catcol] = CatSimp.transform(newdata[catcol])
# Its time to fill in the missing values in the Numerical Columns:
NumSimp = SimpleImputer(strategy = 'median')

NumSimp.fit(newdata[numcol])
newdata[numcol] = NumSimp.transform(newdata[numcol])
# Finding longitudes and latitudes:
latlong = newdata['Location'].str.strip('()').str.split(', ', expand = True)
newdata['Latitude'] = latlong[[0]]
newdata['Longitude'] = latlong[[1]]

newdata.drop(['Location'], axis = 1, inplace =True)
# Defining a function which produces day, month and year columns from a single date column:

from datetime import date

def getdate(x, c):
    d = []
    m = []
    y = []
    x[c] = pd.to_datetime(x[c])
    for n in range(0,10000):
        d.append(x[c].iloc[n].day)
        m.append(x[c].iloc[n].month)
        y.append(x[c].iloc[n].year)
        
        
    x[c + " " + "Day"] = d
    x[c + " " + "Month"] = m
    x[c + " " + "Year"] = y
    x.drop([c], axis = 1, inplace = True)
    return 
# Defining dates as the collection all the features with dates:
dates = ['Permit Creation Date', 'First Construction Document Date', 'Filed Date', 'Current Status Date', 'Issued Date']

# Lets produce some day, month and year columns:
for d in dates:
    getdate(newdata, d)
# Since the Values of Longitude and Latitudes were in the "object" dtype, lets convert it to float:
newdata['Longitude'] = newdata['Longitude'].astype('float64')
newdata['Latitude'] = newdata['Latitude'].astype('float64')
# I dont't want to mess up the newdata dataset since converting its categorical values into numerical values will affect data visualization,
# lets copy it to another dataset:
encdata = pd.DataFrame(newdata, columns = newdata.columns)
PermitNumber = encdata['Permit Number']
encdata.drop(['Permit Number'], axis = 1, inplace = True)
# Defining a function that converts a dataset's categorical values to Numerical Values
def convertcat(data, target):

# Defining the Categorical Columns:
    c = (data.dtypes == 'object')
    datacatcol = list(c[c].index)


# Importing the bad boy needed to convert Categorical Values to Numerical Values:
    from category_encoders import CatBoostEncoder
    cbe = CatBoostEncoder()

# CatBoostEncoder demands a target value becuase it changes the cate
    for i in datacatcol:
        cbe.fit(data[i], data[target])
        data[i] = cbe.transform(data[i], data[target])

    return
# Convertin encdat:
convertcat(encdata, 'Estimated Cost')
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize  = (17, 20))
sns.set_context("paper", font_scale = 1)
sns.heatmap(encdata.corr())
# Visualizing the permits on a map with their Current Status:

import plotly.express as px

fig = px.scatter_mapbox(newdata, lat = "Latitude", lon="Longitude",
                        zoom = 10, height = 300, 
                        color = "Current Status")
fig.update_layout(mapbox_style = "carto-positron")
fig.update_layout(margin = {"r":0, "t":0, "l":0, "b":0})
fig.show()
# Visualizing this again but on a map to have a better idea:

import plotly.express as px

fig = px.scatter_mapbox(newdata, lat = "Latitude", lon="Longitude",
                        zoom = 10, height = 300, 
                        color = "Plansets")
fig.update_layout(mapbox_style = "carto-positron")
fig.update_layout(margin = {"r":0, "t":0, "l":0, "b":0})
fig.show()
# Plotting a countplot for the Current Status' of permits:

plt.figure(figsize = (20, 6))
sns.set_context("poster", font_scale = 0.9)
sns.countplot(newdata['Current Status'])
# Plotting the Countplot for Filed Date Month:

plt.figure(figsize=(12,6))
sns.set_context("poster")
sns.countplot(newdata['Filed Date Month'])
# Plotting the Countplot for Issued Date Month:

plt.figure(figsize=(12,6))
sns.set_context("poster")
sns.countplot(newdata['Issued Date Month'])
# Countplot for the days on which permits are issued:

plt.figure(figsize=(20,5))
sns.countplot(newdata['Issued Date Day'])