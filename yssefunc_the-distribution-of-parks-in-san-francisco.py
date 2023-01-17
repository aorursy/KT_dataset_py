#Importing library
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
# read dataset
park_data = pd.read_csv('../input/SF_Park_Scores.csv',encoding="latin-1")
#chech the head of data
park_data.head()
# rename the some columns
park_data = park_data.rename(columns={'Facility Type':'FacilityType','Square Feet':'SquareFeet','Perimeter Length':'PerimeterLength'})
# tail of the data
park_data.tail()
# Missing ratio
all_data_na = (park_data.isnull().sum() / len(park_data)) * 100
all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)[:30]
missing_data = pd.DataFrame({'Missing Ratio' :all_data_na})
missing_data.head(20)
#Data Cleaning Process
# I will drop the Floor Count    
park_data = park_data.drop(["Floor Count"], axis = 1)
park_data = park_data.dropna()
#I will put 0 for numeric data
park_data["Latitude"] = park_data["Latitude"].fillna(0)
park_data["Longitude"] = park_data["Longitude"].fillna(0)
park_data["Acres"] = park_data["Acres"].fillna(0)
park_data["Perimeter Length"] = park_data["PerimeterLength"].fillna(0)
park_data["Square Feet"] = park_data["SquareFeet"].fillna(0)
park_data["Zipcode"] = park_data["Zipcode"].fillna(0)
#I will put None for strings
park_data["State"] = park_data["State"].fillna("None")
park_data["Address"] = park_data["Address"].fillna("None")
park_data["Facility Name"] = park_data["Facility Name"].fillna("None")
park_data["FacilityType"] = park_data["FacilityType"].fillna("None")

#info about the dataset
park_data.info()
# check the data whether clean or not
all_data_na = (park_data.isnull().sum() / len(park_data)) * 100
all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)[:30]
missing_data = pd.DataFrame({'Missing Ratio' :all_data_na})
missing_data.head(20)
# finding unique zip co
park_data.Zipcode.unique()
#the distrubution of parks in san francisco in terms of longtitude and latitude
longitude = list(park_data.Longitude) 
latitude = list(park_data.Latitude)
plt.figure(figsize = (10,10))
plt.plot(longitude,latitude,'.', alpha = 0.4, markersize = 30)
plt.show()
# Which zip code with highest parks
plt.subplots(figsize=(22,12))
sns.countplot(y=park_data['Zipcode'],order=park_data['Zipcode'].value_counts().index)
plt.show()
#What kind of facility type of SF Parks
print("The number of unique Facility type is %d"%len(np.unique(park_data.FacilityType)))
print("Facility Types       Frequency in the dataset")
print(park_data.FacilityType.value_counts()[0:10])

#Visualiztion always good
park_data.FacilityType.value_counts().head(10).plot.bar()
#SF Parks Facility Score points
park_data['Score'].value_counts().sort_index().plot.line(figsize=(12, 6),color='mediumvioletred',fontsize=16,title='SF Parks Score')
#SF Parks Facility Score points with using rugplot
sns.rugplot(park_data['Score'])
# Word Cloud for Park
import matplotlib as mpl
from wordcloud import WordCloud, STOPWORDS

mpl.rcParams['font.size']=12                
mpl.rcParams['savefig.dpi']=100             
mpl.rcParams['figure.subplot.bottom']=.1 
stopwords = set(STOPWORDS)
wordcloud = WordCloud(
                          background_color='white',
                          stopwords=stopwords,
                          max_words=200,
                          max_font_size=40, 
                          random_state=42
                         ).generate(str(park_data['Park']))

print(wordcloud)
fig = plt.figure(1)
plt.imshow(wordcloud)
plt.axis('off')
plt.show()
fig.savefig("word1.png", dpi=1000)
# Which Public Administation has the highest parks 
import plotly.graph_objs as go
import plotly.offline as py
py.init_notebook_mode(connected=True) # this is important

z = {'PSA1': 'PSA1', 'PSA2': 'PSA2', 'PSA3': 'PSA3','PSA4': 'PSA4','PSA5': 'PSA5','PSA6': 'PSA6','GGP': 'GGP'}
data = [go.Bar(
            x = park_data.PSA.map(z).unique(),
            y = park_data.PSA.value_counts().values,
            marker= dict(colorscale='Jet',
                         color = park_data.PSA.value_counts().values
                        ),
           
    )]

layout = go.Layout(
    title='Target variable distribution'
)

fig = go.Figure(data=data, layout=layout)

py.iplot(fig, filename='basic-bar')