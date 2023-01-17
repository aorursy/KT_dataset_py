import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
plt.style.use('fivethirtyeight')
from collections import OrderedDict
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
color = sns.color_palette()
from numpy import array
from matplotlib import cm
from scipy.misc import imread
import base64
from sklearn import preprocessing
from mpl_toolkits.basemap import Basemap
from wordcloud import WordCloud, STOPWORDS
import plotly.plotly as py1
import plotly.offline as py
py.init_notebook_mode(connected=True)
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.offline as offline
offline.init_notebook_mode()
from plotly import tools


import warnings
warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
df_donations = pd.read_csv('../input/io/Donations.csv')
df_donors = pd.read_csv('../input/io/Donors.csv')
df_resources = pd.read_csv('../input/io/Resources.csv')
df_schools = pd.read_csv('../input/io/Schools.csv')
df_teachers = pd.read_csv('../input/io/Teachers.csv')
df_projects = pd.read_csv('../input/io/Projects.csv')
pd.DataFrame({'Dataset':['Donations','Donors','Resources','Schools','Teachers', 'Projects'],
             'Datapoints':[df_donations.shape[0], df_donors.shape[0],df_resources.shape[0],
                     df_schools.shape[0], df_teachers.shape[0], df_projects.shape[0]],
             'Features':[df_donations.shape[1], df_donors.shape[1],df_resources.shape[1],
                     df_schools.shape[1], df_teachers.shape[1], df_projects.shape[1]]})
df_donations.head(3)
df_donors.head(3)
df_resources.head(3)
df_schools.head(3)
df_teachers.head(3)
df_projects.head(3)
df_teachers['Teacher First Project Posted Date'] = pd.to_datetime(
    df_teachers['Teacher First Project Posted Date'], errors='coerce')

df_teachers.dtypes
((df_teachers['Teacher ID'].value_counts().values)>1).any()
plt.figure(figsize=(6,6))
plt.bar(df_teachers['Teacher Prefix'].value_counts().index, 
       df_teachers['Teacher Prefix'].value_counts(),
       color=sns.color_palette('viridis'))
plt.xlabel('Teacher Prefix')
plt.ylabel('Counts')
plt.title('Teacher Prefix Distribution')
plt.tight_layout()
df_teachers['weekdays'] = df_teachers['Teacher First Project Posted Date'
                                     ].dt.dayofweek

df_teachers['month'] = df_teachers['Teacher First Project Posted Date'].dt.month

df_teachers['year'] = df_teachers['Teacher First Project Posted Date'].dt.year

weekdays = {0:'Monday',1:'Tuesday',2:'Wednesday',3:'Thursday',
            4:'Friday',5:'Saturday',6:'Sunday'}

months= {1 :"Jan",2 :"Feb",3 :"Mar",4 :"Apr",5 : "May",6 : "Jun",
          7 : "Jul",8 :"Aug", 9 :"Sep",10 :"Oct",11 :"Nov",12 :"Dec"}

df_teachers['weekdays']=df_teachers['weekdays'].map(weekdays)
df_teachers['month']=df_teachers['month'].map(months)

df_teachers.head(3)
plt.figure(figsize=(10,6))

plt.bar(df_teachers['weekdays'].value_counts().index, 
        df_teachers['weekdays'].value_counts(),
        color=sns.color_palette('viridis'))
plt.xlabel('Days of the Week')
plt.ylabel('Counts')
plt.title('Most Popular Days for Teacher Project Creation')
plt.tight_layout()
pd.DataFrame(df_teachers['weekdays'].value_counts())
plt.figure(figsize=(10,6))
plt.bar(df_teachers['month'].value_counts().index, 
        df_teachers['month'].value_counts(),
        color=sns.color_palette('plasma'))
plt.xlabel('Months')
plt.ylabel('Counts')
plt.title('Most Popular Months for Teacher Project Creation')
plt.tight_layout()
pd.DataFrame(df_teachers['month'].value_counts())
ts = df_teachers.groupby('year').agg({'Teacher ID' : 'count'}).reset_index()
plt.figure(figsize=(10,6))
plt.plot(ts['year'][:-1],ts['Teacher ID'][:-1], 
         color=sns.color_palette('plasma')[0] )
plt.xlabel('Years')
plt.ylabel('Counts')
plt.title('Teacher Project Creation Over Time')
plt.tight_layout()
plt.figure(figsize=(10,6))
pref = ['Mrs.','Ms.','Mr.','Teacher','Dr.','Mx.']
for i in range(len(pref)):
    ts = df_teachers[df_teachers['Teacher Prefix'] == pref[i]].groupby('year').agg({'Teacher ID' : 'count'}).reset_index()
    plt.plot(ts['year'][:-1],ts['Teacher ID'][:-1], marker = 'o',markersize = 8,
             color=sns.color_palette('plasma')[i], label = pref[i])
    plt.xlabel('Years')
    plt.ylabel('Counts')
    
plt.legend()
plt.title('Teacher Project Creation Over Time, by Teacher Prefix')
plt.tight_layout()
gender = {'Mrs.':'Female','Ms.':'Female','Mr.':'Male','Dr.':'Unknown','Mx.':'Unknown'}
df_teachers['gender'] = df_teachers['Teacher Prefix']
df_teachers['gender'] = df_teachers['gender'].map(gender)
plt.figure(figsize=(10,6))

genders = ['Female','Male','Unknown']

for i in range(len(genders)):
    ts = df_teachers[df_teachers['gender'] == genders[i]].groupby('year').agg({'Teacher ID' : 'count'}).reset_index()
    
    plt.plot(ts['year'][:-1],ts['Teacher ID'][:-1], marker = 'o',markersize = 8,
             color=sns.color_palette('plasma')[i], label = genders[i])
    plt.xlabel('Years')
    plt.ylabel('Counts')
    
plt.legend()
plt.title('Teacher Project Creation Over Time, by Teacher Gender')
plt.tight_layout()
plt.figure(figsize=(10,6))

pref = ['Mrs.','Ms.','Mr.','Teacher','Dr.','Mx.']
ts = df_teachers[df_teachers['year'] == 2018]

for i in range(len(pref)):
    daily = ts[ts['Teacher Prefix']==pref[i]].groupby(
        ts['Teacher First Project Posted Date']).agg({'Teacher ID' : 'count'}).reset_index()
    
    plt.plot(daily['Teacher First Project Posted Date'],
             daily['Teacher ID'],
             color=sns.color_palette('Dark2')[i], label = pref[i])
    plt.xlabel('Days')
    plt.ylabel('Counts')
    
plt.legend()
plt.title('Teacher Project Creation in 2018')
plt.tight_layout()
plt.figure(figsize=(10,6))
plt.bar(df_schools['School Metro Type'].value_counts().index, 
        df_schools['School Metro Type'].value_counts(),
        color=sns.color_palette('plasma'))
plt.xlabel('Metro Type')
plt.ylabel('Counts')
plt.title('Schools Metro Type Distribution')
plt.tight_layout()
plt.figure(figsize=(10,6))
df_schools['School Percentage Free Lunch'].hist(bins = 50, color=sns.color_palette('plasma')[2])
plt.xlabel('School Percentage Free Lunch')
plt.ylabel('Counts')
plt.title('School Percentage Free Lunch')
plt.tight_layout()
pd.DataFrame(df_schools['School Percentage Free Lunch'].describe())
plt.figure(figsize=(10,14))

plt.barh(np.arange(0,816,16), 
        df_schools['School State'].value_counts()[::-1], height = 12, 
        color=sns.color_palette('plasma'), align='center')

plt.yticks(np.arange(0,816,16),df_schools['School State'].value_counts()[::-1].index)
plt.xlabel('Counts')
plt.ylabel('State')
plt.title('School State Distribution')
plt.tight_layout()
plt.figure(figsize=(10,6))
plt.bar(df_donors['Donor Is Teacher'].value_counts().index, 
        df_donors['Donor Is Teacher'].value_counts(),
        color=sns.color_palette('plasma'))
plt.xlabel('Donor Is Teacher')
plt.ylabel('Counts')
plt.title('Donor-Teacher Distribution')
plt.tight_layout()
plt.figure(figsize=(10,14))
plt.barh(np.arange(0,832,16),
        df_donors['Donor State'].value_counts()[::-1], height = 12, 
         color=sns.color_palette('plasma'), align='center')
plt.yticks(np.arange(0,832,16), df_donors['Donor State'].value_counts()[::-1].index)
plt.xlabel('Counts')
plt.ylabel('State')
plt.title('Donors State Distribution')
plt.tight_layout()
plt.figure(figsize=(10,14))
plt.barh(np.arange(0, 240, 16),
        df_donors['Donor City'].value_counts()[:15][::-1], height = 12, 
         color=sns.color_palette('plasma'), align='center')
plt.yticks(np.arange(0, 256, 16), df_donors['Donor City'].value_counts()[:15][::-1].index)
plt.xlabel('Counts')
plt.ylabel('City')
plt.title('Donors City Distribution')
plt.tight_layout()
pd.DataFrame(df_donors['Donor City'].value_counts()[:15])
# Donation Time per Location of Donor
print("Donors Features:",df_donors.columns)
print("Donations Features:",df_donations.columns)
df_donations[['Donor ID', 'Donation Received Date']].head()
df_donors[['Donor ID', 'Donor State', 'Donor City']].head()
df_temp = pd.merge(df_donors, df_donations, on=['Donor ID'])[['Donor ID', 'Donor State', 'Donor City', 'Donation Received Date']]
print(df_donations.shape)
print(df_donors.shape)
print(df_temp.shape)
df_temp.iloc[:, -1] = pd.to_datetime(df_temp.iloc[:, -1])
df_temp.head()
df_temp['Donation Received Time'] = [d.time() for d in df_temp['Donation Received Date']]
df_temp['Donation Received Year'] = [d.year for d in df_temp['Donation Received Date']]
df_temp['Donation Received Month'] = [d.month for d in df_temp['Donation Received Date']]
df_temp['Donation Received Day'] = [d.day for d in df_temp['Donation Received Date']]
df_temp.head()
states = set(df_temp['Donor State'].values)
len(states)
# Select Random States to Visualize
import random
random.seed(33)
states_5 = random.sample(states, 5)
states_5
# Plotting 1st 1000 datapoints per state in states_5
for state in states_5:
    plt.figure(figsize=(15,3))
    a = df_temp[df_temp['Donor State'] == state][:1000]
    #b = a[a['Donation Received Year']==2015]
    #b = b[b['Donation Received Month']]
    ax = sns.swarmplot(y = 'Donor State', x = 'Donation Received Date', hue = 'Donor City', data=a, palette='Set2')
    plt.title(state)
    ax.legend_.remove()
    plt.show()
df_donations['Donation Received Date'] = pd.to_datetime(
    df_donations['Donation Received Date'], errors='coerce')

df_donations['year'] = df_donations['Donation Received Date'].dt.year
df_donations['day-formated'] = df_donations['Donation Received Date'].dt.strftime('%m/%d/%Y')
plt.figure(figsize=(10,6))
df_donations['Donation Amount'].hist(bins = 50, range = (0,500), color=sns.color_palette('plasma')[2])
plt.xlabel('Donation Amount')
plt.ylabel('Counts')
plt.title('Donations Amount Distribution')
plt.tight_layout()
plt.figure(figsize=(10,6))
plt.bar(df_donations['Donation Included Optional Donation'].value_counts().index, 
        df_donations['Donation Included Optional Donation'].value_counts(),
        color=sns.color_palette('plasma'))
plt.xlabel('Donation Included Optional Donation')
plt.ylabel('Counts')
plt.title('Donations Included Optional Donation Distribution')
plt.tight_layout()
plt.figure(figsize=(10,6))

included = ['Yes','No']
ts = df_donations[df_donations['year']==2018]

for i in range(len(included)):
    daily = ts[ts['Donation Included Optional Donation']==included[i]].groupby(
        ts['day-formated']).agg({'Donation ID' : 'count'}).reset_index()
    
    plt.plot(daily['day-formated'],
             daily['Donation ID'],
             color=sns.color_palette('Dark2')[i], label = included[i])
    plt.xlabel('Days')
    plt.ylabel('Counts')

plt.xticks([0,25, 50,75,100,125,150])
plt.legend()
plt.title('Donations Received Trend in 2018')
plt.tight_layout()
df = df_resources

fig = plt.figure(figsize=(10,6))
ax2 = fig.add_subplot(111)
ax2.hist(df[df['Resource Unit Price']<100]['Resource Unit Price'],100, color=sns.color_palette('viridis')[1]);

ax3 = fig.add_axes([0.55,0.5,0.4,0.4])
ax3.hist(df['Resource Unit Price'].dropna(), 50, color=sns.color_palette('viridis')[0]);
ax2.set_xlabel("Resource Unit Price ($)")
ax2.set_ylabel('Frequency')

ax3.set_xlabel("Resource Unit Price ($)")
ax3.set_ylabel('Frequency')
ax2.set_title('Distribution of Resource Unit Price', size = 18)
plt.tight_layout()
fig = plt.figure(figsize=(10,6))
ax4 = fig.add_subplot(111)
ax4.scatter(df['Resource Quantity'],df['Resource Unit Price'], color='#aa3333')
ax4.set_xlabel('Resource Quantity')
ax4.set_ylabel('Resource Unit Price')
ax4.set_title('Relationship of Resource Unit Price vs. Resource Quantity', size =18)
ax4.set_xlim(0,600);
df_projects['Project Posted Date'] = pd.to_datetime(
    df_projects['Project Posted Date'], errors='coerce')

df_projects['Project Expiration Date'] = pd.to_datetime(
    df_projects['Project Expiration Date'], errors='coerce')

df_projects['Project Fully Funded Date'] = pd.to_datetime(
    df_projects['Project Fully Funded Date'], errors='coerce')

df_projects.dtypes
df_projects['year-posted'] = df_projects['Project Posted Date'].dt.year
df_projects['day-posted-formated'] = df_projects['Project Posted Date'].dt.strftime('%m/%d/%Y')

df_projects['year-expiry'] = df_projects['Project Expiration Date'].dt.year
df_projects['day-expiry-formated'] = df_projects['Project Expiration Date'].dt.strftime('%m/%d/%Y')

df_projects['year-funded'] = df_projects['Project Fully Funded Date'].dt.year
df_projects['day-funded-formated'] = df_projects['Project Fully Funded Date'].dt.strftime('%m/%d/%Y')

df_projects['delta-days-before-expiry'] = (df_projects['Project Expiration Date'] - df_projects['Project Posted Date']).dt.days
df_projects['delta-days-before-funded'] = (df_projects['Project Fully Funded Date'] - df_projects['Project Posted Date']).dt.days
df_projects.columns.tolist()
pd.DataFrame(df_projects['Teacher ID'].value_counts().describe())
plt.figure(figsize=(10,6))
plt.bar(df_projects['Project Type'].value_counts().index, 
        df_projects['Project Type'].value_counts(),
        color=sns.color_palette('viridis'))
plt.xlabel('Project Type')
plt.ylabel('Counts')
plt.title('Projects Type Distribution')
plt.tight_layout()
plt.figure(figsize=(10,6))
plt.barh(df_projects['Project Subject Category Tree'].value_counts()[:15].index, 
        df_projects['Project Subject Category Tree'].value_counts()[:15],
        color=sns.color_palette('viridis'))
plt.xlabel('Counts')
plt.ylabel('Project Subject Category Tree')
plt.title('Projects Subject Category Tree Distribution')
plt.tight_layout()
plt.figure(figsize=(10,6))
plt.barh(df_projects['Project Subject Subcategory Tree'].value_counts()[:15].index, 
        df_projects['Project Subject Subcategory Tree'].value_counts()[:15],
        color=sns.color_palette('viridis'))
plt.xlabel('Counts')
plt.ylabel('Project Subject Subcategory Tree')
plt.title('Projects Subject Subcategory Tree Distribution')
plt.tight_layout()
temp = df_projects['Project Grade Level Category'].value_counts()
fig = {
  "data": [
    {
      "values": temp.values,
      "labels": temp.index,
      "domain": {"x": [0, .48]},
      "name": "Grade Level Category",
      #"hoverinfo":"label+percent+name",
      'marker': {'colors': ['rgb(45, 35, 113)',
                                  'rgb(0, 208, 110)',
                                  'rgb(0, 208, 202)',
                                  'rgb(83, 158, 196)',
                                  'rgb(124, 231, 87)']},
      "hole": .7,
      "type": "pie"
    },
    
    ],
  "layout": {
        "title":"Distribution of Projects Grade Level Category",
        "annotations": [
            {
                "font": {
                    "size": 20
                },
                "showarrow": False,
                "text": "Grade Level Categories",
                "x": 0.11,
                "y": 0.5
            }
            
        ]
    }
}
iplot(fig, filename='donut')
plt.figure(figsize=(10,6))
plt.barh(df_projects['Project Resource Category'].value_counts().index, 
        df_projects['Project Resource Category'].value_counts(),
        color=sns.color_palette('viridis'))
plt.xlabel('Counts')
plt.ylabel('Project Resource Category')
plt.title('Projects Resource Category Distribution')
plt.tight_layout()
plt.figure(figsize=(10,6))
plt.barh(df_projects['Project Current Status'].value_counts().index, 
        df_projects['Project Current Status'].value_counts(),
        color=sns.color_palette('viridis'))
plt.xlabel('Counts')
plt.ylabel('Project Current Status')
plt.title('Projects Current Status Distribution')
plt.tight_layout()
df = df_projects

fig = plt.figure(figsize=(10,6))
ax2 = fig.add_subplot(111)
ax2.hist(df[df['Project Cost']<5000]['Project Cost'],100, color=sns.color_palette('viridis')[1]);

ax3 = fig.add_axes([0.55,0.5,0.4,0.4])
ax3.hist(df['Project Cost'].dropna(), 50, color=sns.color_palette('viridis')[0]);
ax2.set_xlabel("Project Cost ($)")
ax2.set_ylabel('Frequency')

ax3.set_xlabel("Project Cost ($)")
ax3.set_ylabel('Frequency')
ax2.set_title('Distribution of Project Cost', size = 18)
plt.tight_layout()
pd.DataFrame(df_projects['Project Cost'].describe())
df = df_projects

fig = plt.figure(figsize=(10,6))
ax2 = fig.add_subplot(111)
ax2.hist(df['delta-days-before-funded'].dropna(),100, color=sns.color_palette('viridis')[1]);

#ax3 = fig.add_axes([0.55,0.5,0.4,0.4])
#ax3.hist(df['Project Cost'].dropna(), 50, color=sns.color_palette('viridis')[0]);
ax2.set_xlabel("Number of days before fully funded")
ax2.set_ylabel('Frequency')

#ax3.set_xlabel("Project Cost ($)")
#ax3.set_ylabel('Frequency')
ax2.set_title('Distribution of days it took before project funding', size = 18)
plt.tight_layout()
pd.DataFrame(df['delta-days-before-funded'].describe())
# donations_resources = pd.merge(df_donations,df_resources, how='left',on='Project ID')
# donations_resources_donors = pd.merge(donations_resources, df_donors, how='left', on='Donor ID')
# donations_resources_donors_projects = pd.merge(donations_resources_donors, df_projects, how='left', on='Project ID')
# donations_resources_donors_projects_teachers = pd.merge(donations_resources_donors_projects, df_teachers, how='left', on='Teacher ID')
# merged_df = pd.merge(donations_resources_donors_projects_teachers, df_schools, how='left', on='School ID')
# merged_df.head(1)
# merged_df.sample(n = 50000, axis = 0).to_pickle('sampled_df.pickle')
df = pd.read_csv("../input/sampled-dataset-50k/sampled_df_jojie.csv").iloc[:,1:]
df.shape
n_rows = 10000
df_sample_1 = df.sample(n = n_rows, random_state = 123, axis = 0)
df_sample_1.shape
not_useful = ['Project ID', 'Donation ID','Donor ID',
          'Donor Cart Sequence','Resource Vendor Name','Resource Item Name', 
              'Teacher Project Posted Sequence','School ID', 'Teacher ID',
              'School Name', 'Donor Zip', 'School Zip', 'Unnamed: 0.1']

date_feat = ['Donation Received Date','Project Posted Date', 'Project Expiration Date',
        'Project Fully Funded Date','Teacher First Project Posted Date']

donor_feat = ['Donation ID', 'Donor ID',
       'Donation Included Optional Donation', 'Donation Amount',
       'Donor Cart Sequence', 'Donation Received Date', 
       'Donor City', 'Donor State', 'Donor Is Teacher', 'Donor Zip']

project_feat = ['Resource Item Name',
       'Resource Quantity', 'Resource Unit Price', 'Resource Vendor Name',
       'Project Type', 'Project Title', 'Project Essay',
       'Project Short Description', 'Project Need Statement',
       'Project Subject Category Tree', 'Project Subject Subcategory Tree',
       'Project Grade Level Category', 'Project Resource Category',
       'Project Cost', 'Project Posted Date', 'Project Expiration Date',
       'Project Current Status', 'Project Fully Funded Date', 'Teacher Prefix',
       'Teacher First Project Posted Date', 'School Name', 'School Metro Type',
       'School Percentage Free Lunch', 'School State', 'School Zip',
       'School City', 'School County', 'School District']

cat_feat = ['Donor City', 'Donor State', 'Donor Is Teacher', 'Donor Zip',
            'Project Type','Project Subject Category Tree', 'Donation Included Optional Donation',
            'Project Subject Subcategory Tree',
            'Project Grade Level Category', 'Project Resource Category',
            'Project Current Status','Teacher Prefix','School Metro Type',
            'School State', 'School Zip','School City','School County', 'School District']
df_sample = df_sample_1.drop(labels=not_useful,axis = 1)
cat_feat_new = []
for x in df_sample.columns.tolist(): 
    if x in cat_feat:
        cat_feat_new.append(x)
df_sample.dropna(axis=0, inplace=True)
#df_new[pd.isnull(df_new).any(axis=1)]
checker1 = df_sample[df_sample['School State']=='Alaska'].index.tolist()
checker2 = df_sample[df_sample['School Metro Type']=='town'].index.tolist()
from sklearn.preprocessing import LabelEncoder

labels = {}
le = LabelEncoder()

for cat in cat_feat_new:
    le.fit(df_sample[cat].values)
    
    if df_sample[cat].dtype == 'float64' or df_sample[cat].dtype == 'int':
        df_sample[cat] = le.transform(df_sample[cat])
    
    else:
        df_sample[cat] = le.transform(df_sample[cat].astype(str))
    
    labels[cat] = list(le.classes_)
df_sample['Project Posted Date'] = pd.to_datetime(
    df_sample['Project Posted Date'], errors='coerce')

df_sample['Project Expiration Date'] = pd.to_datetime(
    df_sample['Project Expiration Date'], errors='coerce')

df_sample['Project Fully Funded Date'] = pd.to_datetime(
    df_sample['Project Fully Funded Date'], errors='coerce')

df_sample['Donation Received Date'] = pd.to_datetime(
    df_sample['Donation Received Date'], errors='coerce')

df_sample['Teacher First Project Posted Date'] = pd.to_datetime(
    df_sample['Teacher First Project Posted Date'], errors='coerce')
df_sample['delta-days-before-expiry'] = (df_sample['Project Expiration Date'] - df_sample['Project Posted Date']).dt.days
df_sample['delta-days-before-funded'] = (df_sample['Project Fully Funded Date'] - df_sample['Project Posted Date']).dt.days
df_sample['delta-days-before-donating'] = (df_sample['Donation Received Date'] - df_sample['Project Posted Date']).dt.days
X = df_sample[['Resource Quantity', 'Resource Unit Price','Project Type',
       'Project Subject Category Tree', 'Project Subject Subcategory Tree',
       'Project Grade Level Category', 'Project Resource Category',
       'Project Cost', 'Teacher Prefix',
       'School Metro Type','School Percentage Free Lunch', 'School State', 'School City',
       'School County', 'School District']]
X.shape
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters = 5).fit(X)
plt.figure(figsize=(10,6))
for i in range(len(kmeans.cluster_centers_)):
    plt.plot(kmeans.cluster_centers_[i][0],kmeans.cluster_centers_[i][1],'x', markersize=12, label=i);
    plt.xticks([], [])
    plt.yticks([], [])
    plt.legend()
plt.title('KMeans Clustering (n_clusters = 5)');
from sklearn.metrics import silhouette_score
sse = []
silhouette = []
krange = range(2, 15)
plt.figure(figsize=(10,6))
for k in krange:
    kmeans = KMeans(n_clusters = k).fit(X)
    sse.append(kmeans.inertia_)
    
    labels = kmeans.predict(X)
    sl = silhouette_score(X, labels)
    silhouette.append(sl)
    
plt.plot(krange, sse, label='SS', c='orange')
lines, labels = plt.gca().get_legend_handles_labels()
plt.twinx()
plt.plot(krange, silhouette, label = 'Silhouette', c='blue')
lines2, labels2 = plt.gca().get_legend_handles_labels()
plt.legend(lines+lines2, labels+labels2)
plt.title("Validation Measures per k Clusters")
plt.show()

n_clusters = 4
kmeans = KMeans(n_clusters = n_clusters).fit(X)
plt.figure(figsize=(10,6))
for i in range(len(kmeans.cluster_centers_)):
    plt.plot(kmeans.cluster_centers_[i][0],kmeans.cluster_centers_[i][1],'x', markersize=12, label=i);
    plt.xticks([], [])
    plt.yticks([], [])
    plt.legend()
plt.title('KMeans Clustering (n_clusters = {})'.format(n_clusters));
clusters_list = {}
for c in range(n_clusters):
    clusters_list[c] = df.iloc[X.iloc[kmeans.labels_ == c, :].index]['Donor ID']
X['Label'] = kmeans.labels_
X.shape
X.head()
X.tail()
from sklearn.decomposition import PCA
pca = PCA(n_components = 3).fit_transform(X)
plt.figure(figsize=(10,6))
plt.scatter(pca[:,0],pca[:,1], c = kmeans.labels_);
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
pd.options.display.float_format = '{:,.2g}'.format
from collections import Counter
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
X_ = X.drop('Label', axis = 1)
y_ = X['Label']
def ml_class(feature, target, ml_type='knn_class', show_PCC=False,
             param_range=range(1, 30), seed_settings=range(0, 30),
             plot=False, report=True, penalty='l2'):
    """
    Plot accuracy vs parameter for test and training data. Print
    maximum accuracy and corresponding parameter value. Print number of trials.

    Inputs
    ======
    feature: Dataframe of features
    target: Series of target values
    show_PCC: Boolean. will show PCC on plot if True
    param_range: Range of values for parameters
    seed_settings: Range of seed settings to run
    plt: Boolean. Will show plot if True
    report: Boolean. Will show report if True
    penalty: String either l1 for L1 norm or l2 for L2 norm

    Outputs
    =======
    Plot of accuracy vs parameter for test and training data
    Report showing number of maximum accuracy, optimal parameters, PCC, and
        no. of iterations
    """

    train_acc = []
    test_acc = []

    # Initiate counter for number of trials
    iterations = 0

    # create an array of cols: parameters and rows: seeds
    for seed in seed_settings:

        # count one trial
        iterations += 1

        # split data into test and training sets
        X_train, X_test, y_train, y_test = train_test_split(feature,
                                                            target,
                                                            random_state=seed)
        train = []
        test = []

        # make a list of accuracies for different parameters
        for param in param_range:
            # build the model
            if ml_type == 'knn_class':
                clf = KNeighborsClassifier(n_neighbors=param)

            elif ml_type == 'log_reg':
                clf = LogisticRegression(C=param, penalty=penalty)

            elif ml_type == 'svc':
                clf = LinearSVC(C=param, penalty=penalty, dual=False)

            clf.fit(X_train, y_train)

            # record training set accuracy
            train.append(clf.score(X_train, y_train))
            # record generalization accuracy
            test.append(clf.score(X_test, y_test))

        # append the list to _acc arrays
        train_acc.append(train)
        test_acc.append(test)

    # compute mean and error across columns
    train_all = np.mean(train_acc, axis=0)
    test_all = np.mean(test_acc, axis=0)

    # compute standard deviation
    var_train = np.var(train_acc, axis=0)
    var_test = np.var(test_acc, axis=0)

    # compute pcc
    state_counts = Counter(target)
    df_state = pd.DataFrame.from_dict(state_counts, orient='index')
    num = (df_state[0] / df_state[0].sum())**2
    pcc = 1.25 * num.sum()

    if plot == True:
        plt.figure(figsize=(10,6))
        # plot train and errors and standard devs
        plt.plot(param_range, train_all, c='b',
                 label="training set", marker='.')
        plt.fill_between(param_range,
                         train_all + var_train,
                         train_all - var_train,
                         color='b', alpha=0.1)

        # plot test and errors and standard devs
        plt.plot(param_range, test_all, c='r', label="test set", marker='.')
        plt.fill_between(param_range,
                         test_all + var_test,
                         test_all - var_test,
                         color='r', alpha=0.1)

        # plot pcc line
        if show_PCC == True:
            plt.plot(param_range, [pcc] * len(param_range),
                     c='tab:gray', label="pcc", linestyle='--')

        plt.xlabel('Parameter Value')
        plt.ylabel('Accuracy')
        plt.title(ml_type + ": Accuracy vs Parameter Value")
        plt.legend(loc=0)

        plt.tight_layout()
        plt.show()

    max_inds = np.argmax(test_all)
    acc_max = np.amax(test_all)
    param_max = (param_range)[max_inds]

    if report == True:
        print('Report:')
        print('=======')
        print("Max average accuracy: {}".format(
            np.round(acc_max, 4)))
        print("Var of accuracy at optimal parameter: {0:.4f}".format(
            var_test[max_inds]))
        print("Optimal parameter: {0:.4f}".format(param_max))
        if ml_type != "knn_class":
            print("Regularization: ", penalty)
        print('1.25 x PCC: {0:.4f}'.format(pcc))
        print('Total iterations: {}'.format(iterations))
        

    # return maximum accuracy and corresponding parameter value
    return np.round(acc_max, 4), param_max  # best_feat
C = [1e-3,0.1, 0.75, 1, 5, 10]

acc_max, param_max = ml_class(X_, y_, ml_type='log_reg', show_PCC=True,
                            param_range=C, seed_settings=range(0, 20),
                            plot=True, report=True, penalty='l1');
X_train, X_test, y_train, y_test = train_test_split(X_, y_)
clf = LogisticRegression(C=param_max, penalty='l1')
clf.fit(X_train, y_train)

plt.figure(figsize=(10,6))
for i in range(n_clusters):
    sorted_i = np.argsort(abs(clf.coef_[i]))[::-1]
    
    plt.plot(X_.columns[sorted_i], clf.coef_[i][sorted_i], marker='o', linestyle='none', linewidth=1, label = "Cluster "+str(i))
    plt.bar(X_.columns[sorted_i], clf.coef_[i][sorted_i], alpha=0.3, width=0.1)
    plt.legend()
    plt.xticks(rotation=90)
    plt.ylabel("Coefficient")
    plt.title("Logistic Regression Feature Coefficients per Cluster")
plt.show()
df['Project Posted Date'] = pd.to_datetime(
    df['Project Posted Date'], errors='coerce')

df['Project Expiration Date'] = pd.to_datetime(
    df['Project Expiration Date'], errors='coerce')

df['Project Fully Funded Date'] = pd.to_datetime(
    df['Project Fully Funded Date'], errors='coerce')

df['Donation Received Date'] = pd.to_datetime(
    df['Donation Received Date'], errors='coerce')

df['Teacher First Project Posted Date'] = pd.to_datetime(
    df['Teacher First Project Posted Date'], errors='coerce')

df['delta-days-before-expiry'] = (df['Project Expiration Date'] - df['Project Posted Date']).dt.days
df['delta-days-before-funded'] = (df['Project Fully Funded Date'] - df['Project Posted Date']).dt.days
df['delta-days-before-donating'] = (df['Donation Received Date'] - df['Project Posted Date']).dt.days
clusters_by_state = {}
clusters_by_org_loyalty = {}
clusters_by_early_donors = {}
clusters_by_late_donors = {}
clusters_by_all = {}

for c in range(len(clusters_list)):
    clusters_by_state[c] = df.iloc[clusters_list[c].index][['Donor ID','Donor State']]
    clusters_by_org_loyalty[c] = df.iloc[clusters_list[c].index][['Donor ID','Donation Included Optional Donation']]
    clusters_by_early_donors[c] = df.iloc[clusters_list[c].index][['Donor ID','delta-days-before-donating']][\
    df['delta-days-before-donating']<30]
    clusters_by_late_donors[c] = df.iloc[clusters_list[c].index][['Donor ID','delta-days-before-donating']][\
    df['delta-days-before-donating']>30]
    clusters_by_all[c] = df.iloc[clusters_list[c].index][['Donor ID','Donor State','Donation Included Optional Donation','delta-days-before-donating']]
import pandas as pd
X_test = pd.DataFrame(columns=['Resource Quantity', 'Resource Unit Price','Project Type',
       'Project Subject Category Tree', 'Project Subject Subcategory Tree',
       'Project Grade Level Category', 'Project Resource Category',
       'Project Cost', 'Teacher Prefix',
       'School Metro Type','School Percentage Free Lunch', 'School State', 'School City',
       'School County', 'School District'])

X_test['Resource Quantity'] = [10]
X_test['Resource Unit Price'] = [10]
X_test['Project Type'] = ['Teacher-Led']
X_test['Project Subject Category Tree'] = ['Health & Sports']
X_test['Project Subject Subcategory Tree'] = ['Gym & Fitness, Health & Wellness']
X_test['Project Grade Level Category'] = ['Grades 9-12']
X_test['Project Resource Category'] = ['Sports & Exercise Equipment']
X_test['Project Cost'] = [53.3]
X_test['Teacher Prefix'] = ['Mrs.']
X_test['School Metro Type'] = ['suburban']
X_test['School Percentage Free Lunch'] = [65]
X_test['School State'] = ['New York']
X_test['School City'] = ['New York City']
X_test['School County'] = ['Queens']
X_test['School District'] = ['New York Dept Of Education']
temp = X_test
temp.T
def donors_to_recommend(X, ind_ = 0, cluster_disp = False):
    cat_feat = [
            'Project Type','Project Subject Category Tree',
            'Project Subject Subcategory Tree',
            'Project Grade Level Category', 'Project Resource Category',
            'Teacher Prefix','School Metro Type',
            'School State','School City','School County', 'School District']
    
    X_test_transformed = X_test.copy()

    le = LabelEncoder()

    for cat in cat_feat:
        le.fit(X_test_transformed[cat].values)
    
        if X_test_transformed[cat].dtype == 'float64' or X_test_transformed[cat].dtype == 'int':
            X_test_transformed[cat] = le.transform(X_test_transformed[cat])

        else:
#             print(cat)
#             print(X_test_transformed[cat].dtype)
#             print(X_test_transformed[cat].values.astype(str))
            X_test_transformed[cat] = le.transform(X_test_transformed[cat].astype(str))
    
    y = clf.predict(X_test_transformed)
    if cluster_disp == True:
        print("Cluster Num", y[ind_])
    else:
        pass
    
    return clusters_by_all[y[ind_]]              
donors_pred = donors_to_recommend(X_test, ind_ = 0, cluster_disp = True)
display(donors_pred.head())
print("No. of donors returned:", len(donors_pred))
