import pandas as pd
import numpy as np
import matplotlib
%matplotlib inline

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.offline as offline
offline.init_notebook_mode(connected=True)

import plotly.graph_objs as go
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf,pacf
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from keras.layers import Dense, LSTM, Dropout
from keras.models import Sequential
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

from geopy.geocoders import Nominatim
import gc
import time
# Data directory
data_dir = '../input/'
sns.set()
donations = pd.read_csv(data_dir+"Donations.csv",error_bad_lines=False)
projects = pd.read_csv(data_dir+"Projects.csv",error_bad_lines=False)
donors = pd.read_csv(data_dir+"Donors.csv",error_bad_lines=False)
resources = pd.read_csv(data_dir+"Resources.csv",error_bad_lines=False)
schools = pd.read_csv(data_dir+"Schools.csv",error_bad_lines=False)
teachers = pd.read_csv(data_dir+"Teachers.csv",error_bad_lines=False)
donors.describe()
donations.describe(include='all')
projects.describe(include='all')
resources.describe(include='all')
schools.describe(include='all')
teachers.describe(include='all')
names = ['Donations','Donors','Projects','Resources','Schools','Teachers']
datum = [donations,donors,projects,resources,schools,teachers]
fig = plt.figure(figsize=(20,25))
for i,data in enumerate(datum):
    ax = fig.add_subplot(int(str(32)+str(i+1)))
    null_pct = data.isnull().sum()
    sns.barplot(null_pct.values,null_pct.index,ax = ax)

    for idx,value in enumerate(null_pct.values):
        ax.text(-0.055,idx,"{:.2f}%".format(float(value)/len(data)*100),fontsize=18)
    ax.set_xlabel("number of missing data",fontsize=18)
    ax.set_ylabel("Column Name",fontsize=18)
    ax.set_title("Missing data for {} dataframe".format(names[i]),fontsize=20)
    
plt.tight_layout()
plt.figure(figsize=(8,8))
threshold = 500

temp = donations[["Donation Amount"]].dropna(inplace=False)
temp["large_amount"] = temp["Donation Amount"] > threshold
temp = temp.groupby(["large_amount"]).count().reset_index()
temp['Donation Amount'].plot(kind='pie',autopct="%.2f%%",labels=['donation <= ${}'.format(threshold),'donation > ${}'.format(threshold)])
plt.ylabel("")
plt.figure(figsize=(15,8))
temp = donations[donations["Donation Amount"] <= threshold]['Donation Amount']
ax = sns.distplot(temp,bins=50,kde=False)
ax.set_xlim([-1,threshold])
plt.xlabel("Donation amounts",fontsize=18)
plt.title("Distribution of donation amounts",fontsize=20)
plt.figure(figsize=(8,8))
donation_optional = donations['Donation Included Optional Donation'].value_counts().reset_index()
donation_optional['Donation Included Optional Donation'].plot(kind='pie',autopct='%.2f%%',labels=['Yes','No'])
plt.ylabel("")
plt.title("Donation Included Optional Donation",fontsize=14)
top = 20
plt.figure(figsize=(18,8))
donation_cart = donations['Donor Cart Sequence'].dropna().value_counts().reset_index()[:top]
sns.barplot(x='index',y='Donor Cart Sequence',data=donation_cart)
for i,v in enumerate(donation_cart['Donor Cart Sequence'].values):
    plt.text(i-0.5,0,"{:.2f}%".format(float(v)/donation_cart['Donor Cart Sequence'].values.sum()*100),fontsize=14)
#plt.xticks(rotation=90)
plt.ylabel("")
plt.xlabel("Donor Cart Sequence",fontsize=18)
plt.title("Top {} Donor Cart Sequence".format(top),fontsize=20)
print("Number of unique donations = {}".format(donations['Donation ID'].nunique()))
print("Number of donations = {}".format(len(donations)))
print("Number of unique donations = {}".format(donations['Donor ID'].nunique()))
print("Number of donations = {}".format(len(donations)))
temp = donations['Donor ID'].value_counts()
print("Number of donors donating once only = {}".format(sum(temp==1)))
print("Number of donors donating more than 1000 times = {}".format(sum(temp > 1000)))
print("Number of unique projects = {}".format(donations['Project ID'].nunique()))
print("Number of projects recorded in Donations = {}".format(len(donations)))
temp = donations['Project ID'].value_counts().sort_values()
plt.figure(figsize=(15,6))
temp.plot(kind='hist',bins=100)
print("Number of unique projects = {}".format(projects['Project ID'].nunique()))
print("Number of total projects in projects dataframe = {}".format(len(projects)))
temp = projects['Project ID'].value_counts().sort_values(ascending=False)
ids = []
for i,v in enumerate(temp):
    if v > 1: ids.append(temp.index[i])
print(projects[projects['Project ID'].isin(ids)])
redundant_ids = ["99c07777fdcf63d3a0fdb4a0deb4b012","c940d0e78b7559573aca536db90c0646"]
projects = projects[~(projects['Project ID'].isin(redundant_ids) & 
                      (projects['Project Type'] == 'Professional Development'))]
print("Number of unique schools = {}".format(projects['School ID'].nunique()))
print("Number of projects = {}".format(len(projects)))
temp = projects['School ID'].value_counts().sort_values()
print("Number of schools with only one project = {}".format(sum(temp==1)))
print("Number of schools with more than 250 projects = {}".format(sum(temp>250)))
print("Number of schools = {}".format(len(temp)))
data = temp[temp <= 250]
plt.figure(figsize=(15,6))
data.plot(kind='hist',bins=100)
plt.xlim([0,250])
print("Number of unique teachers = {}".format(projects['Teacher ID'].nunique()))
print("Number of projects = {}".format(len(projects)))
temp = projects['Teacher ID'].value_counts().sort_values(ascending=False)
print(temp.head())
print(temp.tail())
print("Number of teachers with only one project = {}".format(sum(temp==1)))
plt.figure(figsize=(15,6))
temp.plot(kind='hist',bins=50)
top = 20
temp = projects["Teacher Project Posted Sequence"].value_counts().sort_values(ascending=False)[:top]
plt.figure(figsize=(18,8))
sns.barplot(temp.index,temp.values)
plt.ylabel("")
plt.xlabel("Teacher Project Posted Sequence",fontsize=18)
plt.title("Top {} Teacher Project Posted Sequence".format(top),fontsize=20)
total = temp.sum()
for i,v in enumerate(temp):
    plt.text(i-0.5,0,"{:.2f}%".format(float(v)/total*100))
temp = projects["Project Type"].value_counts().sort_values(ascending=False).reset_index()
plt.figure(figsize=(9,9))
temp['Project Type'].plot(kind='pie',autopct='%.2f%%',labels=temp['index'])
plt.ylabel("")
plt.xlabel("")
plt.title("Distribution of Project Types",fontsize=18)
temp = projects['Project Subject Category Tree'].value_counts().sort_values(ascending=False).reset_index()
plt.figure(figsize=(15,20))
sns.barplot(x='Project Subject Category Tree',y='index',data=temp)
total = temp['Project Subject Category Tree'].sum()
plt.xlabel("")
plt.ylabel("Main Categories",fontsize=18)
plt.title("Distribution of project main categories",fontsize=20)
for i,v in enumerate(temp['Project Subject Category Tree']):
    plt.text(0,i,"{:.2f}%".format(float(v)/total*100),fontsize=14)
top = 20
temp = projects['Project Subject Subcategory Tree'].value_counts().sort_values(ascending=False).reset_index()[:top]
plt.figure(figsize=(15,8))
sns.barplot(x='Project Subject Subcategory Tree',y='index',data=temp)
total = temp['Project Subject Subcategory Tree'].sum()
plt.xlabel("")
plt.ylabel("Project Subcategories",fontsize=18)
plt.title("Top {} Project Subject Subcategory".format(top),fontsize=20)
for i,v in enumerate(temp['Project Subject Subcategory Tree']):
    plt.text(0,i,"{:.2f}%".format(float(v)/total*100),fontsize=14)
temp = projects['Project Grade Level Category'].value_counts().sort_values(ascending=False).reset_index()
plt.figure(figsize=(8,8))
temp['Project Grade Level Category'].plot(kind='pie',autopct='%.2f%%',labels=temp['index'])
plt.title("Distribution of Project Grade Level Category",fontsize=14)
plt.ylabel("")
temp = projects['Project Resource Category'].value_counts().sort_values(ascending=False).reset_index()
plt.figure(figsize=(15,8))
sns.barplot(x='Project Resource Category',y='index',data=temp)
total = temp['Project Resource Category'].sum()
plt.xlabel("")
plt.ylabel("Project Resource Category",fontsize=18)
plt.title("Distribution of Project Resource Category",fontsize=20)
for i,v in enumerate(temp['Project Resource Category']):
    plt.text(0,i,"{:.2f}%".format(float(v)/total*100),fontsize=14)
threshold = 3000
temp = projects[['Project Cost']]
plt.figure(figsize=(15,8))
sns.distplot(temp[temp['Project Cost'] < threshold],bins=100,kde=False)
plt.xlim([0,3001])
plt.title("Distribution of project costs",fontsize=20)
temp = projects['Project Current Status'].value_counts().sort_values(ascending=False).reset_index()[:20]
plt.figure(figsize=(8,8))
temp['Project Current Status'].plot(kind='pie',autopct='%.2f%%',labels=temp['index'])
plt.xlabel("")
plt.ylabel("")
plt.title("Distribution of Project Current Status",fontsize=14)
print("Number of unique donors = {}".format(donors['Donor ID'].nunique()))
print("Number of donors = {}".format(len(donors)))
us_state_abbrev = {
    'Alabama': 'AL',
    'Alaska': 'AK',
    'Arizona': 'AZ',
    'Arkansas': 'AR',
    'California': 'CA',
    'Colorado': 'CO',
    'Connecticut': 'CT',
    'District of Columbia': 'DC',
    'Delaware': 'DE',
    'Florida': 'FL',
    'Georgia': 'GA',
    'Hawaii': 'HI',
    'Idaho': 'ID',
    'Illinois': 'IL',
    'Indiana': 'IN',
    'Iowa': 'IA',
    'Kansas': 'KS',
    'Kentucky': 'KY',
    'Louisiana': 'LA',
    'Maine': 'ME',
    'Maryland': 'MD',
    'Massachusetts': 'MA',
    'Michigan': 'MI',
    'Minnesota': 'MN',
    'Mississippi': 'MS',
    'Missouri': 'MO',
    'Montana': 'MT',
    'Nebraska': 'NE',
    'Nevada': 'NV',
    'New Hampshire': 'NH',
    'New Jersey': 'NJ',
    'New Mexico': 'NM',
    'New York': 'NY',
    'North Carolina': 'NC',
    'North Dakota': 'ND',
    'Ohio': 'OH',
    'Oklahoma': 'OK',
    'Oregon': 'OR',
    'Pennsylvania': 'PA',
    'Rhode Island': 'RI',
    'South Carolina': 'SC',
    'South Dakota': 'SD',
    'Tennessee': 'TN',
    'Texas': 'TX',
    'Utah': 'UT',
    'Vermont': 'VT',
    'Virginia': 'VA',
    'Washington': 'WA',
    'West Virginia': 'WV',
    'Wisconsin': 'WI',
    'Wyoming': 'WY',
}
donors_state = donors['Donor State'].value_counts().drop('other',axis=0)
data = [dict(
        type = "choropleth",
        #locations = donors_state.index.str,
        locations = donors_state.index.map(lambda x: us_state_abbrev[x]),
        locationmode = 'USA-states',
        z = donors_state.values,
        #text = donors_state.index,
        colorscale = 'Red',
        marker = dict(line=dict(width=0.7)),
        colorbar = dict(title='Number of Donors')
    )]
layout = dict(
        title = 'Number of Donor Distribution Among States',
        geo = dict(
            scope='usa',
            showframe = False,
            showcoastlines = True,
            projection=dict( type='albers usa' ),
        )
    )
fig = dict(data=data, layout=layout)
offline.iplot(fig)
temp = donors['Donor State'].value_counts().sort_values(ascending=False).reset_index()
plt.figure(figsize=(10,12))
sns.barplot(y='index',x='Donor State',data=temp)
plt.xlabel("")
plt.ylabel("")
donor_teacher = donors['Donor Is Teacher'].value_counts().reset_index()
plt.figure(figsize=(8,8))
donor_teacher['Donor Is Teacher'].plot(kind='pie',autopct='%.2f%%',labels=donor_teacher['index'])
plt.xlabel("")
plt.ylabel("")
plt.title("Donor Is Teacher",fontsize=14)
donors_location = donors[donors['Donor State'] != 'other'][['Donor City','Donor State']]
donors_location = donors_location.loc[~donors_location['Donor City'].isnull(),]
donors_city = donors_location['Donor City'].astype(str) + ', ' + donors_location['Donor State'].astype(str)
donors_city = donors_city.value_counts().sort_values(ascending=False)[:100].to_frame()
donors_city.columns = ['Count']
# Get the location of each city and plot it
#geolocator = Nominatim()
#for city in donors_city.index:
#    loc = geolocator.geocode(city)
#    if loc:
#        donors_city.loc[city,'lon'] = loc.longitude
#        donors_city.loc[city,'lat'] = loc.latitude
#        time.sleep(1) # we cannot query geolocator too frequently

# we load the longitude and latitude of cities directly, which comes from the above commented codes
donors_city['lon'] = [ 
    -87.6244212,   -73.9866136,   -73.9495823,   -84.11721147,  -73.3676149,
    -95.3676974,  -122.3300624,   -75.1635755,   -84.3901849,  -122.6741949,
    -86.1583502,   -97.7436995,   -80.8431268,   -77.0366456,   -80.1936589,
    -117.1627714,  -112.0773456,   -93.2654692,  -104.984696,    -96.7968559,
    -84.0665224,   -97.5103397,   -90.1978889,  -122.2713563,   -98.4951405,
    -115.149225,    -82.458444,    -79.9900861,   -85.759407,    -81.3794368,
    -78.6390989,   -93.1015026,   -76.610759,    -87.922497,    -71.0595678,
    -121.4943996,   -81.655651,    -83.0007065,   -84.5124602,   -74.1496048,
    -73.83669616,  -97.3327459,   -80.1433786,   -90.0516285,   -94.5630298,
    -77.43428,     -78.9018115,  -111.9783931,  -119.0194639,  -110.9262353,
    -86.7743531,   -77.09024765,  -81.6934446,   -89.3837613,   -77.0841585,
    -118.15804932,  -95.9929113,   -89.97500545, -106.6509851,   -86.8024326,
    -82.3984882,   -95.2621553,  -122.2728639,  -114.9819235,   -79.9402728,
    -79.7919754,   -94.63275393,  -78.7811925,   -95.8243956,  -117.8259819,
    -80.2440518,  -121.9885719,   -75.9774183,   -81.0998342,   -85.6678639,
    -95.4172549,  -105.0166498,   -81.0343313,  -111.58606618,  -95.9378732,
    -111.8992365,   -80.0533746,   -84.5496148,  -122.064963,    -78.8783922,
    -80.1247667,  -118.0000166,  -104.8253485,   -91.154551,    -84.2747329,
    -80.1494901,   -83.9210261,  -117.1884542,   -84.1916069,  -121.8746789,
    -84.4970393,  -121.9999606,  -149.8948523,  -118.3406288,  -122.4886034 ]
donors_city['lat'] = [
    41.8755546,  40.7306458,  40.6501038,   9.9970987,   8.6545394,  29.7589382,
    47.6038321,  39.9524152,  33.7490987,  45.5202471,  39.7683331,  30.2711286,
    35.2270869,  38.8949549,  25.7742658,  32.7174209,  33.4485866,  44.9772995,
    39.7391428,  32.7762719,   9.9327612,  35.5377266,  38.6272733,  37.8044557,
    29.4246002,  36.1662859,  27.9477595,  40.4416941,  38.2542376,  28.5423999,
    35.7803977,  44.9504037,  39.2908816,  43.0349931,  42.3604823,  38.5815719,
    30.3321838,  39.9622601,  39.1014537,  40.5834557,  40.85703325, 32.753177,
    26.1223084,  35.1490215,  39.0844687,  37.5385087,  35.9966551,  40.7879394,
    35.3738712,  32.2218917,  36.1622296,  38.8147596,  41.5051613,  43.074761,
    38.8903961,  33.78538945, 36.1556805,  30.03280175, 35.0841034,  33.5206824,
    34.851354,   29.9988312,  37.8708393,  36.0391456,  32.7876012,  36.0726355,
    38.99134745, 35.7882973,  29.7857853,  33.6856969,  36.0998131,  37.5482697,
    36.8529841,  32.0835407,  42.9632405,  30.0798826,  39.613321,   34.0007493,
    33.436188,   41.2587317,  33.5091215,  26.7153425,  33.9528472,  37.9063131,
    42.8867166,  26.2378597,  33.6783336,  38.8339578,  30.4507462,  34.0709576,
    26.0112014,  35.9603948,  33.5777524,  39.7589478,  37.6624312,  38.0464066,
    37.8215929,  61.2163129,  33.8358492,  37.5053381 ]
text = donors_city.index.astype(str) + ', Number of donors: ' + donors_city.Count.astype(str)
data = [dict(
        type = 'scattergeo',
        locationmode = 'USA-states',
        lon = donors_city.lon,
        lat = donors_city.lat,
        text = text,
        mode = 'markers',
        marker = dict(
            size = donors_city.Count.div(1000),
            opacity = 0.8,
            reversescale = True,
            autocolorscale = False,
            #symbol = 'square',
            line = dict(
                width=1,
                color='rgba(102, 102, 102)'
            ),
            cmin = 0,
            color = donors_city.Count,
            cmax = max(donors_city.Count),
            colorbar=dict(
                title="Number of Donors"
            )
        ))]

layout = dict(
        title = 'Distribution of Donors Across Cities',
        geo = dict(
            scope='usa',
            projection=dict( type='albers usa' ),
            showland = True,
            landcolor = "rgb(250, 250, 250)",
            subunitcolor = "rgb(217, 217, 217)",
            countrycolor = "rgb(217, 217, 217)",
            countrywidth = 0.5,
            subunitwidth = 0.5
        ),
    )

fig = dict(data=data, layout=layout)
offline.iplot(fig)
print("Number of unique projects in resources = {}".format(resources['Project ID'].nunique()))
print("Number of resource records = {}".format(len(resources)))
temp = resources['Project ID'].value_counts().sort_values(ascending=False)
plt.figure(figsize=(15,6))
temp.plot(kind='hist',bins=100)
temp = resources[~resources['Resource Quantity'].isnull() & ~resources['Resource Unit Price'].isnull()][['Resource Quantity','Resource Unit Price']]
temp['Resource Quantity'] = temp['Resource Quantity'].astype(float)
temp['Resource Unit Price'] = temp['Resource Unit Price'].astype(float)
temp['total_cost'] = temp['Resource Quantity'].mul(temp['Resource Unit Price'])

# 
threshold = 1000
print("Number of resources is {}".format(len(temp)))
print("Number of resources greater than {} is {}".format(threshold,(temp['total_cost'] > threshold).sum()))
temp = temp[temp['total_cost'] <= 1000]
plt.figure(figsize=(15,8))
sns.distplot(temp['total_cost'],bins=20,kde=False)
plt.xlabel("Resource Costs",fontsize=18)
plt.title("Distribution of Resource Item Costs",fontsize=20)
top = 20
temp = resources[~resources['Resource Vendor Name'].isnull()]
temp.loc[:,'total_costs'] = temp.loc[:,'Resource Quantity'].mul(temp.loc[:,'Resource Unit Price'])
data = temp.groupby('Resource Vendor Name')['total_costs'].sum().sort_values(ascending=False)[:20].reset_index()
plt.figure(figsize=(12,8))
ax = sns.barplot(y='Resource Vendor Name',x='total_costs',data=data)
plt.title("Top {} Resource Vendors".format(top),fontsize=20,y=1.05)
plt.ylabel("Vendor Name",fontsize=18)
plt.xlabel("")
ax.xaxis.set_label_position('top') 
ax.xaxis.tick_top()
plt.tight_layout()
print("Number of unique schools = {}".format(schools['School ID'].nunique()))
print("Size of School dataframe = {}".format(len(schools)))
school_type = schools['School Metro Type'].value_counts().reset_index()
plt.figure(figsize=(8,8))
school_type['School Metro Type'].plot(kind='pie',autopct='%.2f%%',labels=school_type['index'])
plt.title("School Metro Type",fontsize=14)
plt.xlabel("")
plt.ylabel("")
plt.figure(figsize=(15,6))
sns.distplot(schools["School Percentage Free Lunch"].dropna(),bins=20,kde=False)
plt.title("Distribution of School Percentage Free Lunch",fontsize=16)
plt.xlabel("")
school_state = schools['School State'].dropna().value_counts()
data = [dict(
        type = "choropleth",
        #locations = donors_state.index.str,
        locations = school_state.index.map(lambda x: us_state_abbrev[x]),
        locationmode = 'USA-states',
        z = school_state.values,
        colorscale = 'Blue',
        marker = dict(line=dict(width=0.7)),
        colorbar = dict(title='Number of Schools')
    )]
layout = dict(
        title = 'Number of Schools Among States',
        geo = dict(
            scope='usa',
            showframe = False,
            showcoastlines = True,
            projection=dict( type='albers usa' ),
        )
    )
fig = dict(data=data, layout=layout)
offline.iplot(fig)
print("Number of unique teacher ID = {}".format(teachers['Teacher ID'].nunique()))
print("Size of Teacher dataframe = {}".format(len(teachers)))
teacher_gender = teachers['Teacher Prefix'].dropna().value_counts().reset_index()
plt.figure(figsize=(15,6))
sns.barplot(x='index',y='Teacher Prefix',data=teacher_gender)
plt.xlabel("")
plt.ylabel("")
total = teacher_gender['Teacher Prefix'].sum()
plt.title("Distribution of Teacher Prefix",fontsize=18)
for i,v in enumerate(teacher_gender['Teacher Prefix']):
    plt.text(i-0.2,0,"{:.2f}%".format(float(v)/total*100),fontsize=14)
simplified_donation = donations[['Donation Amount','Project ID']].groupby("Project ID").sum().reset_index()
merged = projects.merge(simplified_donation,
                       how='inner',
                       on='Project ID')
merged['Donation Amount'] = merged['Donation Amount'].astype(float)
merged['funding_gap'] = merged['Project Cost'].subtract(merged['Donation Amount'])
merged.head()
temp = merged[merged['Project Current Status'] == 'Fully Funded'][['Project Current Status','funding_gap']]
temp['well_funded'] = (temp['funding_gap'] <= 0)
counts = temp['well_funded'].value_counts().reset_index()
total = counts['well_funded'].sum()
plt.figure(figsize=(10,6))
sns.barplot(x='index',y='well_funded',data=counts)
plt.xlabel("Projects get fully funded from donations",fontsize=16)
plt.ylabel("")
plt.title("Distribution of Fully Funded Projects",fontsize=16)
for i,v in enumerate(counts['well_funded']):
    plt.text(i-0.1,0,"{:.2f}%".format(float(v)/total*100),fontsize=16)
lb,ub = 0,1000
temp = merged[(lb <= merged['funding_gap']) & (merged['funding_gap'] < ub)]['funding_gap']
plt.figure(figsize=(15,8))
sns.distplot(temp,bins = 50,kde=False)
plt.xlim([0,1000])
plt.xlabel("Amount of funding gaps",fontsize=18)
plt.title("Distribution of funding gaps between {} and {}".format(lb,ub),fontsize=20)
simplified_src = resources[['Project ID','Resource Quantity','Resource Unit Price']]
simplified_src.loc[:,'resource_prices'] = simplified_src['Resource Quantity'].mul(simplified_src['Resource Unit Price'])
simplified_src = simplified_src[['Project ID','resource_prices']].groupby('Project ID').sum().reset_index()
merged = projects.merge(simplified_src,
                       how = 'inner',
                       on = 'Project ID')
merged['excess_cost'] = merged['Project Cost'].subtract(merged['resource_prices'])
merged['have_excess_cost'] = (merged['excess_cost'] > 0)
temp = merged['have_excess_cost'].value_counts().reset_index()
total = temp['have_excess_cost'].sum()
plt.figure(figsize=(10,6))
sns.barplot(x='index',y='have_excess_cost',data=temp,order=[True,False])
plt.title("Projects Having Costs Besides Resources",fontsize=18)
plt.xlabel("")
plt.ylabel("")
for i,v in enumerate(temp['have_excess_cost']):
    plt.text(i-0.1,0,"{:.2f}%".format(float(v)/total*100),fontsize=16)
data = donations.groupby('Donation Included Optional Donation')['Donation Amount'].agg(['sum','mean']).reset_index()
fig = plt.figure(figsize=(15,6))
ax1 = fig.add_subplot("121")
sns.barplot(x='Donation Included Optional Donation',y='sum',data=data,ax=ax1)
ax1.set_ylabel("")
ax1.set_xlabel("")
ax1.set_title("Sum")
ax2 = fig.add_subplot("122")
sns.barplot(x='Donation Included Optional Donation',y='mean',data=data,ax=ax2)
ax2.set_ylabel("")
ax2.set_xlabel("")
ax2.set_title("Mean")
donation_donor = donations.merge(donors,how='left',on='Donor ID')
data = donation_donor.groupby('Donor Is Teacher')['Donation Amount'].agg(['mean']).reset_index()
plt.figure(figsize=(10,6))
sns.barplot(x='Donor Is Teacher',y='mean',data=data)
plt.ylabel("")
plt.xlabel("")
plt.title("Mean")
project_teacher = projects.merge(schools,how='left',on='School ID')
donation_teacher = donations.merge(project_teacher,how='left',on='Project ID')
data = donation_teacher.groupby('School Metro Type')['Donation Amount'].agg(['sum','mean']).reset_index()
fig = plt.figure(figsize=(15,7))
ax1 = fig.add_subplot(121)
ax1 = data['sum'].plot(kind='pie',autopct='%.2f%%',labels=data['School Metro Type'],ax=ax1)
ax1.set_ylabel("")
ax1.set_xlabel("")
ax1.set_title("Sum")
ax2 = fig.add_subplot(122)
sns.barplot(x='School Metro Type',y='mean',data=data,ax=ax2)
ax2.set_ylabel("Amounts",fontsize=14)
ax2.set_xlabel("")
ax2.set_title("Mean")
left = pd.to_datetime(projects['Project Posted Date']).to_frame()
left.loc[:,'posted_count'] = 1
left.sort_values(by='Project Posted Date',inplace=True)
left.set_index(keys='Project Posted Date',inplace=True)
left = left.resample('W').count()
left.set_index(left.index.to_period(freq='W'),inplace=True)

right = pd.to_datetime(projects['Project Expiration Date']).to_frame()
right.loc[:,'expired_count'] = 1
right.sort_values(by='Project Expiration Date',inplace=True)
right.set_index(keys='Project Expiration Date',inplace=True)
right = right.resample('W').count()
right.set_index(right.index.to_period(freq='W'),inplace=True)

merged = left.merge(right,left_index=True,right_index=True)
merged.head()
merged.plot(figsize=(15,8))
plt.xlabel('Time',fontsize=16)
plt.ylabel('Number of cases per week',fontsize=16)
plt.legend(['Posted','Expired'])
plt.title("Trend of Posted and Expired Projects",fontsize=20)
temp = pd.to_datetime(projects['Project Fully Funded Date']).to_frame()
temp.loc[:,'count'] = 1
temp.sort_values(by='Project Fully Funded Date',inplace=True)
temp.set_index(keys='Project Fully Funded Date',inplace=True)
temp = temp.resample('W').count()
temp.set_index(temp.index.to_period(freq='W'),inplace=True)
temp.plot(figsize=(15,8))
plt.xlabel('Time',fontsize=18)
plt.ylabel('Number of cases per week',fontsize=18)
plt.legend(['Fully Funded'])
plt.title("Trend of Fully Funded Projects",fontsize=20)
temp = pd.to_datetime(teachers['Teacher First Project Posted Date']).to_frame()
temp.loc[:,'count'] = 1
temp.sort_values(by='Teacher First Project Posted Date',inplace=True)
temp.set_index(keys='Teacher First Project Posted Date',inplace=True)
#temp.head()
temp = temp.resample('W').count()
temp.set_index(temp.index.to_period(freq='W'),inplace=True)
temp.plot(figsize=(15,8))
plt.xlabel('Time',fontsize=18)
plt.ylabel('Number of teachers per week',fontsize=18)
plt.title("Trend of Teacher First Project Posted Date",fontsize=20)
temp = donations.loc[:,['Donation Amount','Donation Received Date']]
temp.loc[:,'Donation Received Date'] = pd.to_datetime(temp['Donation Received Date'])
temp.sort_values(by='Donation Received Date',inplace=True)
temp.set_index(keys='Donation Received Date',inplace=True)
temp = temp.resample('W').agg(['sum','count'])
temp.set_index(temp.index.to_period(freq='W'),inplace=True)
temp.columns = temp.columns.droplevel(0)
ax1 = temp['sum'].plot(figsize=(15,8))
plt.xlabel('Time',fontsize=18)
plt.ylabel('Donation amounts per week',fontsize=18)
plt.title("Trend of Donations Amounts",fontsize=20)
ax1 = temp['count'].plot(figsize=(15,8))
plt.xlabel('Time',fontsize=18)
plt.ylabel('Number of Donations per week',fontsize=18)
plt.title("Trend of Donations Counts",fontsize=20)
temp = donations[['Donation Amount','Donation Received Date']]
temp.loc[:,'Donation Received Date'] = pd.to_datetime(temp['Donation Received Date'])
temp.loc[:,'Month'] = temp['Donation Received Date'].dt.month
count_mean = temp.groupby('Month')['Donation Amount'].agg(['count','mean']).sort_index()
count_mean.rename(columns={'count':'Counts','mean':'Average'},inplace=True)
fig = plt.figure(figsize=(10,12))
for idx,col in enumerate(count_mean.columns):
    ax = fig.add_subplot(int(str(21)+str(idx+1)))
    sns.barplot(x=count_mean.index,y=col,data=count_mean,ax=ax)
    ax.set_title("Distribution of Donation {} Among Months".format(col),fontsize=16)
    plt.ylabel("")
temp = donations[['Donation Amount','Donation Received Date']]
temp.loc[:,'Donation Received Date'] = pd.to_datetime(temp['Donation Received Date'])
temp.loc[:,'Month'] = temp['Donation Received Date'].dt.weekday
count_mean = temp.groupby('Month')['Donation Amount'].agg(['count','mean']).sort_index()
count_mean.rename(columns={'count':'Counts','mean':'Average'})
count_mean.rename(index={0:'Monday',1:'Tuesday',2:'Wednesday',3:'Thursday',4:'Friday',5:'Saturday',6:'Sunday'},inplace=True)
fig = plt.figure(figsize=(10,12))
for idx,col in enumerate(count_mean.columns):
    ax = fig.add_subplot(int(str(21)+str(idx+1)))
    sns.barplot(count_mean.index,count_mean[col],ax=ax)
    ax.set_title("Distribution of Donation {} Among Weekdays".format(col),fontsize=16)
    plt.ylabel("")
    plt.xlabel("")
temp = pd.to_datetime(projects['Project Posted Date']).to_frame()
temp.loc[:,'count'] = 1
temp.sort_values(by='Project Posted Date',inplace=True)
temp.set_index(keys='Project Posted Date',inplace=True)
temp = temp.resample('W').count()
temp.set_index(temp.index.to_period(freq='W'),inplace=True)
ax1 = temp['count'].plot(figsize=(20,8))
mean = temp['count'].rolling(window=10).mean()
mean.plot(ax=ax1,color='r')
plt.legend(['original','rolling mean'])
plt.xlabel("")
shifted = 52
temp.loc[:,'shifted_count'] = temp['count'].shift(shifted)
temp.loc[:,'diff_count'] = temp['count'].subtract(temp['shifted_count'])
temp.diff_count.plot(figsize=(20,8))
plt.xlabel("")
test = adfuller(temp.diff_count.dropna(), autolag='AIC')
output = pd.Series(test[0:4], index=['Test Statistic','p-value','#Lags','Number of Observations Used'])
for k,v in test[4].items():
    output['Critical Value (%s)'%k] = v
print(output)
fig = plt.figure(figsize=(10,10))
ax1 = fig.add_subplot(211)
fig = plot_acf(temp.diff_count.dropna(), lags=100, ax=ax1)
ax2 = fig.add_subplot(212)
fig = plot_pacf(temp.diff_count.dropna(), lags=100, ax=ax2)
training = np.array(temp.diff_count.dropna())
test_len = 20
data = [x for x in training[:len(training)-test_len]]
predictions = []
for i in range(test_len):
    try:
        model = ARIMA(data, order=(1,2,0))
        ret = model.fit(disp=0)
        forecast = ret.forecast()
        prediction = forecast[0]
        predictions.append(prediction)
        data.append(testing[i])
    except:
        continue
        
# our prediction is shifted difference, we recover it by adding with the original series
values = temp['count'].values
for i in range(test_len):
    predictions[i] += values[len(temp)-shifted-test_len+i]
        
plt.figure(figsize=(15,6))
plt.plot(values)
plt.plot(pd.Series(predictions,index=range(len(temp)-test_len,len(temp))), color='red')
plt.legend(['Original','Prediction'])
temp = pd.to_datetime(projects['Project Expiration Date']).to_frame()
temp.loc[:,'count'] = 1
temp.sort_values(by='Project Expiration Date',inplace=True)
temp.set_index(keys='Project Expiration Date',inplace=True)
temp = temp.resample('W').count()
temp.set_index(temp.index.to_period(freq='W'),inplace=True)
ax1 = temp['count'].plot(figsize=(20,8))
mean = temp['count'].rolling(window=10).mean()
mean.plot(ax=ax1,color='r')
plt.legend(['original','rolling mean'])
plt.xlabel("")
temp = temp.iloc[:-18,]
ax1 = temp['count'].plot(figsize=(20,8))
mean = temp['count'].rolling(window=10).mean()
mean.plot(ax=ax1,color='r')
plt.legend(['original','rolling mean'])
plt.xlabel("")
counts = temp['count'].values
fig = plt.figure(figsize=(20,10))
ax1 = fig.add_subplot(211)
fig = plot_acf(counts, lags=110, ax=ax1)
ax2 = fig.add_subplot(212)
fig = plot_pacf(counts, lags=110, ax=ax2)
de_seasonality = temp['count'].subtract(temp['count'].shift(52)).dropna()
fig = plt.figure(figsize=(20,15))
ax1 = fig.add_subplot(311)
de_seasonality.plot(ax=ax1)
ax2 = fig.add_subplot(312)
fig = plot_acf(de_seasonality, lags=110, ax=ax2)
ax3 = fig.add_subplot(313)
fig = plot_pacf(de_seasonality, lags=110, ax=ax3)

# stationarity test
test = adfuller(de_seasonality, autolag='AIC')
output = pd.Series(test[0:4], index=['Test Statistic','p-value','#Lags','Number of Observations Used'])
for k,v in test[4].items():
    output['Critical Value (%s)'%k] = v
print(output)
test_len = 30

# Since there is a spike at delay=1 for both ACF and PACF plot with de-seasoned series, we set order=(1,0,1)
# For the seasonal component, we try the most common seasonal_order=(0,1,1,period).
# You can use grid search to find best seasonal order.
model = SARIMAX(counts, trend='t', order=(1,0,1), seasonal_order=(0,1,1,52))
ret = model.fit()
predictions = ret.predict(start = len(counts)-test_len, end = len(counts)-1, dynamic = True)  
        
plt.figure(figsize=(15,6))
plt.plot(counts)
plt.plot(pd.Series(predictions,index=range(len(counts)-test_len,len(counts))), color='red')
plt.legend(['Original','Prediction'])
temp = pd.to_datetime(teachers['Teacher First Project Posted Date']).to_frame()
temp.loc[:,'count'] = 1
temp.sort_values(by='Teacher First Project Posted Date',inplace=True)
temp.set_index(keys='Teacher First Project Posted Date',inplace=True)
temp = temp.resample('W').count().dropna()
data = temp.loc['2013':,:]

# Plot the original time series, ACF and PACF
fig = plt.figure(figsize=(20,15))
ax1 = fig.add_subplot(311)
data.plot(ax=ax1)
ax2 = fig.add_subplot(312)
fig = plot_acf(data, lags=110, ax=ax2)
ax3 = fig.add_subplot(313)
fig = plot_pacf(data, lags=110, ax=ax3)
# de-season by taking shifted difference
de_seasonality = temp['count'].subtract(temp['count'].shift(52)).dropna()

# stationarity test
test = adfuller(de_seasonality, autolag='AIC')
output = pd.Series(test[0:4], index=['Test Statistic','p-value','#Lags','Number of Observations Used'])
for k,v in test[4].items():
    output['Critical Value (%s)'%k] = v
print(output)
plt.figure(figsize=(15,6))
de_seasonality.plot()
# detrend the sequence by taking one more diff
de_trend = de_seasonality.diff().dropna()
# stationarity test
test = adfuller(de_trend, autolag='AIC')
output = pd.Series(test[0:4], index=['Test Statistic','p-value','#Lags','Number of Observations Used'])
for k,v in test[4].items():
    output['Critical Value (%s)'%k] = v
print(output)
fig = plt.figure(figsize=(20,15))
ax1 = fig.add_subplot(311)
de_trend.plot(ax=ax1)
ax2 = fig.add_subplot(312)
fig = plot_acf(de_trend, lags=110, ax=ax2)
ax3 = fig.add_subplot(313)
fig = plot_pacf(de_trend, lags=110, ax=ax3)
test_len = 30

# There is no significant spike after delay=2 in PACF and after delay=1 in ACF, so choose (2,d,1)
model = SARIMAX(temp['count'].values, trend='t', order=(2,2,1), seasonal_order=(0,1,0,52))
ret = model.fit()
predictions = ret.predict(start = len(temp)-test_len, end = len(temp)-1, dynamic = True)  
        
plt.figure(figsize=(20,6))
plt.plot(temp['count'].values)
plt.plot(pd.Series(predictions,index=range(len(temp)-test_len,len(temp))), color='red')
plt.legend(['Original','Prediction'])
# parameters
time_step = 52
batch_size = 1
test_size = 10
epochs = 100

# create dataset for LSTM
data = donations.loc[:,['Donation Amount','Donation Received Date']]
data.loc[:,'Donation Received Date'] = pd.to_datetime(data.loc[:,'Donation Received Date'])
data.sort_values(by='Donation Received Date',inplace=True)
data.set_index(keys='Donation Received Date',inplace=True)
data = data.resample('W').sum()
data.set_index(data.index.to_period(freq='W'),inplace=True)

X, y = [], []
temp = data.values
scaler = MinMaxScaler(feature_range=(0, 1))
temp = scaler.fit_transform(temp)
for i in range(len(temp)-time_step):
    X.append(temp[i:(i+time_step),0])
    y.append(temp[(i+time_step),0])
X,y = np.array(X), np.array(y)
X_train, X_test = X[:(X.shape[0]-test_size),:], X[(X.shape[0]-test_size):,:]
y_train, y_test = y[:(y.shape[0]-test_size)], y[(y.shape[0]-test_size):]
X_train = np.reshape(X_train,(X_train.shape[0],X_train.shape[1],1))
X_test = np.reshape(X_test,(X_test.shape[0],X_test.shape[1],1))

# build LSTM model
model = Sequential()
model.add(LSTM(12,
               batch_input_shape = (batch_size,X_train.shape[1],1),
               stateful = True))
model.add(Dense(1,activation='sigmoid'))
model.compile(loss='mean_squared_error', optimizer='adam')

# fitting model
for i in range(epochs):
    model.fit(X_train, y_train, epochs=1, batch_size=1, verbose=0, shuffle=False)
    model.reset_states()
    
# forecast
prediction = model.predict(X_test, batch_size)
prediction = scaler.inverse_transform(prediction)
y = scaler.inverse_transform([y])
y = y.flatten()

# plot prediction
plt.figure(figsize=(15,8))
plt.plot(y)
plt.plot([None]*(len(y)-test_size) + [x[0] for x in prediction],color='red')
