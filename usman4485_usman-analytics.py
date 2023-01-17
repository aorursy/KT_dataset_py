# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
#SchoolData = pd.read_csv('/kaggle/input/io/Schools.csv')
DonationData = pd.read_csv('/kaggle/input/io/Donations.csv')
DonorsData = pd.read_csv('/kaggle/input/io/Donors.csv')
#ProjectsData = pd.read_csv('/kaggle/input/io/Projects.csv')
TeachersData = pd.read_csv('/kaggle/input/io/Teachers.csv')
#ResourcesData = pd.read_csv('/kaggle/input/io/Resources.csv')
DonationData.head()
DonorsData.head()
TeachersData.head()
#Count each donar Donations
DonorsData['Donor City'].value_counts()
DonorsData['Donor ID'].value_counts().to_frame()
donorid = DonationData['Donor ID'].value_counts().to_frame()
donorid
donorX = len(donorid[donorid['Donor ID']>1])
print("The Total numbers and percent of donors having >1 Donations are : {} and {}%".format(donorX,int((donorX/donorid.shape[0])*100)))
print("The Total numbers and percent of donors having single donations are : {} and {}%".format(donorid.shape[0]-donorX,100-int((donorX/donorid.shape[0])*100)))
import folium 
from folium import plugins
from io import StringIO
#adding latitude and longitude
statesll=StringIO("""State,Latitude,Longitude
Alabama,32.806671,-86.791130
Alaska,61.370716,-152.404419
Arizona,33.729759,-111.431221
Arkansas,34.969704,-92.373123
California,36.116203,-119.681564
Colorado,39.059811,-105.311104
Connecticut,41.597782,-72.755371
Delaware,39.318523,-75.507141
District of Columbia,38.897438,-77.026817
Florida,27.766279,-81.686783
Georgia,33.040619,-83.643074
Hawaii,21.094318,-157.498337
Idaho,44.240459,-114.478828
Illinois,40.349457,-88.986137
Indiana,39.849426,-86.258278
Iowa,42.011539,-93.210526
Kansas,38.526600,-96.726486
Kentucky,37.668140,-84.670067
Louisiana,31.169546,-91.867805
Maine,44.693947,-69.381927
Maryland,39.063946,-76.802101
Massachusetts,42.230171,-71.530106
Michigan,43.326618,-84.536095
Minnesota,45.694454,-93.900192
Mississippi,32.741646,-89.678696
Missouri,38.456085,-92.288368
Montana,46.921925,-110.454353
Nebraska,41.125370,-98.268082
Nevada,38.313515,-117.055374
New Hampshire,43.452492,-71.563896
New Jersey,40.298904,-74.521011
New Mexico,34.840515,-106.248482
New York,42.165726,-74.948051
North Carolina,35.630066,-79.806419
North Dakota,47.528912,-99.784012
Ohio,40.388783,-82.764915
Oklahoma,35.565342,-96.928917
Oregon,44.572021,-122.070938
Pennsylvania,40.590752,-77.209755
Rhode Island,41.680893,-71.511780
South Carolina,33.856892,-80.945007
South Dakota,44.299782,-99.438828
Tennessee,35.747845,-86.692345
Texas,31.054487,-97.563461
Utah,40.150032,-111.862434
Vermont,44.045876,-72.710686
Virginia,37.769337,-78.169968
Washington,47.400902,-121.490494
West Virginia,38.491226,-80.954453
Wisconsin,44.268543,-89.616508
Wyoming,42.755966,-107.302490""")

DonorsData['Donor State'].value_counts().to_frame()
tempdf = states_df = DonorsData['Donor State'].value_counts()
t1 = pd.DataFrame()
t1['Donor State'] = tempdf.index
t1['Donor Count'] = tempdf.values
sdf = pd.read_csv(statesll).rename(columns = {'State':'Donor State'})
sdf = sdf.merge(t1,on='Donor State',how='inner')
sdf
map4 = folium.Map(location=[39.50, -98.35], tiles='CartoDB dark_matter', zoom_start=3.5)
for j, rown in sdf.iterrows():
    rown = list(rown)
    folium.CircleMarker([float(rown[1]), float(rown[2])], popup="<b>State:</b>" + rown[0].title() +"<br> <b>Donors:</b> "+str(int(rown[3])), radius=float(rown[3])*0.0001, color='#be0eef', fill=True).add_to(map4)
map4
DonationData['Donation Amount'].max(),DonationData['Donation Amount'].min()
min_donors = sdf[sdf['Donation Amount'] <= 6000]
max_donors = DonationData[DonationData['Donation Amount'] >= 6000]
DonationData.head()
x = list(DonationData.columns)
x.append('State')
DonorAmountData = pd.DataFrame(columns =x)
for donor_id in DonorsData['Donor ID']:
    k = DonationData[DonationData['Donor ID']==donor_id]
    donor_state = DonorsData[DonorsData['Donor ID']==donor_id]['Donor State'].values
    k['State']=donor_state
    DonorAmountData = pd.concat([DonorAmountData,k])
DonorAmountData

state, idx = DonorsData[['Donor State','Donor ID']].iloc[11]
state,idx
#DonrAmountData = pd.DataFrame()
d = DonationData[DonationData['Donor ID']==idx]
#d.join(donor_state.to_frame)
#DonorAmountData.concatenate(d)
d['state'] = state
d
DonrAmountData = d
DonrAmountData = pd.concat([DonrAmountData,d])
DonrAmountData

