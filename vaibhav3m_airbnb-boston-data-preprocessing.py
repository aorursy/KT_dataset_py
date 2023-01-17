import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

import os
%matplotlib inline
pd.set_option('max_columns', None)

try:
   df = pd.read_csv('../input/airbnb-boston/boston_listings.csv')
except CParserError:
    print("Something wrong the file")

df.head(2)
df.info()
#drop columns where more than 70% column is null
df = df.drop(columns = df.columns[df.isna().mean() > 0.70])
#add location data not useful or all are none
drop_cols =  ['country_code', 'country', 'state','experiences_offered']
df = df.drop(columns = drop_cols)
df.head(1)
#fixing price
df['price'] = df['price'].map(lambda p: int(p[1:-3].replace(",", "")))
#If Fee type is nan that is then it is supposed that there are no charge for the service
df['cleaning_fee'] = df['cleaning_fee'].fillna('$0.00').map(lambda p: int(p[1:-3].replace(",", "")))
df['security_deposit'] = df['security_deposit'].fillna('$0.00').map(lambda p: int(p[1:-3].replace(",", "")))
df['extra_people'] = df['extra_people'].fillna('$0.00').map(lambda p: int(p[1:-3].replace(",", "")))
#separating amenities
def amenities_separtor(x):
    arr = x.split(',')
    result = [s.replace('"', '').replace("{","").replace('}', '') for s in arr]
    return result
df['amenities'] = df['amenities'].apply(amenities_separtor)
pd.Series(np.concatenate(df['amenities'])).value_counts().plot(kind='bar')
#All type of amenities
all_amenities = np.unique(np.concatenate(df['amenities']))[1:]
all_amenities
#creating a list of features for amenities
amenity_list = np.array([df['amenities'].map(lambda amns: a in amns) for a in all_amenities])
#add columns to df
df = pd.concat([df,pd.DataFrame(amenity_list.T, columns=all_amenities)], axis =1)
df = df.drop(columns=['amenities'])
#fixing which are saved as strings of the form "t" or "f".
for tf_feature in ['host_is_superhost', 'host_identity_verified', 'host_has_profile_pic',
                   'is_location_exact', 'requires_license', 'instant_bookable',
                   'require_guest_profile_picture', 'require_guest_phone_verification']:
    df[tf_feature] = df[tf_feature].map(lambda s: False if s == "f" else True)
#create dummy variables
categorical_features = ['neighbourhood_cleansed', 'property_type', 'room_type', 'bed_type','cancellation_policy']
for feature in categorical_features:
    df = pd.concat([df, pd.get_dummies(df[feature])], axis=1)
df = df.drop(columns =categorical_features)
#removing columns with text data for now
# many could have been useful such as transit, notes, interaction, 
columns_withtext = ['summary','description','space','neighborhood_overview','notes','transit','interaction',  
                    'house_rules','host_name','host_about','host_location','host_neighbourhood','street','neighbourhood','market',
                   'smart_location','calendar_updated','calendar_last_scraped','first_review', 'last_review','access',
                    'name', 'host_verifications', 'city', 'zipcode']
columns_withurl = ['xl_picture_url','host_url','thumbnail_url','medium_url','host_picture_url','host_thumbnail_url',
                  'picture_url','listing_url']
columns_nouse = ['id', 'host_id','scrape_id','host_listings_count','last_scraped']
df = df.drop(columns = (columns_withtext + columns_withurl+ columns_nouse))
#converting string data to date time 
df['host_since'] = df['host_since'].apply(lambda x: pd.to_datetime(x))
#converting to ordinal form
import datetime as dt
df['host_since'] = df['host_since'].map(dt.datetime.toordinal)
#handling host_response_time, converting to numeric 
def response_time_cat(x):
    if x == 'within an hour' or x == 'within a few hours':
        return 1
    elif x == 'within a day':
        return 0.5
    return 0

df['host_response_time'] = df['host_response_time'].apply(response_time_cat)
# replacing nan values with 0.0% and converting to float
df['host_response_rate'] = df['host_response_rate'].fillna('0%').map(lambda x: float(x.replace('%',''))/100)
df['host_acceptance_rate'] = df['host_acceptance_rate'].fillna('0%').map(lambda x: float(x.replace('%',''))/100)
#columns with NaN values
for col in df.columns[df.isnull().mean() > 0]:
    print(col + ' = {:.2f} %'.format(df[col].isnull().mean()*100))
#filling all the columns with median of respective column
for col in df.columns[df.isnull().any()]:
    df[col] = df[col].fillna(df[col].median())
df.head(5)
df.info()
df.to_csv('boston_listings_updated.csv')
