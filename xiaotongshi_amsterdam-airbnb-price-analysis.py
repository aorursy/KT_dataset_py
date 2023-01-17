import folium # map rendering library
import requests # library to handle requests
import pandas as pd
import numpy as np
import seaborn as sns
%matplotlib inline
import matplotlib.pyplot as plt
# The coordinates of Amsterdam center
latitude = 52.372952
longitude = 4.906080
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
ab_data = pd.read_csv('/kaggle/input/airbnb-amsterdam/listings_details.csv')
print(ab_data.shape)
ab_data.head()
ab_data.columns
import geopandas as gpd
geo_ams = gpd.read_file('/kaggle/input/airbnb-amsterdam/neighbourhoods.geojson')
geo_ams["longitude"] = geo_ams.centroid.x
geo_ams["latitude"] = geo_ams.centroid.y
geo_ams.drop('neighbourhood_group', axis=1, inplace=True)
geo_ams
ab_data.columns
columns = ['id', 'name',
       'neighbourhood_cleansed', 'latitude', 'longitude', 'property_type', 'room_type', 'accommodates',
       'bathrooms', 'bedrooms', 'beds', 'bed_type', 'square_feet',
       'price', 'cleaning_fee', 'guests_included', 'extra_people', 
       'minimum_nights', 'maximum_nights', 
       'has_availability', 'availability_30', 'availability_60', 'availability_90', 'availability_365', 
       'number_of_reviews', 'review_scores_rating',
       'review_scores_accuracy', 'review_scores_cleanliness',
       'review_scores_checkin', 'review_scores_communication',
       'review_scores_location', 'review_scores_value', 'instant_bookable',
       'is_business_travel_ready', 'cancellation_policy']
data_columns = ab_data[columns]
print(data_columns.shape)
data_columns = data_columns[data_columns['has_availability'] == 't']
data_columns.shape
data_columns = data_columns[data_columns['availability_30'] != 0]
data_columns = data_columns[data_columns['availability_60'] != 0]
data_columns = data_columns[data_columns['availability_90'] != 0]
data_columns = data_columns[data_columns['availability_365'] != 0]
data_columns.shape
data_columns.drop(['has_availability', 'availability_30', 'availability_60', 'availability_90', 'availability_365'], axis=1, inplace=True)
data_columns.shape
data_columns = data_columns[data_columns['price'] != 0]
data_columns = data_columns[data_columns['number_of_reviews'] != 0]
data_columns = data_columns[data_columns['review_scores_rating'].notnull()]
data_columns.shape
def examine_missing_values(data):
    data_na= data.isnull().sum().sort_values(ascending=False)
    data_na_percent = (data.isnull().sum()/len(data)*100).sort_values(ascending=False)
    missing_data = pd.concat([data_na, data_na_percent], axis=1, keys=['Total', 'Percent'])
    return missing_data
missing_data = examine_missing_values(data_columns)
missing_data.head(20)
data_columns.drop(['square_feet'], axis=1, inplace=True)
data_columns.shape
data_columns['review_scores_value'].fillna(data_columns['review_scores_rating'], inplace=True)
data_columns['review_scores_location'].fillna(data_columns['review_scores_rating'], inplace=True)
data_columns['review_scores_communication'].fillna(data_columns['review_scores_rating'], inplace=True)
data_columns['review_scores_checkin'].fillna(data_columns['review_scores_rating'], inplace=True)
data_columns['bathrooms'].fillna(0, inplace=True)
data_columns['bedrooms'].fillna(0, inplace=True)
data_columns['beds'].fillna(0, inplace=True)
data_columns['cleaning_fee'].fillna(data_columns['cleaning_fee'].mode()[0], inplace=True)
missing_data = examine_missing_values(data_columns)
missing_data.head(20)
data_columns.shape
numeric_features = list(data_columns.dtypes[data_columns.dtypes != 'object'].index)
numeric_features.remove('id')
numeric_features.remove('latitude')
numeric_features.remove('longitude')
numeric_features
category_features = list(data_columns.dtypes[data_columns.dtypes == 'object'].index)
category_features.remove('name')
category_features
data_columns['price'] = data_columns['price'].apply(lambda x:x.lstrip('$'))
data_columns['price'] = data_columns['price'].apply(lambda x:x.replace(',',''))
data_columns['price'] = data_columns['price'].astype('float')
data_columns['cleaning_fee'] = data_columns['cleaning_fee'].apply(lambda x:x.lstrip('$'))
data_columns['cleaning_fee'] = data_columns['cleaning_fee'].astype('float')
data_columns['extra_people'] = data_columns['extra_people'].apply(lambda x:x.lstrip('$'))
data_columns['extra_people'] = data_columns['extra_people'].astype('float')
numeric_features.extend(['cleaning_fee', 'extra_people'])
numeric_features
category_features.remove('price')
category_features.remove('cleaning_fee')
category_features.remove('extra_people')
data_columns.columns = ['neighbourhood' if x=='neighbourhood_cleansed' else x for x in data_columns.columns]
features = data_columns
print(features.shape)
features.head()
category_features.remove('neighbourhood_cleansed')
category_features.append('neighbourhood')
category_features
from scipy.stats import skew, boxcox_normmax
from scipy.special import boxcox1p
skew_features = features[numeric_features].apply(lambda x: skew(x)).sort_values(ascending=False)
high_skew = skew_features[skew_features > 0.5]
skew_index = high_skew.index

for i in skew_index:
    features[i] = boxcox1p(features[i], boxcox_normmax(features[i] + 1))
def one_hot_encode(data, columns):
    onehot = pd.get_dummies(data[columns])
    onehot['id'] = data['id']
    # move id column to the first column
    fixed_columns = [onehot.columns[-1]] + list(onehot.columns[:-1])
    onehot = onehot[fixed_columns]
    return onehot
features_onehot = one_hot_encode(features, category_features)
features_onehot.head()
from scipy import stats
from scipy.stats import norm
import matplotlib.pyplot as plt

def check_dist(price):
    print('Checking the statistical distribution of prices')
    print(price.describe())
    
    print('Fitting the prices into normal distribution')
    sns.distplot(price, fit=norm)

    # Get the fitted parameters used by the function
    (mu, sigma) = norm.fit(price)
    print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))

    #Now plot the distribution
    plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
                loc='best')
    plt.ylabel('Frequency')
    plt.title('Price distribution')

    #Get also the QQ-plot
    fig = plt.figure()
    res = stats.probplot(price, plot=plt)
    plt.show()
check_dist(features['price'])
features = features[features['price']<2000]
def correct_dist(price):
    price = np.log1p(price)

    #Check the new distribution 
    sns.distplot(price , fit=norm);

    # Get the fitted parameters used by the function
    (mu, sigma) = norm.fit(price)
    print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))

    #Now plot the distribution
    plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
                loc='best')
    plt.ylabel('Frequency')
    plt.title('Price distribution')

    #Get also the QQ-plot
    fig = plt.figure()
    res = stats.probplot(price, plot=plt)
    plt.show()
    return price
features['corrected_price'] = correct_dist(features['price'])
neighbourhood_count = pd.DataFrame({'neighbourhood': features['neighbourhood'].value_counts().index, 
                                    'count': features['neighbourhood'].value_counts().values})
neighbourhood_count.head()
map_ams_count = folium.Map(location=[latitude, longitude], zoom_start=12)
map_ams_count.choropleth(
    geo_data=r'/kaggle/input/airbnb-amsterdam/neighbourhoods.geojson',
    data=neighbourhood_count,
    columns=['neighbourhood', 'count'],
    key_on='feature.properties.neighbourhood',
    fill_color='YlOrRd', 
    fill_opacity=0.7, 
    line_opacity=0.2,
    legend_name='The number of properites',
    reset=True
)
map_ams_count
neighbourhood_price = features.groupby('neighbourhood').median()['price']
neighbourhood_price = pd.DataFrame({'neighbourhood':neighbourhood_price.index, 'price': neighbourhood_price.T.values})
neighbourhood_price.sort_values('price', ascending=False, inplace=True)
neighbourhood_price.head()
map_ams_price = folium.Map(location=[latitude, longitude], zoom_start=12)
map_ams_price.choropleth(
    geo_data=r'/kaggle/input/airbnb-amsterdam/neighbourhoods.geojson',
    data=neighbourhood_price,
    columns=['neighbourhood', 'price'],
    key_on='feature.properties.neighbourhood',
    fill_color='YlOrRd', 
    fill_opacity=0.7, 
    line_opacity=0.2,
    legend_name='The average_price',
    reset=True
)
map_ams_price
CLIENT_ID = 'H201PLIGMNTAP5ZKN2DZK1QSDVTSZNLH4SGVA0VBPFO00MFT' # your Foursquare ID
CLIENT_SECRET = 'JAPGNRKSTAJEQUKPATJJCNFETSEJQBAQRPDODZDNU1CN1MLP' # your Foursquare Secret
VERSION = '20180605' # Foursquare API version

print('Your credentails:')
print('CLIENT_ID: ' + CLIENT_ID)
print('CLIENT_SECRET:' + CLIENT_SECRET)
# categories = {'Arts & Entertainment': '4d4b7104d754a06370d81259', 
#                   'College & University': '4d4b7105d754a06372d81259', 
#                   'Event': '4d4b7105d754a06373d81259', 
#                   'Food': '4d4b7105d754a06374d81259',
#                   'Nightlife Spot': '4d4b7105d754a06376d81259',
#                   'Outdoors & Recreation': '4d4b7105d754a06377d81259',
#                   'Professional & Other Places': '4d4b7105d754a06375d81259',
#                   'Residence': '4e67e38e036454776db1fb3a',
#                   'Shop & Service': '4d4b7105d754a06378d81259',
#                   'Travel & Transport': '4d4b7105d754a06379d81259'}
categories = {'Arts & Entertainment': '4d4b7104d754a06370d81259', 
                  'Event': '4d4b7105d754a06373d81259', 
                  'Food': '4d4b7105d754a06374d81259',
                  'Nightlife Spot': '4d4b7105d754a06376d81259',
                  'Outdoors & Recreation': '4d4b7105d754a06377d81259',
                  'Shop & Service': '4d4b7105d754a06378d81259',
                  'Travel & Transport': '4d4b7105d754a06379d81259'}
def getNearbyVenues(data, categories, radius=500, limit=10):
    
    venues_list=[]
    print('Obtaining venues around the neighbourhoods: ', end='')
    for name, lat, lng in zip(data['neighbourhood'], data['latitude'], data['longitude']):
        print('.', end='')
        # create the API request URL
        venues = {'neighbourhood':name}
        for category, category_id in categories.items():
            url = 'https://api.foursquare.com/v2/venues/explore?&client_id={}&client_secret={}&v={}&ll={},{}&categoryId={}&radius={}&limit={}'.format(
                CLIENT_ID, 
                CLIENT_SECRET, 
                VERSION, 
                lat, 
                lng, 
                category_id,
                radius, 
                limit)

            # make the GET request
            results = requests.get(url).json()["response"]['groups'][0]['items'] 
            venues[category] = len(results)
        venues_list.append(venues)

    venues_list = pd.DataFrame(venues_list)
    return venues_list
neighbourhood_venues = getNearbyVenues(geo_ams, categories, radius=1000, limit=100)
print()
print(neighbourhood_venues.shape)
neighbourhood_venues.head()
neighbourhood_venues.iloc[:, 1:] = neighbourhood_venues.iloc[:, 1:].apply(lambda x: x/x.sum(), axis=1)
neighbourhood_venues.head()
neighbourhood_venues.fillna(0, inplace=True)
neighbourhood_venues.head()
features_merged = pd.merge(features, neighbourhood_venues)
features_merged.head()
features_merged_dummy = pd.merge(features_merged, features_onehot)
features_merged_dummy.head()
features_merged_dummy.drop(category_features, axis=1, inplace=True)
features_merged_dummy.head()
y = features_merged_dummy['corrected_price']
X = features_merged_dummy.drop(['id', 'name', 'latitude', 'longitude', 'price', 'corrected_price'], axis=1)
X.shape
overfit = []
for i in X.columns:
    counts = X[i].value_counts()
    zeros = counts.iloc[0]
    if zeros / len(X) * 100 > 95:
        overfit.append(i)
overfit
X.drop(overfit, axis=1, inplace=True)
X.shape
X.columns
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error

# Root Mean Squared Logarithmic Error ，RMSLE
def rmsle(y, y_pred):
    return np.sqrt(mean_squared_error(y, y_pred))

#
def cv_rmse(model):
    rmse = np.sqrt(-cross_val_score(model, X_train, y_train, scoring="neg_mean_squared_error", cv=10))
    return (rmse)
from sklearn.linear_model import Lasso, Ridge, ElasticNet
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
#ridge
ridge = Ridge()

#lasso
lasso = Lasso()

#elastic net
elasticnet = ElasticNet()

#GradientBoosting
gbr = GradientBoostingRegressor(n_estimators=3000)


#lightgbm
lightgbm = LGBMRegressor(
    objective='regression',
    num_leaves=4,
    learning_rate=0.01,
    n_estimators=5000)

#xgboost（
xgb = XGBRegressor(learning_rate=0.01, 
                   booster='gbtree',
                   objective='reg:linear',
                   eval_metric='rmse',
                   max_depth=3,
                   min_child_weight=0,
                   n_estimators=3000)
print('TEST score')

score = cv_rmse(ridge) 
print("Ridge score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()) ) 

score = cv_rmse(lasso)
print("Lasso score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()) ) 

score = cv_rmse(elasticnet)
print("ElasticNet score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

score = cv_rmse(lightgbm)
print("Lightgbm score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

score = cv_rmse(gbr)
print("GradientBoosting score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

score = cv_rmse(xgb)
print("Xgboost score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
print('START Fit')
print('ridge')
ridge_model_full_data = ridge.fit(X_train, y_train)
print('GradientBoosting')
gbr_model_full_data = gbr.fit(X_train, y_train)
print( 'xgboost')
xgb_model_full_data = xgb.fit(X_train, y_train)
print('lightgbm')
lgb_model_full_data = lightgbm.fit(X_train, y_train)
print('ridge', rmsle(y_train, ridge_model_full_data.predict(X_train)))
print('gbr', rmsle(y_train, gbr_model_full_data.predict(X_train)))
print('xgb', rmsle(y_train, xgb_model_full_data.predict(X_train)))
print('lgb', rmsle(y_train, lgb_model_full_data.predict(X_train)))
def blend_models_predict(X):
    return ((0.1 * ridge_model_full_data.predict(X)) + 
            (0.3 * gbr_model_full_data.predict(X)) + 
            (0.3 * xgb_model_full_data.predict(X)) + 
            (0.3 * lgb_model_full_data.predict(X)) )
            
print('RMSLE score on train data:')
print(rmsle(y_train, blend_models_predict(X_train)))
y_pred = blend_models_predict(X_test)
rmsle(y_test, y_pred)
linear_model_list = {'ridge': ridge_model_full_data}
nonlinear_model_list = {'GradientBoosting': gbr_model_full_data,
                       'XGBoosting': xgb_model_full_data,
                       'Lightgbm': lgb_model_full_data}
feature_importance = []
for model_name, model in linear_model_list.items():
    feature_importance.append(model.coef_)
for model_name, model in nonlinear_model_list.items():
    feature_importance.append(model.feature_importances_)
feature_importance = pd.DataFrame(feature_importance, columns=X_train.columns)
feature_importance.index = list(linear_model_list.keys()) + list(nonlinear_model_list.keys())
feature_importance
def return_most_important_features(features_importance, top):
    features_sorted = features_importance.sort_values(ascending=False)
    return features_sorted.index.values[0:top]
features_sorted = []
for ind in np.arange(feature_importance.shape[0]):
    features_sorted.append(return_most_important_features(feature_importance.iloc[ind, :], top=10))
def return_columns(top):
    indicators = ['st', 'nd', 'rd']

    columns = []
    for ind in np.arange(top):
        try:
            columns.append('{}{} Important Feature'.format(ind+1, indicators[ind]))
        except:
            columns.append('{}th Important Feature'.format(ind+1))
    return columns
features_sorted = pd.DataFrame(features_sorted, columns=return_columns(top=10))
features_sorted.index = feature_importance.index
features_sorted
map_ams_price
venues_sorted = []
for model_name, model_feature_importance in feature_importance.iterrows():
    venues_sorted.append(return_most_important_features(model_feature_importance[list(categories.keys())], top=7))
venues_sorted = pd.DataFrame(venues_sorted, columns=return_columns(top=7))
venues_sorted.index = feature_importance.index
venues_sorted
