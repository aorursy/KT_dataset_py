import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import seaborn as sns
%matplotlib inline
pd.set_option('max_columns', None)
df_original = pd.read_csv('../input/airbnb-boston/boston_listings.csv')
df = pd.read_csv('../input/boston-preprocessed/boston_listings_updated.csv')
df = df.drop(columns=['Unnamed: 0'])
df.head(2)
print(df['price'].describe()) 
df['price'].plot(kind ='box')
Q1 = df['price'].quantile(0.25)
Q3 = df['price'].quantile(0.75)

IQR = Q3 -Q1

Max = Q3 + 1.5*IQR
Min = Q1 - 1.5*IQR
print('Min value {} , Max value {}'.format(Min,Max))
print('Total count for price higher than 422 = ' + str(df[df['price'] > 422]['price'].count()))
print('A look at the outlier prices :' + str(np.sort(df[df['price'] > 422]['price'].unique())))
df = df.query('price < 500')
!pip install sklearn
!pip install pycaret
import pycaret
from pycaret.regression import *
bnb_setup = setup(df, target='price',silent = True)
compare_models(fold=5)
catboost_regressor = create_model('catboost')
tuned_catboost_regressor = tune_model('catboost',optimize ='mse')
predictions = predict_model(tuned_catboost_regressor)

final_model = finalize_model(tuned_catboost_regressor)
final_model.get_params()
save_model(final_model, 'airbnb_catboost_1')
interpret_model(final_model)
features = final_model.get_feature_importance(prettified = True).set_index('Feature Id').to_dict()['Importances']

#sort by decreasing importance 
feature_importance_dict = dict([(v[0],v[1]) for v in sorted(features.items(), key=lambda kv: (-kv[1], kv[0]))])
amenity_list = ['24-Hour Check-in', 'Air Conditioning', 'Breakfast',
       'Buzzer/Wireless Intercom', 'Cable TV', 'Carbon Monoxide Detector',
       'Cat(s)', 'Dog(s)', 'Doorman', 'Dryer', 'Elevator in Building',
       'Essentials', 'Family/Kid Friendly', 'Fire Extinguisher',
       'First Aid Kit', 'Free Parking on Premises',
       'Free Parking on Street', 'Gym', 'Hair Dryer', 'Hangers',
       'Heating', 'Hot Tub', 'Indoor Fireplace', 'Internet', 'Iron',
       'Kitchen', 'Laptop Friendly Workspace', 'Lock on Bedroom Door',
       'Other pet(s)', 'Paid Parking Off Premises', 'Pets Allowed',
       'Pets live on this property', 'Pool', 'Safety Card', 'Shampoo',
       'Smoke Detector', 'Smoking Allowed', 'Suitable for Events', 'TV',
       'Washer', 'Washer / Dryer', 'Wheelchair Accessible',
       'Wireless Internet', 'translation missing: en.hosting_amenity_49',
       'translation missing: en.hosting_amenity_50']

amenity_importance = []
for amenity in amenity_list:
    for col in feature_importance_dict.keys():
        if amenity in col:
            if 'False' in col:
                amenity_importance.append((col[:-6], -feature_importance_dict[col]))
            else:
                amenity_importance.append((col[:-5], feature_importance_dict[col]))
    
    
amenity_importance.sort(key=lambda tup: tup[1], reverse = True) 

pd.DataFrame(amenity_importance, columns = ['amenity','Importance']).drop_duplicates().set_index('amenity').plot(kind='bar', figsize = (30,15))
plt.xticks(fontsize=20)
plt.xlabel('amenity', fontsize=30)
plt.ylabel('Importance', fontsize=30)

df_original['price'] = df_original['price'].map(lambda p: int(p[1:-3].replace(",", "")))
df_original = df_original.query('price < 500')
neighborhoods = df_original['neighbourhood_cleansed']

for n in neighborhoods:
    n_price.append((n, df_original[df_original['neighbourhood_cleansed'] == n]['price'].mean()))
pd.DataFrame(n_price, columns = ['Neighborhood','Average Price']).drop_duplicates().set_index('Neighborhood').sort_values(by = 'Average Price').plot(kind='bar', figsize = (30,15))
plt.xticks(fontsize=20)
plt.xlabel('Neighborhood', fontsize=30)
plt.ylabel('Average Price', fontsize=30)
