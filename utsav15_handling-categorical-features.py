import pandas as pd
import numpy as np
data = pd.DataFrame(
        [['female', 'New York', 'low', 4], ['female', 'London', 'medium', 3], ['male', 'New Delhi', 'high', 2]],
        columns=['Gender', 'City', 'Temperature', 'Rating'])
data
from sklearn.preprocessing import LabelEncoder

data['City_encoded'] = LabelEncoder().fit_transform(data['City'])
data[['City', 'City_encoded']] # special syntax to get just these two columns
encoder = LabelEncoder()
encoder.fit(data['City'])
encoder.classes_
data['City_encoded'] = encoder.transform(data['City']) # transform as a separate step from fit
data[['City', 'City_encoded']]
data['Temperature_encoded'] = data['Temperature'].map( {'low':0, 'medium':1, 'high':2})
data[['Temperature', 'Temperature_encoded']]
data['Male'] = data['Gender'].map( {'male':1, 'female':0} )
data[['Gender', 'Male']]
pd.get_dummies(data['City'], prefix='City')
data = pd.concat([data, pd.get_dummies(data['City'], prefix='City')], axis=1)
data[['City', 'City_London', 'City_New Delhi', 'City_New York']]