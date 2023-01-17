import pandas as pd

import numpy as np

from sklearn.preprocessing import MinMaxScaler

from sklearn.preprocessing import  StandardScaler

df= pd.read_csv('../input/indian-food-101/indian_food.csv')

df.head()
df.shape
ingre = set()

for i in df['ingredients']:

    ingre.update(str(i).lower().split(","))

    

print("Total unique ingredients in dataset",len(ingre),sep=": ")
def count_ingredient(column):

    return float(len(column.split(",")))

df['ingredient_count'] = df['ingredients'].apply(count_ingredient)

df.head()
df.drop('ingredients', axis=1, inplace=True)

df.head()
df.describe()
df.describe(include='object')
df.replace(-1, np.NaN, inplace = True)

df.replace("-1", np.NaN, inplace = True)

df.nunique()
data = df.dropna()
data.shape
data.describe()
data.dtypes
num = (data.dtypes == 'float64')

numerical = list(num[num].index)

print("Numerical variables are:")

print(numerical)
num_data = data[numerical]

num_data.head()
scaler = MinMaxScaler()

num_data_values = num_data.values

num_data_scaled = scaler.fit_transform(num_data_values)

normalized_df = pd.DataFrame(num_data_scaled)

normalized_df.head()
normalized_df.describe()
std_scaler = StandardScaler()

num_data_values = num_data.values

num_data_std= std_scaler.fit_transform(num_data_values)

standardized_df = pd.DataFrame(num_data_std)

standardized_df.head()
standardized_df.describe()
data.describe(include='object')
cat = (data.dtypes == 'object')

objects = list(cat[cat].index)

print("Categorical variables are:")

print(objects)
cat_data = data[objects]

cat_data.head()
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()

cat_data['course'] = label_encoder.fit_transform(cat_data['course'])

cat_data['state'] = label_encoder.fit_transform(cat_data['state'])

cat_data.head()
f_pro = {'sweet':1,'spicy':2, 'bitter':3, 'sour':4}

cat_data = cat_data.replace({'flavor_profile':f_pro})

cat_data.head()
ndiet={'vegetarian':0,'non-vegitarian':1}

cat_data= cat_data.replace({'diet':ndiet})

cat_data.head()
cat_data = pd.get_dummies(cat_data,columns=['region'],prefix = ['cat'])

cat_data.head()
cat_data.describe()