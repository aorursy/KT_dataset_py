import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt #plotting
% matplotlib inline
plt.style.use('ggplot')
import seaborn as sb #plotting
data = pd.read_csv('../input/food_coded.csv',encoding='latin-1')

data.describe()
data.breakfast.value_counts()
new = data.loc[data.breakfast == 2]
new
new.Gender.value_counts().plot(kind='bar',figsize=(10,10))
plt.scatter(new.Gender, new.weight)


new.groupby(['breakfast','Gender']).size().to_frame('count').reset_index()
new.groupby(['calories_day','Gender']).size().to_frame('count').reset_index()



new.loc[:,['comfort_food_reasons' , 'comfort_food']]

new.loc[:,['food_childhood' , 'comfort_food']]

x = new.loc[:,('food_childhood' , 'parents_cook', 'mother_profession', 'fav_food', 'father_profession')]
x
new.groupby(['parents_cook']).size().to_frame('count').reset_index()
new.groupby(['fav_food']).size().to_frame('count').reset_index()
z = new.loc[:,('cuisine' , 'comfort_food')]
z