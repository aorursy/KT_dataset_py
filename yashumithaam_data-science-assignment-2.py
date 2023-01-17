import pandas as pd

import numpy as np

import matplotlib.pyplot as plt
data  = pd.read_csv("/kaggle/input/housesalesprediction/kc_house_data.csv")

data.dropna()
brooms = data[ 'bathrooms'] #extract the values of columns with the name 'bathrooms'

price = data['price'] #extract the values of columns with the name 'SalePrice'

plt.plot(brooms,price)

plt.show()