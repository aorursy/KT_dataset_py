import numpy as np
import pandas as pd
# pandas defaults
pd.options.display.max_columns = 500
pd.options.display.max_rows = 500
import warnings
warnings.filterwarnings('ignore')
df=pd.read_csv('Steam_games_original.csv')
df.head(10)
df=pd.read_csv('Steam_games.csv')
df.head(10)
df=df.fillna(method='ffill',axis=1)
# let's rename the columns so that they make sense
df.rename(columns={'sale_price':'orig_price',}, inplace=True)
df.head()
#checking Data types
df.dtypes

# Converting data type to suitable types
convert_dict = {'Name':str, 'rel_date': str, 'orig_price': float, 'discounted_price': float
               } 
  
df = df.astype(convert_dict) 
print(df.dtypes) 
# Creating another column with discount percentage
df['discount%']= (((df['orig_price']-df['discounted_price'])*100)/df['orig_price'])
df.head(10)
#Replacing and rounding of values to 2 digit
df['discount%'] = df['discount%'].replace(np.nan, 0) 
df['discount%']=df['discount%'].round(2)
df.head(10)
df.to_csv('Steam_games_final.csv')
df = pd.read_csv("Steam_games_final.csv")
df.describe()
