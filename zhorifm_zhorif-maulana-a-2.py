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
!pip install pandas-profiling
online_sales = pd.read_csv('/kaggle/input/uisummerschool/Online_sales.csv', sep=',') #data dipisah dengan koma

online_sales.head() #5 data pertama
# Menampilkan beberapa kolom

online_sales [['Date', 'Product SKU', 'Quantity', 'Revenue', 'Tax']]
# Menyimpan data ke variabel baru

test = online_sales [['Date', 'Product SKU', 'Quantity', 'Revenue', 'Tax','Delivery']]
test['Net Income'] = test['Revenue'] - test['Tax'] - test['Delivery']

test.tail(5)
## Update kolom yg ada berdasar kondisi



kondisi = test['Tax'].isnull() 

test.loc[kondisi, ['Tax']] = 1
# Group by untuk aggregasi



test = online_sales.groupby(['Date'])['Quantity'].sum().reset_index() #jumlah di hari tertentu

test.head()
#Group by multiple kolom

test = online_sales.groupby(['Date', 'Product SKU'])['Quantity'].sum().reset_index()

test.head()
# Agregasi beberapa tipe

test = online_sales.groupby(['Date']).agg({'Quantity': 'sum',

                                                      'Revenue': 'sum',

                                                      'Tax': 'sum',

                                                      'Product SKU': 'count',

                                                      'Transaction ID': 'count',

                                                     }).reset_index() #count hitungan per item, sum jumlah nilainya
test.head()
#Sort data berdasar quantity descending

online_sales.sort_values(by=['Quantity'], ascending = False).head(15)
test.head(2)
#ganti nama kolom, dann delete kolom

test.rename(index=str, columns={"Quantity": "Total Quantity", "Revenue": "Total Revenue"}, inplace = True)



test.drop(columns=['Product SKU', 'Transaction ID'], inplace = True)

test.head()
## Tahap 1 Olah data hingga mendapat total revenue per hari

online_sales = online_sales = pd.read_csv('/kaggle/input/uisummerschool/Online_sales.csv')
daily_online_revenue = online_sales.groupby(['Date'])['Revenue'].sum().reset_index()

daily_online_revenue.tail()
import seaborn as sns
daily_online_revenue['Date'] = daily_online_revenue['Date'].astype(str) #cast tanggal jadi string

daily_online_revenue['Date'] = pd.to_datetime(daily_online_revenue['Date'])

g = sns.lineplot(data = daily_online_revenue, x= 'Date', y = 'Revenue')  
## Tahap 2 Tambahkan data yang akan diprediksi

add_data = [['2017-12-01', 0], ['2017-12-02', 0], ['2017-12-03', 0],

            ['2017-12-04', 0], ['2017-12-05', 0], ['2017-12-06', 0],

            ['2017-12-07', 0], ['2017-12-08', 0], ['2017-12-09', 0],

            ['2017-12-10', 0], ['2017-12-11', 0], ['2017-12-10', 0],

            ['2017-12-13', 0], ['2017-12-14', 0]

           ] 
# Create the pandas DataFrame 

add_data_df = pd.DataFrame(add_data, columns = ['Date', 'Revenue']) #add_data untuk date dan revenue

add_data_df['Date'] = add_data_df['Date'].astype(str) #date cast jadi string

add_data_df['Date'] = pd.to_datetime(add_data_df['Date'])
daily_online_revenue = daily_online_revenue.append(add_data_df) #nambahin add_data ke daily_online_revenue
daily_online_revenue.tail(5)
## 3 Persiapan data prediksi dan training



## Add feature 

daily_online_revenue['H - 1'] = 0

daily_online_revenue['H - 2'] = 0

daily_online_revenue['H - 3'] = 0

daily_online_revenue['H - 4'] = 0

daily_online_revenue['H - 5'] = 0
daily_online_revenue.head(3)
end_of_training_date = "2017-11-16" 

daily_online_revenue = daily_online_revenue.copy()

train_data = daily_online_revenue.sample(frac=0.75, random_state=1)

test_data = daily_online_revenue.drop(train_data.index)
test_data.head()
# Pisahkan kolom yang ingin diprediksi (biasa disebut label menjadi y), dan variabel lain menjadi x

x_train = train_data['Date']

y_train = train_data['Revenue']



x_test = test_data['Date']

y_test = test_data['Revenue']
from sklearn.ensemble import RandomForestRegressor



def fit(x_train, y_train):

    model = RandomForestRegressor(random_state=0)  #14045

    model.fit(x_train, y_train)

    return model
def predict(model, x_test):

    y_pred = model.predict(x_test)

    return y_pred



model = fit(x_train, y_train)