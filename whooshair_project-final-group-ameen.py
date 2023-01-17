# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df= pd.read_csv('../input/r-left1/r_left.csv')
import matplotlib.pyplot as plt
plt.figure(figsize=(20,10)) 
plt.plot(df['IRRADIATION'],df['AMBIENT_TEMPERATURE'],marker='o',color='lightpink', linestyle='', label = 'Ambient Temperature')

plt.xlabel= 'Irradiation'
plt.ylabel= 'Ambient Temperature'
plt.title='Irradiation vs Ambient Temperature '
plt.legend()
plt.grid()
plt.margins(0.05)
plt.figure(figsize=(20,10))
plt.plot(df['IRRADIATION'],df['AC_POWER'],marker='o',color='darkorange',linestyle='', label = 'AC Power')

plt.xlabel= 'Irradiation'
plt.ylabel= 'AC Power'
plt.title='Irradiation vs AC Power '
plt.legend()
plt.grid()
plt.margins(0.05)
plt.figure(figsize=(20,10))
plt.plot(df['AMBIENT_TEMPERATURE'],df['AC_POWER'],marker='o',color='crimson',linestyle='', label = 'AC Power')

plt.xlabel= 'Ambient Temperature'
plt.ylabel= 'AC Power'
plt.title='Ambient Temperature vs AC Power '
plt.legend()
plt.grid()
plt.margins(0.05)
x =df.iloc[:,13:14].values # X is our INPUT  #X1 = r_left.iloc[:,[15]]  # AMBIENT TEMP COLUMN
y =df.iloc[:,4].values  # y is our OUTPUT #AC POWER COLUMN

from sklearn.model_selection import train_test_split as tts
x_train,x_test,y_train,y_test =tts(x,y,test_size =0.3,random_state=0) #xtrain,xtest,yrain,ytest
from sklearn.linear_model import LinearRegression
lin_reg=LinearRegression()
lin_reg.fit(x_train,y_train)
y_pred =lin_reg.predict(x_test)
y_pred    # prediction value
y_test
#1
import pickle
pickle.dump(lin_reg,open('model.pkl','wb'))  #write bit for wb

 #2
!pip install streamlit
!pip install pyngrok
%%writefile amn.py
import streamlit as st
import pickle 
st.title('AC POWER OUTPUT PREDICTOR')
st.subheader('This app will take your ambient temperature as input and predict how much AC Power you can expect to generate.')
st.write('Ambient temperature vs AC Power')
xyz = pickle.load(open('model.pkl','rb')) #read bit -rd
at = st.number_input('Enter AMBIENT TEMP')
xyz.predict([[at]])
if st.button('Predict'):
    st.title(xyz.predict([[at]]))
from pyngrok import ngrok
url = ngrok.connect(port = '8501')
url
!streamlit run amn.py
pw = pd.read_csv('../input/solar-power-generation-data/Plant_1_Weather_Sensor_Data.csv')
pg = pd.read_csv('../input/solar-power-generation-data/Plant_1_Generation_Data.csv')