# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# iletilen dosyada ki https://www.kaggle.com/lisphilar/covid-19-data-with-sir-model modelde sır grafiğini çıkarmak için gerekli olan kütüphaneleri programa atıyoruz
from collections import defaultdict
from datetime import timedelta
from dateutil.relativedelta import relativedelta
import functools
from IPython.display import display, Markdown
import math
import os
from pprint import pprint
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib
from matplotlib.ticker import ScalarFormatter
%matplotlib inline
import numpy as np
import pandas as pd
import dask.dataframe as dd
pd.plotting.register_matplotlib_converters()
import seaborn as sns
import scipy as sci
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import sympy as sym


for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#link üzerinden gerekli dökümanları kaggle notebooka a input data olarak atıyoruz https://github.com/lisphilar/covid19-sir

!pip install git+https://github.com/lisphilar/covid19-sir#egg=covsirphy
#https://github.com/lisphilar/covid19-sir uzantısı üzerinden çekilen covsirphy kütüphanesini kullanıyoruz
import covsirphy as cs
#Kaynak olan .csv uzantılı exceli programa tanıtıyoruz

daily_corona_data_in_turkey = pd.read_csv("/kaggle/input/covid19-in-turkey/covid_19_data_tr.csv")


#Şehir bazlı program olmadığı için şehir sekmesini çıkarıyoruz(ödevde şehir önemli bir kısım değil)

daily_corona_data_in_turkey.drop("Province/State", axis = 1, inplace = True)

#dataları yeniden adlandırıyoruz(daha rahat işlem yapabilmek için akılda kalacak şekilde yeniden adlandırdık)

daily_corona_data_in_turkey.rename(columns = {"Country/Region":"Turkey","Last_Update" : "Date", "Confirmed" : "confirmed_count", "Deaths" : "death_count", "Recovered" : "recovered_count"}, inplace = True)


#kontrol(sonuç olarak hatalı bir değer görünmemektedir)

print(daily_corona_data_in_turkey)

#kontrol (bütün çıktılar doğru olarak görünmekte)

#Programming in Python 3: A Complete Introduction to the Python Language Page:"145" we learn how to listing data to print 
#Burda günlük kaç hasta olduğunu gösteriyoruz


confirmedlist= []
for confirmingcounter in daily_corona_data_in_turkey["confirmed_count"]:
    confirmedlist.append(confirmingcounter)
    confirmedinTurkey = confirmedlist[-1]
print("Türkiyede bulaşan hasta sayısı:")
print(confirmedinTurkey)

#kontrol (hasta sayısı gündelik olarak güncellenecektir)
#Burda günlük kaç ölü olduğunu gösteriyoruz

deathlist= []
for deathcounter in daily_corona_data_in_turkey["death_count"]:
    deathlist.append(deathcounter)
    deathinTurkey = deathlist[-1]
print("Türkiyede ölen hasta sayısı:")
print(deathinTurkey)

#kontrol (ölü sayısı gündelik olarak güncellenecektir)
#Burda günlük kaç iyileşen olduğunu öğreniyoruz

recoveredlist= []
for recoveringcounter in daily_corona_data_in_turkey["recovered_count"]:
    recoveredlist.append(recoveringcounter)
    recoveredinTurkey = recoveredlist[-1]
print("Türkiyede iyileşen hasta sayısı:")
print(recoveredinTurkey)

#kontrol (iyileşen sayısı gündelik olarak güncellenecektir)
#Türkiye nüfusunu belirtiyoruz
N=83154997
#Türkiye nüfusunu grafikte kullanırken hata yaşamamak için belirtiyoruz
eg_population=83154997

#Bize iletilen https://www.kaggle.com/lisphilar/covid-19-data-with-sir-model kaggle notebooktaki soruce code üzerinde bulunan In[61],In[63] girdilerini kullanıyoruz

#SIR-F için gerekli olan grafiği, dökümandan anlık olarak çekmesi için gereken işlemleri manuel olarak değilde anlık olarak programa tanıtıyoruz
dtetavalue=(confirmedinTurkey+recoveredinTurkey)/deathinTurkey
kappavalue= (confirmedinTurkey-recoveredinTurkey)
eg_r0, eg_theta, eg_kappa, eg_rho = (deathinTurkey/N, deathinTurkey/recoveredinTurkey, deathinTurkey/confirmedinTurkey, deathinTurkey/kappavalue)
eg_sigma = eg_rho / eg_r0 - eg_kappa
eg_initials = (0.999, 0.001, 0, 0)
#Belirtilen simgeler klavye üzerinde tanımlı olmadığı için tanıtım
display(Markdown(rf"$\theta = {eg_theta},\ \kappa = {eg_kappa},\ \rho = {eg_rho},\ \sigma = {eg_sigma}$."))
print("Türkiye Nüfus")
print(N)
#kontrol (Nüfus değerinde iki farklı değer bulunduğu için kontrol ediyoruz)
print("SIRF-F Grafiği")

sirf_param_dict = {
    "theta": eg_theta, "kappa": eg_kappa, "rho": eg_rho, "sigma": eg_sigma
}
sirf_simulator = cs.ODESimulator(country="Example", province="SIR-F")
sirf_simulator.add(
    model=cs.SIRF, step_n= int(dtetavalue), population=eg_population,
    param_dict=sirf_param_dict,
    y0_dict={"Susceptible": eg_population, "Infected": confirmedinTurkey, "Recovered": recoveredinTurkey, "Fatal": deathinTurkey}
)
sirf_simulator.run()
sirf_simulator.non_dim().tail()

cs.line_plot(
    sirf_simulator.non_dim().set_index("t"),
    title=r"SIR-F: $R_0={0}\ (\theta={1}, \kappa={2}, \rho={3}, \sigma={4})$".format(
        eg_r0, eg_theta, eg_kappa, eg_rho, eg_sigma
    ),
    ylabel="",
    h=1
)

#grafikteki değerlerin temsilini belirtiyoruz
print("Kırmızı toplam bulaşma ihtimali olan nüfus")
print("Yeşil iyileşen hasta sayısı")
print("Mavi enfekte olan hasta sayısı")
print("Sarı ölen hasta sayısı")

#programın sonu

