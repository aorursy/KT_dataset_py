import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import plotly.graph_objects as go

from plotly.offline import init_notebook_mode, iplot

import plotly.express as px

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



        import warnings

warnings.filterwarnings('ignore')
province_data= pd.read_csv("/kaggle/input/number-of-cases-in-the-city-covid19-turkey/number_of_cases_in_the_city.csv")

data = pd.read_csv("/kaggle/input/covid19-in-turkey/covid_19_data_tr.csv")
data.tail()
data.info()
data.rename(columns = {"Last_Update" : "Tarih", "Confirmed" : "Vaka", "Deaths" : "Vefat", "Recovered" : "Tedavi_Edilen"}, inplace = True)

data.drop("Province/State", axis = 1, inplace = True)
data.drop([127,128], inplace = True)
test_sayisi = [0,0,0,0,0,0,0,0,1981,3656,2953,1738,3672,3952,5035,7286,7533,7641,9982,11535,15422,14396,18757,16160,19664

               ,20065,21400,20023,24900,28578,30864,33170,35720,34456,33070,34090,40427,40270,40520,35344,39703,39429,37535

               ,40962,38351,38308,30177,20143,29230,43498,42004,41431,36318,24001,35771,33283,30303,30395,33687,35605,36187

               ,32722,37351,33332,34821,38565,42236,35369,25141,25382,20838,33633,37507,40178,24589,21492,19853,21043,33559

               ,36155,39230,35600,31525,32325,52305,54234,57829,35846,35335,39361,37225,36521,49190,41013,45092,45176,42032

               ,46800,52901,48412,41316,41142,40469,41413,42982,53486,52303,51198,45213,48309,51014,50492,52313,49714,52141

               ,48248,46414,52193,50545,49302,50103,48787,48813,45232,46492,43231,42320]

data['Test Sayısı'] = test_sayisi #Test sayısı açıklanmayan günler 0 olarak belirtilmiştir. 
toplamYogunBakim = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,344,445,568,725,847,979,1101,1251,1311,1381,1415,1474,1492,1552,1667,1626,1665,1786,1809,1820,1854,1845,1894

                    ,1922,1909,1865,1814,1816,1790,1782,1776,1736,1621,1574,1514,1480,1445,1424,1384,1338,1278,1260,1219,1168,1154,1126,1045,998,963,944,906,914

                    ,903,882,877,820,800,775,769,756,739,723,683,662,649,648,651,633,612,602,592,591,613,625,642,631,643,664,684,717,722,732,745, 755,769,781,803

                    ,846,893,914,941,963,984,996,1018,1026,1035,1067,1082,1093,1127,1130,1152,1172,1179,1182,1194,1209,1223,1204,1206]

data['toplamYogunBakim'] = toplamYogunBakim #Yogun bakim hastasi sayısı açıklanmayan günler 0 olarak belirtilmiştir.
toplamEntubeSayisi = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,241,309,394,523,622,692,783,867,909,935,966,987,995,1017,1062,1021,978,1063,1087

                      ,1052,1040,1014,1054,1031,1033,1006,985,982,929,900,883,882,845,831,803,818,778,766,727,707,669,665,653,628,598,578

                      ,576,535,508,490,474,468,463,455,445,424,401,388,358,371,338,331,339,324,308,287,283,271,261,265,269,264,274,261,281

                      ,280,266,282,284,290,291,303,306,311,310,318,327,345,362,356,369,382,366,381,375,368,362,372,374,378,392,395,400,406

                      ,399,402,401,409,402,396,401]

data['toplamEntubeSayisi'] = toplamEntubeSayisi #Entübe sayısı açıklanmayan günler 0 olarak belirtilmiştir.
vaka_oran = [0] #ilk değerden önce bir değer olmadığı için ilk değeri 0 verdik

olu_oran = [0]

vaka_artis= [0]

olu_artis = [0]

test_oran = [0]

test_artis= [0]

yogun_bakim_oran = [0]

yogun_bakim_artis = [0]

entube_hasta_oran = [0]

entube_hasta_artis = [0]

yogun_bakim_vaka_orani = [0]

entube_vaka_orani = [0]

aktif_vaka = [0]



for i in range(len(data)-1):

    testOran =  round((data["Test Sayısı"][i+1] - data["Test Sayısı"][i]) / data["Test Sayısı"][i], 2)

    testArtis = data["Test Sayısı"][i+1] - data["Test Sayısı"][i] 

    vakaOran = round((data["Vaka"][i+1] - data["Vaka"][i]) / data["Vaka"][i], 2)

    vakaArtis = data["Vaka"][i+1] - data["Vaka"][i]

    oluOran =  round((data["Vefat"][i+1] - data["Vefat"][i]) / data["Vefat"][i], 2)

    olumArtis = data["Vefat"][i+1] - data["Vefat"][i]

    yogunBakimOran =  round((data["toplamYogunBakim"][i+1] - data["toplamYogunBakim"][i]) / data["toplamYogunBakim"][i], 2)

    yogunBakimArtis = data["toplamYogunBakim"][i+1] - data["toplamYogunBakim"][i] 

    entubehastaOran =  round((data["toplamEntubeSayisi"][i+1] - data["toplamEntubeSayisi"][i]) / data["toplamEntubeSayisi"][i], 2)

    entubeHastaArtis = data["toplamEntubeSayisi"][i+1] - data["toplamEntubeSayisi"][i]

    yogunBakimVakaOrani = round((data["toplamYogunBakim"][i+1] - data["toplamYogunBakim"][i]) / data["Vaka"][i], 5)

    entubeVakaOrani = round((data["toplamEntubeSayisi"][i+1] - data["toplamEntubeSayisi"][i]) / data["Vaka"][i], 5)

    aktifVaka = data["Vaka"][i] - data["Tedavi_Edilen"][i]

    



    test_oran.append(testOran)

    test_artis.append(testArtis)

    vaka_oran.append(vakaOran)

    olu_oran.append(oluOran)

    vaka_artis.append(vakaArtis)

    olu_artis.append(olumArtis)

    yogun_bakim_oran.append(yogunBakimOran)

    yogun_bakim_artis.append(yogunBakimArtis)

    entube_hasta_oran.append(entubehastaOran)

    entube_hasta_artis.append(entubeHastaArtis)

    yogun_bakim_vaka_orani.append(yogunBakimVakaOrani)

    entube_vaka_orani.append(entubeVakaOrani)

    aktif_vaka.append(aktifVaka)





data["Test Artış Sayısı"] = test_artis

data["Test Artış Oranı"] = test_oran

data["Vaka Artış Sayısı"] = vaka_artis

data["Vaka Artış Oranı"] = vaka_oran

data["Vefat Artış Sayısı"] = olu_artis

data["Vefat Artış Oranı"] = olu_oran

data["Yogun Bakim Artış Sayısı"] = yogun_bakim_artis

data["Yogun Bakim Artış Oranı"] = yogun_bakim_oran

data["Entube Hasta Artış Sayısı"] = entube_hasta_artis

data["Entube Hasta Artış Oranı"] = entube_hasta_oran

data["Yoğun Bakım/Vaka Oranı"] = yogun_bakim_vaka_orani

data["Entube/Vaka Oranı"] = entube_vaka_orani

data['Aktif Vaka'] = aktif_vaka



data.fillna(0, inplace = True) # 0/0'dan kaynaklanan NaN degerlerini 0 ile degistirdik.

data = data.replace([np.inf, -np.inf], np.nan) #sayi/0 sonsuzlugunu 0 olarak degistirdik.
test_pozitif=[]

for i in range(len(data)):

    test_pozitif_orani =  round((data["Vaka Artış Sayısı"][i] / data["Test Sayısı"][i]), 2)

    test_pozitif.append(test_pozitif_orani)

        

data["Pozitif/Test Oranı"] = test_pozitif

data = data.replace([np.inf, -np.inf], np.nan)
data.fillna(0, inplace = True)

data
fig = go.Figure(data=[

    go.Bar(name='Vaka', x=data['Tarih'], y=data['Vaka'], marker_color='rgba(135, 206, 250, 0.8)'),

    go.Bar(name='Vefat', x=data['Tarih'], y = data['Vefat'], marker_color='rgba(255, 0, 0, 0.8)'),

    go.Bar(name='Tedavi_Edilen', x=data['Tarih'], y=data['Tedavi_Edilen'], marker_color='rgba(0, 255, 0, 0.8)')

])

fig.update_layout(barmode='group', title_text='Türkiye Günlük Vaka, Vefat ve Taburcu Hasta Sayıları', xaxis_tickangle=-45)

fig.show()
data2 = data.tail(60)

fig = go.Figure(data=[

    go.Bar(name='Vaka', x=data2['Tarih'], y=data2['Vaka'], marker_color='rgba(135, 206, 250, 0.8)'),

    go.Bar(name='Vefat', x=data2['Tarih'], y = data2['Vefat'], marker_color='rgba(255, 0, 0, 0.8)'),

    go.Bar(name='Tedavi_Edilen', x=data2['Tarih'], y=data2['Tedavi_Edilen'], marker_color='rgba(0, 255, 0, 0.8)')

])

fig.update_layout(barmode='group', title_text='Türkiye Günlük Vaka, Vefat ve Taburcu Hasta Sayıları(Son 60 Gün)', xaxis_tickangle=-45)

fig.show()
fig = go.Figure(data=[

    go.Bar(name='Vefat', x=data['Tarih'], y = data['Vefat'], marker_color='rgb(255, 0, 0)'),

    go.Bar(name='Tedavi_Edilen', x=data['Tarih'] , y=data['Tedavi_Edilen'], marker_color='rgb(0, 255, 0)')

])

fig.update_layout(barmode='group', title_text='Türkiye Günlük Vefat ve Taburcu Hasta Sayıları', xaxis_tickangle=-45)

fig.show()
data2 = data.tail(60)

fig = go.Figure(data=[

    go.Bar(name='Vefat', x=data2['Tarih'], y = data2['Vefat'], marker_color='rgb(255, 0, 0)'),

    go.Bar(name='Tedavi_Edilen', x=data2['Tarih'] , y=data2['Tedavi_Edilen'], marker_color='rgb(0, 255, 0)')

])

fig.update_layout(barmode='group', title_text='Türkiye Günlük Vefat ve Taburcu Hasta Sayıları(Son 60 Gün)', xaxis_tickangle=-45)

fig.show()
fig = go.Figure(data=[go.Bar(name='Aktif Vaka', x=data['Tarih'], y=data['Aktif Vaka'], marker_color='rgba(135, 206, 250, 0.8)'),])

fig.update_layout(barmode='group', title_text='Türkiye Günlük Aktif Vaka Sayıları', xaxis_tickangle=-45, xaxis= dict(title= 'Tarih / Date'),

                  yaxis= dict(title= 'Kişi Sayısı'))

fig.show()
mart17Sonrasi = data.loc[data['Tarih'] > '3/17/2020']

mart15Sonrasi = data.loc[data['Tarih'] > '3/15/2020']

mart26Sonrasi = data.loc[data['Tarih'] > '3/27/2020']
vaka = go.Scatter(x = data.Tarih,

                    y = data.Vaka,

                    mode = "lines+markers",

                    name = "Vaka / Cases",

                    marker = dict(color = 'rgba(135, 206, 250, 0.8)'),

                    text= data.Vaka

                   )

olum = go.Scatter(x = data.Tarih,

                    y = data.Vefat,

                    mode = "lines+markers",

                    name = "Vefat / Death",

                    marker = dict(color = 'rgba(255, 0, 0, 0.8)'),

                    text= data.Vefat

                   )

taburcu = go.Scatter(x = data.Tarih,

                    y = data.Tedavi_Edilen,

                    mode = "lines+markers",

                    name = "Tedavi/Recovered",

                    marker = dict(color = 'rgba(0, 255, 0, 0.8)'),

                    text= data.Tedavi_Edilen

                   )

data2 = [vaka, olum, taburcu]

layout = dict(title = "Türkiye'deki Covid-19 Vaka, Vefat ve Tedavi Sayıları -  Covid-19 Number of Case and Deaths in Turkey", 

              xaxis= dict(title= 'Tarih / Date'), yaxis= dict(title= 'Kişi Sayısı'), xaxis_tickangle=-45)

fig = dict(data = data2, layout = layout)

iplot(fig)
olum = go.Scatter(x = data.Tarih,

                    y = data.Vefat,

                    mode = "lines+markers",

                    name = "Vefat / Death",

                    marker = dict(color = 'rgba(255, 0, 0, 0.8)'),

                    text= data.Vefat

                   )

taburcu = go.Scatter(x = data.Tarih,

                    y = data.Tedavi_Edilen,

                    mode = "lines+markers",

                    name = "Tedavi/Recovered",

                    marker = dict(color = 'rgba(0, 255, 0, 0.8)'),

                    text= data.Tedavi_Edilen

                   )

data2 = [olum,taburcu]

layout = dict(title = "Türkiye'deki Covid-19 Vefat ve Tedavi Sayıları - Covid-19 Number of Deaths and Recovered in Turkey", 

              xaxis= dict(title= 'Tarih / Date'), yaxis= dict(title= 'Kişi Sayısı'), xaxis_tickangle=-45)

fig = dict(data = data2, layout = layout)

iplot(fig)
vaka = go.Scatter(x =mart15Sonrasi.Tarih,

                    y = mart15Sonrasi.Vaka,

                    mode = "lines+markers",

                    name = "Vaka / Cases",

                    marker = dict(color = 'rgba(135, 206, 250, 0.8)'),

                    text= mart15Sonrasi.Vaka

                   )

layout = dict(title = "Türkiye'deki Covid-19 Vaka Sayıları (Logaritmik) -  Covid-19 Number of Case in Turkey (Logarithmic)", 

              xaxis= dict(title= 'Tarih / Date'), yaxis= dict(title= 'Kişi Sayısı'), xaxis_tickangle=-45, yaxis_type="log")

fig = dict(data = vaka, layout = layout)

iplot(fig)
Vefat = go.Scatter(x =mart17Sonrasi.Tarih,

                    y = mart17Sonrasi.Vefat,

                    mode = "lines+markers",

                    name = "Vefat / Death",

                    marker = dict(color = 'rgba(255, 0, 0, 0.8)'),

                    text= mart17Sonrasi.Vefat

                   )

layout = dict(title = "Türkiye'deki Covid-19 Vefat Sayıları (Logaritmik) -  Covid-19 Number of Death in Turkey (Logarithmic)", 

              xaxis= dict(title= 'Tarih / Date'), yaxis= dict(title= 'Kişi Sayısı'), xaxis_tickangle=-45, yaxis_type="log")

fig = dict(data = Vefat, layout = layout)

iplot(fig)
yogunBakim = go.Scatter(x =mart26Sonrasi.Tarih,

                    y = mart26Sonrasi.toplamYogunBakim,

                    mode = "lines+markers",

                    name = "toplamYogunBakim",

                    marker = dict(color = 'rgba(72,118 ,255, 0.8)'),

                    text= mart26Sonrasi.toplamYogunBakim

                   )

entubeHasta = go.Scatter(x =mart26Sonrasi.Tarih,

                    y = mart26Sonrasi.toplamEntubeSayisi,

                    mode = "lines+markers",

                    name = "toplamEntubeSayisi",

                    marker = dict(color = 'rgba(139, 90, 43, 0.8)'),

                    text= mart26Sonrasi.toplamEntubeSayisi

                   )

data2 = [yogunBakim,entubeHasta]

layout = dict(title = "Türkiye'deki Covid-19 Yogun Bakım ve Entube Hasta Sayıları", 

              xaxis= dict(title= 'Tarih / Date'), yaxis= dict(title= 'Kişi Sayısı'), xaxis_tickangle=-45)

fig = dict(data = data2, layout = layout)

iplot(fig)
yogunBakim = go.Scatter(x =mart26Sonrasi.Tarih,

                    y = mart26Sonrasi.toplamYogunBakim,

                    mode = "lines+markers",

                    name = "toplamYogunBakim",

                    marker = dict(color = 'rgba(72,118 ,255,0.8)'),

                    text= mart26Sonrasi.toplamYogunBakim

                   )

entubeHasta = go.Scatter(x =mart26Sonrasi.Tarih,

                    y = mart26Sonrasi.toplamEntubeSayisi,

                    mode = "lines+markers",

                    name = "toplamEntubeSayisi",

                    marker = dict(color = 'rgba(139, 90, 43, 0.8)'),

                    text= mart26Sonrasi.toplamEntubeSayisi

                   )

data2 = [yogunBakim,entubeHasta]

layout = dict(title = "Türkiye'deki Covid-19 Yogun Bakım ve Entube Hasta Sayıları(Logaritmik)", 

              xaxis= dict(title= 'Tarih / Date'), yaxis= dict(title= 'Kişi Sayısı'), xaxis_tickangle=-45, yaxis_type="log")

fig = dict(data = data2, layout = layout)

iplot(fig)
yogunBakimOran = go.Scatter(x =mart26Sonrasi.Tarih,

                    y = mart26Sonrasi['Yoğun Bakım/Vaka Oranı'],

                    mode = "lines+markers",

                    name = "Yoğun Bakım/Vaka Oranı",

                    marker = dict(color = 'rgba(72,118 ,255, 0.8)'),

                    text= mart26Sonrasi['Yoğun Bakım/Vaka Oranı']

                   )

entubeHastaOran = go.Scatter(x =mart26Sonrasi.Tarih,

                    y = mart26Sonrasi['Entube/Vaka Oranı'],

                    mode = "lines+markers",

                    name = "Entube/Vaka Oranı",

                    marker = dict(color = 'rgba(139, 90, 43, 0.8)'),

                    text= mart26Sonrasi['Entube/Vaka Oranı']

                   )

data2 = [yogunBakimOran,entubeHastaOran]

layout = dict(title = "Türkiye'deki Covid-19 Yogun Bakım ve Entube Hasta Sayılarının Vaka Sayılarına Oranlariı", 

              xaxis= dict(title= 'Tarih / Date'), yaxis= dict(title= 'Kişi Sayısı'), xaxis_tickangle=-45)

fig = dict(data = data2, layout = layout)

iplot(fig)
vaka = go.Scatter(x = mart26Sonrasi.Tarih,

                    y = mart26Sonrasi.Vaka,

                    mode = "lines+markers",

                    name = "Vaka / Cases",

                    marker = dict(color = 'rgba(135, 206, 250, 0.8)'),

                    text= mart26Sonrasi.Vaka

                   )

yogunBakim = go.Scatter(x =mart26Sonrasi.Tarih,

                    y = mart26Sonrasi.toplamYogunBakim,

                    mode = "lines+markers",

                    name = "toplamYogunBakim",

                    marker = dict(color = 'rgba(72,118 ,255, 0.8)'),

                    text= mart26Sonrasi.toplamYogunBakim

                   )

entubeHasta = go.Scatter(x =mart26Sonrasi.Tarih,

                    y = mart26Sonrasi.toplamEntubeSayisi,

                    mode = "lines+markers",

                    name = "toplamEntubeSayisi",

                    marker = dict(color = 'rgba(139, 90, 43, 0.8)'),

                    text= mart26Sonrasi.toplamEntubeSayisi

                   )

data2 = [vaka,yogunBakim,entubeHasta]

layout = dict(title = "Türkiye'deki Covid-19 Vaka, Yogun Bakım ve Entube Hasta Sayıları(Logaritmik)", 

              xaxis= dict(title= 'Tarih / Date'), yaxis= dict(title= 'Kişi Sayısı'), xaxis_tickangle=-45, yaxis_type="log")

fig = dict(data = data2, layout = layout)

iplot(fig)
vaka_oran_ = go.Scatter(x = data.Tarih,

                    y = data['Vaka Artış Oranı'],

                    mode = "lines+markers",

                    name = "Vaka Artış Oranı",

                    marker = dict(color = 'rgba(135, 206, 250, 0.8)'),

                    text= data['Vaka Artış Oranı']

                   )

olu_oran_ = go.Scatter(x = data.Tarih,

                    y = data['Vefat Artış Oranı'],

                    mode = "lines+markers",

                    name = "Vefat Artış Oranı",

                    marker = dict(color = 'rgba(255, 0, 0, 0.8)'),

                    text= data['Vefat Artış Oranı']

                   )

pozitif_test_oran = go.Scatter(x = data.Tarih,

                    y = data['Pozitif/Test Oranı'],

                    mode = "lines+markers",

                    name = "Pozitif/Test Oranı",

                    marker = dict(color = 'rgba(220, 218, 68, 0.8)'),

                    text= data['Pozitif/Test Oranı']

                   )

data2 = [vaka_oran_, olu_oran_, pozitif_test_oran]

layout = dict(title = "Türkiye'deki Covid-19 Günlük Oranlar - Covidi-19 Daily Rates in Turkey", 

              xaxis= dict(title= 'Tarih / Date'), yaxis= dict(title= 'Oran / Rate'), xaxis_tickangle=-45)

fig = dict(data = data2, layout = layout)

iplot(fig)
vaka_oran_ = go.Scatter(x = data.Tarih,

                    y = data['Vaka Artış Oranı'],

                    mode = "lines+markers",

                    name = "Vaka Artış Oranı",

                    marker = dict(color = 'rgba(135, 206, 250, 0.8)'),

                    text= data['Vaka Artış Oranı']

                   )

layout = dict(title = "Türkiye'deki Covid-19 Vaka Artış Oranı", 

              xaxis= dict(title= 'Tarih / Date'), yaxis= dict(title= 'Oran / Rate'), xaxis_tickangle=-45)

fig = dict(data = vaka_oran_, layout = layout)

iplot(fig)
olu_oran_ = go.Scatter(x = data.Tarih,

                    y = data['Vefat Artış Oranı'],

                    mode = "lines+markers",

                    name = "Vefat Artış Oranı",

                    marker = dict(color = 'rgba(255, 0, 0, 0.8)'),

                    text= data['Vefat Artış Oranı']

                   )

layout = dict(title = "Türkiye'deki Covid-19 Vefat Artış Oranı", 

              xaxis= dict(title= 'Tarih / Date'), yaxis= dict(title= 'Oran / Rate'), xaxis_tickangle=-45)

fig = dict(data = olu_oran_, layout = layout)

iplot(fig)
test = go.Scatter(x = data.Tarih,

                    y = data['Test Sayısı'],

                    mode = "lines+markers",

                    name = "Test Sayısı",

                    marker = dict(color = 'rgba(220, 218, 68, 0.8)'),

                    text= data['Test Sayısı']

                   )

layout = dict(title = "Türkiye'deki Covid-19 Test Sayısı", 

              xaxis= dict(title= 'Tarih / Date'), yaxis= dict(title= 'Sayı'), xaxis_tickangle=-45)

fig = dict(data = test, layout = layout)

iplot(fig)
pozitif_test_oran = go.Scatter(x = data.Tarih,

                    y = data['Pozitif/Test Oranı'],

                    mode = "lines+markers",

                    name = "Pozitif/Test Oranı",

                    marker = dict(color = 'rgba(220, 218, 68, 0.8)'),

                    text= data['Pozitif/Test Oranı']

                   )

layout = dict(title = "Türkiye'deki Covid-19 Pozitif/Test Oranı", 

              xaxis= dict(title= 'Tarih / Date'), yaxis= dict(title= 'Oran / Rate'), xaxis_tickangle=-45)

fig = dict(data = pozitif_test_oran, layout = layout)

iplot(fig)
vaka_artis = go.Scatter(x = data.Tarih,

                        y = data['Vaka Artış Sayısı'],

                    mode = "lines+markers",

                    name = "Vaka Artış Sayısı",

                    marker = dict(color = 'rgba(135, 206, 250, 0.8)'),

                    text= data['Vaka Artış Sayısı']

                   )

test_artis = go.Scatter(x = data.Tarih,

                        y = data['Test Artış Sayısı'],

                    mode = "lines+markers",

                    name = "Test Artış Sayısı",

                    marker = dict(color = 'rgba(220, 218, 68, 0.8)'),

                    text= data['Test Artış Sayısı']

                   )

data2 =[vaka_artis,test_artis] 

layout = dict(title = "Türkiye'deki Covid-19 Vaka Artış ve Test Artış Sayıları", 

              xaxis= dict(title= 'Tarih / Date'), yaxis= dict(title= 'Oran / Rate'), xaxis_tickangle=-45)

fig = dict(data = data2, layout = layout)

iplot(fig)
province_data.head()
province_data.info()
province_data.rename(columns = {"Province" : "Sehir", "Number of Case" : "Vaka Sayisi"}, inplace = True)
province_data.sort_values(by=['Vaka Sayisi'], ascending=False, inplace = True)
#province_df = province_data.head(10)

fig = px.pie(province_data.head(10), values='Vaka Sayisi', names='Sehir', title='Şehirlerdeki Vaka Sayıları')

fig.show()
province_df2 = province_data[1:]

fig = px.pie(province_df2, values='Vaka Sayisi', names='Sehir', title='İstanbul Dışındaki Şehirlerin Vaka Sayıları', 

             hover_data=['Vaka Sayisi'])

fig.update_traces(textposition='inside', textinfo='percent+label')

fig.show()
fig = px.bar(province_data, x="Sehir", y="Vaka Sayisi", title='Şehirlerdeki Vaka Sayıları')

fig.update_layout(barmode='group')

fig.show()
fig = px.bar(province_df2, x="Sehir", y="Vaka Sayisi", title='İstanbul Dışındaki Şehirlerin Vaka Sayıları')

fig.update_layout(barmode='group')

fig.show()
fig = px.bar(province_df2.head(15), x="Sehir", y="Vaka Sayisi", title='İstanbul Dışındaki İlk 15 Şehirin Vaka Sayıları')

fig.update_layout(barmode='group')

fig.show()