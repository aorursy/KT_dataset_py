import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



ds = pd.read_csv("/kaggle/input/youtube-new/USvideos.csv")



#actividad 1

ds.tail(10)



#actividad 2

ds.loc[ds.channel_title == 'The Deal Guy']



#actividad 3

ds.iloc[5000]



#actividad 4

ds.loc[ds.likes>= 5000000]



#actividad 5

sum(ds.likes[ds.channel_title=='iHasCupquake'])

       

#actividad 6

ds.plot.hist([ds.channel_title=='iHasCupquake'])

ds.groupby('likes')['trending_date'].sum().plot(kind='bar',legend='Reverse')
import pandas as pd

CAvideos = pd.read_csv("../input/youtube-new/CAvideos.csv")

DEvideos = pd.read_csv("../input/youtube-new/DEvideos.csv")

FRvideos = pd.read_csv("../input/youtube-new/FRvideos.csv")

GBvideos = pd.read_csv("../input/youtube-new/GBvideos.csv")

INvideos = pd.read_csv("../input/youtube-new/INvideos.csv")

JPvideos = pd.read_csv("../input/youtube-new/JPvideos.csv")

KRvideos = pd.read_csv("../input/youtube-new/KRvideos.csv")

MXvideos = pd.read_csv("../input/youtube-new/MXvideos.csv")

RUvideos = pd.read_csv("../input/youtube-new/RUvideos.csv")

USvideos = pd.read_csv("../input/youtube-new/USvideos.csv")