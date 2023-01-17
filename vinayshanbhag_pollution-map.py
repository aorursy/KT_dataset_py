import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
df = pd.read_csv('../input/IndiaAffectedWaterQualityAreas.csv',encoding='latin1',skiprows=[0], names=['state','district','block','panchayat','village','habitation','pollutant','year'])
import matplotlib.pyplot as plt

from mpl_toolkits.basemap import Basemap

from geopy import geocoders

import math

import re

%matplotlib inline
df['district'] = df.district.apply(lambda x: str(x).split('(')[0])

df['block'] = df.block.apply(lambda x: str(x).split('(')[0])

df['panchayat'] = df.panchayat.apply(lambda x: str(x).split('(')[0])

df['village'] = df.village.apply(lambda x: str(x).split('(')[0])

df['habitation'] = df.habitation.apply(lambda x: str(x).split('(')[0])



# Keep just the year. All dates are 1/4/20XX anyway

df['year'] = df.year.apply(lambda x: str(x).split('/')[-1])
g = df.groupby(['pollutant','district','year']).size().unstack()

g.fillna('NA',inplace=True)

g
from IPython.display import Image

from IPython.core.display import HTML 

Image(url= "https://raw.githubusercontent.com/vinayshanbhag/coordinates/master/India-water-pollution-plot.png", width=700,height=700)