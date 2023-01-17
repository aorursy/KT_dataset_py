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
import pandas as pd

import folium

from folium import plugins

import requests, json

from datetime import datetime



colors = ['rgb(171,221,164)','rgb(254,224,139)','rgb(253,174,97)','rgb(244,109,67)','rgb(213,62,79)']

def get_color(f, volume, location):

    """get marker color based on traffic volume"""

    q = f[(f['location_name']==location)].Volume.quantile([0.25,0.5,0.75,0.9]).values

    if volume < q[0]:

        return colors[0]

    elif volume > q[0] and volume < q[1]:    

        return colors[1]

    elif volume > q[1] and volume < q[2]:    

        return colors[2]

    elif volume > q[2] and volume < q[3]:    

        return colors[3]

    else:

        return colors[4]



# Lane coordinates

lane_coord = {"2021 BLK KINNEY AVE (NW 300ft NW of Lamar)": {"None": [[-97.769321, 30.250531], [-97.770732, 30.248283]]}, "CAPITAL OF TEXAS HWY / LAKEWOOD DR": {"NB": [[-97.783021, 30.373691], [-97.787913, 30.369489]], "SB": [[-97.783289, 30.373951], [-97.788245, 30.369813]]}, "400 BLK AZIE MORTON RD (South of Barton Springs Rd)": {"SB": [[-97.765106, 30.264723], [-97.766051, 30.264064]], "NB": [[-97.765065, 30.264691], [-97.76601, 30.264034]]}, "BURNET RD / PALM WAY (IBM DRIVEWAY)": {"SB": [[-97.717408, 30.40287], [-97.718401, 30.401129]], "NB": [[-97.717271, 30.402814], [-97.718264, 30.401063]]}, "1000 BLK W CESAR CHAVEZ ST (H&B Trail Underpass)": {"EB": [[-97.761858, 30.269488], [-97.75866, 30.268068]], "WB": [[-97.761788, 30.269595], [-97.758603, 30.268185]]}, "LAMAR BLVD / SANDRA MURAIDA WAY (Lamar Bridge)": {"NB": [[-97.7570102, 30.2646134], [-97.7569485, 30.2647964], [-97.7568975, 30.2649331], [-97.75686, 30.2650814], [-97.7563369, 30.2663393], [-97.7562833, 30.2664528], [-97.7562055, 30.2666451], [-97.756117, 30.2668397], [-97.7555618, 30.2680327]], "SB": [[-97.7556771, 30.2680188], [-97.7563637, 30.2664899], [-97.7569056, 30.265174], [-97.7569243, 30.2651184], [-97.7569377, 30.2650651], [-97.7569672, 30.2649725], [-97.7571255, 30.2646389]]}, "100 BLK S CONGRESS AVE (Congress Bridge)": {"NB": [[-97.745429, 30.260732], [-97.746706, 30.257173]], "SB": [[-97.745665, 30.260834], [-97.746931, 30.257387]]}, "LAMAR BLVD / N LAMAR SB TO W 15TH RAMP": {"NB": [[-97.750308, 30.280748], [-97.750542, 30.278855]], "SB": [[-97.750682, 30.280951], [-97.750878, 30.278861]]}, "CONGRESS AVE / JOHANNA ST (Fulmore Middle School)": {"SB": [[-97.751431, 30.245143], [-97.752028, 30.24352]], "NB": [[-97.751262, 30.245105], [-97.751891, 30.243416]]}, "1612 BLK S LAMAR BLVD (Collier)": {"SB": [[-97.764003, 30.252651], [-97.76661, 30.250158]], "NB": [[-97.763778, 30.252465], [-97.766374, 30.249945]]}, "LAMAR BLVD / SHOAL CREEK BLVD": {"SB": [[-97.7487034, 30.2957997], [-97.7479631, 30.2946418], [-97.7472174, 30.2935162], [-97.747137, 30.2933356], [-97.7470994, 30.2931827], [-97.7470994, 30.2930577], [-97.7471048, 30.2929372], [-97.7470994, 30.2928677], [-97.7471155, 30.2927983], [-97.7471262, 30.2927334], [-97.7471638, 30.29265], [-97.7472442, 30.2925342], [-97.7473462, 30.2924462], [-97.7474213, 30.2923768], [-97.7474749, 30.2923397], [-97.7475607, 30.292298], [-97.7476466, 30.2922471], [-97.7477646, 30.29221], [-97.7486658, 30.2920201], [-97.7498835, 30.2917885]], "NB": [[-97.7498567, 30.2916727], [-97.7488375, 30.2918533], [-97.748124, 30.2920016], [-97.7478504, 30.2920618], [-97.7477056, 30.2920988], [-97.7475768, 30.2921452], [-97.7474481, 30.2922054], [-97.7473408, 30.2922795], [-97.7472496, 30.292349], [-97.7471799, 30.2924416], [-97.747094, 30.2925481], [-97.747035, 30.2926593], [-97.7469867, 30.2927797], [-97.7469546, 30.2929048], [-97.7469385, 30.293016], [-97.7469492, 30.2931318], [-97.7469492, 30.2932059], [-97.7469707, 30.2932846], [-97.7469975, 30.2933634], [-97.747035, 30.2934421], [-97.7472764, 30.2938451], [-97.7485693, 30.2958368]]}, "700 BLK E CESAR CHAVEZ ST": {"WB": [[-97.738738, 30.261854], [-97.736822, 30.261315]], "EB": [[-97.738778, 30.261727], [-97.736862, 30.261177]]}, "LAMAR BLVD / MANCHACA RD": {"NB": [[-97.780499, 30.244798], [-97.78322, 30.242842]], "SB": [[-97.780651, 30.244934], [-97.783367, 30.242983]]}, "CAPITAL OF TEXAS HWY / WALSH TARLTON LN": {"SB": [[-97.816003, 30.262221], [-97.810875, 30.255669]], "NB": [[-97.815402, 30.262499], [-97.810349, 30.256031]]}, "LAMAR BLVD / ZENNIA ST": {"SB": [[-97.729588, 30.321111], [-97.73158, 30.317937]], "NB": [[-97.729272, 30.321058], [-97.731325, 30.31777]]}, "BURNET RD / RUTLAND DR": {"NB": [[-97.72312, 30.386833], [-97.72461, 30.380611]], "SB": [[-97.723421, 30.386971], [-97.724816, 30.3809]]}, "CAPITAL OF TEXAS HWY / CEDAR ST": {"SB": [[-97.802593, 30.341246], [-97.804953, 30.337514]], "NB": [[-97.802292, 30.341172], [-97.804738, 30.337357]]}, "3201 BLK S LAMAR BLVD (BROKEN SPOKE)": {"None": [[-97.78467, 30.241911], [-97.789262, 30.238682]]}}
df = pd.read_csv("/kaggle/input/radar-traffic-data/Radar_Traffic_Counts.csv")

# cleanup leading space in names

df['location_name']=df.location_name.apply(lambda x: x.strip()) 

df.sample()
hourly_vol = df.groupby(['location_name','location_latitude','location_longitude','Direction','Hour']).agg({'Volume':'mean'}).reset_index()

hourly_vol.sample(2)
# Date slider - We are visualizing volume at each location by hour of day, using several years of data. Date can be ignored.

dt = datetime.today().replace(hour=0, minute=0, second=0, microsecond=0)



# construct geo json

lanes = [

    {

      "type": "Feature",

      "geometry": {

        "type": "LineString",

        "coordinates": lane_coord[row[0]][row[3]],

      },

      "properties": {

        'style':{

          'color': get_color(hourly_vol,row[5],row[0]),

          'weight': 3

        },

        "times": [dt.replace(hour=row[4], second=i).strftime("%Y-%m-%dT%T") for i in range(len(lane_coord[row[0]][row[3]]))]

      }

    }

    for idx, row in enumerate(hourly_vol.values)

  ]



data = {

  "type": "FeatureCollection",

  "features": lanes

}



m = folium.Map([30.264428, -97.751054], 

               tiles='CartoDB positron', 

               zoom_start=16,

               control_scale=True,

               attr='Radar traffic sensor data<br>from <a href="https://data.austintexas.gov/">City of Austin</a>\'s public dataset'

              )

m.add_child(plugins.TimestampedGeoJson(data

                                       ,period='PT1H'

                                       ,date_options='HH:00'

                                       ,transition_time=500

                                       ,add_last_point=False)

)

m