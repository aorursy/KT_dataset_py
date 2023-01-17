#Author Justin Neumann



import json

import pandas as pd

import gpxpy as gpx

import gpxpy.gpx
#import training data

with open('../input/train.json') as data_file:

    data = json.load(data_file)

train = pd.DataFrame(data)



train.head(1)
# create gpx file as from https://pypi.python.org/pypi/gpxpy/0.8.8

gpx = gpxpy.gpx.GPX()



for index, row in train.iterrows():

    #print (row['latitude'], row['longitude'])



    if row['interest_level'] == 'high': #opting for all nominals results in poor performance of Google Earth

        gps_waypoint = gpxpy.gpx.GPXWaypoint(row['latitude'],row['longitude'],elevation=10)

        gpx.waypoints.append(gps_waypoint)



# You can add routes and waypoints, too...
filename = "test.gpx"

FILE = open(filename,"w")

FILE.writelines(gpx.to_xml())

FILE.close()

print ('Created GPX:')