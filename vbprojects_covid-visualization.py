

import plotly.express as ex

import pandas as pd

import numpy as np

import csv

import json        
sc_to_co = dict()

i = 0

with open("/kaggle/input/state-countiess-to-coordinates/geocodes.csv") as csvfile:

    csv_reader = csv.reader(csvfile, delimiter=',')

    for row in csv_reader:

        if i == 0:

            i+=1

            continue

        row[0] = row[0].strip()

        if row[0] not in sc_to_co:

            row[3] = row[3].lower().strip()

            sc_to_co[row[0]] = dict()

            sc_to_co[row[0]][row[3]] = dict()

            sc_to_co[row[0]][row[3]]["lat"] = float(row[1]) 

            sc_to_co[row[0]][row[3]]["long"] = float(row[1]) 

        elif row[3] not in sc_to_co[row[0]]:

            if row[0] == "LA":

                KAL = row[3].find("Parish")

                if KAL != -1:

                    row[3] = row[3][0:KAL]

            row[3] = row[3].lower().strip()

            sc_to_co[row[0]][row[3]] = dict()

            sc_to_co[row[0]][row[3]]["lat"] = float(row[1]) 

            sc_to_co[row[0]][row[3]]["long"] = float(row[2])

    print("done wiht csv read")    

sc_to_co["NY"]["new york city"] = dict()

sc_to_co["NY"]["new york city"]["lat"] = 40.71

sc_to_co["NY"]["new york city"]["long"] = -74.01

# print(sc_to_co)

# #with open('COORDS.json', 'w+') as fp:

# #    json.dump(sc_to_co, fp)

# #print("done")


data = pd.read_csv("/kaggle/input/us-counties-covid-19-dataset/us-counties.csv")



data["county"] = data["county"].str.lower().str.strip()

#print(data)

us_state_abbrev = {

    'Alabama': 'AL',

    'Alaska': 'AK',

    'American Samoa': 'AS',

    'Arizona': 'AZ',

    'Arkansas': 'AR',

    'California': 'CA',

    'Colorado': 'CO',

    'Connecticut': 'CT',

    'Delaware': 'DE',

    'District of Columbia': 'DC',

    'Florida': 'FL',

    'Georgia': 'GA',

    'Guam': 'GU',

    'Hawaii': 'HI',

    'Idaho': 'ID',

    'Illinois': 'IL',

    'Indiana': 'IN',

    'Iowa': 'IA',

    'Kansas': 'KS',

    'Kentucky': 'KY',

    'Louisiana': 'LA',

    'Maine': 'ME',

    'Maryland': 'MD',

    'Massachusetts': 'MA',

    'Michigan': 'MI',

    'Minnesota': 'MN',

    'Mississippi': 'MS',

    'Missouri': 'MO',

    'Montana': 'MT',

    'Nebraska': 'NE',

    'Nevada': 'NV',

    'New Hampshire': 'NH',

    'New Jersey': 'NJ',

    'New Mexico': 'NM',

    'New York': 'NY',

    'North Carolina': 'NC',

    'North Dakota': 'ND',

    'Northern Mariana Islands':'MP',

    'Ohio': 'OH',

    'Oklahoma': 'OK',

    'Oregon': 'OR',

    'Pennsylvania': 'PA',

    'Puerto Rico': 'PR',

    'Rhode Island': 'RI',

    'South Carolina': 'SC',

    'South Dakota': 'SD',

    'Tennessee': 'TN',

    'Texas': 'TX',

    'Utah': 'UT',

    'Vermont': 'VT',

    'Virgin Islands': 'VI',

    'Virginia': 'VA',

    'Washington': 'WA',

    'West Virginia': 'WV',

    'Wisconsin': 'WI',

    'Wyoming': 'WY'

}

data["state"] = data["state"].apply(lambda x : us_state_abbrev[x])

#print(data)

data = data[(data.state != 'unknown') & (data.county != 'unknown')]

#print(data)

#data.to_pickle("uscovid")
#sc_to_co:dict

#with open("COORDS.json","r") as json_file:

#    sc_to_co = json.load(json_file)

#print(sc_to_co)

cdf = data

#cdf["cases"] = cdf["cases"].apply(lambda i : i if i == 0 else np.log(i))

#print(cdf)

def check_keys(x,y):

    try:

        return (sc_to_co[x][y]["lat"],sc_to_co[x][y]["long"])

    except Exception as e:

        #print("COULD NOT FIND %s %s" % (x,y))

        return float('nan'),float('nan')



gords = [check_keys(x,y) for x,y in zip(cdf["state"],cdf["county"])]

lats = [x[0] for x in gords]

longs = [x[1] for x in gords]

cdf["lats"] = lats

cdf["longs"] = longs

#print(cdf)

cdf = cdf[-((np.isnan(cdf.lats)) | (np.isnan(cdf.longs)))] 

#print(cdf)

#cdf.to_csv("FINAL.csv")
data = cdf

#print(data)

data.to_csv("END.csv")

fig = ex.scatter_geo(data, lat="lats", lon="longs",size="cases",animation_frame="date",scope='usa')

fig.show()