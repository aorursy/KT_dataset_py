citytracker = {'Atlanta': 486290, 'Boston': 685094, 'Chicago': 2749360, 'Denver': 619968, 'Seattle': 724725}
print(citytracker["Chicago"])
citytracker["Seattle"] = 724725 + 17500 # This works, but doesn't seem to be a proper approach for intances when we don't want to list out the old value or don't know the old value. I can't find anything in the lecture slides though to point to a more elegant solution.

print(citytracker["Seattle"])
citytracker["Los Angeles"] = 45000

print(citytracker["Los Angeles"])
print("Denver: " + str(citytracker["Denver"]))
for key in citytracker:

    print(key)
for key in citytracker:

    print(key, ":", citytracker[key])
def in_dict(key):

    if key in citytracker:

        print(key + ":", citytracker[key])

    else:

        print("Sorry, that is not in the tracker.")



in_dict("New York")

in_dict("Atlanta")
potentialcities = ['Cleveland','Phoenix','Nashville','Philadelphia','Milwaukee']
for x in potentialcities:

    if x in citytracker:

        print(x + ":", citytracker[x])

    else:

        print(0)

for key in citytracker:

    print(key, ",", citytracker[key], sep="") # One way to do it



for key in citytracker:

    print('%s,%s' % (key, citytracker[key])) # Another way to do it
import os



### Add your code here



import csv

with open("popreport.csv","w",newline='') as f:

    w = csv.DictWriter(f, fieldnames = ["city", "pop"])

    w.writeheader()

    for key in citytracker:

        w.writerow({"city":key,"pop":citytracker[key]})



### This will print out the list of files in your /working directory to confirm you wrote the file.

### You can also examine the right sidebar to see your file.



for dirname, _, filenames in os.walk('/kaggle/working'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
from kaggle_secrets import UserSecretsClient

user_secrets = UserSecretsClient()

secret_value_0 = user_secrets.get_secret("OpenCage secret key") # make sure this matches the Label of your key

key1 = secret_value_0



from opencage.geocoder import OpenCageGeocode

geocoder = OpenCageGeocode(key1)

query = 'Seattle'  # replace this city with cities from the names in your citytracker dictionary

results = geocoder.geocode(query)

lat = str(results[0]['geometry']['lat'])

lng = str(results[0]['geometry']['lng'])

print ("Lat: %s, Lon: %s" % (lat, lng))
# This code retrieves your key from your Kaggle Secret Keys file

from kaggle_secrets import UserSecretsClient

user_secrets = UserSecretsClient()

secret_value_0 = user_secrets.get_secret("DarkSky secret key") # make sure this matches the Label of your key



import urllib.error, urllib.parse, urllib.request, json, datetime



def safeGet(url):

    try:

        return urllib.request.urlopen(url)

    except urllib2.error.URLError as e:

        if hasattr(e,"code"):

            print("The server couldn't fulfill the request.")

            print("Error code: ", e.code)

        elif hasattr(e,'reason'):

            print("We failed to reach a server")

            print("Reason: ", e.reason)

        return None



# lat and lon below are for UW

def getForecast(lat="47.656648",lng="-122.310233"): # default values are for UW

    # https://api.darksky.net/forecast/[key]/[latitude],[longitude]

    key2 = secret_value_0

    url = "https://api.darksky.net/forecast/"+key2+"/"+lat+","+lng

    return safeGet(url)



data = json.load(getForecast(lat,lng))

current_time = datetime.datetime.now() 

print("Retrieved at: %s" %current_time)

print(data['currently']['summary'])

print("Temperature: " + str(data['currently']['temperature']))

print(data['minutely']['summary'])
