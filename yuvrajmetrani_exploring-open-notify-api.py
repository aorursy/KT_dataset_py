import requests # will be needed to send requests and get json data
import datetime # will be needed to convert UTC epoch time to formatted date time
import time
import pytz

people = requests.get('http://api.open-notify.org/astros.json')
print(request.status_code)
print(people.text)
sat = requests.get('http://api.open-notify.org/iss-now.json')
(sat.json())
pos = (sat.json())['iss_position']
pos
lat = pos['latitude']
lon = pos['longitude']
print (lat, lon)
dicp = (people.json())
dicp
# number of people in the ISS
dicp['number']
# people currently aboard the ISS
dicp['people']
for p in dicp['people']:
    print (p["name"])
#my location
lat = '12'
lon = '57'
alt_above_msl = '780'
n= '20' #number_of_times_i_want_to_see_iss

#passes = requests.get('http://api.open-notify.org/iss-pass.json?lat=12&lon=57&alt=780&n=20')
passes = requests.get('http://api.open-notify.org/iss-pass.json?lat=' + lat + '&lon=' + lon + '&alt=' + alt_above_msl + '&n=' + n +'')

passes
# Gets the time and duration of the next few passes of the ISS over a given location
passes.json()
#I used the timezone name as specified in this available list
# source : http://pytz.sourceforge.net/

from pytz import all_timezones   
all_timezones[:]


ist = pytz.timezone('Asia/Kolkata')
isspasses = passes.json()

for s in isspasses['response']:
    spottime = (s['risetime'])
    # UTC
    print(time.strftime('%Y-%m-%d %H:%M:%S %Z%z', time.localtime(spottime))) 

    # Indian Standard Time
    tim = datetime.datetime.fromtimestamp(spottime, tz= pytz.timezone('Asia/Kolkata'))
    print(tim)
