import pandas as pd
import re
js_file = open('../input/javascript-data/BMC-ContainmentZones.js')
data = js_file.read()
marker_id = []
longitude = []
latitude = []

pattern = re.compile(r'var marker_([a-z\d]+) = L.marker\W+([\d+\.*]+), ([\d+\.*]+)\W,\W+\.addTo\Wfeature_group_([a-z\d]+)\W+')
matches = pattern.finditer(data)

for match in matches:
    cord_data = match.group(0)
    print(cord_data)
    marker_id.append(match.group(1))
    latitude.append(match.group(2))
    longitude.append(match.group(3))
coordinates = pd.DataFrame({'Marker ID':marker_id,
                            'Latitude':latitude,
                            'Longitide':longitude})
coordinates.head()
coordinates.describe()
coordinates.to_csv('coordinates_data.csv')