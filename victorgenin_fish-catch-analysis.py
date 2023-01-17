import matplotlib.pyplot as plt
import pandas as pd
import re
import urllib
from xml.etree import ElementTree

try:
	fishCatchesDS = pd.read_csv(fishCatchesURL)
	fishCodesDS = pd.read_csv(fishCodesURL, sep='\t')
	countryCodesDS = pd.read_csv(countryCodesURL)

	# load area codes from xml
	file = urllib.urlopen(northAtlanticFAOCodesURL)
	data = file.read()
	file.close()
	root = ElementTree.fromstring(data)
	ns = {'role': 'http://www.fao.org/fi/figis/devcon/'}
	codes = {}
	for area in root.findall('./role:Area/role:AreaProfile/role:AreaStruct/role:Area/role:AreaIdent', ns):
	    for i, child in enumerate(area):
	        if i is 0:
	            name = re.sub('\(.*\)', '', child.text)
	            match = re.search(",", name)
	            if (match is not None):
	                name = name[:match.start()]
	        else:
	            code = child.attrib['Code']
	    codes[code] = name
	northAtlanticFAOCodesDS = pd.DataFrame(codes.items(), columns=['Code', 'Name'])
except:
    pass
try:
	# slice ds
	fishCodesDS = fishCodesDS[['3A_CODE', 'English_name']]
	countryCodesDS = countryCodesDS[countryCodesDS.columns[0:2]]
	fishCatchesDS = fishCatchesDS[fishCatchesDS.columns[0:13]]
	fishCatchesDS.drop(['Units'], axis=1, inplace=True)

	# remove all subdivisions
	fishCatchesDS = fishCatchesDS.dropna()
	fishCatchesDS = fishCatchesDS[fishCatchesDS.Area.str.contains('^27\.\d+$')]
except:
	pass
try:
	# plot catches by countries
	fishCatchesByCountry = fishCatchesDS.groupby(['Country']).sum()
	fishCatchesByCountry = pd.merge(fishCatchesByCountry, countryCodesDS, right_on='Code', left_index=True)
	plt.figure()
	ax = fishCatchesByCountry.plot(kind='bar', x=10, stacked=True, figsize=(15, 5), title='Nominal catches by country')
	ax.set_xlabel("Country")
except:
	pass
try:
	# plot catches by year
	plt.figure()
	fishCatchesByYear = fishCatchesDS.ix[:, 3:11].sum()
	ax = fishCatchesByYear.plot(kind='line', figsize=(10, 5), title='Nominal catches by year', legend=False)
except:
	pass
try:
	# plot catches by fish species
	fishCatchesBySpecies = fishCatchesDS.groupby(['Species']).sum()
	fishCatchesBySpecies = fishCatchesBySpecies.sum(axis=1)
	fishCatchesBySpecies = fishCatchesBySpecies.order(ascending=False).head(10)
	fishCatchesBySpecies = pd.merge(pd.DataFrame(fishCatchesBySpecies), fishCodesDS, how='left', right_on='3A_CODE', left_index=True)
	plt.figure()
	ax = fishCatchesBySpecies.plot(kind='bar', x='English_name', stacked=True, figsize=(10, 5), title='Nominal catches by species', legend=False)
	ax.set_xlabel("Species")
except:
	pass
try:
	# plot catches by area
	fishCatchesByArea = fishCatchesDS.groupby(['Area']).sum()
	fishCatchesByArea = fishCatchesByArea.sum(axis=1)
	fishCatchesByArea = fishCatchesByArea.order(ascending=False).head(10)
	fishCatchesByArea = pd.merge(pd.DataFrame(fishCatchesByArea), northAtlanticFAOCodesDS, how='left', right_on='Code', left_index=True)
	plt.figure()
	ax = fishCatchesByArea.plot(kind='bar', x='Name', stacked=True, figsize=(10, 5), title='Nominal catches by area', legend=False)
	ax.set_xlabel("Area")
except:
	pass