import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#
#  Read list derived from the The Guardian's memorial list of healthcare workers
#  who died of COVID-19
#
guardianList = pd.read_csv('../input/list-of-uk-health-workers-dead-from-covid19/guardian_list.csv', encoding='cp1252')

#
#  Plot raw number of deceased workers
#
fig,axes = plt.subplots(1,1,figsize=(20,10))
plot1 = guardianList['MED_SPEC'].value_counts().plot(kind='bar')
plt.title('Guardian UK raw # Health Workers who died of COVID-19 by specialty', fontsize=24)
plt.ylabel('number of workers', fontsize=20)
plt.xticks(fontsize=20)
plt.tight_layout()

#
# Read the medical and nursing specialties lists from the EU
# and combine them into one dataframe
#
medSpecsByCountry = pd.read_csv('../input/eu-physicians-by-medical-specialty/hlth_rs_spec_1_Data_fixed.csv')
medSpecsByCountry  = medSpecsByCountry.rename(columns={'total_number_of_docs' : 'total_number_of_workers'}) # make specialty column names match

nurseSpecsByCountry = pd.read_csv('../input/eu-nursing-and-caring-professionals/hlth_rs_prsns_1_Data_fixed.csv')
nurseSpecsByCountry = nurseSpecsByCountry.query('WSTATUS == "Practising" and UNIT=="Number"') # practising nurses only
nurseSpecsByCountry = nurseSpecsByCountry.drop(['WSTATUS'], axis=1) # column doesn't exist in doctors specialty list
nurseSpecsByCountry  = nurseSpecsByCountry.rename(columns={'total_number_of_nurses' : 'total_number_of_workers'}) # make specialty column names match
nurseSpecsByCountry  = nurseSpecsByCountry.rename(columns={'ISCO08' : 'MED_SPEC'}) # make specialty column names match
medSpecsByCountry = medSpecsByCountry.append(nurseSpecsByCountry)
#
#  Extract totals by occupation for just the UK, for just 2016.
#
specsUK = medSpecsByCountry.query("GEO=='United Kingdom' and TIME==2016") # data is spotty after 2016
#
#  Calculate totals in the Guardian list by specialty/occupation
#
specCounts = guardianList.groupby('MED_SPEC').count().name
specCountsDF = pd.DataFrame(specCounts)
specCountsDF = specCountsDF.rename(columns={'name':'deceased_workers'})
specCountsDF.index.names=['MED_SPEC']
#
# Calculate the fraction of workers in each specialty/occupation who have died
#
fracDF = specsUK.merge(specCountsDF, on='MED_SPEC')
fracDF = fracDF.assign(fraction_deceased = fracDF.deceased_workers/fracDF.total_number_of_workers)

#
# Count the workers in the guardian-derived list by specialty
#
specCounts = guardianList.groupby('MED_SPEC').count().name
specCountsDF = pd.DataFrame(specCounts)
specCountsDF = specCountsDF.rename(columns={'name':'deceased_workers'})

specCountsDF.index.names=['MED_SPEC']
fracDF = specsUK.merge(specCountsDF, on='MED_SPEC')
fracDF = fracDF.assign(fraction_deceased = fracDF.deceased_workers/fracDF.total_number_of_workers)
fracDF.index.names=['MED_SPEC']

plt.figure()
fig,axes = plt.subplots(1,1,figsize=(20,10))
plot2 = plt.bar(fracDF.MED_SPEC, fracDF.fraction_deceased)
plt.xticks(rotation='vertical', fontsize=20)
plt.title('Fraction of UK Health Workers in Guardian list who died of COVID-19, by specialty', fontsize=24)
plt.ylabel('Fraction of workers', fontsize=20)
plt.tight_layout()
UK_summary_table = fracDF[['MED_SPEC','deceased_workers','total_number_of_workers','fraction_deceased']]
UK_summary_table
fnomceoList = pd.read_csv('../input/list-of-doctors-in-italy-dead-from-covid19/fnomceo_memorial_list.csv', encoding='cp1252')

#
#  Plot raw number of deceased workers
#
fig,axes = plt.subplots(1,1,figsize=(20,10))
plot1 = fnomceoList['MED_SPEC'].value_counts().plot(kind='bar')
plt.title('FNOMCeO #physicians who died of COVID-19 by EU specialties', fontsize=24)
plt.ylabel('number of physicians', fontsize=20)
plt.xticks(fontsize=20)
plt.tight_layout()

specsByCountry = pd.read_csv('../input/eu-physicians-by-medical-specialty/hlth_rs_spec_1_Data_fixed.csv')
specsItaly = specsByCountry.query("GEO=='Italy' and TIME==2016") # data is spotty after 2016
specCounts = fnomceoList.groupby('MED_SPEC').count().name
specCountsDF = pd.DataFrame(specCounts)
specCountsDF = specCountsDF.rename(columns={'name':'deceased_docs'})

specCountsDF.index.names=['MED_SPEC']
fracDF = specsItaly.merge(specCountsDF, on='MED_SPEC')
fracDF = fracDF.assign(fraction_deceased = fracDF.deceased_docs/fracDF.total_number_of_docs)

plt.figure()
fig,axes = plt.subplots(1,1,figsize=(20,10))
plot2 = plt.bar(fracDF.MED_SPEC, fracDF.fraction_deceased)
plt.xticks(rotation='vertical')
plt.title('Fraction of Italian physicians in FNOMCeO list who died of COVID-19 by EU specialties', fontsize=24)
plt.ylabel('Fraction of physicians', fontsize=20)
plt.xticks(fontsize=20)
plt.tight_layout()
#plt.xticks(fracDF.fraction_deceased, fracDF.MED_SPEC, rotation='vertical')
italy_summary_table = fracDF[['MED_SPEC','deceased_docs','total_number_of_docs']]
italy_summary_table
from IPython.display import display, Image
display(Image(filename='../input/usrawnumbyspecpng/USrawNumBySpec.png'))
from IPython.display import display, Image
display(Image(filename='../input/usfracbyspecpng/USFracBySpec.png'))
from IPython.display import display, Image
display(Image(filename='../input/iranrawnumbyspecpng/IranRawNumBySpec.png'))