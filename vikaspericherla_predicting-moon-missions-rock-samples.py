import pandas as pd

import numpy as np

import matplotlib.pyplot as mpl
rock_samples = pd.read_csv('../input/moon-rock-samples/rocksamples.csv')
rock_samples.head(10)
rock_samples.info()
rock_samples['Weight(g)'] = rock_samples['Weight(g)'].apply(lambda x : x* 0.001)

rock_samples.rename(columns={'Weight(g)':'Weight(Kg)'},inplace = True)

rock_samples.head()
rock_samples.groupby('Type')['Weight(Kg)'].count()
missions = pd.DataFrame()

missions['Mission'] = rock_samples['Mission'].unique()

print(missions)
sample_total_weight = rock_samples.groupby('Mission')['Weight(Kg)'].sum()

missions =pd.merge(missions,sample_total_weight, on='Mission')

missions.rename(columns={'Weight(Kg)':'Sample_Weight(Kg)'}, inplace = True)

missions
missions['weight_diff'] = missions['Sample_Weight(Kg)'].diff()

missions
missions['weight_diff'] = missions['weight_diff'].fillna(value=0)

missions

missions['Lunar Module (LM)'] = {'Eagle (LM-5)', 'Intrepid (LM-6)', 'Antares (LM-8)', 'Falcon (LM-10)', 'Orion (LM-11)', 'Challenger (LM-12)'}

missions['LM Mass (kg)'] = {15103, 15235, 15264, 16430, 16445, 16456}

missions['LM Mass Diff'] = missions['LM Mass (kg)'].diff()

missions['LM Mass Diff'] = missions['LM Mass Diff'].fillna(value=0)



missions['Command Module (CM)'] = {'Columbia (CSM-107)', 'Yankee Clipper (CM-108)', 'Kitty Hawk (CM-110)', 'Endeavor (CM-112)', 'Casper (CM-113)', 'America (CM-114)'}

missions['CM Mass (kg)'] = {5560, 5609, 5758, 5875, 5840, 5960}

missions['CM Mass Diff'] = missions['CM Mass (kg)'].diff()

missions['CM Mass Diff'] = missions['CM Mass Diff'].fillna(value=0)



missions
missions['Total Weight(Kg)'] = missions['CM Mass (kg)'] + missions['LM Mass (kg)']

missions['Total Weight Diff'] = missions['LM Mass Diff'] + missions['CM Mass Diff']

missions
missions.describe()
# Sample-to-weight ratio

saturnVPayload = 43500

missions['Crewed Area : Payload'] = missions['Total Weight(Kg)'] / saturnVPayload

missions['Sample : Crewed Area'] = missions['Sample_Weight(Kg)'] / missions['Total Weight(Kg)']

missions['Sample : Payload'] = missions['Sample_Weight(Kg)'] / saturnVPayload

missions
crewedArea_payload_ratio = missions['Crewed Area : Payload'].mean()

sample_crewedArea_ratio = missions['Sample : Crewed Area'].mean()

sample_payload_ratio = missions['Sample : Payload'].mean()

print(crewedArea_payload_ratio)

print(sample_crewedArea_ratio)

print(sample_payload_ratio)
artemis_crewedArea = 26520

artemis_mission = pd.DataFrame({'Mission':['artemis1','artemis1b','artemis2'],

                                 'Total Weight (kg)':[artemis_crewedArea,artemis_crewedArea,artemis_crewedArea],

                                 'Payload (kg)':[26988, 37965, 42955]})

artemis_mission
artemis_mission['Sample Weight from Total (kg)'] = artemis_mission['Total Weight (kg)'] * sample_crewedArea_ratio

artemis_mission['Sample Weight from Payload (kg)'] = artemis_mission['Payload (kg)'] * sample_payload_ratio

artemis_mission
artemis_mission['Estimated Sample Weight (kg)'] = (artemis_mission['Sample Weight from Payload (kg)'] + artemis_mission['Sample Weight from Total (kg)'])/2

artemis_mission