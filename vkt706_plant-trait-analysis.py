!pip install lofo-importance
import warnings
warnings.filterwarnings("ignore")
from lofo import LOFOImportance, plot_importance
from scipy import stats
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import os
import matplotlib.pyplot as plt
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
data = pd.read_csv("/kaggle/input/plant-trait-dataset/combined_data.csv", chunksize = 15000000, low_memory=False)
for i,d in enumerate(data):
    #print(i)
    df = d
    break
data_names = df['DataName'].unique()
imp_cols =['DatasetID','AccSpeciesID', 'AccSpeciesName', 'TraitID','TraitName', 'DataID', 
       'DataName', 'OriglName', 'OrigValueStr','OrigUnitStr']
N_fix_cap = df[df['DataName'] == 'Nitrogen-fixation capacity']
N_fix_cap = N_fix_cap[imp_cols]

unique_vals = N_fix_cap['OrigValueStr'].unique()
N_fix_cap = N_fix_cap[N_fix_cap['OrigValueStr'].notna()]
N_fix_cap.head()
# Cleaning the data
vals_to_replace = {'NO-N-fixer':True, 'N-FIXER':True, 'not N2 fixing':False, 'N2 fixing?':True, '0':False, '1':True, '2':True,
                   'High':True, 'Low':False, 'N2 fixing':True, 'no':False, 'n':False, 'y':True, 'no, not an N fixer':False,
                   'yes, an N fixer':True, 'Medium':True, 'No':False, 'Yes':True, 'yes':True, True:True, False:False, 'N':False, 'Y':True}
N_fix_cap['OrigValueStr'] = N_fix_cap['OrigValueStr'].map(vals_to_replace)
N_fix_cap['OrigUnitStr'] = 'boolean'
N_values = pd.concat([N_fix_cap['AccSpeciesName'],pd.get_dummies(N_fix_cap['OrigValueStr'])], axis = 1)
N_grouped = N_values.groupby('AccSpeciesName').sum().reset_index(drop=False)
N_grouped.columns = ['AccSpeciesName','False','True']

N_grouped['%True'] = N_grouped['True']*100/(N_grouped['True']+N_grouped['False'])
N_grouped['%False'] = 100 - N_grouped['%True'] 
N_grouped.head()
N_fixed_species = N_grouped[N_grouped['True'] > N_grouped['False']]['AccSpeciesName'].values
N_not_fixed_species = N_grouped[N_grouped['True'] < N_grouped['False']]['AccSpeciesName'].values
unknown_N_fixed_species = N_grouped[N_grouped['True'] == N_grouped['False']]['AccSpeciesName'].values

print("Plants that have nitrogen(N) fixation:  ",N_fixed_species)
print()
print("Plants don't have nitrogen(N) fixation:  ",N_not_fixed_species)
# Plants growth form
plant_growth = df[df['DataName'] == 'Plant growth form']
plant_growth = plant_growth[imp_cols]

plant_growth['OrigValueStr'] = plant_growth['OrigValueStr'].apply(lambda x: str(x).lower())

tree = ['t','tree (evergreen)','tre','Drink','tree | tree','trees','tree (woody >4m)','smtree']
tree_shurb = ['tree shrub intermediate','tree / shrub','deciduous shrub or tree','tree, shrub','shrub|tree','tree/large shrub','shrub / tree','shrub, tree','shrub or tree','shrub|shrub|tree','tree|shrub|shrub','subshrub, shrub, tree']
shrub = ['low to high shrub','s','subshrub, shrub','large shrub','subshrub (woody <1m)','sub-shrub (chamaephyte)','sub-shrub','shrub (woody 1-4m)','sh']
grass = ['grasslike','c4 grass','c3 grass','grass (poaceae only)','grass (tussock)','annual grass']
plant_growth['OrigValueStr'] = plant_growth['OrigValueStr'].replace("f",'forb').replace('forb (herbaceous, with or without woody base)','forb').replace('shrub, graminoid','shrub/graminoid').replace(dict.fromkeys(tree, 'tree')).replace(dict.fromkeys(tree_shurb, 'shrub/tree')).replace('erect dwarf shrub','dwarf shrub').replace(dict.fromkeys(shrub, 'shrub')).replace(dict.fromkeys(grass, 'grass'))
# Grouping the species by most frequent Plant growth form
plant_growth_grouped = plant_growth.groupby('AccSpeciesName')['OrigValueStr'].agg(lambda x:x.value_counts().index[0]).reset_index(drop=False)
plant_growth_grouped.columns = ['AccSpeciesName','Plant growth form']
plant_growth_grouped.head()
n_species = 10
bar_data = pd.DataFrame(plant_growth_grouped['Plant growth form'].value_counts()).reset_index(drop=False)
bar_data.columns = ['Plant growth Form','Number of Species']
sns.barplot( x= bar_data['Plant growth Form'][:n_species],y = bar_data['Number of Species'][:n_species]).set_title('Species with different growth forms')
plt.xticks(rotation=90)
plt.tight_layout()
nitrogen_content = df[df['TraitName'] == 'Leaf nitrogen (N) content per leaf dry mass']
nitrogen_content = nitrogen_content[imp_cols]
nitrogen_content = nitrogen_content[nitrogen_content['OrigValueStr'].notna()] # dropping nan values
# Cleaning the data
mg_g = ['mg g-1','mg / g','mg_g-1','mg N g-1','mg/g dry mass','g/kg','g kg-1','g mg-1']
percent = ['%','% mass/mass','mg/mg *100']
nitrogen_content['OrigUnitStr']  = nitrogen_content['OrigUnitStr'].replace(dict.fromkeys(mg_g, 'mg/g')).replace('g N g-1 DW', 'g/g').replace(dict.fromkeys(percent, 'percent'))
# Convert OrigValueStr values string to float type
arr = []
for i in nitrogen_content['OrigValueStr']:
    try: 
        float(i)
    except:
        if i not in arr:
            arr.append(i)
            
nitrogen_content['OrigValueStr'] = nitrogen_content['OrigValueStr'].replace('20-30','25').replace('10-20','15').replace('5-10','7.5').replace('>30','30').replace('<5','5')
nitrogen_content['OrigValueStr'] = nitrogen_content['OrigValueStr'].astype(float)
# Converting values with different units to mg/g

n_percent = nitrogen_content[nitrogen_content['OrigUnitStr'] == 'percent']
n_percent['OrigValueStr'] = n_percent['OrigValueStr']*10
nitrogen_content[nitrogen_content['OrigUnitStr'] == 'percent'] = n_percent
nitrogen_content['OrigUnitStr'] = nitrogen_content['OrigUnitStr'].replace('percent','mg/g')

n_g_g = nitrogen_content[nitrogen_content['OrigUnitStr'] ==  'g/g']
n_g_g['OrigValueStr'] = n_g_g['OrigValueStr']*1000
nitrogen_content[nitrogen_content['OrigUnitStr'] == 'g/g'] = n_g_g
nitrogen_content['OrigUnitStr'] = nitrogen_content['OrigUnitStr'].replace( 'g/g','mg/g')

# mmol/g contains <0.01%, for now, we drop mmol/g
nitrogen_content = nitrogen_content[nitrogen_content['OrigUnitStr'] == 'mg/g']
# Distribution of Nitrogen Content in different_species

import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

# Species
species_list = list(dict(nitrogen_content['AccSpeciesName'].value_counts()).keys())

off_set = 1  
shape = (2,4)
fig, axes =plt.subplots(shape[0],shape[1], figsize=(20,6), sharex=True)
axes = axes.flatten()
object_bol = df.dtypes == 'object'
num_species = shape[0]*shape[1]
for ax, species_name in zip(axes, species_list[num_species*off_set: num_species*(1+off_set)]):
    #sns.countplot(y=catplot, data=df, ax=ax, order=np.unique(df.values))
    species_data = nitrogen_content[nitrogen_content['AccSpeciesName'] == species_name]['OrigValueStr'].values
    sns.distplot(species_data, ax=ax).set_title(species_name)

plt.tight_layout()  
plt.show()

nitrogen_content_grouped = nitrogen_content.groupby(['AccSpeciesName'], as_index=False).agg({'OrigValueStr':['mean','std']})
nitrogen_content_grouped.columns = ['AccSpeciesName','nitrogen_content_mean','nitrogen_content_std']
nitrogen_content_grouped.head()
%matplotlib inline
importance_df = nitrogen_content_grouped.copy()
importance_df.columns= ['feature','importance_mean','importance_std']
plot_importance(importance_df[:20], figsize=(12, 8))
fig,ax = plt.subplots(figsize=(12,4))
sns.distplot(nitrogen_content_grouped['nitrogen_content_mean'], bins = 300).set_title("Distribution of Avg. Nitrogen Content in different species")
ax.set(xlim=(0, 100))
phosy_rate = df[df['TraitName'] == 'Leaf photosynthesis rate per leaf area']
phosy_rate = phosy_rate[imp_cols]
phosy_rate = phosy_rate[phosy_rate['OrigValueStr'].notna()] # dropping nan values
mmms = ['micro mol m-2 s-1','micromol m-2 s-1','micro mol/m2/s','micromol/m2/s','micro mol CO2 m-2 s-1','micro mol C m-2 s-1','micromol. m-2. s-1', ' micro mol m-2 s-1','micromoles/m2/s','micromolco2 m-2 s-1','micromol CO2 m-2 s-1','micromol CO2 m-1s-1','umol CO2 / m2 / sec','umol CO2/m^2 s','umolCO2/m2-s','umol/m2/s','umol CO2 m-2 s-1','umol CO2/m2/s','mol(CO2) m-2 s-1 ']
gmd = ['g m-2 day-1','g/m2/day','g/m2/d']
gcmd = ['g/cm2/d','g/cm2/day']

phosy_rate.OrigUnitStr = phosy_rate.OrigUnitStr.replace(dict.fromkeys(mmms, 'mmol/m2/s')).replace(dict.fromkeys(gmd, 'g/m2/day')).replace(dict.fromkeys(gcmd, 'g/cm2/day')).replace(np.nan, 'mmol/m2/s')

print(phosy_rate.OrigUnitStr.value_counts())
phosy_rate = phosy_rate[phosy_rate['OrigUnitStr']=='mmol/m2/s']

phosy_rate['OrigValueStr'] = phosy_rate['OrigValueStr'].astype('float')
phosy_rate.head()
# Distribution of Leaf photosynthesis rate per leaf area in different_species

import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

# Species
species_list = list(dict(phosy_rate['AccSpeciesName'].value_counts()).keys())

off_set = 1  
shape = (2,4)
fig, axes =plt.subplots(shape[0],shape[1], figsize=(20,6), sharex=True)
axes = axes.flatten()
object_bol = df.dtypes == 'object'
num_species = shape[0]*shape[1]

for ax, species_name in zip(axes, species_list[num_species*off_set: num_species*(1+off_set)]):
    #sns.countplot(y=catplot, data=df, ax=ax, order=np.unique(df.values))
    species_data = phosy_rate[phosy_rate['AccSpeciesName'] == species_name]['OrigValueStr'].values
    sns.distplot(species_data, ax=ax).set_title(species_name)

plt.tight_layout()  
plt.show()
phosy_rate_grouped = phosy_rate.groupby(['AccSpeciesName'], as_index=False).agg({'OrigValueStr':['mean','std']})
phosy_rate_grouped.columns = ['AccSpeciesName','phosy_rate_mean','phosy_rate_std']
phosy_rate_grouped.head()
fig,ax = plt.subplots(figsize=(12,4))
sns.distplot(phosy_rate_grouped[phosy_rate_grouped['phosy_rate_mean']<100].phosy_rate_mean, bins = 100).set_title("Distribution of Leaf Photosynthesis rate in all species")
ax.set(xlim=(0, 100))
photosyn_pathway = df[df['TraitName']=='Leaf photosynthesis pathway'][imp_cols]
c3 = ['C3?','C3.','3','c3']
c4 = ['C4','C4?','c4']
nan = ['http://tropical.theferns.info/viewtropical.php?id=Vochysia+haenkeana','unknown','no','yes','C3/C4/CAM','C3/C4','C3/CAM','C4/CAM']
photosyn_pathway['OrigValueStr'] = photosyn_pathway['OrigValueStr'].replace(dict.fromkeys(c3, 'C3')).replace('CAM?','CAM').replace(dict.fromkeys(c4, 'C4')).replace(dict.fromkeys(nan, np.nan))
photosyn_pathway = photosyn_pathway[photosyn_pathway['OrigValueStr'].notna()]

photosyn_pathway_grouped = photosyn_pathway.groupby('AccSpeciesName')['OrigValueStr'].agg(lambda x:x.value_counts().index[0]).reset_index(drop=False)
bar_data = pd.DataFrame(photosyn_pathway_grouped.OrigValueStr.value_counts()).reset_index(drop=False)
bar_data.columns = ['type','No of species']
sns.barplot(x = bar_data['type'], y= bar_data['No of species']).set_title('Species with different Leaf photosynthesis pathway')
carbon_content = df[df['TraitName'] == 'Leaf carbon (C) content per leaf dry mass'][imp_cols]
carbon_content = carbon_content[carbon_content['OrigValueStr'].notna()]
carbon_content = carbon_content[carbon_content['OrigValueStr']!='na']
carbon_content.OrigValueStr = carbon_content.OrigValueStr.astype('float')
mg_g = ['mg/g','g/kg','mg g-1','mg/g dry mass','g mg-1']
g_g = ['g/g','g C g-1 DW']
percent = ['%','percent','mg/mg *100']
carbon_content['OrigUnitStr'] = carbon_content['OrigUnitStr'].replace(dict.fromkeys(mg_g, 'mg/g')).replace(dict.fromkeys(g_g, 'g/g')).replace(dict.fromkeys(percent, 'percent'))
g_g_carbon = carbon_content[carbon_content.OrigUnitStr=='g/g'].OrigValueStr
g_g_carbon *= 100
carbon_content.OrigValueStr[carbon_content.OrigUnitStr=='g/g'] = g_g_carbon
mg_g_carbon = carbon_content[carbon_content.OrigUnitStr=='mg/g'].OrigValueStr 
mg_g_carbon /= 10
carbon_content.OrigValueStr[carbon_content.OrigUnitStr=='mg/g'] = mg_g_carbon
carbon_content['OrigUnitStr'] = carbon_content['OrigUnitStr'].replace(['mg/g','g/g'],['percent','percent'])
carbon_content = carbon_content[carbon_content['OrigUnitStr']=='percent']
# Distribution of carbon content in different_species

import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

# Species
species_list = list(dict(carbon_content['AccSpeciesName'].value_counts()).keys())

off_set = 1  
shape = (2,4)
fig, axes =plt.subplots(shape[0],shape[1], figsize=(20,6), sharex=True)
axes = axes.flatten()
object_bol = df.dtypes == 'object'
num_species = shape[0]*shape[1]

for ax, species_name in zip(axes, species_list[num_species*off_set: num_species*(1+off_set)]):
    #sns.countplot(y=catplot, data=df, ax=ax, order=np.unique(df.values))
    species_data = carbon_content[carbon_content['AccSpeciesName'] == species_name]['OrigValueStr'].values
    sns.distplot(species_data, ax=ax).set_title(species_name)

plt.tight_layout()  
plt.show()


carbon_content_grouped = carbon_content.groupby(['AccSpeciesName'], as_index=False).agg({'OrigValueStr':['mean','std']})
carbon_content_grouped.columns = ['AccSpeciesName','carbon_content_mean','carbon_content_std']
fig,ax = plt.subplots(figsize=(12,4))
sns.distplot(carbon_content_grouped[carbon_content_grouped['carbon_content_mean']<100].carbon_content_mean, bins = 100).set_title("Distribution of carbon content in all species")
#ax.set(xlim=(0, 100))
%matplotlib inline
importance_df = carbon_content_grouped.copy()
importance_df.columns= ['feature','importance_mean','importance_std']
plot_importance(importance_df[:20], figsize=(12, 8))
phosphorus_content = df[df['TraitName'] == 'Leaf phosphorus (P) content per leaf dry mass'][imp_cols]
phosphorus_content = phosphorus_content.replace(['1-2','2-3','<1','>3','nd'],['1.5','2.5','1','3',np.nan])
phosphorus_content = phosphorus_content[phosphorus_content['OrigValueStr'].notna()]
phosphorus_content.OrigValueStr = phosphorus_content.OrigValueStr.astype('float')
percent = ['%','% ','% mass/mass','percent']
g_g = ['g P g-1 DW','g/g']
mg_g = ['mg/g','mg g-1','g/kg','mg_g-1']
mg_kg = ['mg kg-1','mg/kg']

phosphorus_content.OrigUnitStr = phosphorus_content.OrigUnitStr.replace(dict.fromkeys(percent, 'percent')).replace(dict.fromkeys(g_g, 'g/g')).replace(dict.fromkeys(mg_g, 'mg/g')).replace(dict.fromkeys(mg_kg, 'mg/kg'))

# converting values with mg/10g, mmol/kg and ppm units to mg/g

mg_10g = phosphorus_content[phosphorus_content.OrigUnitStr=='mg/10g'].OrigValueStr
mg_10g /= 10
phosphorus_content.OrigValueStr[phosphorus_content.OrigUnitStr=='mg/10g'] = mg_10g
phosphorus_content.OrigUnitStr[phosphorus_content.OrigUnitStr=='mg/10g'] = 'mg/g'

phosphorus_content[phosphorus_content.OrigUnitStr=='ppm'].OrigValueStr
ppm = phosphorus_content[phosphorus_content.OrigUnitStr=='ppm'].OrigValueStr
ppm /= 1000
phosphorus_content.OrigValueStr[phosphorus_content.OrigUnitStr=='ppm'] = ppm
phosphorus_content.OrigUnitStr[phosphorus_content.OrigUnitStr=='ppm'] = 'mg/g'

phosphorus_content[phosphorus_content.OrigUnitStr=='mmol/kg'].OrigValueStr
mmol_kg = phosphorus_content[phosphorus_content.OrigUnitStr=='mmol/kg'].OrigValueStr
mmol_kg *= 0.031
phosphorus_content.OrigValueStr[phosphorus_content.OrigUnitStr=='mmol/kg'] = mmol_kg
phosphorus_content.OrigUnitStr[phosphorus_content.OrigUnitStr=='mmol/kg'] = 'mg/g'


phosphorus_content[phosphorus_content.OrigUnitStr=='percent'].OrigValueStr
percent = phosphorus_content[phosphorus_content.OrigUnitStr=='percent'].OrigValueStr
percent *= 10
phosphorus_content.OrigValueStr[phosphorus_content.OrigUnitStr=='percent'] = percent
phosphorus_content.OrigUnitStr[phosphorus_content.OrigUnitStr=='percent'] = 'mg/g'

phosphorus_content[phosphorus_content.OrigUnitStr=='g/g'].OrigValueStr
g_g = phosphorus_content[phosphorus_content.OrigUnitStr=='g/g'].OrigValueStr
g_g *= 1000
phosphorus_content.OrigValueStr[phosphorus_content.OrigUnitStr=='g/g'] = g_g
phosphorus_content.OrigUnitStr[phosphorus_content.OrigUnitStr=='g/g'] = 'mg/g'

phosphorus_content[phosphorus_content.OrigUnitStr=='mg/kg'].OrigValueStr
mg_kg = phosphorus_content[phosphorus_content.OrigUnitStr=='mg/kg'].OrigValueStr
mg_kg /= 1000
phosphorus_content.OrigValueStr[phosphorus_content.OrigUnitStr=='mg/kg'] = mg_kg
phosphorus_content.OrigUnitStr[phosphorus_content.OrigUnitStr=='mg/kg'] = 'mg/g'
# Distribution of phosphorus content in different_species

import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

# Species
species_list = list(dict(phosphorus_content['AccSpeciesName'].value_counts()).keys())

off_set = 1  
shape = (2,4)
fig, axes =plt.subplots(shape[0],shape[1], figsize=(20,6), sharex=False)
axes = axes.flatten()
object_bol = df.dtypes == 'object'
num_species = shape[0]*shape[1]

for ax, species_name in zip(axes, species_list[num_species*off_set: num_species*(1+off_set)]):
    #sns.countplot(y=catplot, data=df, ax=ax, order=np.unique(df.values))
    species_data = phosphorus_content[phosphorus_content['AccSpeciesName'] == species_name]['OrigValueStr'].values
    sns.distplot(species_data, ax=ax).set_title(species_name)

plt.tight_layout()  
plt.show()
phosphorus_content_grouped = phosphorus_content.groupby(['AccSpeciesName'], as_index=False).agg({'OrigValueStr':['mean','std']})
phosphorus_content_grouped.columns = ['AccSpeciesName','phosphorus_content_mean','phosphorus_content_std']
phosphorus_content_grouped.head()