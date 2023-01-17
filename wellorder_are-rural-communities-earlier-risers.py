import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Polygon, PathPatch
from matplotlib.collections import PatchCollection
from mpl_toolkits.basemap import Basemap
STATE_CODES = {'01': 'Alabama',
               '02': 'Alaska',
               '04': 'Arizona',
               '05': 'Arkansas',
               '06': 'California',
               '08': 'Colorado',
               '09': 'Connecticut',
               '10': 'Delaware',
               '11': 'District of Columbia',
               '12': 'Florida',
               '13': 'Georgia',
               '15': 'Hawaii',
               '16': 'Idaho',
               '17': 'Illinois',
               '18': 'Indiana',
               '19': 'Iowa',
               '20': 'Kansas',
               '21': 'Kentucky',
               '22': 'Louisiana',
               '23': 'Maine',
               '24': 'Maryland',
               '25': 'Massachusetts',
               '26': 'Michigan',
               '27': 'Minnesota',
               '28': 'Mississippi',
               '29': 'Missouri',
               '30': 'Montana',
               '31': 'Nebraska',
               '32': 'Nevada',
               '33': 'New Hampshire',
               '34': 'New Jersey',
               '35': 'New Mexico',
               '36': 'New York',
               '37': 'North Carolina',
               '38': 'North Dakota',
               '39': 'Ohio',
               '40': 'Oklahoma',
               '41': 'Oregon',
               '42': 'Pennsylvania',
               '44': 'Rhode Island',
               '45': 'South Carolina',
               '46': 'South Dakota',
               '47': 'Tennessee',
               '48': 'Texas',
               '49': 'Utah',
               '50': 'Vermont',
               '51': 'Virginia',
               '53': 'Washington',
               '54': 'West Virginia',
               '55': 'Wisconsin',
               '56': 'Wyoming',
               #'72': 'Puerto Rico'
               }

JWDP_TIMES = {'001': '12:00 a.m.',
           '002': '12:30 a.m.',
           '003': '1:00 a.m.',
           '004': '1:30 a.m.',
           '005': '2:00 a.m.',
           '006': '2:30 a.m.',
           '007': '3:00 a.m.',
           '008': '3:10 a.m.',
           '009': '3:20 a.m.',
           '010': '3:30 a.m.',
           '011': '3:40 a.m.',
           '012': '3:50 a.m.',
           '013': '4:00 a.m.',
           '014': '4:10 a.m.',
           '015': '4:20 a.m.',
           '016': '4:30 a.m.',
           '017': '4:40 a.m.',
           '018': '4:50 a.m.',
           '019': '5:00 a.m.',
           '020': '5:05 a.m.',
           '021': '5:10 a.m.',
           '022': '5:15 a.m.',
           '023': '5:20 a.m.',
           '024': '5:25 a.m.',
           '025': '5:30 a.m.',
           '026': '5:35 a.m.',
           '027': '5:40 a.m.',
           '028': '5:45 a.m.',
           '029': '5:50 a.m.',
           '030': '5:55 a.m.',
           '031': '6:00 a.m.',
           '032': '6:05 a.m.',
           '033': '6:10 a.m.',
           '034': '6:15 a.m.',
           '035': '6:20 a.m.',
           '036': '6:25 a.m.',
           '037': '6:30 a.m.',
           '038': '6:35 a.m.',
           '039': '6:40 a.m.',
           '040': '6:45 a.m.',
           '041': '6:50 a.m.',
           '042': '6:55 a.m.',
           '043': '7:00 a.m.',
           '044': '7:05 a.m.',
           '045': '7:10 a.m.',
           '046': '7:15 a.m.',
           '047': '7:20 a.m.',
           '048': '7:25 a.m.',
           '049': '7:30 a.m.',
           '050': '7:35 a.m.',
           '051': '7:40 a.m.',
           '052': '7:45 a.m.',
           '053': '7:50 a.m.',
           '054': '7:55 a.m.',
           '055': '8:00 a.m.',
           '056': '8:05 a.m.',
           '057': '8:10 a.m.',
           '058': '8:15 a.m.',
           '059': '8:20 a.m.',
           '060': '8:25 a.m.',
           '061': '8:30 a.m.',
           '062': '8:35 a.m.',
           '063': '8:40 a.m.',
           '064': '8:45 a.m.',
           '065': '8:50 a.m.',
           '066': '8:55 a.m.',
           '067': '9:00 a.m.',
           '068': '9:05 a.m.',
           '069': '9:10 a.m.',
           '070': '9:15 a.m.',
           '071': '9:20 a.m.',
           '072': '9:25 a.m.',
           '073': '9:30 a.m.',
           '074': '9:35 a.m.',
           '075': '9:40 a.m.',
           '076': '9:45 a.m.',
           '077': '9:50 a.m.',
           '078': '9:55 a.m.',
           '079': '10:00 a.m.',
           '080': '10:10 a.m.',
           '081': '10:20 a.m.',
           '082': '10:30 a.m.',
           '083': '10:40 a.m.',
           '084': '10:50 a.m.',
           '085': '11:00 a.m.',
           '086': '11:10 a.m.',
           '087': '11:20 a.m.',
           '088': '11:30 a.m.',
           '089': '11:40 a.m.',
           '090': '11:50 a.m.',
           '091': '12:00 p.m.',
           '092': '12:10 p.m.',
           '093': '12:20 p.m.',
           '094': '12:30 p.m.',
           '095': '12:40 p.m.',
           '096': '12:50 p.m.',
           '097': '1:00 p.m.',
           '098': '1:10 p.m.',
           '099': '1:20 p.m.',
           '100': '1:30 p.m.',
           '101': '1:40 p.m.',
           '102': '1:50 p.m.',
           '103': '2:00 p.m.',
           '104': '2:10 p.m.',
           '105': '2:20 p.m.',
           '106': '2:30 p.m.',
           '107': '2:40 p.m.',
           '108': '2:50 p.m.',
           '109': '3:00 p.m.',
           '110': '3:10 p.m.',
           '111': '3:20 p.m.',
           '112': '3:30 p.m.',
           '113': '3:40 p.m.',
           '114': '3:50 p.m.',
           '115': '4:00 p.m.',
           '116': '4:10 p.m.',
           '117': '4:20 p.m.',
           '118': '4:30 p.m.',
           '119': '4:40 p.m.',
           '120': '4:50 p.m.',
           '121': '5:00 p.m.',
           '122': '5:10 p.m.',
           '123': '5:20 p.m.',
           '124': '5:30 p.m.',
           '125': '5:40 p.m.',
           '126': '5:50 p.m.',
           '127': '6:00 p.m.',
           '128': '6:10 p.m.',
           '129': '6:20 p.m.',
           '130': '6:30 p.m.',
           '131': '6:40 p.m.',
           '132': '6:50 p.m.',
           '133': '7:00 p.m.',
           '134': '7:30 p.m.',
           '135': '8:00 p.m.',
           '136': '8:30 p.m.',
           '137': '9:00 p.m.',
           '138': '9:10 p.m.',
           '139': '9:20 p.m.',
           '140': '9:30 p.m.',
           '141': '9:40 p.m.',
           '142': '9:50 p.m.',
           '143': '10:00 p.m.',
           '144': '10:10 p.m.',
           '145': '10:20 p.m.',
           '146': '10:30 p.m.',
           '147': '10:40 p.m.',
           '148': '10:50 p.m.',
           '149': '11:00 p.m.',
           '150': '11:30 p.m.'
    }
df1 = pd.read_csv('../input/pums//ss13pusa.csv', usecols=['ST','PUMA','JWDP','INDP'])
df2 = pd.read_csv('../input/pums//ss13pusb.csv', usecols=['ST','PUMA','JWDP','INDP'])
df = pd.concat([df1,df2], ignore_index=True)
df_grouped = df.groupby(['ST','PUMA'])
time_leave = df_grouped.JWDP.median()
fig = plt.figure(figsize=(20,10))
ax = fig.add_subplot(111, frame_on=False, axisbg='w')
fig.suptitle('Median time departing for work', fontsize=20)
m = Basemap(width=5000000,height=3500000,
             resolution='l',projection='aea',\
             lat_1=30.,lat_2=50,lon_0=-96,lat_0=38)

# Next we read in shapefiles for the PUMAs (i.e. statistical regions) in each state.
# We also keep track of the median departure time for each PUMA, 
# as well as the size of each PUMA (for reasons that will later be clear).

patches = np.array([])
medians = np.array([])
puma_areas = []
for key in STATE_CODES.keys():
    if int(key) == 72: continue
    m.readshapefile('../input/shapefiles/pums/tl_2013_{0}_puma10'.format(key),
                    name='state', drawbounds=True, default_encoding='latin-1')
    puma_areas += m.state_info
    state_patches = np.asarray([Polygon(np.array(coords), True) for coords in m.state])
    state_medians = np.asarray([time_leave.loc[int(key),int(info['PUMACE10'])] for info in m.state_info])
    patches = np.append(patches, state_patches)
    medians = np.append(medians, state_medians)
    
# In this step, we use some built-in functions to color the PUMAs according to
# the median time people depart for work.  In particular, set_array will automatically 
# make sure that the color range in the PatchCollection corresponds to the range of 
# numbers between the minimum and maximum values of the array passed to it.

pc = PatchCollection(patches, cmap='YlOrBr', edgecolor='k', linewidths=0.0, zorder=2)
pc.set_array(medians)
ax.add_collection(pc)

# Next, we create a colorbar so people know what times the different colors represent.

scale = mpl.cm.ScalarMappable(cmap='YlOrBr')
scale.set_array(medians)
cb = m.colorbar(scale)
#colorbar() automatically creates labels based on the data in scale.  We want to
#display different labels, so we first clear out the old labels.
cb.set_ticks([])
cb.set_ticklabels([])
ticks = np.append(np.arange(medians.min(),medians.max(),5),[medians.max()])
#the keys in JWDP_TIMES are strings of the form '073'
tick_labels = [JWDP_TIMES[str(int(x)).zfill(3)] for x in ticks]
cb.set_ticks(ticks)
cb.set_ticklabels(tick_labels)
areas = pd.DataFrame(puma_areas)
areas['STATEFP10'] = areas['STATEFP10'].astype(int)
areas['PUMACE10'] = areas['PUMACE10'].astype(int)
areas_grouped = areas.groupby(['STATEFP10', 'PUMACE10'])
#Note that some PUMAs get multiple entries in the shapefile.
#However, each entry lists the same landmass (as one would expect),
#so the median of the group will give us the landmass.
pd.DataFrame({'JWDP median': df_grouped.JWDP.median(), 'landmass': areas_grouped.ALAND10.median()}).corr()
def agr_pct_vs_departure(group):
    #Looking at the data dictionary, one finds that agricultural jobs have the codes seen below.
    return pd.DataFrame({'% ag workers': len(group[group.INDP.isin([170,180,190,270,280,290])])/len(group),
                         'median time departing for work': group.JWDP.median()}, index=group.index)
    
df_grouped.apply(agr_pct_vs_departure).corr()