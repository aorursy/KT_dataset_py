import pandas as pd
from pandas_highcharts.core import serialize
from pandas_highcharts.display import display_charts
from IPython.display import display
import numpy as np
pd.options.display.max_columns = None
#pd.options.display.max_rows = None
# Reading csv data
dataHous2011    = pd.read_csv('../input/india_census_housing-hlpca-full.csv', encoding='utf-8')
dataDis2011     = pd.read_csv('../input/india-districts-census-2011.csv', encoding='utf-8')
dataElemSchool  = pd.read_csv('../input/elementary_2015_16.csv',encoding='utf-8')

# Reading different states gdp data
AndhraPradesh1   = pd.read_csv('../input/gdp_AndhraPradesh1.csv' , encoding = 'utf-8')
AndhraPradesh2   = pd.read_csv('../input/gdp_AndhraPradesh2.csv' , encoding = 'utf-8')
ArunachalPradesh1= pd.read_csv('../input/gdp_ArunachalPradesh.csv' , encoding = 'utf-8')
Assam1           = pd.read_csv('../input/gdp_Assam1.csv' , encoding = 'utf-8')
Assam2           = pd.read_csv('../input/gdp_Assam2.csv' , encoding = 'utf-8')
Bihar1           = pd.read_csv('../input/gdp_Bihar1.csv' , encoding = 'utf-8')
Bihar2           = pd.read_csv('../input/gdp_Bihar2.csv' , encoding = 'utf-8')
Chattisgarh1     = pd.read_csv('../input/gdp_Chattisgarh.csv' , encoding = 'utf-8')
Haryana1         = pd.read_csv('../input/gdp_Haryana.csv' , encoding = 'utf-8')
HimachalPradesh1 = pd.read_csv('../input/gdp_HimachalPradesh.csv' , encoding = 'utf-8')
Jharkhand1       = pd.read_csv('../input/gdp_Jharkhand.csv' , encoding = 'utf-8')
Karnataka1       = pd.read_csv('../input/gdp_Karnataka1.csv' , encoding = 'utf-8')
Karnataka2       = pd.read_csv('../input/gdp_Karnataka2.csv' , encoding = 'utf-8')
Kerala1          = pd.read_csv('../input/gdp_Kerala1.csv' , encoding = 'utf-8')
Kerala2          = pd.read_csv('../input/gdp_Kerala2.csv' , encoding = 'utf-8')
MadhyaPradesh1   = pd.read_csv('../input/gdp_MadhyaPradesh.csv' , encoding = 'utf-8')
Maharashtra1     = pd.read_csv('../input/gdp_Maharashtra1.csv' , encoding = 'utf-8')
Maharashtra2     = pd.read_csv('../input/gdp_Maharashtra2.csv' , encoding = 'utf-8')
Manipur1         = pd.read_csv('../input/gdp_Manipur.csv' , encoding = 'utf-8')
Meghalaya1       = pd.read_csv('../input/gdp_Meghalaya.csv' , encoding = 'utf-8')
Mizoram1         = pd.read_csv('../input/gdp_Mizoram.csv' , encoding = 'utf-8')
Odisha1          = pd.read_csv('../input/gdp_Odisha1.csv' , encoding = 'utf-8')
Odisha2          = pd.read_csv('../input/gdp_Odisha2.csv' , encoding = 'utf-8')
Punjab1          = pd.read_csv('../input/gdp_Punjab1.csv' , encoding = 'utf-8')
Punjab2          = pd.read_csv('../input/gdp_Punjab2.csv' , encoding = 'utf-8')
Rajasthan1       = pd.read_csv('../input/gdp_Rajasthan1.csv' , encoding = 'utf-8')
Rajasthan2       = pd.read_csv('../input/gdp_Rajasthan2.csv' , encoding = 'utf-8')
Sikkim1          = pd.read_csv('../input/gdp_Sikkim.csv' , encoding = 'utf-8')
Tamilnadu1       = pd.read_csv('../input/gdp_Tamilnadu.csv' , encoding = 'utf-8')
Uttarakhand1     = pd.read_csv('../input/gdp_Uttarakhand.csv' , encoding = 'utf-8')
UttarPradesh1    = pd.read_csv('../input/gdp_UttarPradesh1.csv' , encoding = 'utf-8')
UttarPradesh2    = pd.read_csv('../input/gdp_UttarPradesh2.csv' , encoding = 'utf-8')
WestBengal1      = pd.read_csv('../input/gdp_WestBengal1.csv' , encoding = 'utf-8')
WestBengal2      = pd.read_csv('../input/gdp_WestBengal2.csv' , encoding = 'utf-8')

# Manupulating India census district data
dataDis2011.columns = map(str.upper, dataDis2011.columns)
dataDis2011['DISTRICT NAME'] = dataDis2011['DISTRICT NAME'].str.upper()
dataDis2011.replace('NCT OF DELHI','DELHI',inplace=True)
dataDis2011.replace('ORISSA','ODISHA',inplace=True)

# Manupulating India census housing data
dataHous2011 = dataHous2011.replace('&','AND',regex=True)
dataHous2011.replace('NCT OF DELHI','DELHI',inplace=True)
dataHous2011.replace('PUDUCHERRY','PONDICHERRY',inplace=True)
dataHous2011.columns = map(str.upper, dataHous2011.columns)
dataHous2011['DISTRICT NAME'] = dataHous2011['DISTRICT NAME'].str.upper()
dataHous2011 = dataHous2011.loc[dataHous2011['RURAL/URBAN'] == 'Total']
#Combining housing and district census 2011
dataCensus2011 = dataDis2011.merge(dataHous2011,on=['STATE NAME','DISTRICT NAME'],how='left')
dataCensus2011['STATE NAME'] = dataCensus2011['STATE NAME'].str.strip()
dataCensus2011.head()
dataElemSchool = dataElemSchool.replace(np.NaN,'NA',regex=True)
dataElemSchool.head()
dataSocEle = dataCensus2011.merge(dataElemSchool,on=['STATE NAME','DISTRICT NAME'],how='left')
dataSocEle = dataSocEle.replace(np.NaN,'NA',regex=True)
dataSocEle.head()
dataSocEle.replace('NA',np.NaN,inplace=True)
dataSocEle = dataSocEle.replace(',','',regex=True)
dataSocEle = dataSocEle.fillna(method = 'bfill', axis=0).fillna(0)
dataSocEle['STATE NAME'] = dataSocEle['STATE NAME'].str.title()
dataSocEle.head()
SWPIIdfg =  dataSocEle.groupby(['STATE NAME'])[['MALE','FEMALE','POPULATION']].sum().rename(columns={'POPULATION':'TOTAL'})
display_charts(SWPIIdfg, title="India's population", kind="bar",figsize = (1000, 700))
dataSocEle['SEX RATIO'] = (dataSocEle['FEMALE'].div(dataSocEle['MALE'])).multiply(1000)
SROFPTM = dataSocEle.groupby(['STATE NAME'])[['SEX RATIO']].mean().round()
display_charts(SROFPTM, title="Sex ratio of females per 1000 males in different states of India", kind="bar",figsize = (1000, 700))
SWDAGI = dataSocEle.groupby(['STATE NAME'])[['AGE_GROUP_0_29','AGE_GROUP_30_49','AGE_GROUP_50']].sum().rename(columns={'AGE_GROUP_0_29':'0-29 YEARS','AGE_GROUP_30_49':'30-49 YEARS','AGE_GROUP_50':'50 YEARS & ABOVE'})
display_charts(SWDAGI, title="State wise different age groups in India", kind="barh",figsize = (1000, 700))
AGPI   = dataSocEle[['AGE_GROUP_0_29','AGE_GROUP_30_49','AGE_GROUP_50']].rename(columns={'AGE_GROUP_0_29':'0-29 YEARS','AGE_GROUP_30_49':'30-49 YEARS','AGE_GROUP_50':'50 YEARS & ABOVE'}).sum()
AGPIdf = pd.DataFrame(AGPI)
AGPIdf = AGPIdf.rename(columns={0:'AGE'})
display_charts(AGPIdf, kind='pie', title="Different age groups percentage in India", tooltip={'pointFormat': '{series.name}: <b>{point.percentage:.1f}%</b>'})
IRPdf  = dataSocEle[['HINDUS','MUSLIMS','CHRISTIANS','SIKHS','BUDDHISTS','JAINS','OTHERS_RELIGIONS']].sum()
IRPdfg = pd.DataFrame(IRPdf)
IRPdfg = IRPdfg.rename(columns={0:'RELIGIONS'})
display_charts(IRPdfg, kind='pie', title="Religions in India", tooltip={'pointFormat': '{series.name}: <b>{point.percentage:.1f}%</b>'})
SWSPIIdfg = dataSocEle.groupby(['STATE NAME'])[['SC']].sum()
display_charts(SWSPIIdfg, kind='pie', title="State wise SC populations in India", tooltip={'pointFormat': '{series.name}: <b>{point.percentage:.1f}%</b>'})
SWSPIIdfg = dataSocEle.groupby(['STATE NAME'])[['ST']].sum()
display_charts(SWSPIIdfg, kind='pie', title="State wise ST populations in India", tooltip={'pointFormat': '{series.name}: <b>{point.percentage:.1f}%</b>'})
dataSocEle['LITERACY RATE'] = dataSocEle['LITERATE']/dataSocEle['POPULATION']*100
SWLRIIdfg = dataSocEle.groupby(['STATE NAME'])[['LITERACY RATE']].mean().round(2)
display_charts(SWLRIIdfg, title="State wise literacy rate in India", kind="bar",figsize = (1000, 700))
TELAPS = dataSocEle.groupby(['STATE NAME'])[['BELOW_PRIMARY_EDUCATION','PRIMARY_EDUCATION','MIDDLE_EDUCATION','SECONDARY_EDUCATION','HIGHER_EDUCATION','GRADUATE_EDUCATION','OTHER_EDUCATION']].sum()
display_charts(TELAPS , kind="line", title="Types of  education level atended populations in different states of India",figsize = (1000, 700))
PODTWI = dataSocEle.groupby(['STATE NAME'])[['MAIN_WORKERS','MARGINAL_WORKERS','CULTIVATOR_WORKERS','AGRICULTURAL_WORKERS','HOUSEHOLD_WORKERS','NON_WORKERS']].sum()
display_charts(PODTWI , kind="area", title="Populations of different Types of  Workers In different states of India ",figsize = (1000, 700))
SWHSIDS = dataSocEle.groupby(['STATE NAME'])[['HOUSEHOLD_SIZE_1_PERSON_HOUSEHOLDS','HOUSEHOLD_SIZE_1_TO_2_PERSONS','HOUSEHOLD_SIZE_2_PERSONS_HOUSEHOLDS','HOUSEHOLD_SIZE_3_PERSONS_HOUSEHOLDS','HOUSEHOLD_SIZE_3_TO_5_PERSONS_HOUSEHOLDS','HOUSEHOLD_SIZE_4_PERSONS_HOUSEHOLDS','HOUSEHOLD_SIZE_5_PERSONS_HOUSEHOLDS','HOUSEHOLD_SIZE_6_8_PERSONS_HOUSEHOLDS','HOUSEHOLD_SIZE_9_PERSONS_AND_ABOVE_HOUSEHOLDS']].sum().rename(columns={'HOUSEHOLD_SIZE_1_PERSON_HOUSEHOLDS':'1 Person','HOUSEHOLD_SIZE_1_TO_2_PERSONS':'1-2 Persons','HOUSEHOLD_SIZE_2_PERSONS_HOUSEHOLDS':'2 Persons','HOUSEHOLD_SIZE_3_PERSONS_HOUSEHOLDS':'3 Persons','HOUSEHOLD_SIZE_3_TO_5_PERSONS_HOUSEHOLDS':'3-5 Persons','HOUSEHOLD_SIZE_4_PERSONS_HOUSEHOLDS':'4 Persons','HOUSEHOLD_SIZE_5_PERSONS_HOUSEHOLDS':'5 Persons','HOUSEHOLD_SIZE_6_8_PERSONS_HOUSEHOLDS':'6-8 Persons','HOUSEHOLD_SIZE_9_PERSONS_AND_ABOVE_HOUSEHOLDS':'9 Persons & Above'})
display_charts(SWHSIDS  , kind="line", title="State wise HouseHold size in different states of india ",figsize = (1000, 700))
SWPPHDWF =  dataSocEle.groupby(['STATE NAME'])[['DW_TFTS','DW_TFUS','DW_CW','DW_UW','DW_HANDPUMP','DW_TB','DW_SPRING','DW_RC','DW_TPL','DW_OS']].mean().round(2).rename(columns={'DW_TFTS':'Tap water treated source','DW_TFUS':'Tap water untreated source','DW_CW':'Covered well','DW_UW':'Uncovered Well','DW_HANDPUMP':'Handpump','DW_TB':'Tubewell','DW_SPRING':'Spring','DW_RC':'River/Canal','DW_TPL':'Tank/Lake','DW_OS':'Other source'})
display_charts(SWPPHDWF , kind="area", title="State wise population percentage having different drinking water source in Indiaa ",figsize = (1000, 700))
# Function to append two gdp csv file
def appendGdp(state1,state2):
    state1 =  state1.rename(columns=lambda x: x.strip())
    state2 =  state2.rename(columns=lambda x: x.strip())
    state  =  state1.append(state2)[state1.columns.tolist()].reset_index()
    state  =  state.drop(columns=['index'], axis = 0)
    return state;

# Function to change column header of gdp dataframe
def changeGdp(stateName):
    stateName         = stateName.rename(columns=lambda x: x.strip())
    stateName         = stateName.transpose().reset_index()
    stateName.columns = stateName.iloc[1] + '|' + stateName.iloc[0]
    stateName         = stateName.rename(columns={'Description|Year':'DISTRICT NAME'})
    stateName         = stateName.drop([0, 1])
    return stateName;

# Manupulating different states gdp data
AndhraPradesh = appendGdp(AndhraPradesh1 ,AndhraPradesh2)
AndhraPradesh = AndhraPradesh.drop([5,6,7,8,14,15,16])
AndhraPradesh = changeGdp(AndhraPradesh)
AndhraPradesh['STATE NAME'] = 'ANDHRA PRADESH'

ArunachalPradesh = changeGdp(ArunachalPradesh1)
ArunachalPradesh['STATE NAME'] = 'ARUNACHAL PRADESH'

Assam         = appendGdp(Assam1,Assam2)
Assam         = changeGdp(Assam)
Assam['STATE NAME'] = 'ASSAM'

Bihar         = appendGdp(Bihar1,Bihar2)
Bihar         = Bihar.drop([5])
Bihar         = changeGdp(Bihar)
Bihar['STATE NAME'] = 'BIHAR'

Chattisgarh = changeGdp(Chattisgarh1)
Chattisgarh['STATE NAME'] = 'CHATTISGARH'

Haryana     =  changeGdp(Haryana1)
Haryana['STATE NAME'] = 'HARYANA'

HimachalPradesh = changeGdp(HimachalPradesh1)
HimachalPradesh['STATE NAME'] = 'HIMACHAL PRADESH'

Jharkhand    = changeGdp(Jharkhand1)
Jharkhand['STATE NAME'] = 'JHARKHAND'

Karnataka   = appendGdp(Karnataka1,Karnataka2)
Karnataka   = changeGdp(Karnataka)
Karnataka['STATE NAME'] = 'KARNATAKA'

Kerala      =  appendGdp(Kerala1,Kerala2)
Kerala      = Kerala.drop([6,7,14])
Kerala      = changeGdp(Kerala)
Kerala['STATE NAME'] = 'KERALA'

MadhyaPradesh = changeGdp(MadhyaPradesh1)
MadhyaPradesh['STATE NAME'] = 'MADHYA PRADESH'

Maharashtra  = appendGdp(Maharashtra1,Maharashtra2)
Maharashtra  = Maharashtra.drop([6,7,14])
Maharashtra  = changeGdp(Maharashtra)
Maharashtra['STATE NAME'] = 'MAHARASHTRA'

Manipur      = changeGdp(Manipur1)
Manipur['STATE NAME'] = 'MANUPUR'

Meghalaya    = changeGdp(Meghalaya1)
Meghalaya['STATE NAME'] = 'MEGHALAYA'

Mizoram     = changeGdp(Mizoram1)
Mizoram['STATE NAME'] = 'MIZORAM'

Odisha      = appendGdp(Odisha1,Odisha2)
Odisha      = Odisha.drop([5,6,7,13,14])
Odisha      = changeGdp(Odisha)
Odisha['STATE NAME'] = 'ODISHA'

Punjab      = appendGdp(Punjab1,Punjab2)
Punjab      = Punjab.drop([5,6,12])
Punjab      = changeGdp(Punjab)
Punjab['STATE NAME'] = 'PUNJAB'

Rajasthan   = appendGdp(Rajasthan1,Rajasthan2)
Rajasthan   = Rajasthan.drop([5,6,12])
Rajasthan   = changeGdp(Rajasthan)
Rajasthan['STATE NAME'] = 'RAJASTHAN'

Sikkim      = changeGdp(Sikkim1)
Sikkim['STATE NAME'] = 'SIKKIM'

Tamilnadu   = changeGdp(Tamilnadu1)
Tamilnadu['STATE NAME'] = 'TAMIL NADU'

Uttarakhand = changeGdp(Uttarakhand1)
Uttarakhand['STATE NAME'] = 'UTTARAKHAND'

UttarPradesh = appendGdp(UttarPradesh1,UttarPradesh2)
UttarPradesh = UttarPradesh.drop([5,6,12])
UttarPradesh = changeGdp(UttarPradesh)
UttarPradesh['STATE NAME'] = 'UTTAR PRADESH'

WestBengal  = appendGdp(WestBengal1,WestBengal2)
WestBengal  = WestBengal.drop([5,6,7,13,14])
WestBengal  = changeGdp(WestBengal)
WestBengal['STATE NAME'] = 'WEST BENGAL'

# Appending all the states into single dataframe
dataGDP = AndhraPradesh.append([ArunachalPradesh,Assam,Bihar,Chattisgarh,Haryana,HimachalPradesh,Jharkhand,Karnataka,
          Kerala,MadhyaPradesh,Maharashtra,Manipur,Meghalaya,Mizoram,Odisha,Punjab,Rajasthan,Sikkim,Tamilnadu,Uttarakhand,
          UttarPradesh,WestBengal])
dataGDP['DISTRICT NAME'] = dataGDP['DISTRICT NAME'].str.upper()
dataGDP['STATE NAME']    = dataGDP['STATE NAME'].str.title()
dataGDP.head()
dataSocEleGdp = dataSocEle.merge(dataGDP,on=['STATE NAME','DISTRICT NAME'],how='left')
dataSocEleGdp = dataSocEleGdp.replace(np.NaN,'NA',regex=True)
dataSocEleGdp.head()
dataSocEleGdp.replace('NA',np.NaN,inplace=True)
dataSocEleGdp = dataSocEleGdp.fillna(method = 'bfill', axis=0).fillna(0)
dataSocEleGdp.head()
from sklearn import preprocessing
x = dataSocEleGdp[['POPULATION','MALE','FEMALE','LITERATE','MALE_LITERATE','FEMALE_LITERATE']].values.astype(float)
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
df_normalized = pd.DataFrame(x_scaled,columns=['POPULATION','MALE','FEMALE','LITERATE','MALE_LITERATE','FEMALE_LITERATE'])
df_normalized['STATE NAME']    = dataSocEleGdp['STATE NAME']
df_normalized['DISTRICT NAME'] = dataSocEleGdp['DISTRICT NAME']
df_normalized
POESAN = df_normalized.groupby(['STATE NAME'])[['MALE','FEMALE','POPULATION']].mean().rename(columns={'POPULATION':'TOTAL'})
display_charts(POESAN, title="India's population", kind="bar",figsize = (1000, 700))