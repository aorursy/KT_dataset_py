import pandas as pd
import numpy as np
from functools import reduce
datacensus2001 = pd.read_csv('../input/india-pca-data/census_2001.csv', encoding='UTF-8')
datacensus2011 = pd.read_csv('../input/india-pca-data/census_2011.csv', encoding='UTF-8')

# Reading Geography data
geography      = pd.read_csv('../input/india-pca-data/geography.csv',encoding='UTF-8')

# Method to get default year data
def getYearData(year):
    censustemp     = geography[['name']]
    censustemp     = censustemp.rename(index=str, columns={'name': 'NAME'})
    censustemp.insert(1,'YEAR',year)
    #censustemp = censustemp.drop([0])
    return censustemp
datacensuscom = pd.concat([datacensus2001,datacensus2011],sort=False)
datacensuscom['YEAR'] = datacensuscom['YEAR'].astype(str)
datacensuscom['NAME'] = datacensuscom['NAME'].str.strip()
datacensuscom.head()
cols = list(datacensuscom.columns.values)
cols.remove('AREA')
cols.remove('NAME')
datacom2001_11 = datacensuscom.groupby(['NAME','YEAR'],sort=False, as_index=False)[cols].sum()
datacom2001_11.head()
datadefault2001 = getYearData('2001')
datadefault2002 = getYearData('2002')
datadefault2003 = getYearData('2003')
datadefault2004 = getYearData('2004')
datadefault2005 = getYearData('2005')
datadefault2006 = getYearData('2006')
datadefault2007 = getYearData('2007')
datadefault2008 = getYearData('2008')
datadefault2009 = getYearData('2009')
datadefault2010 = getYearData('2010')
datadefault2011 = getYearData('2011')
datadefault2012 = getYearData('2012')
datadefault2013 = getYearData('2013')
datadefault2014 = getYearData('2014')
datadefault2015 = getYearData('2015')
datadefault2016 = getYearData('2016')
datadefault2017 = getYearData('2017')
datacomcenFinal = pd.concat([datacom2001_11,datadefault2002,datadefault2003,datadefault2004,datadefault2005,datadefault2006,datadefault2007,datadefault2008,datadefault2009,datadefault2010,datadefault2012,datadefault2013,datadefault2014,datadefault2015,datadefault2016,datadefault2017],sort=False)
datacomcenFinal.head()
dataVioCrm       = pd.read_csv('../input/india-crimes-data/violent_crime_incidence_2016.csv')
dataRapeVictm    = pd.read_csv('../input/india-crimes-data/rape_victims_2016.csv')
dataKidnapping   = pd.read_csv('../input/india-crimes-data/kidnapping_2016.csv')
dataTrafficing   = pd.read_csv('../input/india-crimes-data/trafficing_2016.csv')
dataMissChildren = pd.read_csv('../input/india-crimes-data/missing_traced_children_2016.csv')
dataJuvArrest    = pd.read_csv('../input/india-crimes-data/juvenile_arrested_bckgrnd_2016.csv')
dataMurderVictm  = pd.read_csv('../input/india-crimes-data/murder_victims_2016.csv')
dataMrdrMotive   = pd.read_csv('../input/india-crimes-data/murder_motives_2016.csv')
crimedf      = [dataVioCrm,dataTrafficing,dataMissChildren,dataRapeVictm,dataKidnapping,dataMurderVictm,dataJuvArrest,dataMrdrMotive]
crimedfCom   = reduce(lambda left,right: pd.merge(left,right, on=['STATE NAME']),crimedf)
crimedfCom.insert(2,'YEAR','2016')
crimedfCom   = crimedfCom.rename(index=str, columns={'STATE NAME': 'NAME',})
crimeDF2016  = crimedfCom.drop(columns=['Unnamed: 0_x','Unnamed: 0_y'])
crimeDF2016.head()
datacomcrimeFinal = pd.concat([crimeDF2016,datadefault2001,datadefault2002,datadefault2003,datadefault2004,datadefault2005,datadefault2006,datadefault2007,datadefault2008,datadefault2009,datadefault2010,datadefault2011,datadefault2012,datadefault2013,datadefault2014,datadefault2015,datadefault2017],sort=False)
datacomcrimeFinal.head()
AlloDocPHCS        = pd.read_csv('../input/india-primary-health-care-data/allo-doc-PHCS_2017.csv', encoding='utf-8')
DisSubdisDoc       = pd.read_csv('../input/india-primary-health-care-data/dis-subdis-doctors_2017.csv', encoding='utf-8')
PhysicianCHCS      = pd.read_csv('../input/india-primary-health-care-data/physician-CHCS_2017.csv', encoding='utf-8')
SurgeonCHCS        = pd.read_csv('../input/india-primary-health-care-data/surgeons-CHCS_2017.csv', encoding='utf-8')
WorkerMaleSubCen   = pd.read_csv('../input/india-primary-health-care-data/worker-male-subcen_2017.csv', encoding='utf-8')
WorkerFemaleSubCen = pd.read_csv('../input/india-primary-health-care-data/worker-female-subcen_2017.csv', encoding='utf-8')
AssisMalePHCS      = pd.read_csv('../input/india-primary-health-care-data/assistant-male-PHCS_2017.csv', encoding='utf-8')
AssisFemalePHCS    = pd.read_csv('../input/india-primary-health-care-data/assistant-female-PHCS_2017.csv', encoding='utf-8')
NurseStaffPHCCHC   = pd.read_csv('../input/india-primary-health-care-data/nursing-staff-PHCS-CHCS_2017.csv', encoding='utf-8')
FacilitiesCHCS     = pd.read_csv('../input/india-primary-health-care-data/facilities-CHCS_2017.csv', encoding='utf-8')
FacilitiesPHCS     = pd.read_csv('../input/india-primary-health-care-data/facilities-PHCS_2017.csv', encoding='utf-8')
RadiographerCHCS   = pd.read_csv('../input/india-primary-health-care-data/radiographers-CHCS_2017.csv', encoding='utf-8')
VillageCovered     = pd.read_csv('../input/india-primary-health-care-data/villg-coveredby-centre_2017.csv', encoding='utf-8')
AreaCovered        = pd.read_csv('../input/india-primary-health-care-data/rural-area-covered-centre_2017.csv', encoding='utf-8')
InfantMorRate      = pd.read_csv('../input/india-primary-health-care-data/infant-mortality-rate_2017.csv', encoding='utf-8')
RuralPopCovered    = pd.read_csv('../input/india-primary-health-care-data/rural-population-centre_2017.csv', encoding='utf-8')
PharmacistPHCCHC   = pd.read_csv('../input/india-primary-health-care-data/pharmacists-PHCS-CHCS_2017.csv', encoding='utf-8')
FunctioningPHCS    = pd.read_csv('../input/india-primary-health-care-data/functioning-PHCS_2017.csv', encoding='utf-8')
#FunctioningPHCS    = pd.read_csv('../input/india-primary-health-care-data/year-wise-PHCS.csv', encoding='utf-8')
#FunctioningCHCS    = pd.read_csv('../input/india-primary-health-care-data/year-wise-CHCS.csv', encoding='utf-8')
#FunctioningSubcen  = pd.read_csv('../input/india-primary-health-care-data/year-wise-Subcentre.csv', encoding='utf-8')
def getHealthCols(df,colheader):
    df.columns = df.columns.str.replace('Number of Primary Health Centres - ', '')
    df = df.add_prefix(colheader)
    cols = list(df.columns.values)[:2]
    df = df.rename(index=str, columns={cols[0]:'S. No.',cols[1]:'State'})
    return df

def getFacilitiesCols(df,colheader):
    df.columns = df.columns.str.replace('Number of Community Health Centres - ', '')
    df.columns = df.columns.str.replace('No. of CHC having a regular supply of - ', '')
    df = df.add_prefix(colheader)
    cols = list(df.columns.values)[:2]
    df = df.rename(index=str, columns={cols[0]:'State'})
    return df
AlloDocPHCSf       = getHealthCols(AlloDocPHCS,'Allopathic_doctor_')
DisSubdisDoc       = DisSubdisDoc.rename(index=str, columns={'State/ UT':'State'})
PhysicianCHCSf     = getHealthCols(PhysicianCHCS,'Physician_CHCS_')
SurgeonCHCSf       = getHealthCols(SurgeonCHCS,'Surgeon_CHCS_')
WorkerMaleSubCenf  = getHealthCols(WorkerMaleSubCen,'Workers_male_subcentre_')
WorkerFemaleSubCenf= getHealthCols(WorkerFemaleSubCen,'Workers_female_subcentre_')
AssisMalePHCSf     = getHealthCols(AssisMalePHCS,'Assistant_male_PHCS_')
AssisFemalePHCSf   = getHealthCols(AssisFemalePHCS,'Assistant_female_PHCS_')
NurseStaffPHCCHCf  = getHealthCols(NurseStaffPHCCHC,'Nurse_PHCS_CHCS_')
FacilitiesCHCSf    = getFacilitiesCols(FacilitiesCHCS,'CHCS_facilities_')
FacilitiesPHCSf    = getHealthCols(FacilitiesPHCS,'PHCS_facilities_')
RadiographerCHCSf  = getHealthCols(RadiographerCHCS,'Radiographer_CHCS_')
VillageCovered     = VillageCovered.rename(index=str, columns={'State/ UT':'State'})
AreaCovered        = AreaCovered.rename(index=str, columns={'State/ UT':'State'})
InfantMorRate      = InfantMorRate.rename(index=str, columns={'State/ UT':'State'})
RuralPopCovered    = RuralPopCovered.rename(index=str, columns={'State/ UT':'State'})
PharmacistPHCCHCf  = getHealthCols(PharmacistPHCCHC,'Pharmacist_PHCS_CHCS_')
FunctioningPHCS    = FunctioningPHCS.rename(index=str, columns={'State/ UT':'State'})

# Combining all the df
healthdf    =  [AlloDocPHCSf,DisSubdisDoc,PhysicianCHCSf,SurgeonCHCSf,WorkerMaleSubCenf,WorkerFemaleSubCenf,AssisMalePHCSf,AssisFemalePHCSf,NurseStaffPHCCHCf,FacilitiesCHCSf,FacilitiesPHCSf,RadiographerCHCSf,VillageCovered,AreaCovered,InfantMorRate,RuralPopCovered,PharmacistPHCCHCf,FunctioningPHCS]
healthdf2017 = reduce(lambda left,right: pd.merge(left,right, on=['State']),healthdf)
healthdf2017.replace('A & N Island','ANDAMAN & NICOBAR ISLANDS',inplace=True)
healthdf2017.replace('Delhi','NCT OF DELHI',inplace=True)
healthdf2017  = healthdf2017.drop(columns=['S. No._x','S. No._y','Note of State/ UT_y','Note of State/ UT_x'])
healthdf2017['State'] = healthdf2017['State'].str.upper()
healthdf2017 = healthdf2017.rename(index=str, columns={'State':'NAME'})
healthdf2017.insert(1,'YEAR','2017')
healthdf2017.head()
datacomhealthFinal = pd.concat([healthdf2017,datadefault2001,datadefault2002,datadefault2003,datadefault2004,datadefault2005,datadefault2006,datadefault2007,datadefault2008,datadefault2009,datadefault2010,datadefault2011,datadefault2012,datadefault2013,datadefault2014,datadefault2015,datadefault2016],sort=False)
datacomhealthFinal.head()
# Reading different states gdp data
AndhraPradesh1   = pd.read_csv('../input/all-census-data/gdp_AndhraPradesh1.csv' , encoding = 'utf-8')
AndhraPradesh2   = pd.read_csv('../input/all-census-data/gdp_AndhraPradesh2.csv' , encoding = 'utf-8')
ArunachalPradesh1= pd.read_csv('../input/all-census-data/gdp_ArunachalPradesh.csv' , encoding = 'utf-8')
Assam1           = pd.read_csv('../input/all-census-data/gdp_Assam1.csv' , encoding = 'utf-8')
Assam2           = pd.read_csv('../input/all-census-data/gdp_Assam2.csv' , encoding = 'utf-8')
Bihar1           = pd.read_csv('../input/all-census-data/gdp_Bihar1.csv' , encoding = 'utf-8')
Bihar2           = pd.read_csv('../input/all-census-data/gdp_Bihar2.csv' , encoding = 'utf-8')
Chattisgarh1     = pd.read_csv('../input/all-census-data/gdp_Chattisgarh.csv' , encoding = 'utf-8')
Haryana1         = pd.read_csv('../input/all-census-data/gdp_Haryana.csv' , encoding = 'utf-8')
HimachalPradesh1 = pd.read_csv('../input/all-census-data/gdp_HimachalPradesh.csv' , encoding = 'utf-8')
Jharkhand1       = pd.read_csv('../input/all-census-data/gdp_Jharkhand.csv' , encoding = 'utf-8')
Karnataka1       = pd.read_csv('../input/all-census-data/gdp_Karnataka1.csv' , encoding = 'utf-8')
Karnataka2       = pd.read_csv('../input/all-census-data/gdp_Karnataka2.csv' , encoding = 'utf-8')
Kerala1          = pd.read_csv('../input/all-census-data/gdp_Kerala1.csv' , encoding = 'utf-8')
Kerala2          = pd.read_csv('../input/all-census-data/gdp_Kerala2.csv' , encoding = 'utf-8')
MadhyaPradesh1   = pd.read_csv('../input/all-census-data/gdp_MadhyaPradesh.csv' , encoding = 'utf-8')
Maharashtra1     = pd.read_csv('../input/all-census-data/gdp_Maharashtra1.csv' , encoding = 'utf-8')
Maharashtra2     = pd.read_csv('../input/all-census-data/gdp_Maharashtra2.csv' , encoding = 'utf-8')
Manipur1         = pd.read_csv('../input/all-census-data/gdp_Manipur.csv' , encoding = 'utf-8')
Meghalaya1       = pd.read_csv('../input/all-census-data/gdp_Meghalaya.csv' , encoding = 'utf-8')
Mizoram1         = pd.read_csv('../input/all-census-data/gdp_Mizoram.csv' , encoding = 'utf-8')
Odisha1          = pd.read_csv('../input/all-census-data/gdp_Odisha1.csv' , encoding = 'utf-8')
Odisha2          = pd.read_csv('../input/all-census-data/gdp_Odisha2.csv' , encoding = 'utf-8')
Punjab1          = pd.read_csv('../input/all-census-data/gdp_Punjab1.csv' , encoding = 'utf-8')
Punjab2          = pd.read_csv('../input/all-census-data/gdp_Punjab2.csv' , encoding = 'utf-8')
Rajasthan1       = pd.read_csv('../input/all-census-data/gdp_Rajasthan1.csv' , encoding = 'utf-8')
Rajasthan2       = pd.read_csv('../input/all-census-data/gdp_Rajasthan2.csv' , encoding = 'utf-8')
Sikkim1          = pd.read_csv('../input/all-census-data/gdp_Sikkim.csv' , encoding = 'utf-8')
Tamilnadu1       = pd.read_csv('../input/all-census-data/gdp_Tamilnadu.csv' , encoding = 'utf-8')
Uttarakhand1     = pd.read_csv('../input/all-census-data/gdp_Uttarakhand.csv' , encoding = 'utf-8')
UttarPradesh1    = pd.read_csv('../input/all-census-data/gdp_UttarPradesh1.csv' , encoding = 'utf-8')
UttarPradesh2    = pd.read_csv('../input/all-census-data/gdp_UttarPradesh2.csv' , encoding = 'utf-8')
WestBengal1      = pd.read_csv('../input/all-census-data/gdp_WestBengal1.csv' , encoding = 'utf-8')
WestBengal2      = pd.read_csv('../input/all-census-data/gdp_WestBengal2.csv' , encoding = 'utf-8')
# Function to append two gdp csv file
def appendGdp(state1,state2):
    state1 =  state1.rename(columns=lambda x: x.strip())
    state2 =  state2.rename(columns=lambda x: x.strip())
    state  =  state1.append(state2,sort=False)[state1.columns.tolist()].reset_index()
    state  =  state.drop(columns=['index'], axis = 0)
    return state

# Calculating geographic data
def getGdpData(df):
    df = pd.melt(df, id_vars=['State', 'Year','Description'], var_name='District', value_name="GDP")
    df['Year'] = df['Year'].str.strip().str.split('-').str[0]
    df  =  df.loc[df['Description'] == 'GDP (in Rs. Cr.)']
    df  = df[['State','Year','District','GDP']]
    
    cols = list(df.columns.values)[3:]
    df = df.replace(',','',regex=True)
    df[cols] = df[cols].astype(float)
    df = df.rename(index=str, columns={'State': 'NAME'})
    dfState = df.groupby(['NAME','Year'])[cols].sum().reset_index()
    dfState['NAME'] = dfState['NAME'].str.upper()
   
    dfDis   = df.groupby(['District','Year'])[cols].sum().reset_index()
    dfDis   = dfDis.rename(index=str, columns={'District': 'NAME'})
    dfDis['NAME'] = dfDis['NAME'].str.title()
    
    dfCom = pd.concat([dfState, dfDis])
    return dfCom
# Manupulating different states gdp data
AndhraPradesh = appendGdp(AndhraPradesh1 ,AndhraPradesh2)
AndhraPradesh = AndhraPradesh.drop([5,6,7,8,14,15,16])
AndhraPradesh['State'] = 'ANDHRA PRADESH'
AndhraPradeshF = getGdpData(AndhraPradesh)

ArunachalPradesh1['State'] = 'ARUNACHAL PRADESH'
ArunachalPradeshF = getGdpData(ArunachalPradesh1)

Assam         = appendGdp(Assam1,Assam2)
Assam['State'] = 'ASSAM'
AssamF = getGdpData(Assam)


Bihar         = appendGdp(Bihar1,Bihar2)
Bihar         = Bihar.drop([5])
Bihar['State']= 'BIHAR'
BiharF        = getGdpData(Bihar)

Chattisgarh1['State'] = 'CHATTISGARH'
ChattisgarhF          = getGdpData(Chattisgarh1)


Haryana1['State'] = 'HARYANA'
HaryanaF          = getGdpData(Haryana1)

HimachalPradesh1['State'] = 'HIMACHAL PRADESH'
HimachalPradeshF = getGdpData(HimachalPradesh1)

Jharkhand1['State'] = 'JHARKHAND'
JharkhandF = getGdpData(Jharkhand1)

Karnataka   = appendGdp(Karnataka1,Karnataka2)
Karnataka['State'] = 'KARNATAKA'
KarnatakaF = getGdpData(Karnataka)

Kerala      =  appendGdp(Kerala1,Kerala2)
Kerala      = Kerala.drop([6,7,14])
Kerala['State'] = 'KERALA'
KeralaF = getGdpData(Kerala)

MadhyaPradesh1['State'] = 'MADHYA PRADESH'
MadhyaPradeshF = getGdpData(MadhyaPradesh1)

Maharashtra  = appendGdp(Maharashtra1,Maharashtra2)
Maharashtra  = Maharashtra.drop([6,7,14])
Maharashtra['State'] = 'MAHARASHTRA'
MaharashtraF = getGdpData(Maharashtra)

Manipur1['State'] = 'MANIPUR'
ManipurF = getGdpData(Manipur1)

Meghalaya1['State']= 'MEGHALAYA'
MeghalayaF = getGdpData(Meghalaya1)

Mizoram1['State'] = 'MIZORAM'
MizoramF = getGdpData(Mizoram1)

Odisha      = appendGdp(Odisha1,Odisha2)
Odisha      = Odisha.drop([5,6,7,13,14])
Odisha['State'] = 'ODISHA'
OdishaF = getGdpData(Odisha)

Punjab      = appendGdp(Punjab1,Punjab2)
Punjab      = Punjab.drop([5,6,12])
Punjab['State'] = 'PUNJAB'
PunjabF = getGdpData(Punjab)

Rajasthan   = appendGdp(Rajasthan1,Rajasthan2)
Rajasthan   = Rajasthan.drop([5,6,12])
Rajasthan['State'] = 'RAJASTHAN'
RajasthanF = getGdpData(Rajasthan)

Sikkim1['State'] = 'SIKKIM'
SikkimF = getGdpData(Sikkim1)

Tamilnadu1['State'] = 'TAMIL NADU'
TamilnaduF = getGdpData(Tamilnadu1)

Uttarakhand1['State'] = 'UTTARAKHAND'
UttarakhandF = getGdpData(Uttarakhand1)

UttarPradesh = appendGdp(UttarPradesh1,UttarPradesh2)
UttarPradesh = UttarPradesh.drop([5,6,12])
UttarPradesh['State'] = 'UTTAR PRADESH'
UttarPradeshF = getGdpData(UttarPradesh)

WestBengal  = appendGdp(WestBengal1,WestBengal2)
WestBengal  = WestBengal.drop([5,6,7,13,14])
WestBengal['State'] = 'WEST BENGAL'
WestBengalF = getGdpData(WestBengal)
WestBengalF

# Appending all the states into single dataframe
dataGDP = AndhraPradeshF.append([ArunachalPradeshF,AssamF,BiharF,ChattisgarhF,HaryanaF,HimachalPradeshF,JharkhandF,KarnatakaF,
          KeralaF,MadhyaPradeshF,MaharashtraF,ManipurF,MeghalayaF,MizoramF,OdishaF,PunjabF,RajasthanF,SikkimF,TamilnaduF,UttarakhandF,
          UttarPradeshF,WestBengalF],sort=False)

dataGDP = dataGDP[(dataGDP.Year != "1999") & (dataGDP.Year != "2000")]
dataGDP = dataGDP.sort_values(by=['Year'])
dataGDP   = dataGDP.rename(index=str, columns={'Year': 'YEAR'})
dataGDP.head()
datacomgdpFinal = pd.concat([dataGDP,datadefault2013,datadefault2014,datadefault2015,datadefault2016,datadefault2017],sort=False)
datacomgdpFinal.head()
# Reading school data
dataEl_2015 = pd.read_csv('../input/india-elementary-school-data/elementary_2015_16.csv', encoding='utf-8')
dataEl_2013 = pd.read_csv('../input/india-elementary-school-data/elementary_2013_14.csv', encoding='utf-8')
dataEl_2012 = pd.read_csv('../input/india-elementary-school-data/elementary_2012_13.csv', encoding='utf-8')
dataEl_2011 = pd.read_csv('../input/india-elementary-school-data/elementary_2011_12.csv', encoding='utf-8')
dataEl_2010 = pd.read_csv('../input/india-elementary-school-data/elementary_2010_11.csv', encoding='utf-8')
dataEl_2009 = pd.read_csv('../input/india-elementary-school-data/elementary_2009_10.csv', encoding='utf-8')
dataEl_2008 = pd.read_csv('../input/india-elementary-school-data/elementary_2008_09.csv', encoding='utf-8')
dataEl_2007 = pd.read_csv('../input/india-elementary-school-data/elementary_2007_08.csv', encoding='utf-8')
dataEl_2006 = pd.read_csv('../input/india-elementary-school-data/elementary_2006_07.csv', encoding='utf-8')
dataEl_2005 = pd.read_csv('../input/india-elementary-school-data/elementary_2005_06.csv', encoding='utf-8')

# drop unused columns
dataEl_2013  = dataEl_2013.drop(columns=['Unnamed: 0','12'])

# Renaming the school data column
dataEl_2012 = dataEl_2012.rename(index=str, columns={'PRIMARY WITH UPPER PRIMARY SEC AND HIGHER SEC. (SCHGOVT3)': 'PRIMARY WITH UPPER PRIMARY SEC/HIGHER SEC. (SCHGOVT3)','UPPER PRIMARY WITH SEC. AND HIGHER SEC. (SCHGOVT5)':'UPPER PRIMARY WITH SEC./HIGHER SEC. (SCHGOVT5)'})
dataEl_2013 = dataEl_2013.rename(index=str, columns={'PRIMARY WITH UPPER PRIMARY SEC AND HIGHER SEC. (SCHGOVT3)': 'PRIMARY WITH UPPER PRIMARY SEC/HIGHER SEC. (SCHGOVT3)','UPPER PRIMARY WITH SEC. AND HIGHER SEC. (SCHGOVT5)':'UPPER PRIMARY WITH SEC./HIGHER SEC. (SCHGOVT5)'})
dataEl_2015 = dataEl_2015.rename(index=str, columns={'PRIMARY ONLY (SCH1G)':'PRIMARY (SCHGOVT1)','PRIMARY WITH UPPER PRIMARY (SCH2G)':'PRIMARY WITH UPPER PRIMARY (SCHGOVT2)','PRIMARY WITH UPPER PRIMARY SEC/H.SEC (SCH3G)':'PRIMARY WITH UPPER PRIMARY SEC/HIGHER SEC. (SCHGOVT3)','UPPER PRIMARY ONLY (SCH4G)':'UPPER PRIMARY ONLY (SCHGOVT4)','UPPER PRIMARY WITH SEC./H.SEC (SCH5G)':'UPPER PRIMARY WITH SEC./HIGHER SEC. (SCHGOVT5)','PRIMARY ONLY (SCH1P)':'PRIMARY (SCHPVT1)','PRIMARY WITH UPPER PRIMARY (SCH2P)':'PRIMARY WITH UPPER PRIMARY (SCHPVT2)','PRIMARY WITH UPPER PRIMARY SEC/H.SEC (SCH3P)':'PRIMARY WITH UPPER PRIMARY SEC/HIGHER SEC. (SCHPVT3)','UPPER PRIMARY ONLY (SCH4P)':'UPPER PRIMARY ONLY (SCHPVT4)','UPPER PRIMARY WITH SEC./H.SEC (SCH5P)':'UPPER PRIMARY WITH SEC./HIGHER SEC. (SCHPVT5)'})
dataEl_2009 = dataEl_2009.rename(index=str, columns={'PRIMARY (ENR GOVT1)':'PRIMARY (ENR_GOVT1)','PRIMARY WITH UPPER PRIMARY (ENR GOVT2)':'PRIMARY WITH UPPER PRIMARY (ENR_GOVT2)','PRIMARY WITH UPPER PRIMARY SEC/HIGHER SEC. (ENR GOVT3)':'PRIMARY WITH UPPER PRIMARY SEC/HIGHER SEC. (ENR_GOVT3)','UPPER PRIMARY ONLY (ENR GOVT4)':'UPPER PRIMARY ONLY (ENR_GOVT4)','UPPER PRIMARY WITH SEC./HIGHER SEC. (ENR GOVT5)':'UPPER PRIMARY WITH SEC./HIGHER SEC. (ENR_GOVT5)'})
dataEl_2010 = dataEl_2010.rename(index=str, columns={'PRIMARY (ENR GOVT1)':'PRIMARY (ENR_GOVT1)','PRIMARY WITH UPPER PRIMARY (ENR GOVT2)':'PRIMARY WITH UPPER PRIMARY (ENR_GOVT2)','PRIMARY WITH UPPER PRIMARY SEC/HIGHER SEC. (ENR GOVT3)':'PRIMARY WITH UPPER PRIMARY SEC/HIGHER SEC. (ENR_GOVT3)','UPPER PRIMARY ONLY (ENR GOVT4)':'UPPER PRIMARY ONLY (ENR_GOVT4)','UPPER PRIMARY WITH SEC./HIGHER SEC. (ENR GOVT5)':'UPPER PRIMARY WITH SEC./HIGHER SEC. (ENR_GOVT5)'})
dataEl_2011 = dataEl_2011.rename(index=str, columns={'PRIMARY (ENR GOVT1)':'PRIMARY (ENR_GOVT1)','PRIMARY WITH UPPER PRIMARY (ENR GOVT2)':'PRIMARY WITH UPPER PRIMARY (ENR_GOVT2)','PRIMARY WITH UPPER PRIMARY SEC/HIGHER SEC. (ENR GOVT3)':'PRIMARY WITH UPPER PRIMARY SEC/HIGHER SEC. (ENR_GOVT3)','UPPER PRIMARY ONLY (ENR GOVT4)':'UPPER PRIMARY ONLY (ENR_GOVT4)','UPPER PRIMARY WITH SEC./HIGHER SEC. (ENR GOVT5)':'UPPER PRIMARY WITH SEC./HIGHER SEC. (ENR_GOVT5)'})
dataEl_2012 = dataEl_2012.rename(index=str, columns={'PRIMARY (ENR GOVT1)':'PRIMARY (ENR_GOVT1)','PRIMARY WITH UPPER PRIMARY (ENR GOVT2)':'PRIMARY WITH UPPER PRIMARY (ENR_GOVT2)','PRIMARY WITH UPPER PRIMARY SEC/HIGHER SEC. (ENR GOVT3)':'PRIMARY WITH UPPER PRIMARY SEC/HIGHER SEC. (ENR_GOVT3)','UPPER PRIMARY ONLY (ENR GOVT4)':'UPPER PRIMARY ONLY (ENR_GOVT4)','UPPER PRIMARY WITH SEC./HIGHER SEC. (ENR GOVT5)':'UPPER PRIMARY WITH SEC./HIGHER SEC. (ENR_GOVT5)'})
dataEl_2013 = dataEl_2013.rename(index=str, columns={'PRIMARY (ENR GOVT1)':'PRIMARY (ENR_GOVT1)','PRIMARY WITH UPPER PRIMARY (ENR GOVT2)':'PRIMARY WITH UPPER PRIMARY (ENR_GOVT2)','PRIMARY WITH UPPER PRIMARY SEC/HIGHER SEC. (ENR GOVT3)':'PRIMARY WITH UPPER PRIMARY SEC/HIGHER SEC. (ENR_GOVT3)','UPPER PRIMARY ONLY (ENR GOVT4)':'UPPER PRIMARY ONLY (ENR_GOVT4)','UPPER PRIMARY WITH SEC./HIGHER SEC. (ENR GOVT5)':'UPPER PRIMARY WITH SEC./HIGHER SEC. (ENR_GOVT5)'})
dataEl_2015 = dataEl_2015.rename(index=str, columns={'PRIMARY ONLY (ENR1G)':'PRIMARY (ENR_GOVT1)','PRIMARY WITH UPPER PRIMARY (ENR2G)':'PRIMARY WITH UPPER PRIMARY (ENR_GOVT2)','PRIMARY WITH UPPER PRIMARY SEC/H.SEC (ENR3G)':'PRIMARY WITH UPPER PRIMARY SEC/HIGHER SEC. (ENR_GOVT3)','UPPER PRIMARY ONLY (ENR4G)':'UPPER PRIMARY ONLY (ENR_GOVT4)','UPPER PRIMARY WITH SEC./H.SEC (ENR5G)':'UPPER PRIMARY WITH SEC./HIGHER SEC. (ENR_GOVT5)'})
dataEl_2009 = dataEl_2009.rename(index=str, columns={'PRIMARY (ENR PVT1)':'PRIMARY (ENR_PVT1)','PRIMARY WITH UPPER PRIMARY (ENR PVT2)':'PRIMARY WITH UPPER PRIMARY (ENR_PVT2)','PRIMARY WITH UPPER PRIMARY SEC/HIGHER SEC. (ENR PVT3)':'PRIMARY WITH UPPER PRIMARY SEC/HIGHER SEC. (ENR_PVT3)','UPPER PRIMARY ONLY (ENR PVT4)':'UPPER PRIMARY ONLY (ENR_PVT4)','UPPER PRIMARY WITH SEC./HIGHER SEC. (ENR PVT5)':'UPPER PRIMARY WITH SEC./HIGHER SEC. (ENR_PVT5)'})
dataEl_2010 = dataEl_2010.rename(index=str, columns={'PRIMARY (ENR PVT1)':'PRIMARY (ENR_PVT1)','PRIMARY WITH UPPER PRIMARY (ENR PVT2)':'PRIMARY WITH UPPER PRIMARY (ENR_PVT2)','PRIMARY WITH UPPER PRIMARY SEC/HIGHER SEC. (ENR PVT3)':'PRIMARY WITH UPPER PRIMARY SEC/HIGHER SEC. (ENR_PVT3)','UPPER PRIMARY ONLY (ENR PVT4)':'UPPER PRIMARY ONLY (ENR_PVT4)','UPPER PRIMARY WITH SEC./HIGHER SEC. (ENR PVT5)':'UPPER PRIMARY WITH SEC./HIGHER SEC. (ENR_PVT5)'})
dataEl_2011 = dataEl_2011.rename(index=str, columns={'PRIMARY (ENR PVT1)':'PRIMARY (ENR_PVT1)','PRIMARY WITH UPPER PRIMARY (ENR PVT2)':'PRIMARY WITH UPPER PRIMARY (ENR_PVT2)','PRIMARY WITH UPPER PRIMARY SEC/HIGHER SEC. (ENR PVT3)':'PRIMARY WITH UPPER PRIMARY SEC/HIGHER SEC. (ENR_PVT3)','UPPER PRIMARY ONLY (ENR PVT4)':'UPPER PRIMARY ONLY (ENR_PVT4)','UPPER PRIMARY WITH SEC./HIGHER SEC. (ENR PVT5)':'UPPER PRIMARY WITH SEC./HIGHER SEC. (ENR_PVT5)'})
dataEl_2012 = dataEl_2012.rename(index=str, columns={'PRIMARY (ENR PVT1)':'PRIMARY (ENR_PVT1)','PRIMARY WITH UPPER PRIMARY (ENR PVT2)':'PRIMARY WITH UPPER PRIMARY (ENR_PVT2)','PRIMARY WITH UPPER PRIMARY SEC/HIGHER SEC. (ENR PVT3)':'PRIMARY WITH UPPER PRIMARY SEC/HIGHER SEC. (ENR_PVT3)','UPPER PRIMARY ONLY (ENR PVT4)':'UPPER PRIMARY ONLY (ENR_PVT4)','UPPER PRIMARY WITH SEC./HIGHER SEC. (ENR PVT5)':'UPPER PRIMARY WITH SEC./HIGHER SEC. (ENR_PVT5)'})
dataEl_2013 = dataEl_2013.rename(index=str, columns={'PRIMARY (ENR PVT1)':'PRIMARY (ENR_PVT1)','PRIMARY WITH UPPER PRIMARY (ENR PVT2)':'PRIMARY WITH UPPER PRIMARY (ENR_PVT2)','PRIMARY WITH UPPER PRIMARY SEC/HIGHER SEC. (ENR PVT3)':'PRIMARY WITH UPPER PRIMARY SEC/HIGHER SEC. (ENR_PVT3)','UPPER PRIMARY ONLY (ENR PVT4)':'UPPER PRIMARY ONLY (ENR_PVT4)','UPPER PRIMARY WITH SEC./HIGHER SEC. (ENR PVT5)':'UPPER PRIMARY WITH SEC./HIGHER SEC. (ENR_PVT5)'})
dataEl_2015 = dataEl_2015.rename(index=str, columns={'PRIMARY ONLY (ENR1P)':'PRIMARY (ENR_PVT1)','PRIMARY WITH UPPER PRIMARY (ENR2P)':'PRIMARY WITH UPPER PRIMARY (ENR_PVT2)','PRIMARY WITH UPPER PRIMARY SEC/H.SEC (ENR3P)':'PRIMARY WITH UPPER PRIMARY SEC/HIGHER SEC. (ENR_PVT3)','UPPER PRIMARY ONLY (ENR4P)':'UPPER PRIMARY ONLY (ENR_PVT4)','UPPER PRIMARY WITH SEC./H.SEC (ENR5P)':'UPPER PRIMARY WITH SEC./HIGHER SEC. (ENR_PVT5)'})

dataEl_2009 = dataEl_2009.rename(index=str, columns={'PRIMARY (TCH GOVT1)':'PRIMARY (TCH_GOVT1)','PRIMARY WITH UPPER PRIMARY (TCH GOVT2)':'PRIMARY WITH UPPER PRIMARY (TCH_GOVT2)','PRIMARY WITH UPPER PRIMARY SEC/HIGHER SEC. (TCH GOVT3)':'PRIMARY WITH UPPER PRIMARY SEC/HIGHER SEC. (TCH_GOVT3)','UPPER PRIMARY ONLY (TCH GOVT4)':'UPPER PRIMARY ONLY (TCH_GOVT4)','UPPER PRIMARY WITH SEC./HIGHER SEC. (TCH GOVT5)':'UPPER PRIMARY WITH SEC./HIGHER SEC. (TCH_GOVT5)'})
dataEl_2010 = dataEl_2010.rename(index=str, columns={'PRIMARY (TCH GOVT1)':'PRIMARY (TCH_GOVT1)','PRIMARY WITH UPPER PRIMARY (TCH GOVT2)':'PRIMARY WITH UPPER PRIMARY (TCH_GOVT2)','PRIMARY WITH UPPER PRIMARY SEC/HIGHER SEC. (TCH GOVT3)':'PRIMARY WITH UPPER PRIMARY SEC/HIGHER SEC. (TCH_GOVT3)','UPPER PRIMARY ONLY (TCH GOVT4)':'UPPER PRIMARY ONLY (TCH_GOVT4)','UPPER PRIMARY WITH SEC./HIGHER SEC. (TCH GOVT5)':'UPPER PRIMARY WITH SEC./HIGHER SEC. (TCH_GOVT5)'})
dataEl_2011 = dataEl_2011.rename(index=str, columns={'PRIMARY (TCH GOVT1)':'PRIMARY (TCH_GOVT1)','PRIMARY WITH UPPER PRIMARY (TCH GOVT2)':'PRIMARY WITH UPPER PRIMARY (TCH_GOVT2)','PRIMARY WITH UPPER PRIMARY SEC/HIGHER SEC. (TCH GOVT3)':'PRIMARY WITH UPPER PRIMARY SEC/HIGHER SEC. (TCH_GOVT3)','UPPER PRIMARY ONLY (TCH GOVT4)':'UPPER PRIMARY ONLY (TCH_GOVT4)','UPPER PRIMARY WITH SEC./HIGHER SEC. (TCH GOVT5)':'UPPER PRIMARY WITH SEC./HIGHER SEC. (TCH_GOVT5)'})
dataEl_2013 = dataEl_2013.rename(index=str, columns={'UPPER PRIMARY WITH SEC. AND HIGHER SEC. (TCH GOVT5)':'UPPER PRIMARY WITH SEC./HIGHER SEC. (TCH_GOVT5)','PRIMARY (TCH GOVT1)':'PRIMARY (TCH_GOVT1)','PRIMARY WITH UPPER PRIMARY (TCH GOVT2)':'PRIMARY WITH UPPER PRIMARY (TCH_GOVT2)','PRIMARY WITH UPPER PRIMARY SEC AND HIGHER SEC. (TCH GOVT3)':'PRIMARY WITH UPPER PRIMARY SEC/HIGHER SEC. (TCH_GOVT3)','UPPER PRIMARY ONLY (TCH GOVT4)':'UPPER PRIMARY ONLY (TCH_GOVT4)','UPPER PRIMARY WITH SEC./HIGHER SEC. (TCH GOVT5)':'UPPER PRIMARY WITH SEC./HIGHER SEC. (TCH_GOVT5)'})
dataEl_2012 = dataEl_2012.rename(index=str, columns={'PRIMARY WITH UPPER PRIMARY SEC AND HIGHER SEC. (TCH_GOVT3)':'PRIMARY WITH UPPER PRIMARY SEC/HIGHER SEC. (TCH_GOVT3)','UPPER PRIMARY WITH SEC. AND HIGHER SEC. (TCH_GOVT5)':'UPPER PRIMARY WITH SEC./HIGHER SEC. (TCH_GOVT5)'})
dataEl_2015 = dataEl_2015.rename(index=str, columns={'PRIMARY ONLY (TCH1G)':'PRIMARY (TCH_GOVT1)','PRIMARY WITH UPPER PRIMARY (TCH2G)':'PRIMARY WITH UPPER PRIMARY (TCH_GOVT2)','PRIMARY WITH UPPER PRIMARY SEC/H.SEC (TCH3G)':'PRIMARY WITH UPPER PRIMARY SEC/HIGHER SEC. (TCH_GOVT3)','UPPER PRIMARY ONLY (TCH4G)':'UPPER PRIMARY ONLY (TCH_GOVT4)','UPPER PRIMARY WITH SEC./H.SEC (TCH5G)':'UPPER PRIMARY WITH SEC./HIGHER SEC. (TCH_GOVT5)'})

dataEl_2009 = dataEl_2009.rename(index=str, columns={'PRIMARY (TCH PVT1)':'PRIMARY (TCH_PVT1)','PRIMARY WITH UPPER PRIMARY (TCH PVT2)':'PRIMARY WITH UPPER PRIMARY (TCH_PVT2)','PRIMARY WITH UPPER PRIMARY SEC/HIGHER SEC. (TCH PVT3)':'PRIMARY WITH UPPER PRIMARY SEC/HIGHER SEC. (TCH_PVT3)','UPPER PRIMARY ONLY (TCH PVT4)':'UPPER PRIMARY ONLY (TCH_PVT4)','UPPER PRIMARY WITH SEC./HIGHER SEC. (TCH PVT5)':'UPPER PRIMARY WITH SEC./HIGHER SEC. (TCH_PVT5)'})
dataEl_2010 = dataEl_2010.rename(index=str, columns={'PRIMARY (TCH PVT1)':'PRIMARY (TCH_PVT1)','PRIMARY WITH UPPER PRIMARY (TCH PVT2)':'PRIMARY WITH UPPER PRIMARY (TCH_PVT2)','PRIMARY WITH UPPER PRIMARY SEC/HIGHER SEC. (TCH PVT3)':'PRIMARY WITH UPPER PRIMARY SEC/HIGHER SEC. (TCH_PVT3)','UPPER PRIMARY ONLY (TCH PVT4)':'UPPER PRIMARY ONLY (TCH_PVT4)','UPPER PRIMARY WITH SEC./HIGHER SEC. (TCH PVT5)':'UPPER PRIMARY WITH SEC./HIGHER SEC. (TCH_PVT5)'})
dataEl_2011 = dataEl_2011.rename(index=str, columns={'PRIMARY (TCH PVT1)':'PRIMARY (TCH_PVT1)','PRIMARY WITH UPPER PRIMARY (TCH PVT2)':'PRIMARY WITH UPPER PRIMARY (TCH_PVT2)','PRIMARY WITH UPPER PRIMARY SEC/HIGHER SEC. (TCH PVT3)':'PRIMARY WITH UPPER PRIMARY SEC/HIGHER SEC. (TCH_PVT3)','UPPER PRIMARY ONLY (TCH PVT4)':'UPPER PRIMARY ONLY (TCH_PVT4)','UPPER PRIMARY WITH SEC./HIGHER SEC. (TCH PVT5)':'UPPER PRIMARY WITH SEC./HIGHER SEC. (TCH_PVT5)'})
dataEl_2013 = dataEl_2013.rename(index=str, columns={'UPPER PRIMARY WITH SEC. AND HIGHER SEC. (TCH PVT5)':'UPPER PRIMARY WITH SEC./HIGHER SEC. (TCH_PVT5)','PRIMARY (TCH PVT1)':'PRIMARY (TCH_PVT1)','PRIMARY WITH UPPER PRIMARY (TCH PVT2)':'PRIMARY WITH UPPER PRIMARY (TCH_PVT2)','PRIMARY WITH UPPER PRIMARY SEC AND HIGHER SEC. (TCH PVT3)':'PRIMARY WITH UPPER PRIMARY SEC/HIGHER SEC. (TCH_PVT3)','UPPER PRIMARY ONLY (TCH PVT4)':'UPPER PRIMARY ONLY (TCH_PVT4)','UPPER PRIMARY WITH SEC./HIGHER SEC. (TCH PVT5)':'UPPER PRIMARY WITH SEC./HIGHER SEC. (TCH_PVT5)'})
dataEl_2012 = dataEl_2012.rename(index=str, columns={'PRIMARY WITH UPPER PRIMARY SEC AND HIGHER SEC. (TCH_PVT3)':'PRIMARY WITH UPPER PRIMARY SEC/HIGHER SEC. (TCH_PVT3)','UPPER PRIMARY WITH SEC. AND HIGHER SEC. (TCH_PVT5)':'UPPER PRIMARY WITH SEC./HIGHER SEC. (TCH_PVT5)'})
dataEl_2015 = dataEl_2015.rename(index=str, columns={'PRIMARY ONLY (TCH1P)':'PRIMARY (TCH_PVT1)','PRIMARY WITH UPPER PRIMARY (TCH2P)':'PRIMARY WITH UPPER PRIMARY (TCH_PVT2)','PRIMARY WITH UPPER PRIMARY SEC/H.SEC (TCH3P)':'PRIMARY WITH UPPER PRIMARY SEC/HIGHER SEC. (TCH_PVT3)','UPPER PRIMARY ONLY (TCH4P)':'UPPER PRIMARY ONLY (TCH_PVT4)','UPPER PRIMARY WITH SEC./H.SEC (TCH5P)':'UPPER PRIMARY WITH SEC./HIGHER SEC. (TCH_PVT5)'})

# Calculating geographic data
def getSchoolData(df):
    
    cols = list(df.columns.values)[3:]
    df = df.replace(',','',regex=True)
    df[cols] = df[cols].astype(float)
    df = df.rename(index=str, columns={'STATE NAME': 'NAME'})
    dfState = df.groupby(['NAME','YEAR'])[cols].sum().reset_index()
    dfState['NAME'] = dfState['NAME'].str.upper()
   
    dfDis   = df.groupby(['DISTRICT NAME','YEAR'])[cols].sum().reset_index()
    dfDis   = dfDis.rename(index=str, columns={'DISTRICT NAME': 'NAME'})
    dfDis['NAME'] = dfDis['NAME'].str.title()
    dfCom = pd.concat([dfState, dfDis])
    return dfCom
dataEl_2015F = getSchoolData(dataEl_2015)
dataEl_2013F = getSchoolData(dataEl_2013)
dataEl_2012F = getSchoolData(dataEl_2012)
dataEl_2011F = getSchoolData(dataEl_2011)
dataEl_2010F = getSchoolData(dataEl_2010)
dataEl_2009F = getSchoolData(dataEl_2009)
dataEl_2008F = getSchoolData(dataEl_2008)
dataEl_2007F = getSchoolData(dataEl_2007)
dataEl_2006F = getSchoolData(dataEl_2006)
dataEl_2005F = getSchoolData(dataEl_2005)
dataSchoolCom = pd.concat([dataEl_2005F,dataEl_2006F,dataEl_2007F,dataEl_2008F,dataEl_2009F,dataEl_2010F,dataEl_2011F,dataEl_2012F,dataEl_2013F,dataEl_2015F,datadefault2001,datadefault2002,datadefault2003,datadefault2004,datadefault2014,datadefault2016,datadefault2017],sort=False)
dataSchoolCom['YEAR'] = dataSchoolCom['YEAR'].astype(str)

dataSchoolCom.replace('ORISSA','ODISHA',inplace=True)
dataSchoolCom.replace('A & N ISLANDS','ANDAMAN & NICOBAR ISLANDS',inplace=True)
dataSchoolCom.replace('D & N HAVELI','DADRA & NAGAR HAVELI',inplace=True)
dataSchoolCom.replace('PONDICHERRY','PUDUCHERRY',inplace=True)
dataSchoolCom.replace('DELHI','NCT OF DELHI',inplace=True)
dataSchoolCom.replace('Middle And North Andamans','North  & Middle Andaman',inplace=True)
dataSchoolCom.replace('North Twenty Four Pargana','North Twenty Four Parganas',inplace=True)
dataSchoolCom.replace('South Twenty Four Pargan','South Twenty Four Parganas',inplace=True)
dataSchoolCom.replace('PONDICHERRY','PUDUCHERRY',inplace=True)

schoolscols = list(dataSchoolCom.columns.values)
dataSchoolComFinal = dataSchoolCom[schoolscols]
dataSchoolComFinal = dataSchoolComFinal.fillna(method = 'bfill', axis=0).fillna(0)
dataSchoolComFinal.head()
dfcomData = [datacomcenFinal,datacomcrimeFinal,datacomhealthFinal,datacomgdpFinal,dataSchoolComFinal]
dfAll     = reduce(lambda left,right: pd.merge(left,right, on=['NAME','YEAR'],how='left'),dfcomData)
dfAll     = dfAll.fillna(0)
dfAll.head()
