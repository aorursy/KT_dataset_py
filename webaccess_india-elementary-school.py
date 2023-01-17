import pandas as pd
from pandas_highcharts.core import serialize
from pandas_highcharts.display import display_charts
from IPython.display import display
pd.options.display.max_columns = None
import numpy as np
from functools import reduce

# Reading excel file
#dataSocio   = pd.read_csv('../input/all.csv', encoding='utf-8')
fileEL05_06 = pd.ExcelFile('../input/elementary_data_2005-06.xls')
fileEL06_07 = pd.ExcelFile('../input/elementary_data_2006-07.xls')
fileEL07_08 = pd.ExcelFile('../input/elementary_data_2007-08.xls')
fileEL08_09 = pd.ExcelFile('../input/elementary_data_2008-09.xls')
fileEL09_10 = pd.ExcelFile('../input/elementary_data_2009-10.xls')
fileEL10_11 = pd.ExcelFile('../input/elementary_data_2010-11.xls')
fileEL11_12 = pd.ExcelFile('../input/elementary_data_2011-12.xlsx')
#fileEL12_13 = pd.ExcelFile('data/elementary_data_2012-13.xlsx')
fileEL13_14 = pd.ExcelFile('../input/elementary_data_2013-14.xlsx')
fileEL14_15 = pd.ExcelFile('../input/elementary_data_2014-15.xlsx')
fileEL15_16 = pd.ExcelFile('../input/elementary_data_2015-16.xlsx')

# Getting data from multiple excel file for different year
dataEL1_2005_06 = pd.read_excel(fileEL05_06, 'School')
dataEL2_2005_06 = pd.read_excel(fileEL05_06, 'Enrolment')
dataEL3_2005_06 = pd.read_excel(fileEL05_06, 'Teachers')

dataEL1_2006_07 = pd.read_excel(fileEL06_07, 'School')
dataEL2_2006_07 = pd.read_excel(fileEL06_07, 'Enrolment')
dataEL3_2006_07 = pd.read_excel(fileEL06_07, 'TEACHERS')

dataEL1_2007_08 = pd.read_excel(fileEL07_08, 'School')
dataEL2_2007_08 = pd.read_excel(fileEL07_08, 'Enrolment')
dataEL3_2007_08 = pd.read_excel(fileEL07_08, 'Teacher')

dataEL1_2008_09 = pd.read_excel(fileEL08_09, 'School')
dataEL2_2008_09 = pd.read_excel(fileEL08_09, 'Enrolment')
dataEL3_2008_09 = pd.read_excel(fileEL08_09, 'Teacher')

dataEL1_2009_10 = pd.read_excel(fileEL09_10, 'School')
dataEL2_2009_10 = pd.read_excel(fileEL09_10, 'Enrolment')
dataEL3_2009_10 = pd.read_excel(fileEL09_10, 'Teacher')

dataEL1_2010_11 = pd.read_excel(fileEL10_11, 'School')
dataEL2_2010_11 = pd.read_excel(fileEL10_11, 'Enrolment')
dataEL3_2010_11 = pd.read_excel(fileEL10_11, 'Teacher')

dataEL1_2011_12 = pd.read_excel(fileEL11_12, 'School')
dataEL2_2011_12 = pd.read_excel(fileEL11_12, 'Enrolment')
dataEL3_2011_12 = pd.read_excel(fileEL11_12, 'Teacher')

#dataEL2_2012_13 = pd.read_excel(fileEL12_13, 'School')
#dataEL3_2012_13 = pd.read_excel(fileEL12_13, 'Enrolment')
#dataEL4_2012_13 = pd.read_excel(fileEL12_13, 'Teacher')

dataEL1_2013_14 = pd.read_excel(fileEL13_14, 'School')
dataEL2_2013_14 = pd.read_excel(fileEL13_14, 'Enrolment')
dataEL3_2013_14 = pd.read_excel(fileEL13_14, 'Teacher')

dataEL1_2014_15 = pd.read_excel(fileEL14_15, '2014-15_1')
dataEL2_2014_15 = pd.read_excel(fileEL14_15, '2014-15_2')

dataEL1_2015_16 = pd.read_excel(fileEL15_16, '2015-16_1')
dataEL2_2015_16 = pd.read_excel(fileEL15_16, '2015-16_2')
# Elementary data for 2005-06
dels05_s = (dataEL1_2005_06.iloc[14:]).replace(np.NaN,'', regex=True).reset_index()
dels05_s.columns = dels05_s.iloc[0].astype(str) + ' '+'('+dels05_s.iloc[1].astype(str) + ')'
dels05_s = dels05_s.drop([0,1]).drop(columns=['State Code (statecd)','District Code (distcd)']).rename(columns={' (ac_year)':'YEAR','State Name (statename)':'State','District Name (distname)':'District'})

dels05_e = (dataEL2_2005_06.iloc[14:]).replace(np.NaN,'', regex=True).reset_index()
dels05_e.columns = dels05_e.iloc[0].astype(str) + ' '+'('+dels05_e.iloc[1].astype(str) + ')'
dels05_e = dels05_e.drop([0,1]).drop(columns=['State Code (statecd)','District Code (distcd)']).rename(columns={'State Name (statename)':'State','District Name (distname)':'District'})

dels05_t = (dataEL3_2005_06.iloc[15:]).replace(np.NaN,'', regex=True).reset_index()
dels05_t.columns = dels05_t.iloc[0].astype(str) + ' '+'('+dels05_t.iloc[1].astype(str) + ')'
dels05_t = dels05_t.drop([0,1]).drop(columns=['State Code (statecd)','District Code (distcd)',' ()']).rename(columns={'State Name (statename)':'State','District Name (distname)':'District'})

dels05_06 = [dels05_s,dels05_e,dels05_t]
dels2005_06 = reduce(lambda left,right: pd.merge(left,right, on=['State','District']),dels05_06)
dels2005_06['State'] = dels2005_06['State'].str.split('.').str[0].str.strip()
dels2005_06['District'] = dels2005_06['District'].str.split('.').str[0].str.strip()
dels2005_06.columns = map(str.upper, dels2005_06.columns)
dels2005_06.columns = dels2005_06.columns.str.replace('_', ' ').str.strip()

# Elementary data for 2006-07
dels06_s = (dataEL1_2006_07.iloc[14:]).replace(np.NaN,'', regex=True).reset_index()
dels06_s.columns = dels06_s.iloc[0].astype(str) + ' '+'('+dels06_s.iloc[1].astype(str) + ')'
dels06_s = dels06_s.drop([0,1]).drop(columns=['State Code (statecd)','District Code (distcd)']).rename(columns={'Academic Year (ac_year)':'YEAR','State Name (statename)':'State','District Name (distname)':'District'})

dels06_e = (dataEL2_2006_07.iloc[14:]).replace(np.NaN,'', regex=True).reset_index()
dels06_e.columns = dels06_e.iloc[0].astype(str) + ' '+'('+dels06_e.iloc[1].astype(str) + ')'
dels06_e = dels06_e.drop([0,1]).drop(columns=['State Code (statecd)','District Code (distcd)']).rename(columns={'State Name (statename)':'State','District Name (distname)':'District'})

dels06_t = (dataEL3_2006_07.iloc[15:]).replace(np.NaN,'', regex=True).reset_index()
dels06_t.columns = dels06_t.iloc[0].astype(str) + ' '+'('+dels06_t.iloc[1].astype(str) + ')'
dels06_t = dels06_t.drop([0,1]).drop(columns=['State Code (statecd)','District Code (distcd)']).rename(columns={'State Name (statename)':'State','District Name (distname)':'District'})

dels06_07 = [dels06_s,dels06_e,dels06_t]
dels2006_07 = reduce(lambda left,right: pd.merge(left,right, on=['State','District']),dels06_07)
dels2006_07['State'] = dels2006_07['State'].str.split('.').str[0].str.strip()
dels2006_07['District'] = dels2006_07['District'].str.split('.').str[0].str.strip()
dels2006_07.columns = map(str.upper, dels2006_07.columns)
dels2006_07.columns = dels2006_07.columns.str.replace('_', ' ').str.strip()

# Elementary data for 2007-08
dels07_s = (dataEL1_2007_08.iloc[21:]).replace(np.NaN,'', regex=True).reset_index()
dels07_s.columns = dels07_s.iloc[0].astype(str) + ' '+'('+dels07_s.iloc[1].astype(str) + ')'
dels07_s = dels07_s.drop([0,1]).drop(columns=['State Code (statecd)','District Code (distcd)']).rename(columns={'Academic Year (ac_year)':'YEAR','State Name (statename)':'State','District Name (distname)':'District'})

dels07_e = (dataEL2_2007_08.iloc[15:]).replace(np.NaN,'', regex=True).reset_index()
dels07_e.columns = dels07_e.iloc[0].astype(str) + ' '+'('+dels07_e.iloc[1].astype(str) + ')'
dels07_e = dels07_e.drop([0,1]).drop(columns=['State Code (statecd)','District Code (distcd)']).rename(columns={'State Name (statename)':'State','District Name (distname)':'District'})

dels07_t = (dataEL3_2007_08.iloc[18:]).replace(np.NaN,'', regex=True).reset_index()
dels07_t.columns = dels07_t.iloc[0].astype(str) + ' '+'('+dels07_t.iloc[1].astype(str) + ')'
dels07_t = dels07_t.drop([0,1]).drop(columns=['18 (19)','State Code (statecd)','District Code (distcd)']).rename(columns={'State Name (statename)':'State','District Name (distname)':'District'})

dels07_08 = [dels07_s,dels07_e,dels07_t]
dels2007_08 = reduce(lambda left,right: pd.merge(left,right, on=['State','District']),dels07_08)
dels2007_08['State'] = dels2007_08['State'].str.split('.').str[0].str.strip()
dels2007_08['District'] = dels2007_08['District'].str.split('.').str[0].str.strip()
dels2007_08.columns = map(str.upper, dels2007_08.columns)
dels2007_08.columns = dels2007_08.columns.str.replace('_', ' ').str.strip()

# Elementary data for 2008-09
dels08_s = (dataEL1_2008_09.iloc[21:]).replace(np.NaN,'', regex=True).reset_index()
dels08_s.columns = dels08_s.iloc[0].astype(str) + ' '+'('+dels08_s.iloc[1].astype(str) + ')'
dels08_s = dels08_s.drop([0,1]).drop(columns=['State Code (Statecd)','District Code (Distcd)',]).rename(columns={'Academic Year (Ac Year)':'YEAR','State Name (Statename)':'State','District Name (Distname)':'District'})

dels08_e = (dataEL2_2008_09.iloc[15:]).replace(np.NaN,'', regex=True).reset_index()
dels08_e.columns = dels08_e.iloc[0].astype(str) + ' '+'('+dels08_e.iloc[1].astype(str) + ')'
dels08_e = dels08_e.drop([0,1]).drop(columns=['State Code (statecd)','District Code (distcd)']).rename(columns={'State Name (statename)':'State','District Name (distname)':'District'})

dels08_t = (dataEL3_2008_09.iloc[18:]).replace(np.NaN,'', regex=True).reset_index()
dels08_t.columns = dels08_t.iloc[0].astype(str) + ' '+'('+dels08_t.iloc[1].astype(str) + ')'
dels08_t = dels08_t.drop([0,1]).drop(columns=['18 (19)','State Code (statecd)','District Code (Distcd)']).rename(columns={'State Name (statename)':'State','District Name (distname)':'District'})

dels08_09 = [dels08_s,dels08_e,dels08_t]
dels2008_09 = reduce(lambda left,right: pd.merge(left,right, on=['State','District']),dels08_09)
dels2008_09['State'] = dels2008_09['State'].str.split('.').str[0].str.strip()
dels2008_09['District'] = dels2008_09['District'].str.split('.').str[0].str.strip()
dels2008_09.columns = map(str.upper, dels2008_09.columns)
dels2008_09.columns = dels2008_09.columns.str.replace('_', ' ').str.strip()

# Elementary data for 2009-10
dels09_s = (dataEL1_2009_10.iloc[21:]).replace(np.NaN, '', regex=True).reset_index()
dels09_s.columns = dels09_s.iloc[0].astype(str) + ' '+'('+dels09_s.iloc[1].astype(str) + ')'
dels09_s = dels09_s.drop([0,1]).drop(columns=['State Code (Statecd)','District Code (Distcd)',]).rename(columns={'State Name (Statename)':'State','District Name (Distname)':'District'})
dels09_s.insert(0,'YEAR','2009-10')

dels09_e = (dataEL2_2009_10.iloc[15:]).replace(np.NaN, '', regex=True).reset_index()
dels09_e.columns = dels09_e.iloc[0].astype(str) + ' '+'('+dels09_e.iloc[1].astype(str) + ')'
dels09_e = dels09_e.drop([0,1]).drop(columns=['State Code (Statecd)','District Code (Distcd)']).rename(columns={ 'State Name (Statename)':'State','District Name (Distname)':'District'})

dels09_t = (dataEL3_2009_10.iloc[18:]).replace(np.NaN, '', regex=True).reset_index()
dels09_t.columns = dels09_t.iloc[0].astype(str) + ' '+'('+dels09_t.iloc[1].astype(str) + ')'
dels09_t = dels09_t.drop([0,1]).drop(columns=['18 (19)','State Code (Statecd)','District Code (Distcd)']).rename(columns={'State Name (Statename)':'State','District Name (Distname)':'District'})

dels09_10 = [dels09_s,dels09_e,dels09_t]
dels2009_10 = reduce(lambda left,right: pd.merge(left,right, on=['State','District']),dels09_10)
dels2009_10['State'] = dels2009_10['State'].str.split('.').str[0].str.strip()
dels2009_10['District'] = dels2009_10['District'].str.split('.').str[0].str.strip()
dels2009_10.columns = map(str.upper, dels2009_10.columns)
dels2009_10.columns = dels2009_10.columns.str.replace('_', ' ').str.strip()

# Elementary data for 2010-11
dels10_s = (dataEL1_2010_11.iloc[20:]).replace(np.NaN, '', regex=True).reset_index()
dels10_s.columns = dels10_s.iloc[1].astype(str) + ' '+'('+dels10_s.iloc[2].astype(str) + ')'
dels10_s =  dels10_s.drop([0,1,2]).drop(columns=['Statecd (1)','District Code (Distcd)',' ()']).rename(columns={ 'Statename (JAMMU AND KASHMIR)':'State', 'District Name (Distname)':'District','Academic Year (Ac Year)':'YEAR'})

dels10_e = (dataEL2_2010_11.iloc[15:]).replace(np.NaN, '', regex=True).reset_index()
dels10_e.columns = dels10_e.iloc[0].astype(str) + ' '+'('+dels10_e.iloc[1].astype(str) + ')'
dels10_e = dels10_e.drop([0,1]).drop(columns=['State Code (Statecd)','Distcd (101)']).rename(columns={ 'State Name (State Name )':'State','Distname (KUPWARA)':'District'})

dels10_t = (dataEL3_2010_11.iloc[18:]).replace(np.NaN, '', regex=True).reset_index()
dels10_t.columns = dels10_t.iloc[0].astype(str) + ' '+'('+dels10_t.iloc[1].astype(str) + ')'
dels10_t = dels10_t.drop([0,1]).drop(columns=['18 (19)','State Code (Statecd)','District Code (Distcd)']).rename(columns={'State Name (Statename)':'State','District Name (Distname)':'District'})

dels10_11 = [dels10_s,dels10_e,dels10_t]
dels2010_11 = reduce(lambda left,right: pd.merge(left,right, on=['State','District']),dels10_11)
dels2010_11['State'] = dels2010_11['State'].str.split('.').str[0].str.strip()
dels2010_11['District'] = dels2010_11['District'].str.split('.').str[0].str.strip()
dels2010_11.columns = map(str.upper, dels2010_11.columns)
dels2010_11.columns = dels2010_11.columns.str.replace('_', ' ').str.strip()

# Elementary data for 2011-12
dels11_s = (dataEL1_2011_12.iloc[21:]).replace(np.NaN, '', regex=True).reset_index()
dels11_s = dels11_s.replace(np.NaN, '', regex=True)
dels11_s.columns = dels11_s.iloc[0].astype(str) + ' '+'('+dels11_s.iloc[1].astype(str) + ')'
dels11_s = dels11_s.drop([0,1]).drop(columns=['State Code (Statecd)','Distcd ()',' ()']).rename(columns={'State Name (Statename)':'State','Distname ()':'District','Academic Year (Ac Year)':'YEAR'})

dels11_e = (dataEL2_2011_12.iloc[15:]).replace(np.NaN, '', regex=True).reset_index()
dels11_e.columns = dels11_e.iloc[0].astype(str) + ' '+'('+dels11_e.iloc[1].astype(str) + ')'
dels11_e = dels11_e.drop([0,1]).drop(columns=['State Code (Statecd)','Distcd (distcd)']).rename(columns={'State Name (State Name )':'State','Distname (distname)':'District'})

dels11_t = (dataEL3_2011_12.iloc[18:]).replace(np.NaN, '', regex=True).reset_index()
dels11_t.columns = dels11_t.iloc[0].astype(str) + ' '+'('+dels11_t.iloc[1].astype(str) + ')'
dels11_t = dels11_t.drop([0,1]).drop(columns=['18 (19)','State Code (Statecd)','District Code (Distcd)']).rename(columns={'State Name (Statename)':'State','District Name (Distname)':'District'})
dels11_12 = [dels11_s,dels11_e,dels11_t]
dels2011_12 = reduce(lambda left,right: pd.merge(left,right, on=['State','District']),dels11_12)
dels2011_12['State'] = dels2011_12['State'].str.split('.').str[0].str.strip()
dels2011_12['District'] = dels2011_12['District'].str.split('.').str[0].str.strip()
dels2011_12.columns = map(str.upper, dels2011_12.columns)
dels2011_12.columns = dels2011_12.columns.str.replace('_', ' ').str.strip()
# Elementary school data for year 2013-14
dels13_s = (dataEL1_2013_14.iloc[21:]).replace(np.NaN, '', regex=True).reset_index()
dels13_s = dels13_s.replace(np.NaN, '', regex=True)
dels13_s.columns = dels13_s.iloc[0].astype(str) + ' '+'('+dels13_s.iloc[1].astype(str) + ')'
dels13_s = dels13_s.drop([0,1]).drop(columns=['State Code ()','District Code ()']).rename(columns={'State Name ()':'State','District Name ()':'District','Academic Year ()':'YEAR'})

dels13_e = (dataEL2_2013_14.iloc[15:]).replace(np.NaN, '', regex=True).reset_index()
dels13_e.columns = dels13_e.iloc[0].astype(str) + ' '+'('+dels13_e.iloc[1].astype(str) + ')'
dels13_e = dels13_e.drop([0,1]).drop(columns=['State Code (Statecd)','Distcd (distcd)']).rename(columns={'State Name (State Name )':'State','Distname (distname)':'District'})

dels13_t = (dataEL3_2013_14.iloc[18:]).replace(np.NaN, '', regex=True).reset_index()
dels13_t.columns = dels13_t.iloc[0].astype(str) + ' '+'('+dels13_t.iloc[1].astype(str) + ')'
dels13_t =dels13_t.drop([0,1]).drop(columns=['18 (19)','State Code (statecd)','District Code (distcd)','Academic Year (ac_year)']).rename(columns={'State Name (statename)':'State','District Name (distname)':'District'})

dels13_14 = [dels13_s,dels13_e,dels13_t]
dels2013_14 = reduce(lambda left,right: pd.merge(left,right, on=['State','District']),dels13_14)
dels2013_14['State'] = dels2013_14['State'].str.split('.').str[0].str.strip()
dels2013_14['District'] = dels2013_14['District'].str.split('.').str[0].str.strip()
dels2013_14.columns = map(str.upper, dels2013_14.columns)
dels2013_14.columns = dels2013_14.columns.str.replace('_', ' ').str.strip()

# Elementary data for year 2014-15
dls3 = dataEL1_2014_15.iloc[16:]
dls3 = dls3.replace(np.NaN, '', regex=True).replace(np.NaN, '', regex=True).reset_index()
dls3.columns = dls3.iloc[0].astype(str) + ' '+'('+dls3.iloc[1].astype(str) + ')'
dls3 = dls3.drop([0,1]).drop(columns=['16 (17)',' (DISTCD)',' (STATCD)',' (DISTRICTS)',' (BLOCKS)',' (VILLAGES)',' (CLUSTERS)'])
dls3 = dls3.rename(columns={' ()':'YEAR',' (STATNAME)':'State',' (DISTNAME)':'District'})

dls4 = dataEL2_2014_15.iloc[16:]
dls4 = dls4.replace(np.NaN, '', regex=True).reset_index()
dls4.columns = dls4.iloc[1].astype(str) + ' '+'('+dls4.iloc[2].astype(str) + ')'
dls4 = dls4.drop([0,1,2]).drop(columns=['17 (18)',' (DISTCD)','Primary Only (TCHM1)','Primary with Upper Primary (TCHM2)','Primary with upper Primary Sec/H.Sec (TCHM3)','Upper Primary Only (TCHM4)','Upper Primary with Sec./H.Sec (TCHM5)','Primary with upper Primary Sec (TCHM6)','Upper Primary with  Sec. (TCHM7)','Primary Only (TCHF1)','Primary with Upper Primary (TCHF2)','Primary with upper Primary Sec/H.Sec (TCHF3)','Primary with upper Primary Sec/H.Sec (TCHF3)','Upper Primary Only (TCHF4)','Upper Primary with Sec./H.Sec (TCHF5)','Primary with upper Primary Sec (TCHF6)','Upper Primary with  Sec. (TCHF7)']).rename(columns={' ()':'YEAR',' (STATNAME)':'State',' (DISTNAME)':'District'})
dels2014_15 = pd.merge(dls3,dls4,how='left',on=['State','District','YEAR'])
dels2014_15['State'] = dels2014_15['State'].str.split('.').str[0].str.strip()
dels2014_15['District'] = dels2014_15['District'].str.split('.').str[0].str.strip()
dels2014_15.columns = map(str.upper, dels2014_15.columns)
dels2014_15.columns = dels2014_15.columns.str.replace('_', ' ').str.strip()

# Elementary data for year 2015-16
dls = dataEL1_2015_16 .iloc[16:]
dls = dls.replace(np.NaN, '', regex=True).reset_index()
dls.columns = dls.iloc[0].astype(str) + ' '+'('+dls.iloc[1].astype(str) + ')'
dls = dls.drop([0,1]).drop(columns=['16 (17)',' (STATCD)',' (DISTCD)',' (DISTRICTS)',' (BLOCKS)',' (VILLAGES)',' (CLUSTERS)']).rename(columns={' ()':'YEAR',' (STATNAME)':'State',' (DISTNAME)':'District'})


dls2 = dataEL2_2015_16.iloc[16:]
dls2 = dls2.replace(np.NaN, '', regex=True).reset_index()
dls2.columns = dls2.iloc[1].astype(str) + ' '+'('+dls2.iloc[2].astype(str) + ')'
dls2 = dls2.drop([0,1,2]).drop(columns=['17 (18)',' (DISTCD)','Primary Only (TCHF1)','Primary with Upper Primary (TCHF2)','Primary with upper Primary Sec/H.Sec (TCHF3)','Upper Primary Only (TCHF4)','Upper Primary with Sec./H.Sec (TCHF5)','Primary with upper Primary Sec (TCHF6)','Upper Primary with  Sec. (TCHF7)','Primary Only (TCHM1)','Primary with Upper Primary (TCHM2)','Primary with upper Primary Sec/H.Sec (TCHM3)','Upper Primary Only (TCHM4)','Upper Primary with Sec./H.Sec (TCHM5)','Primary with upper Primary Sec (TCHM6)','Upper Primary with  Sec. (TCHM7)']).rename(columns={' ()':'YEAR',' (STATNAME)':'State',' (DISTNAME)':'District'})
dels2015_16 = pd.merge(dls,dls2,how='left',on=['State','District','YEAR'])
dels2015_16['State'] = dels2015_16['State'].str.split('.').str[0].str.strip()
dels2015_16['District'] = dels2015_16['District'].str.split('.').str[0].str.strip()
dels2015_16.columns = map(str.upper, dels2015_16.columns)
dels2015_16.columns = dels2015_16.columns.str.replace('_', ' ').str.strip()

dels2015_16
dels2014_15
dels2013_14
SWPQdf = dels2015_16.groupby(['STATE'])[['REGULAR TEACHERS WITH PROFESSIONAL QUALIFICATION : MALE  (PGRMTCH)','REGULAR TEACHERS WITH PROFESSIONAL QUALIFICATION : FEMALE  (PGRFTCH)','TOTAL REGULAR TEACHERS: MALE  (GRMTCH)','TOTAL REGULAR TEACHERS: FEMALE  (GRFTCH)','CONTRACTUAL TEACHERS WITH PROFESSIONAL QUALIFICATION : MALE  (PGCMTCH)','CONTRACTUAL  TEACHERS WITH PROFESSIONAL QUALIFICATION : FEMALE  (PGCFTCH)','TOTAL CONTRACTUAL  TEACHERS: MALE  (PCMTCH)','TOTAL CONTRACTUAL  TEACHERS: FEMALE  (PCFTCH)']].sum()
display_charts(SWPQdf , kind="bar", title="State wise Professionally Qualified Teachers: Government",figsize = (1000, 700))
SWOTSCdf = dels2015_16.groupby(['STATE'])[['PRIMARY ONLY (TCHOBCM1)','PRIMARY WITH UPPER PRIMARY (TCHOBCM2)','PRIMARY WITH UPPER PRIMARY SEC/H.SEC (TCHOBCM3)','UPPER PRIMARY ONLY (TCHOBCM4)','UPPER PRIMARY WITH SEC./H.SEC (TCHOBCM5)','PRIMARY WITH UPPER PRIMARY SEC (TCHOBCM6)','UPPER PRIMARY WITH  SEC. (TCHOBCM7)']].sum()
display_charts(SWOTSCdf, kind="barh", title="State wise  male OBC Teachers by School Category ",figsize = (1000, 700))
SWPQTMdf = dels2015_16.groupby(['STATE'])[['TEACHERS WITH PROFESSIONAL QUALIFICATION : FEMALE  (PPFTCH)','TEACHERS WITH PROFESSIONAL QUALIFICATION : MALE  (PPMTCH)','TOTAL  TEACHERS: MALE  (PMTCH)','TOTAL  TEACHERS: FEMALE  (PFTCH)']].sum()
display_charts(SWPQTMdf, title="State wise Professionally Qualified Teachers: Private", kind="line",figsize = (1000, 700))
SWNCSTdf = dels2015_16.groupby(['STATE'])[['PRIMARY ONLY (CLS1)','PRIMARY WITH UPPER PRIMARY (CLS2)','PRIMARY WITH UPPER PRIMARY SEC/H.SEC (CLS3)','UPPER PRIMARY ONLY (CLS4)','UPPER PRIMARY WITH SEC./H.SEC (CLS5)','PRIMARY WITH UPPER PRIMARY SEC (CLS6)','UPPER PRIMARY WITH  SEC. (CLS7)','TOTAL (CLSTOT)']].sum()
display_charts(SWNCSTdf, title="State wise Number of Classrooms by School Category", kind="bar",figsize = (1000, 700))
SWEESCdf = dels2015_16.groupby(['STATE'])[['PRIMARY ONLY (ENR1)','PRIMARY WITH UPPER PRIMARY (ENR2)','PRIMARY WITH UPPER PRIMARY SEC/H.SEC (ENR3)','UPPER PRIMARY ONLY (ENR4)','UPPER PRIMARY WITH SEC./H.SEC (ENR5)','PRIMARY WITH UPPER PRIMARY SEC (ENR6)','UPPER PRIMARY WITH  SEC. (ENR7)','NO RESPONSE (ENR9)','TOTAL (ENRTOT)']].sum()
display_charts(SWEESCdf, title="State wise Elementary School Enrolment by School Category", kind="bar",figsize = (1000, 700))
SWNOSdf = dels2015_16.groupby(['STATE'])[['TOTAL (SCHTOT)','TOTAL (SCHTOTG)','TOTAL (SCHTOTP)','TOTAL (SCHTOTM)']].sum().rename(columns={'TOTAL (SCHTOT)':'Total','TOTAL (SCHTOTG)':'Government','TOTAL (SCHTOTP)':'Private','TOTAL (SCHTOTM)':'Madarsa & Unrecognised'})
display_charts(SWNOSdf, title="State wise no of schools in India", kind="area",figsize = (1000, 700))
PODTOSdf = dels2015_16[['TOTAL (SCHTOTG)','TOTAL (SCHTOTP)', 'TOTAL (SCHTOTM)']].rename(columns={'TOTAL (SCHTOTG)':'Government','TOTAL (SCHTOTP)':'Private','TOTAL (SCHTOTM)':'Madarsa & Unrecognised'}).sum()
PODTOSdf = pd.DataFrame(PODTOSdf)
display_charts(PODTOSdf, kind='pie', title='Schools In India', tooltip={'pointFormat': '{series.name}: <b>{point.percentage:.1f}%</b>'})
SWGSIRAdf = dels2015_16.groupby(['STATE'])[['PRIMARY ONLY (SCH1GR)','PRIMARY WITH UPPER PRIMARY (SCH2GR)','PRIMARY WITH UPPER PRIMARY SEC/H.SEC (SCH3GR)','UPPER PRIMARY ONLY (SCH4GR)','UPPER PRIMARY WITH SEC./H.SEC (SCH5GR)','PRIMARY WITH UPPER PRIMARY SEC (SCH6GR)','UPPER PRIMARY WITH  SEC. (SCH7GR)','NO RESPONSE (SCH9GR)','TOTAL (SCHTOTGR)']].sum()
display_charts(SWGSIRAdf, title="State wise government school by category in rural areas", kind="bar",figsize = (1000, 700))
SWTBSCdf = dels2014_15.groupby(['STATE'])[['PRIMARY ONLY (TCH1)','PRIMARY WITH UPPER PRIMARY (TCH2)','PRIMARY WITH UPPER PRIMARY SEC/H.SEC (TCH3)','UPPER PRIMARY ONLY (TCH4)','UPPER PRIMARY WITH SEC./H.SEC (TCH5)','PRIMARY WITH UPPER PRIMARY SEC (TCH6)','UPPER PRIMARY WITH  SEC. (TCH7)','TOTAL (TCHTOT)']].sum()
display_charts(SWTBSCdf, title="State wise no of teachers by school category in India", kind="line",figsize = (1000, 700))
SWSCSdf = dels2014_15.groupby(['STATE'])[['PRIMARY ONLY (SCLS1)','PRIMARY WITH UPPER PRIMARY (SCLS2)','PRIMARY WITH UPPER PRIMARY SEC/H.SEC (SCLS3)','UPPER PRIMARY ONLY (SCLS4)','UPPER PRIMARY WITH SEC./H.SEC (SCLS5)','PRIMARY WITH UPPER PRIMARY SEC (SCLS6)','UPPER PRIMARY WITH  SEC. (SCLS7)','TOTAL (SCLSTOT)']].sum()
display_charts(SWSCSdf, title="State wise no of single-classroom schools by school category", kind="bar",figsize = (1000, 700))
SWSTSdf = dels2014_15.groupby(['STATE'])[['PRIMARY ONLY (STCH1)','PRIMARY WITH UPPER PRIMARY (STCH2)','PRIMARY WITH UPPER PRIMARY SEC/H.SEC (STCH3)','UPPER PRIMARY ONLY (STCH4)','UPPER PRIMARY WITH SEC./H.SEC (STCH5)','PRIMARY WITH UPPER PRIMARY SEC (STCH6)','UPPER PRIMARY WITH  SEC. (STCH7)','TOTAL (STCHTOT)']].sum()
display_charts(SWSTSdf, title="State wise no of single teacher school by school category", kind="area",figsize = (1000, 700))
SWSAWRdf = dels2014_15.groupby(['STATE'])[['PRIMARY ONLY (ROAD1)','PRIMARY WITH UPPER PRIMARY (ROAD2)','PRIMARY WITH UPPER PRIMARY SEC/H.SEC (ROAD3)','UPPER PRIMARY ONLY (ROAD4)','UPPER PRIMARY WITH SEC./H.SEC (ROAD5)','PRIMARY WITH UPPER PRIMARY SEC (ROAD6)','UPPER PRIMARY WITH  SEC. (ROAD7)','TOTAL (ROADTOT)']].sum()
display_charts(SWSAWRdf, title="State wise no of schools approachable by all weather road by school category", kind="bar",figsize = (1000, 700))
SWSWPFdf = dels2014_15.groupby(['STATE'])[['PRIMARY ONLY (SPLAY1)','PRIMARY WITH UPPER PRIMARY (SPLAY2)','PRIMARY WITH UPPER PRIMARY SEC/H.SEC (SPLAY3)','UPPER PRIMARY ONLY (SPLAY4)','UPPER PRIMARY WITH SEC./H.SEC (SPLAY5)','PRIMARY WITH UPPER PRIMARY SEC (SPLAY6)','UPPER PRIMARY WITH  SEC. (SPLAY7)','TOTAL (SPLAYTOT)']].sum()
display_charts(SWSWPFdf, title="State wise no of schools with playground facility by school category", kind="barh",figsize = (1000, 700))
SWSWBWdf = dels2014_15.groupby(['STATE'])[['PRIMARY ONLY (SBNDR1)','PRIMARY WITH UPPER PRIMARY (SBNDR2)','PRIMARY WITH UPPER PRIMARY SEC/H.SEC (SBNDR3)','UPPER PRIMARY ONLY (SBNDR4)','UPPER PRIMARY WITH SEC./H.SEC (SBNDR5)','PRIMARY WITH UPPER PRIMARY SEC (SBNDR6)','UPPER PRIMARY WITH  SEC. (SBNDR7)','TOTAL (SBNDRTOT)']].sum()
display_charts(SWSWBWdf, title="State wise no of schools with Boundarywall by school category", kind="area",figsize = (1000, 700))
SWBFdf = dels2014_15[['TOTAL (SGTOILTOT)','TOTAL (SBTOILTOT)','TOTAL (SWATTOT)','TOTAL (SELETOT)','TOTAL (SCOMPTOT)']].rename(columns={'TOTAL (SGTOILTOT)':'School with girls toilet','TOTAL (SBTOILTOT)':'School with boys toilet','TOTAL (SWATTOT)':'School with drinking water facility','TOTAL (SELETOT)':'School with electricity','TOTAL (SCOMPTOT)':'School with computer'}).sum()
SWBFdf = pd.DataFrame(SWBFdf)
display_charts(SWBFdf, kind='pie', title='Schools with basic facility in India', tooltip={'pointFormat': '{series.name}: <b>{point.percentage:.1f}%</b>'})
delsAppend1 = dels2007_08.append([dels2008_09,dels2009_10,dels2010_11,dels2011_12])
delsAppend1
YWGSTIDS = delsAppend1.groupby(['STATE','YEAR'])[['PRIMARY (TCH GOVT1)','PRIMARY WITH UPPER PRIMARY (TCH GOVT2)','PRIMARY WITH UPPER PRIMARY SEC/HIGHER SEC. (TCH GOVT3)','UPPER PRIMARY ONLY (TCH GOVT4)','UPPER PRIMARY WITH SEC./HIGHER SEC. (TCH GOVT5)','NO RESPONSE (TCH GOVT9)']].sum()
display_charts(YWGSTIDS, title="Year wise government school teachers in different states by school category", kind="bar",figsize = (1000, 700))
YWGSTI = delsAppend1.groupby(['YEAR'])[['PRIMARY (TCH GOVT1)','PRIMARY WITH UPPER PRIMARY (TCH GOVT2)','PRIMARY WITH UPPER PRIMARY SEC/HIGHER SEC. (TCH GOVT3)','UPPER PRIMARY ONLY (TCH GOVT4)','UPPER PRIMARY WITH SEC./HIGHER SEC. (TCH GOVT5)','NO RESPONSE (TCH GOVT9)']].sum()
display_charts(YWGSTI, title="Year wise government school teachers in India", kind="bar",figsize = (1000, 700))
YWGSCTG = delsAppend1
YWGSCTG['Government School Teacher'] = YWGSCTG['PRIMARY (TCH GOVT1)'] + YWGSCTG['PRIMARY WITH UPPER PRIMARY (TCH GOVT2)'] + YWGSCTG['PRIMARY WITH UPPER PRIMARY SEC/HIGHER SEC. (TCH GOVT3)'] + YWGSCTG['UPPER PRIMARY ONLY (TCH GOVT4)']+YWGSCTG['UPPER PRIMARY WITH SEC./HIGHER SEC. (TCH GOVT5)']
YWGGST = pd.DataFrame(YWGSCTG.groupby(['YEAR'])['Government School Teacher'].sum())
display_charts(YWGGST, title="Year wise growth in government school teacher in India", kind="bar",figsize = (1000, 700))