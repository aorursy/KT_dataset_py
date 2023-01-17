#imports libraries
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import datetime as dt
import datetime as dt
#displays 500 rows before auto collapsing - used for visual analysis
pd.set_option('display.max_rows', 5000)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

#imports csv file
df = pd.read_csv('../input/gun-violence-data_01-2013_03-2018.csv')

# Any results you write to the current directory are saved as output.
#creates copy of df
df_general = df.copy()
df_gun = df.copy()
df_people = df.copy()
#changes datatypes
df_general['date'] = pd.to_datetime(df_general['date'])
df_general['state_house_district'] = df_general['state_house_district'].astype(object)
df_general['state_senate_district'] = df_general['state_senate_district'].astype(object)
df_general['congressional_district'] = df_general['congressional_district'].astype(object)
df_gun['n_guns_involved'] = df_gun['n_guns_involved'].astype(object)
#creates new column - county
df_general['county'] = df_general.loc[df_general['city_or_county'].str.contains('county', case=False), 'city_or_county']
#renames city or country column to city
df_general.rename(columns={'city_or_county': 'city'}, inplace=True)
#extracts extra characters
df_general['county'] = df_general['county'].str.lower()
df_general['county'] = df_general['county'].str.replace('\(|\)', '')
df_general['county'] = df_general['county'].str.replace('county county', 'county')
#changes str to lower case
df_people['participant_type'] = df_people['participant_type'].str.lower()
#creates new columns for participant_types
df_people['victim'] = df_people.participant_type.str.count('victim').astype(object)
df_people['suspect'] = df_people.participant_type.str.count('Subject-Suspect').astype(object)
df_people = df_people.drop(columns = ['participant_type'])
#changes str to lower case
df_people['participant_gender'] = df_people['participant_gender'].str.lower()
#creates new columns for gender
df_people['male'] = df_people.participant_gender.str.count('male').astype(object)
df_people['female'] = df_people.participant_gender.str.count('female').astype(object)
df_people = df_people.drop(columns = ['participant_gender'])
#changes str to lower case
df_people['participant_status'] = df_people['participant_status'].str.lower()
#makes column for unharmed/arrested
df_people['unharmed_arrested'] = df_people.participant_status.str.count('unharmed, arrested')
#creates columns for singular accounts or unharmed and arrested
df_people['unharmed'] = df_people.participant_status.str.count('unharmed')
df_people['arrested'] = df_people.participant_status.str.count('arrested')
#subtracts duplicates
df_people['arrested'] = df_people.arrested - df_people.unharmed_arrested
df_people['unharmed'] = df_people.unharmed - df_people.unharmed_arrested
#drops unneccesary column
df_people.drop(columns=['participant_status'], inplace=True)
#adds those killed and injured into total_affected column
df_people['total_killed_injured'] = df_people.n_killed + df_people.n_injured
df_people['total_involved'] = df_people.n_killed + df_people.n_injured + df_people.arrested + df_people.unharmed + df_people.unharmed_arrested
#change datatypes
df_people['arrested'] = df_people['arrested'].astype(object)
df_people['unharmed'] = df_people['unharmed'].astype(object)
df_people['unharmed_arrested'] = df_people['unharmed_arrested'].astype(object)
df_people['total_killed_injured'] = df_people['total_killed_injured'].astype(object)
df_people['total_involved'] = df_people['total_involved'].astype(object)
#changes column case
df_gun['gun_type'] = df_gun['gun_type'].str.lower()
#creates gun type columns to keep
df_gun['gun_handgun'] = df_gun.gun_type.str.count('handgun')
df_gun['gun_shotgun'] = df_gun.gun_type.str.count('shotgun')
df_gun['gun_rifle'] = df_gun.gun_type.str.count('rifle')
#creates interim gun type columns
df_gun['3030win'] = df_gun.gun_type.str.count('30-30 win')
df_gun['300win'] = df_gun.gun_type.str.count('308 win')
df_gun['308win'] = df_gun.gun_type.str.count('308 win')
df_gun['ar'] = df_gun.gun_type.str.count('ar-15')
df_gun['ak'] = df_gun.gun_type.str.count('ak-47')
df_gun['10mm'] = df_gun.gun_type.str.count('10mm')
df_gun['32auto'] = df_gun.gun_type.str.count('32 auto')
df_gun['380auto'] = df_gun.gun_type.str.count('380 auto')
df_gun['45auto'] = df_gun.gun_type.str.count('45 auto')
df_gun['25auto'] = df_gun.gun_type.str.count('25 auto')
df_gun['357mag'] = df_gun.gun_type.str.count('357 mag')
df_gun['40sw'] = df_gun.gun_type.str.count('40 sw')
df_gun['12gauge'] = df_gun.gun_type.str.count('12 gauge')
df_gun['16gauge'] = df_gun.gun_type.str.count('16 gauge')
df_gun['20gauge'] = df_gun.gun_type.str.count('20 gauge')
df_gun['28gauge'] = df_gun.gun_type.str.count('28 gauge')
df_gun['410gauge'] = df_gun.gun_type.str.count('410 gauge')
#changes interim columns into aggregated columns
df_gun['gun_shotgun'] = df_gun['gun_shotgun'] + df_gun['410gauge'] + df_gun['28gauge'] + df_gun['20gauge'] +  df_gun['16gauge'] + df_gun['12gauge']
df_gun['gun_rifle'] = df_gun['gun_rifle'] + df_gun['3030win'] + df_gun['300win'] + df_gun['308win'] + df_gun['ar'] + df_gun['ak']
df_gun['gun_handgun'] = df_gun['gun_handgun'] + df_gun['40sw'] + df_gun['357mag'] + df_gun['25auto'] + df_gun['45auto'] + df_gun['380auto'] + df_gun['32auto'] + df_gun['10mm']
#drop interim columns
df_gun.drop(columns=['gun_type', '3030win', '300win', '308win', 'ak', 'ar', '10mm', '32auto', '380auto', '45auto', '25auto', '357mag', '40sw', '12gauge', '16gauge', '20gauge', '28gauge', '410gauge'], inplace=True)
#changes datatypes 
df_gun['gun_shotgun'] = df_gun['gun_shotgun'].astype(object)
df_gun['gun_rifle'] = df_gun['gun_rifle'].astype(object)
df_gun['gun_handgun'] = df_gun['gun_handgun'].astype(object)
# changes column case
df_gun['gun_stolen'] = df_gun['gun_stolen'].str.lower()
#creates interim columns for gun delineations
df_gun['unknown_stolen'] = df_gun.gun_stolen.str.count('unknown')
df_gun['stolen_gun'] = df_gun.gun_stolen.str.count('stolen')
df_gun['not_stolen'] = df_gun.gun_stolen.str.count('not-stolen')
df_gun['gun_stolen'] = df_gun['stolen_gun']
#changes datatypes
df_gun['unknown_stolen'] = df_gun['unknown_stolen'].astype(object)
df_gun['gun_stolen'] = df_gun['gun_stolen'].astype(object)
df_gun['not_stolen'] = df_gun['not_stolen'].astype(object)
#drops interim columns
df_gun.drop(columns=['stolen_gun', 'unknown_stolen', 'not_stolen'], inplace=True)
#changes case
df_people['participant_relationship'] = df_people['participant_relationship'].str.lower()
#splits relationships into separate colums
df_people['relationship_significant_other'] = df_people.participant_relationship.str.count('significant other').astype(object)
df_people['relationship_mass_shooting_known'] = df_people.participant_relationship.str.count('mass shooting - perp').astype(object)
df_people['relationship_family'] = df_people.participant_relationship.str.count('family').astype(object)
df_people['relationship_friend'] = df_people.participant_relationship.str.count('friend').astype(object)
df_people['relationship_home_invasion_known'] = df_people.participant_relationship.str.count('home invasion - - perp knows').astype(object)
df_people['relationship_coworker'] = df_people.participant_relationship.str.count('co-worker').astype(object)
df_people['relationship_aquaintance'] = df_people.participant_relationship.str.count('aquaintance').astype(object)
df_people['relationship_neighbor'] = df_people.participant_relationship.str.count('neighbor').astype(object)
df_people.drop(columns=['participant_relationship'], inplace=True)
#changes case
df_people['incident_characteristics'] = df_people['incident_characteristics'].str.lower()
#creates new columns
df_people['incident_shot'] = df_people.incident_characteristics.str.count('shot')
df_people['incident_driveby'] = df_people.incident_characteristics.str.count('drive-by')
df_people['incident_tsa'] = df_people.incident_characteristics.str.count('tsa action')
df_people['incident_nonshooting'] = df_people.incident_characteristics.str.count('non-shooting')
df_people['incident_domestic'] = df_people.incident_characteristics.str.count('domestic violence')
df_people['incident_standoff'] = df_people.incident_characteristics.str.count('standoff')
df_people['incident_gang'] = df_people.incident_characteristics.str.count('gang')
df_people['incident_carjacking'] = df_people.incident_characteristics.str.count('car-jacking')
df_people['incident_suicide'] = df_people.incident_characteristics.str.count('suicide')
df_people['incident_murdersuicide'] = df_people.incident_characteristics.str.count('murder/suicide')
df_people['incident_accident'] = df_people.incident_characteristics.str.count('accident')
df_people['incident_homeinvasion'] = df_people.incident_characteristics.str.count('home invasion')
df_people['incident_school'] = df_people.incident_characteristics.str.count('school')
df_people['incident_massshooting'] = df_people.incident_characteristics.str.count('mass shooting')
df_people['incident_animal'] = df_people.incident_characteristics.str.count('animal shot/killed')
df_people['incident_roadrage'] = df_people.incident_characteristics.str.count('road rage')
df_people['involved_child'] = df_people.incident_characteristics.str.count('child involved')
df_people['incident_abduction'] = df_people.incident_characteristics.str.count('kidnapping')
df_people['incident_defensive'] = df_people.incident_characteristics.str.count('defensive use')
df_people['incident_sexcrime'] = df_people.incident_characteristics.str.count('sex crime')
df_people['incident_spree'] = df_people.incident_characteristics.str.count('spree shooting')
df_people['incident_hatecrime'] = df_people.incident_characteristics.str.count('hate crime')
df_people['incident_policetarget'] = df_people.incident_characteristics.str.count('police targeted')
df_gun['ghostgun'] = df_gun.incident_characteristics.str.count('ghost gun')
#deletes duplicates for shot column
df_people['incident_shot'] = df_people.incident_shot.replace(2.0, 1.0)
df_people['incident_shot'] = df_people.incident_shot.replace(3.0, 1.0)
df_people['incident_shot'] = df_people.incident_shot.replace(4.0, 1.0)
df_people['incident_shot'] = df_people.incident_shot.replace(5.0, 1.0)
#deletes duplicates for standoff column
df_people['incident_standoff'] = df_people.incident_standoff.replace(2.0, 1.0)
#deletes duplicates for suicide column
df_people['incident_suicide'] = df_people.incident_suicide.replace(2.0, 1.0)
df_people['incident_suicide'] = df_people.incident_suicide.replace(3.0, 1.0)
df_people['incident_suicide'] = df_people.incident_suicide.replace(4.0, 1.0)
df_people['incident_suicide'] = df_people.incident_suicide.replace(5.0, 1.0)
#deletes duplicates for murdersuicide column
df_people['incident_murdersuicide'] = df_people.incident_murdersuicide.replace(2.0, 1.0)
#deletes duplicates for accident column
df_people['incident_accident'] = df_people.incident_accident.replace(2.0, 1.0)
df_people['incident_accident'] = df_people.incident_accident.replace(3.0, 1.0)
df_people['incident_accident'] = df_people.incident_accident.replace(4.0, 1.0)
df_people['incident_accident'] = df_people.incident_accident.replace(5.0, 1.0)
df_people['incident_accident'] = df_people.incident_accident.replace(6.0, 1.0)
#deletes duplicates for homeinvasion column
df_people['incident_homeinvasion'] = df_people.incident_homeinvasion.replace(2.0, 1.0)
df_people['incident_homeinvasion'] = df_people.incident_homeinvasion.replace(3.0, 1.0)
df_people['incident_homeinvasion'] = df_people.incident_homeinvasion.replace(4.0, 1.0)
df_people['incident_homeinvasion'] = df_people.incident_homeinvasion.replace(5.0, 1.0)
#deletes duplicates for school column
df_people['incident_school'] = df_people.incident_school.replace(2.0, 1.0)
df_people['incident_school'] = df_people.incident_school.replace(3.0, 1.0)
df_people['incident_school'] = df_people.incident_school.replace(4.0, 1.0)
df_people['incident_school'] = df_people.incident_school.replace(5.0, 1.0)
#deletes duplicates for defensive column
df_people['incident_defensive'] = df_people.incident_defensive.replace(2.0, 1.0)
df_people['incident_defensive'] = df_people.incident_defensive.replace(3.0, 1.0)
df_people['incident_defensive'] = df_people.incident_defensive.replace(4.0, 1.0)
df_people['incident_defensive'] = df_people.incident_defensive.replace(5.0, 1.0)
#aggregates columns with overlap in armed robbery
df_people['gun_shop'] = df_people.incident_characteristics.str.count('gun shop robbery')
df_people['incident_armed_robbery'] = df_people.incident_characteristics.str.count('armed robbery')
#adds columns
df_people['incident_armed_robbery'] = df_people['incident_armed_robbery'] + df_people['gun_shop']
#deletes duplicates
df_people['incident_armed_robbery'] = df_people.incident_armed_robbery.replace(2.0, 1.0)
#drops interim column
df_people.drop(columns=['gun_shop'], inplace=True)
#aggregates columns with overlap in drugs/alcohol
df_people['drug_involvement'] = df_people.incident_characteristics.str.count('drug involvement')
df_people['involved_drug_alcohol'] = df_people.incident_characteristics.str.count('alcohol or drugs')
#adds columns
df_people['involved_drug_alcohol'] = df_people['involved_drug_alcohol'] + df_people['drug_involvement']
#replaces duplicates
df_people['involved_drug_alcohol'] = df_people.involved_drug_alcohol.replace(2.0, 1.0)
#drops interim column
df_people.drop(columns=['drug_involvement'], inplace=True)
#aggregates columns with overlap in institution
df_people['bar'] = df_people.incident_characteristics.str.count('bar/club')
df_people['gun_range'] = df_people.incident_characteristics.str.count('gun range/gun shop')
df_people['house_party'] = df_people.incident_characteristics.str.count('house party')
df_people['incident_institution'] = df_people.incident_characteristics.str.count('institution/group/business')
#adds columns
df_people['incident_institution'] = df_people['incident_institution'] + df_people['bar'] + df_people['gun_range'] + df_people['house_party']
#replaces duplicates
df_people['incident_institution'] = df_people.incident_institution.replace(2.0, 1.0)
df_people['incident_institution'] = df_people.incident_institution.replace(3.0, 1.0)
#drops interim column
df_people.drop(columns=['bar', 'gun_range', 'house_party'], inplace=True)
#changes datatype
df_people['incident_shot'] = df_people['incident_shot'].astype(object)
df_people['incident_driveby'] = df_people['incident_driveby'].astype(object)
df_people['incident_tsa'] = df_people['incident_tsa'].astype(object)
df_people['incident_nonshooting'] = df_people['incident_nonshooting'].astype(object)
df_people['incident_domestic'] = df_people['incident_domestic'].astype(object)
df_people['incident_standoff'] = df_people['incident_standoff'].astype(object)
df_people['incident_gang'] = df_people['incident_gang'].astype(object)
df_people['incident_carjacking'] = df_people['incident_carjacking'].astype(object)
df_people['incident_suicide'] = df_people['incident_suicide'].astype(object)
df_people['incident_murdersuicide'] = df_people['incident_murdersuicide'].astype(object)
df_people['incident_accident'] = df_people['incident_accident'].astype(object)
df_people['incident_homeinvasion'] = df_people['incident_homeinvasion'].astype(object)
df_people['incident_school'] = df_people['incident_school'].astype(object)
df_people['incident_massshooting'] = df_people['incident_massshooting'].astype(object)
df_people['incident_animal'] = df_people['incident_animal'].astype(object)
df_people['incident_roadrage'] = df_people['incident_roadrage'].astype(object)
df_people['involved_child'] = df_people['involved_child'].astype(object)
df_people['incident_abduction'] = df_people['incident_abduction'].astype(object)
df_people['incident_defensive'] = df_people['incident_defensive'].astype(object)
df_people['incident_sexcrime'] = df_people['incident_sexcrime'].astype(object)
df_people['incident_spree'] = df_people['incident_spree'].astype(object)
df_people['incident_hatecrime'] = df_people['incident_hatecrime'].astype(object)
df_people['incident_policetarget'] = df_people['incident_policetarget'].astype(object)
df_people['involved_drug_alcohol'] = df_people['involved_drug_alcohol'].astype(object)
df_gun['ghostgun'] = df_gun['ghostgun'].astype(object)
#removes extra characters 
df_people['participant_age'] = df_people['participant_age'].str.replace('\d::', '')
#drops unncessary columns in df_general
df_general = df_general[['incident_id', 'date', 'state', 'county', 'city', 'congressional_district', 'incident_url', 'source_url', 'latitude', 'location_description', 'longitude', 'notes', 'sources', 'state_house_district', 'state_senate_district']]
#drops unncessary columns in df_people
df_people.drop(columns=['date', 'state', 'address', 'city_or_county', 'source_url', 'incident_url', 'gun_stolen', 'gun_type', 'participant_name', 'participant_age_group', 'congressional_district', 'incident_url_fields_missing', 'incident_characteristics', 'latitude', 'location_description', 'longitude', 'n_guns_involved', 'notes', 'sources', 'state_house_district', 'state_senate_district'], inplace=True)
#drops unncessary columns in df_gun
df_gun = df_gun[['incident_id', 'gun_stolen', 'n_guns_involved', 'gun_handgun', 'gun_shotgun', 'gun_rifle', 'ghostgun']]
#saves files to csv
df_general.to_csv('gun_violence_general.csv')
df_people.to_csv('gun_violence_people.csv')
df_gun.to_csv('gun_violence_gun.csv')