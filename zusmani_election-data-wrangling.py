import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import fuzzywuzzy
from fuzzywuzzy import process
import chardet

from subprocess import check_output
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory


#print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

NA2 = pd.read_csv("../input/National Assembly 2002 - Updated.csv", encoding = "ISO-8859-1")
NA8 = pd.read_csv("../input/National Assembly 2008.csv", encoding = "ISO-8859-1")
NA13 = pd.read_csv("../input/National Assembly 2013.csv", encoding = "ISO-8859-1")
print("Data Dimensions are: ", NA2.shape)
print("Data Dimensions are: ", NA8.shape)
print("Data Dimensions are: ", NA13.shape)
print("NA 2002.csv")
NA2.info()
print("\nNA 2008.csv")
NA8.info()
print("\nNA 2013.csv")
NA13.info()
print(NA2.head())
print(NA8.head())
print(NA13.head())
print(NA8.columns, "\n>>\n", NA13.columns)

NA8.rename(columns={'Unnamed: 0':'District'}, inplace=True)
NA13.rename(columns={'Unnamed: 0':'District'}, inplace=True)
print("NA 8: ", NA8.columns, "\nNA 13: ", NA13.columns)
#NA13 = NA13.drop('Unnamed: 11', axis=1)
NA8.District = NA8.Seat#.str.split("-", expand=True)[0]
#Add District column
#NA8['District'] = NA8['Seat']
NA8['District'] = NA8['District'].str.replace("."," ") # to deal with D.I. Khan
# remove all those substring with () 
NA8['District'] = NA8['District'].str.replace(r"\(.*\)","")
# remove numeric
NA8['District']  = NA8['District'] .str.replace('[^a-zA-Z -]', '')
#NA8['District'] = NA8['District'].str.replace(r"Cum.*","")
#NA8['District'] = NA8['District'].str.replace(r"cum.*","")
#na18['District'] = na18['District'].str.replace(r"KUM.*","")
# to convert Tribal Area III - Mohman into Tribal Area III
NA8['District'] = NA8['District'].str.replace(r"-.*","")
NA8['District']  = NA8['District'] .str.replace(r" (XX|IX|X?I{0,3})(IX|IV|V?I{0,3})$", '')
NA8['District']  = NA8['District'] .str.replace(r" (XX|IX|X?I{0,3})(IX|IV|V?I{0,3})$", '')
NA8['District'].unique()
NA13.District = NA13.Seat #.str.split("-", expand=True)[0]
#Add District column
#NA13['District'] = NA13['Seat']
NA13['District'] = NA8['District'].str.replace("."," ") # to deal with D.I. Khan
# remove all those substring with () 
NA13['District'] = NA13['District'].str.replace(r"\(.*\)","")
# remove numeric
NA13['District']  = NA13['District'] .str.replace('[^a-zA-Z -]', '')
NA13['District'] = NA13['District'].str.replace(r"Cum.*","")
#na18['District'] = na18['Distirct'].str.replace(r"KUM.*","")
# to convert Tribal Area III - Mohman into Tribal Area III
NA13['District'] = NA13['District'].str.replace(r"-.*","")
NA13['District']  = NA13['District'] .str.replace(r" (XX|IX|X?I{0,3})(IX|IV|V?I{0,3})$", '')
NA13['District']  = NA13['District'] .str.replace(r" (XX|IX|X?I{0,3})(IX|IV|V?I{0,3})$", '')
NA13['District'].unique()
NA13.head()
NA8['Turnout'] = NA8['Turnout'].str.rstrip('%').str.rstrip(' ')
NA13['Turnout'] = NA13['Turnout'].str.rstrip('%').str.rstrip(' ')
NA8['Turnout'] = pd.to_numeric(NA8['Turnout'], errors='coerce')
NA13['Turnout'] = pd.to_numeric(NA13['Turnout'], errors='coerce')
NA2['Year'] = "2002"
NA8['Year'] = "2008"
NA13['Year'] = "2013"
print(NA2.head(), "\n", NA8.head(), "\n", NA13.head())
print("NA2", NA2.isnull().any(), "\nNA8: ", NA8.isnull().any(), "\nNA13:", NA13.isnull().any())
print("\n NA2", NA2.columns, "\n NA8", NA8.columns, "\n NA13", NA13.columns)
NA2.rename(columns={'Constituency_title':'ConstituencyTitle', 'Candidate_Name':'CandidateName', 'Total_Valid_Votes':'TotalValidVotes', 'Total_Rejected_Votes':'TotalRejectedVotes', 'Total_Votes':'TotalVotes', 'Total_Registered_Voters':'TotalRegisteredVoters', }, inplace=True)
NA2.columns
df = pd.concat([NA2, NA8, NA13])
df.shape
df.head()
df.isnull().any()
# get all the unique values in the 'District' column
#df['District'] = df['District'].astype(str)
dist = df['District'].unique()
#dist.sort()
dist
# convert to lower case
df['District'] = df['District'].str.lower()
# remove trailing white spaces
df['District'] = df['District'].str.strip()
dist = df['District'].unique()
#dist.sort()
dist
# get the top 10 closest matches to "charsadda"
matches = fuzzywuzzy.process.extract("charsadda", dist, limit=10, scorer=fuzzywuzzy.fuzz.token_sort_ratio)

# take a look at them
matches
# function to replace rows in the provided column of the provided dataframe
# that match the provided string above the provided ratio with the provided string
def replace_matches_in_column(df, column, string_to_match, min_ratio = 90):
    # get a list of unique strings
    strings = df[column].unique()
    
    # get the top 10 closest matches to our input string
    matches = fuzzywuzzy.process.extract(string_to_match, strings, 
                                         limit=10, scorer=fuzzywuzzy.fuzz.token_sort_ratio)

    # only get matches with a ratio > 90
    close_matches = [matches[0] for matches in matches if matches[1] >= min_ratio]

    # get the rows of all the close matches in our dataframe
    rows_with_matches = df[column].isin(close_matches)

    # replace all rows with close matches with the input matches 
    df.loc[rows_with_matches, column] = string_to_match
    
# use the function we just wrote to replace close matches to "charsadda" 
replace_matches_in_column(df=df, column='District', string_to_match="charsadda")
dist = df['District'].unique()
#dist.sort()
dist
replace_matches_in_column(df=df, column='District', string_to_match="nowshera")
replace_matches_in_column(df=df, column='District', string_to_match="rawalpindi")
replace_matches_in_column(df=df, column='District', string_to_match="sheikhupura")
replace_matches_in_column(df=df, column='District', string_to_match="shikarpur")
replace_matches_in_column(df=df, column='District', string_to_match="nankana sahib")
del dist

pty = df['Party'].unique()
pty.sort()
pty
df['Party'] = df['Party'].replace(['MUTTHIDA\xa0MAJLIS-E-AMAL\xa0PAKISTAN'], 'Muttahidda Majlis-e-Amal Pakistan')
df['Party'] = df['Party'].replace(['Pakistan Muslim League'], 'Pakistan Muslim League (QA)')
#converting text to lower case & removing white spaces
df['Party'] = df['Party'].str.lower()
df['Party'] = df['Party'].str.strip()
# As I coded this earlier, I wouldn't change it due to lower case letters. 
replace_matches_in_column(df=df, column='Party', string_to_match="Balochistan National Movement")
replace_matches_in_column(df=df, column='Party', string_to_match="Independent")
replace_matches_in_column(df=df, column='Party', string_to_match="Istiqlal Party")
replace_matches_in_column(df=df, column='Party', string_to_match="Jamote Qaumi Movement")
replace_matches_in_column(df=df, column='Party', string_to_match="Labour Party Pakistan")
replace_matches_in_column(df=df, column='Party', string_to_match="Mohib-e-Wattan Nowjawan Inqilabion Ki Anjuman (MNAKA)")
replace_matches_in_column(df=df, column='Party', string_to_match="Muttahida Qaumi Movement") # Muttahida Qaumi Movement Pakistan
replace_matches_in_column(df=df, column='Party', string_to_match="Muttahidda Majlis-e-Amal") # Muttahidda Majlis-e-Amal Pakistan
replace_matches_in_column(df=df, column='Party', string_to_match="National Peoples Party")
replace_matches_in_column(df=df, column='Party', string_to_match="Nizam-e-Mustafa Party")
replace_matches_in_column(df=df, column='Party', string_to_match="Pak Muslim Alliance")
replace_matches_in_column(df=df, column='Party', string_to_match="Pakistan Awami Party")
replace_matches_in_column(df=df, column='Party', string_to_match="Pakistan Democratic Party")
# After analyzing each of the below strings.
replace_matches_in_column(df=df, column='Party', string_to_match="Pakistan Muslim League (QA)", min_ratio =97)
replace_matches_in_column(df=df, column='Party', string_to_match="Pakistan Muslim League (N)", min_ratio =97)
replace_matches_in_column(df=df, column='Party', string_to_match="Pakistan Muslim League (J)", min_ratio =97)
replace_matches_in_column(df=df, column='Party', string_to_match="Pakistan Muslim League (F)", min_ratio =97)
replace_matches_in_column(df=df, column='Party', string_to_match="Pakistan Peoples Party Parliamentarians", min_ratio =97)
replace_matches_in_column(df=df, column='Party', string_to_match="Pakistan Peoples Party(Shaheed Bhutto)", min_ratio =95)
replace_matches_in_column(df=df, column='Party', string_to_match="Pakistan Peoples Party(Sherpao)", min_ratio =97)
replace_matches_in_column(df=df, column='Party', string_to_match="Pakistan Tehreek-e-Insaf", min_ratio =95)
replace_matches_in_column(df=df, column='Party', string_to_match="Saraiki Sooba Movement Pakistan", min_ratio =95)

#fuzzywuzzy.process.extract("Pakistan Muslim League (QA)", pty, limit=10, scorer=fuzzywuzzy.fuzz.token_sort_ratio) # acceptance value >97
#fuzzywuzzy.process.extract("Pakistan Muslim League (N)", pty, limit=10, scorer=fuzzywuzzy.fuzz.token_sort_ratio) # acceptance value >97
#fuzzywuzzy.process.extract("Pakistan Muslim League (J)", pty, limit=10, scorer=fuzzywuzzy.fuzz.token_sort_ratio) # acceptance value >97
#fuzzywuzzy.process.extract("Pakistan Muslim League (F)", pty, limit=10, scorer=fuzzywuzzy.fuzz.token_sort_ratio) # acceptance value >97
#fuzzywuzzy.process.extract("Pakistan Peoples Party Parliamentarians", pty, limit=10, scorer=fuzzywuzzy.fuzz.token_sort_ratio) # acceptance value >97
#fuzzywuzzy.process.extract("Pakistan Peoples Party(Shaheed Bhutto)", pty, limit=10, scorer=fuzzywuzzy.fuzz.token_sort_ratio) # acceptance value >95
#fuzzywuzzy.process.extract("Pakistan Peoples Party(Sherpao)", pty, limit=10, scorer=fuzzywuzzy.fuzz.token_sort_ratio) # acceptance value >97
#fuzzywuzzy.process.extract("Pakistan Tehreek-e-Insaf", pty, limit=10, scorer=fuzzywuzzy.fuzz.token_sort_ratio) # acceptance value >95
#fuzzywuzzy.process.extract("Saraiki Sooba Movement Pakistan", pty, limit=10, scorer=fuzzywuzzy.fuzz.token_sort_ratio) # acceptance value >95

df['Party'] = df['Party'].str.lower()
# few fixes taken from https://www.kaggle.com/usman786/exploratory-data-analysis-for-interesting-insights/notebook
df['Party'].replace(['muttahida qaumi movement pakistan'], 'muttahida qaumi movement', inplace = True)
df['Party'].replace(['indeindependentdente','independent (retired)','indepndent'], 'independent',inplace = True)
df['Party'].replace(['indeindependentdente','independent (retired)','indepndent'], 'independent',inplace = True)
df['Party'].replace(['muttahidda majlis-e-amal pakistan','mutthida\xa0majlis-e-amal\xa0pakistan'
                     ,'mutthidaï¿½majlis-e-amalï¿½pakistan'] 
                     ,'muttahidda majlis-e-amal' ,inplace = True)
df['Party'].replace(['nazim-e-mistafa'], 'nizam-e-mustafa party' ,inplace = True)
df['Party'].replace(['pakistan muslim league (qa)'], 'pakistan muslim league (q)' ,inplace = True)
df['Party'].replace(['pakistan muslim league council'], 'pakistan muslim league (c)' ,inplace = True)
df['Party'].replace(['pakistan muslim league \x93h\x94 haqiqi'], 'pakistan muslim league haqiqi' ,inplace = True)
df['Party'].replace(['pakistan muslim league(z)'], 'pakistan muslim league (z)' ,inplace = True)
df['Party'].replace(['pakistan peoples party(shaheed bhutto)'], 'pakistan peoples party (shaheed bhutto)' ,inplace = True)
df['Party'].replace(['pakistan peoples party parliamentarians'], 'pakistan peoples party parliamentarians' ,inplace = True)
df['Party'].replace(['pakistan sariaki party'], 'Pakistan Siraiki Party (T)' ,inplace = True)
df['Party'].replace(['pasban'], 'pasban pakistan' ,inplace = True)
df['Party'].replace(['qaumi watan party (sherpao)'], 'qaumi watan party' ,inplace = True)
df['Party'].replace(['tehreek-e-suba hazara'], 'tehreek-e-suba hazara pakistan' ,inplace = True)
#...
df['Party'].replace(['pashtoonkhwa milli awami party'], 'pakhtoonkhwa milli Awami party' ,inplace = True)
df['Party'].replace(['pakistan amn party'], 'pakistan aman party' ,inplace = True)
df['Party'].replace(['pakistan awami inqelabi'], 'Pakistan Awami Inqelabi League' ,inplace = True)
df['Party'].replace(['pakistan freedom party'], 'pakistan freedom movement' ,inplace = True)
df['Party'].replace(['pakistan insani haqook party (pakistan human rights party)'], 'pakistan human rights party' ,inplace = True)
df['Party'].replace(['awami justice party'], 'awami justice party pakistan' ,inplace = True)
df['Party'].replace(['indeindependentdent'], 'independent' ,inplace = True)
df['Party'].replace(['jamiat ulama-e-pakistan  (noorani)'], 'jamiat ulama-e-pakistan (noorani)' ,inplace = True)
df['Party'].replace(['jumiat ulma-e-islam(nazryati)'], 'jamiat ulma-e-islam nazryati pakistan' ,inplace = True)
df['Party'].replace(['majlis-e-wahdat-e-muslimeen pakistan'], 'Majlis Wahdat-e-Muslimeen Pakistan' ,inplace = True)
df['Party'].replace(['markazi jamat-al-hadais'], 'Markazi Jamiat Ahl-e-Hadith' ,inplace = True)
df['Party'].replace(['mohib-e-wattan nowjawan inqilabion ki anjuman (mnaka)'], 'Muhib-e-Watan Noujawan Anqlabion Ki Anjuman (MNAKA)' ,inplace = True)

pty = df['Party'].unique()
pty.sort()
pty
#del pty
#convert textual content to lower case & remove trailing white spaces
df['CandidateName'] = df['CandidateName'].str.lower()
df['CandidateName'] = df['CandidateName'].str.strip()
df['CandidateName'].head(10)
# remove mr at the beginning of names.
df['CandidateName'] = df.loc[:, 'CandidateName'].replace(regex=True, to_replace="mr ", value="")
df['CandidateName'] = df.loc[:, 'CandidateName'].replace(regex=True, to_replace="mrs ", value="")
df['CandidateName'] = df.loc[:, 'CandidateName'].replace(regex=True, to_replace="miss ", value="")
#df['CandidateName'] = df.loc[:, 'CandidateName'].replace(regex=True, to_replace="mis ", value="")
df['CandidateName'].head(10)
cn = df['CandidateName'].unique()
cn.sort()
print("cn size: ", cn.shape, "\nValues: ", cn) 
df['CandidateName']
fuzzywuzzy.process.extract("zumurad khan", cn, limit=10, scorer=fuzzywuzzy.fuzz.token_sort_ratio) # acceptance value >90
fuzzywuzzy.process.extract("zobaida jalal", cn, limit=10, scorer=fuzzywuzzy.fuzz.token_sort_ratio) # acceptance value >79
#fuzzywuzzy.process.extract("barkat ali", cn, limit=10, scorer=fuzzywuzzy.fuzz.token_sort_ratio) # acceptance value >90
#fuzzywuzzy.process.extract("sher muhammad baloch", cn, limit=10, scorer=fuzzywuzzy.fuzz.token_sort_ratio) # acceptance value >90
#fuzzywuzzy.process.extract("gulab baloch", cn, limit=10, scorer=fuzzywuzzy.fuzz.token_sort_ratio) # acceptance value >90
#fuzzywuzzy.process.extract("babu gulab", cn, limit=10, scorer=fuzzywuzzy.fuzz.token_sort_ratio) # acceptance value >90
replace_matches_in_column(df=df, column='CandidateName', string_to_match="zumurad khan", min_ratio=92)
replace_matches_in_column(df=df, column='CandidateName', string_to_match="zobaida jalal", min_ratio=80)
replace_matches_in_column(df=df, column='CandidateName', string_to_match="barkat ali", min_ratio=90)
replace_matches_in_column(df=df, column='CandidateName', string_to_match="muhammad yasin baloch", min_ratio=90)

for candi in df['CandidateName'].unique(): # 7000
    replace_matches_in_column(df=df, column='CandidateName', string_to_match=candi, min_ratio=90)

# let us know the loop is completed
print("All done!")
#del NA2, NA8, NA13
df.to_csv('NA2002-18.csv', index=None) 
cc = pd.read_csv("../input/National Assembly Candidates List - 2018 Updated.csv", encoding = "ISO-8859-1")
na18 = pd.read_csv("../input/2013-2018 Seat Changes in NA.csv", encoding = "ISO-8859-1") 
pp = pd.read_csv("../input/Political Parties in 2018 Elections - Updated.csv", encoding = "ISO-8859-1")
print(cc.shape, na18.shape, pp.shape)
print(cc.columns, na18.columns)
cc['NA#'] = 'NA-' + cc['NA#'].astype(str)
print(cc['NA#'].unique().shape) # 272
print(na18['2018 Seat Number'].unique().shape) # 273
na18.rename(columns={'2018 Seat Number':'NA#'}, inplace=True)
na18.rename(columns={'Seat Name':'Seat'}, inplace=True)
na18[na18['NA#'] == "Old Constituency Changed Considerably"]
na18 = na18[na18['NA#'] != "Old Constituency Changed Considerably"]
na18['NA#'] = na18.loc[:, 'NA#'].replace(regex=True, to_replace="NA-", value="")
na18['NA#'] = pd.to_numeric(na18['NA#'])
na18['NA#'] = na18['NA#'].astype(np.int64)
na18['NA#'] = 'NA-' + na18['NA#'].astype(str)
#na18['NA#'] = na18.loc[:, 'NA#'].replace(regex=True, to_replace=".0", value="")
na18['NA#'].head()
#Add District column & its cleani
na18['Distirct'] = na18['Seat']
# remove all those substring with () 
na18['Distirct'] = na18['Distirct'].str.replace(r"\(.*\)","")
# remove numeric
na18['Distirct']  = na18['Distirct'].str.replace('[^a-zA-Z -]', '')
na18['Distirct'] = na18['Distirct'].str.replace(r"Cum.*","")
#na18['Distirct'] = na18['Distirct'].str.replace(r"KUM.*","")
# to convert Tribal Area III - Mohman into Tribal Area III
na18['Distirct'] = na18['Distirct'].str.replace(r"-.*","")
na18['Distirct']  = na18['Distirct'] .str.replace(r" (XX|IX|X?I{0,3})(IX|IV|V?I{0,3})$", '')
na18['Distirct']  = na18['Distirct'] .str.replace(r" (XX|IX|X?I{0,3})(IX|IV|V?I{0,3})$", '')
na18['Distirct'].unique()
cc = cc.join(na18.set_index('NA#'), on='NA#')
cc.info()
print(pp.shape)
pp['Name of Political Party'].unique()
pp.rename(columns={'Acronym':'PartyAcro'}, inplace=True)
cc.rename(columns={'Party':'PartyAcro'}, inplace=True)
pp.rename(columns={'Name of Political Party':'Party'}, inplace=True)
# Clean Candidate file
pp['Party'].replace(['pakistan reh-e- haq party'], 'Pakistan Rah-e- Haq Party' ,inplace = True)
pp['Party'].replace(['Pakistan Muslim League SHER-E-BANGAL A.K. Fazal-Ul-Haque'], 'pakistan muslim league (sher-e-bangal)' ,inplace = True)
pp['Party'].replace(['Pakistan Muslim League (Zia-ul-Haq Shaheed)'], 'pakistan muslim league (z)' ,inplace = True)
pp['Party'].replace(['Pakistan Muslim League (Junejo)'], 'pakistan muslim league (j)' ,inplace = True)
pp['Party'].replace(['Pakistan Muslim League (Functional)'], 'pakistan muslim league (f)' ,inplace = True)
pp['Party'].replace(['Pakistan Muslim League (Council)'], 'pakistan muslim league (c)' ,inplace = True)
pp['Party'].replace(['Pakistan Muslim League-Nawaz'], 'pakistan muslim league (n)' ,inplace = True)
pp['Party'].replace(['Pakistan Justice & Democratic Party'], 'Pakistan Justice and Democratic Party' ,inplace = True)
pp['Party'].replace(['Pakistan Kissan Ittehad (Ch. Anwar)'], 'Pakistan Kissan Ittehad' ,inplace = True)
pp['Party'].replace(['Jamiat Ulma-e-Islam Nazryati Pakistan'], 'Jamiat Ulma-e-Islam Nazaryati Pakistan' ,inplace = True)
pp['Party'].replace(['Jamiat Ulma-e-Islam Nazryati Pakistan'], 'Jamiat Ulma-e-Islam Nazaryati Pakistan' ,inplace = True)
pp['Party'].replace(['Jamiat Ulama-e-Islam(F)'], 'Jamiat Ulama-e-Islam (F)' ,inplace = True)
pp['Party'].replace(['Jamiat Ulama-e-Islam(S)'], 'Jamiat Ulama-e-Islam (S)' ,inplace = True)
pp['Party'].replace(['Mohajir Qaumi Movement (Pakistan)'], 'Mohajir Qaumi Movement pakistan' ,inplace = True)
pp['Party'].replace(['Mutahida Majlis-e-Amal'], 'Muttahida Majlis-e-Amal' ,inplace = True)
pp['Party'].replace(['Muttahidda Qaumi Movement Pakistan'], 'Muttahida Qaumi Movement Pakistan' ,inplace = True)

# Remove duplicaties
pp.drop_duplicates(subset=['PartyAcro'], keep="first", inplace=True)
pp.info()
pp
cc[cc['PartyAcro']=='PTI'].head()
#pp[pp['PartyAcro']=='PTEI']
#del cnd
cnd = cc.join(pp.set_index('PartyAcro'), on='PartyAcro')
cnd.info()
cnd.head()
cnd[cnd['PartyAcro']=="PTI"].head()
#remove non-aplhabetic characters from Name
cnd['Name'] = cnd['Name'].str.replace('[^a-zA-Z ]', '')
cnd['Name'] = cnd['Name'].str.lower()
cnd['Name'] = cnd['Name'].str.strip()

cnd['Party'] = cnd['Party'].str.lower()
cnd['Party'] = cnd['Party'].str.strip()

cnd[cnd['PartyAcro']=="PTI"].head()
print(df.columns, cnd.columns)
df.info()
cnd.info()
cnd.rename(columns={'NA#':'ConstituencyTitle'}, inplace=True)
cnd.rename(columns={'Name of Political Party':'Party'}, inplace=True)
cnd.rename(columns={'Name':'CandidateName'}, inplace=True)
cnd.to_csv('Canditates2018.csv', index=None) 
pp.to_csv('Parties_cleand.csv', index=None)

# Reading 2018 Results Data
NA18 = pd.read_csv("../input/NA-Results2018 Ver 2.csv", encoding = "ISO-8859-1")
print("Data Dimensions of NA18 are: ", NA18.shape)

print("\nNA 2018.csv")
NA18.info()

NA18 = NA18.drop('Unnamed: 0', axis=1)
NA18.rename(columns={'district':'District'}, inplace=True)

NA18.District = NA18.Seat
NA18['District'] = NA18['District'].str.replace("."," ") # to deal with D.I. Khan
NA18['District'] = NA18['District'].str.replace(r"\(.*\)","")
NA18['District']  = NA18['District'] .str.replace('[^a-zA-Z -]', '')
NA18['District'] = NA18['District'].str.replace(r"-.*","")
NA18['District']  = NA18['District'] .str.replace(r" (XX|IX|X?I{0,3})(IX|IV|V?I{0,3})$", '')
NA18['District']  = NA18['District'] .str.replace(r" (XX|IX|X?I{0,3})(IX|IV|V?I{0,3})$", '')
NA18['District'].unique()

NA18['Turnout'] = NA18['Turnout'].str.rstrip('%').str.rstrip(' ')
NA18['Turnout'] = pd.to_numeric(NA18['Turnout'], errors='coerce')
NA18.rename(columns={'Constituency_Title':'ConstituencyTitle', 'Candidate_Name':'CandidateName', 'Total_Valid_Votes':'TotalValidVotes', 'Total_Rejected_Votes':'TotalRejectedVotes', 'Total_Votes':'TotalVotes', 'Total_Registered_Voters':'TotalRegisteredVoters', 'Part':'Party' }, inplace=True)
NA18.columns
# convert to lower case
NA18['District'] = NA18['District'].str.lower()
# remove trailing white spaces
NA18['District'] = NA18['District'].str.strip()

# convert to lower case
NA18['CandidateName'] = NA18['CandidateName'].str.lower()
# remove trailing white spaces
NA18['CandidateName'] = NA18['CandidateName'].str.strip()

# convert to lower case
NA18['Party'] = NA18['Party'].str.lower()
# remove trailing white spaces
NA18['Party'] = NA18['Party'].str.strip()
NA18.head()
NA18.to_csv('NA2018_Clean.csv', index=None)