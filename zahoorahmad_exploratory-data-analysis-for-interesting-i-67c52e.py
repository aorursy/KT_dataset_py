# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import fuzzywuzzy
from fuzzywuzzy import process
import chardet

from subprocess import check_output
import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# this cell is zeeshan's script for intial cleaning
# Read data
NA2 = pd.read_csv("../input/National Assembly 2002 - Updated.csv", encoding = "ISO-8859-1")
NA8 = pd.read_csv("../input/National Assembly 2008.csv", encoding = "ISO-8859-1")
NA13 = pd.read_csv("../input/National Assembly 2013.csv", encoding = "ISO-8859-1")

# rename coloumns
NA8.rename(columns={'Unnamed: 0':'District'}, inplace=True)
NA13.rename(columns={'Unnamed: 0':'District'}, inplace=True)

#Get districts separated out of seats
NA8.District = NA8.Seat.str.split("-", expand=True)[0]
NA13.District = NA13.Seat.str.split("-", expand=True)[0]

#Change Data type of turnout
NA8['Turnout'] = NA8['Turnout'].str.rstrip('%').str.rstrip(' ')
NA13['Turnout'] = NA13['Turnout'].str.rstrip('%').str.rstrip(' ')
NA8['Turnout'] = pd.to_numeric(NA8['Turnout'], errors='coerce')
NA13['Turnout'] = pd.to_numeric(NA13['Turnout'], errors='coerce')

#Add Year Column
NA2['Year'] = "2002"
NA8['Year'] = "2008"
NA13['Year'] = "2013"

#Rename coloumns in NA2
NA2.rename(columns={'Constituency_title':'ConstituencyTitle', 'Candidate_Name':'CandidateName', 'Total_Valid_Votes':'TotalValidVotes',
                    'Total_Rejected_Votes':'TotalRejectedVotes', 'Total_Votes':'TotalVotes', 'Total_Registered_Voters':'TotalRegisteredVoters', }, inplace=True)

#Concat all results
df = pd.concat([NA2, NA8, NA13])

df['District'] = df['District'].str.lower()
# remove trailing white spaces
df['District'] = df['District'].str.strip()

#convert textual content to lower case
df['CandidateName'] = df['CandidateName'].str.lower()
df['Party'] = df['Party'].str.lower()
# remove trailing white spaces
df['CandidateName'] = df['CandidateName'].str.strip()
df['Party'] = df['Party'].str.strip()
# Let's write a function to filter parties which fuzzy wuzzy thinks are same but actually they are not
# Parties like Pakistan Muslim League(n),Pakistan Muslim League(qa),Pakistan Muslim League(q) and Pakistan Muslim League will have high similarity scores 
def filter_similarity_exceptions(party_to_match,party_list):
    #find sub party name
    sub_party = party_to_match[party_to_match.find("(")+1:party_to_match.find(")")].strip()
    
    # if party_to_match has no sub party, filter out parties with sub party from party_list
    if(len(party_to_match.split('(')) < 2):
        party_list = [x for x in party_list if len(x.split('(')) < 2]
    
    #make sure sub party names upto length 2 are same, because they can fall in specified similarity score threshold 
    if(len(sub_party) <= 2):
        party_list = [x for x in party_list if x[x.find("(")+1:x.find(")")].strip() == sub_party]
    
    return party_list
# modification of function written by zeeshan to filter similarity exceptions
def replace_matches_in_party(df,string_to_match, min_ratio = 90, column='Party'):
    # get a list of unique strings
    strings = df[column].unique()
    
    # get the top 10 closest matches to our input string
    matches = fuzzywuzzy.process.extract(string_to_match, strings, 
                                         limit=10, scorer=fuzzywuzzy.fuzz.token_sort_ratio)

    # only get matches with a ratio > 90
    close_matches = [matches[0] for matches in matches if matches[1] >= min_ratio]
    
    #filter similarity exceptions
    close_matches = filter_similarity_exceptions(string_to_match,close_matches)
    
    # get the rows of all the close matches in our dataframe
    rows_with_matches = df[column].isin(close_matches)

    # replace all rows with close matches with the input matches 
    df.loc[rows_with_matches, column] = string_to_match
# iterate unique parties and replace the same parties which are spelled differntly
for party in df['Party'].unique():
    replace_matches_in_party(df,party, min_ratio = 93, column='Party')
# have a look at the results after sorting
candidates = df.sort_values('Party')['Party'].unique()
print(candidates)
# Ah it's look like, if we decrease the threshold(min_score) further it may cause problems
# I know these parties have addition of words like Pakitan and Party but are similar, let's replace them
df['Party'].replace(['muttahida qaumi movement pakistan'], 'muttahida qaumi movement', inplace = True)
df['Party'].replace(['indeindependentdente','independent (retired)','indepndent'], 'independent',inplace = True)
df['Party'].replace(['indeindependentdente','independent (retired)','indepndent'], 'independent',inplace = True)
df['Party'].replace(['muttahidda majlis-e-amal pakistan','mutthida\xa0majlis-e-amal\xa0pakistan'
                     ,'mutthidaï¿½majlis-e-amalï¿½pakistan'] 
                     ,'muttahidda majlis-e-amal' ,inplace = True)
df['Party'].replace(['nazim-e-mistafa'], 'nizam-e-mustafa party' ,inplace = True)
# we are all set to meet winners
#reset the index 
df.reset_index(inplace = True)
#find candidates with max votes in a constituency and year, aka the winners
winning_candidates = df.loc[df.groupby(['Year','ConstituencyTitle'])['Votes'].idxmax()].sort_values('ConstituencyTitle')
winning_candidates.head()
# let's find year wise number of seats won by each party 
year_wise_party_results = winning_candidates.groupby(['Party','Year']).size().to_frame('count').sort_values('count')
year_wise_party_results.head()
# find number of times a candidate won from the same constituency
constituency_wise_candidate_wins = winning_candidates.groupby(['ConstituencyTitle','CandidateName']).size().to_frame('wins')
constituency_wise_candidate_wins.head()
# find candidates who won atleast twice from the same constituency
strong_candidates = constituency_wise_candidate_wins[constituency_wise_candidate_wins['wins'] >=2 ].sort_values('wins', ascending = False)
strong_candidates.head()
# find  of number of times a party won from the same constituency
constituency_wise_party_wins = winning_candidates.groupby(['ConstituencyTitle','Party']).size().to_frame('wins')
#find constittuencies where same party won in all three elections
confirmed_constituencies_by_party = constituency_wise_party_wins[constituency_wise_party_wins['wins'] == 3].sort_values('wins', ascending = False)
confirmed_constituencies_by_party.head()
#Swing constituencies are those where maximum wins by the same party are 1 i.e. where no party won twice
#First we will find constituency wise maximum number of wins by a party 
constituency_wise_max_party_wins = constituency_wise_party_wins.groupby(['ConstituencyTitle'])['wins'].max().to_frame('max_wins_by_any_party')
#filter out those constituencies where max wins by any party is 1
swing_constituencies = constituency_wise_max_party_wins[constituency_wise_max_party_wins['max_wins_by_any_party'] == 1]
swing_constituencies.head()
# Time to find Lotas, Ah ha (candidates who changed their parties while competing from the same constituency)
num_parties_by_candidate = df[['CandidateName','ConstituencyTitle','Party']].groupby(['CandidateName','ConstituencyTitle'])['Party'].nunique()\
.to_frame('count').sort_values('count', ascending = False)
#find candidates with party count greater than 1
lotas = num_parties_by_candidate[num_parties_by_candidate['count'] > 1]
#some lotas for you
lotas.head()
# first value in a constituency may have a null value, let's fill it with next value from same constituency and year
df['Turnout'] = df.groupby(['ConstituencyTitle','Year'])['Turnout'].fillna(method = 'bfill')
# find min, max , average turnout and valid votes constituency wise
aggregations_by_constituency = df.groupby('ConstituencyTitle').agg({'TotalValidVotes' : [np.max, np.min, np.average ],'Turnout' : [np.max, np.min, np.average]})
aggregations_by_constituency.head()
# now we will find total number of votes casted to all parties(year wise) and sort them to see popular vote winners
popular_vote_winners = df.groupby(['Party', 'Year'])['Votes'].sum().to_frame('total_votes').sort_values('total_votes', ascending = False)
popular_vote_winners.head()
