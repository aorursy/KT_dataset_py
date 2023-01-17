#import modules

import numpy as np

import pandas as pd 

from requests import get

from bs4 import BeautifulSoup as soup

import re

from datetime import date

from tqdm.notebook import tqdm

import json
#retrieve the main page (if it won't work, try using selenium)

url = "https://www.soccerstats.com"

page = soup(get(url).content, "html.parser")



#retrieve each league

leagues = page.find_all("td", {"align": "center", "valign": "middle", "style":"background-color:#e1e1e1;line-height:1.2em;"})



#retrieve titles

title_regex = re.compile('<span title="(.*?)">')

titles = []



#retrieve abbreviations

abbrs = []



#retrieve links

href_regex = re.compile('href="(.*?)"')

hrefs = []



#fetch data 

for league in leagues:

    titles.append(re.findall(title_regex, str(league))[0])

    hrefs.append(re.findall(href_regex, str(league))[0])

    abbrs.append(league.text)

    

#absolute links to the urls

league_urls = ["/".join([url, href]) for href in hrefs]



#create list of dicts

leagues_info = []

features = ["title", "abbreviation"]

feature_lists = [titles, abbrs]

for index in range(len(titles)):

    new_dict = {}

    for feature_index, feature in enumerate(features):

        new_dict.update({feature: feature_lists[feature_index][index]})

    leagues_info.append(new_dict)

    

#shorthand for loop

inline_for = lambda x: tqdm(range(len(x)))



#get raw content

page_contents = [get(league_urls[x]).content for x in inline_for(league_urls)]



#convert to pd.DataFrame objects

pages = [pd.read_html(page_contents[x]) for x in inline_for(page_contents)]



#the desired columns

cols = ["GP", "W", "D", "L", "GF", "GA", "GD", "Pts", "Form", "PPG", "last 8", "CS", "FTS"]





tables = []

for table_list in pages:

    

    #selects the most similar result 

    likelihoods = []

    for table in table_list:

        likelihood = pd.Series(table.columns).isin(cols).sum()

        likelihoods.append(likelihood)

    table = table_list[np.argmax(likelihoods)]

    

    #a few changes to be made

    try:

        table.rename(columns = {"Unnamed: 1": "Team"}, inplace = True)

    except:

        pass

    try:

        table.drop("Unnamed: 0", 1, inplace = True)

    except:

        pass

    try:

        table["Form"] = list(map(lambda x: re.findall("\d-\d", x), table["Form"].values))

    except:

        pass

    tables.append(table)



#store data as dicts 

scoreboard_dicts = [x.to_dict() for x in tables]



#today as string

today = str(date.today())    



#add items to the already existing json-like list

for i, dix in enumerate(leagues_info):

    if("daily_results" in dix.keys()):

        dix["daily_results"].update({today: scoreboard_dicts[i]})

    else:

        dix.update({"daily_results": {today: scoreboard_dicts[i]}})



#load past data

with open("/kaggle/input/most-popular-soccer-leagues/leagues.json") as read_file:

    data = json.load(read_file)

        

#analyze differences 

def diff_df(data, x, dat):

    df1 = pd.DataFrame(data[x]["daily_results"][dat]).set_index("Team")

    df1 = df1.reindex(index = sorted(df1.index)).select_dtypes(exclude = ["object"]).reset_index(drop = True)

    return df1.values



#implement new changes

def new_scores(data, leagues_info):

    for i in range(len(data)):

        if((pd.Series([x["title"] for x in data]) == pd.Series([x["title"] for x in leagues_info])).sum() != 30): break

        try:

            dat = sorted(list(data[i]["daily_results"].keys()))[-1]

            dif = np.sum(diff_df(data, i, dat) - diff_df(leagues_info, i, today))

            if(dif > 0 or dif < 0):

                data[i]["daily_results"].update({today: leagues_info[i]["daily_results"][today]})

        except: pass

    return data



#run the function

data = new_scores(data, leagues_info)



#save json file

with open("leagues.json", "w") as write_file:

    write_file.write(json.dumps(data))