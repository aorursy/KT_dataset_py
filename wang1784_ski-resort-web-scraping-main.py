import pandas as pd

from ski_scrap_fun import ski_resort as ski



#access main page

url = 'https://ski-resort-stats.com/ski-resorts-in-america/'

skiweb = ski()

soup = skiweb.access(url)



#get website links, the main table and snow info

df_main = skiweb.GetMainTable(soup) #get the main table

info_dict = dict()

for link in df_main['link']: # for each link in the main table

    resortsoup = skiweb.access(link) # access current link

    snow = skiweb.GetSnowData(resortsoup, link, info_dict) # scrap for snow info from current link

    if snow != None: # if snow info is available

        info_dict = snow # update

    else:

        print(link)

        break



    resort = skiweb.GetResortInfo(resortsoup, link, info_dict) #scrap for resort info from current link

    if resort != None:

        info_dict = resort

    else:

        print(link)

        break



# convert snow info to dataframe

df_snow = pd.DataFrame(info_dict)



# save df to csv file

skiweb.SaveDF(df_main, 'ski_main')

skiweb.SaveDF(df_snow, 'ski_snow')