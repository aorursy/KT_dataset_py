#imports

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
#load data

df = pd.read_csv("../input/vgsales.csv", encoding="ISO-8859-1")

pd.options.display.float_format = '{:,.0f}'.format

df.info()
#Removing duplicate lists that has the same games with different consoles.

df_drop = df.drop_duplicates(["Name"])

#Summing the sales of the same game released in different consoles

tmp_df = df[["Global_Sales","EU_Sales","JP_Sales","NA_Sales","Other_Sales"]].groupby(df["Name"]).sum()

tmp_df = tmp_df.sort_values("Global_Sales", ascending=False)

tmp_df["Name"]=tmp_df.index

tmp_df.index = range(0, 11493)

def get_year(x):

    return df_drop[df_drop["Name"] == x]["Year"].values[0].astype(int)  

tmp_df["Year"] = tmp_df["Name"].apply(get_year)
#Dragon Quest

dq = tmp_df[tmp_df["Name"].str.contains("Dragon Quest")]

plt.figure(figsize=(8,6))

#Plot by individual sales

dq_plot1 = dq.sort_values(by="Year").plot.bar(y="Global_Sales", x="Name")

#Plot by year sales

dq_gb = dq.groupby(dq["Year"]).sum()

dq_plot2 = dq_gb.plot.bar(y="Global_Sales")

#Plot by main series sales

dq_list = ["Dragon Quest I & II", "Dragon Quest III: Soshite Densetsu e...",

           "Dragon Quest IV: Michibikareshi Monotachi", "Dragon Quest V: Tenkuu no Hanayome",

           "Dragon Quest VI: Maboroshi no Daichi","Dragon Quest VII: Warriors of Eden",

           "Dragon Quest VIII: Journey of the Cursed King","Dragon Quest IX: Sentinels of the Starry Skies"]

dq_main = dq[dq["Name"].isin(dq_list)]

dq_plot3 = dq_main.sort_values(by="Year").plot.bar(y="Global_Sales", x="Name")

#Seems that the order is not accurate, considering the "Dragon Quest I & II" is earlier than "Dragon Quest V"

#This might be due to the different release dates of American version/Japanese version

#Manually controlling the order (according to the Wikipedia page of Dragon Quest)

dq_list2 = dict({"Dragon Quest I & II":0, "Dragon Quest III: Soshite Densetsu e...":1,

           "Dragon Quest IV: Michibikareshi Monotachi":2, "Dragon Quest V: Tenkuu no Hanayome":3,

           "Dragon Quest VI: Maboroshi no Daichi":4,"Dragon Quest VII: Warriors of Eden":5,

           "Dragon Quest VIII: Journey of the Cursed King":6,"Dragon Quest IX: Sentinels of the Starry Skies":7})

dq_main2 = dq[dq["Name"].isin(dq_list2)]

dq_main2["key"] = ""

for index, row in dq_main2.iterrows():

    dq_main2.set_value(index, 'key', dq_list2[dq_main2.loc[index]['Name']]) 

dq_plot4 = dq_main2.sort_values(by="key").plot.bar(y="Global_Sales", x="Name")
#Final Fantasy

ff = tmp_df[tmp_df["Name"].str.contains("Final Fantasy")]

plt.figure(figsize=(8,6))

#Plot by individual sales

ff_plot1 = ff.sort_values(by="Year").plot.bar(y="Global_Sales", x="Name")

#Plot by year sales

ff_gb = ff.groupby(ff["Year"]).sum()

ff_plot2 = ff_gb.plot.bar(y="Global_Sales")

#Plot by main series sales

ff_list = ["Final Fantasy", "Final Fantasy II",

                "Final Fantasy III", "Final Fantasy IV",

                "Final Fantasy V", "Final Fantasy VII","Final Fantasy VIII",

                "Final Fantasy IX","Final Fantasy X",

                "Final Fantasy X-2","Final Fantasy XII",

                "Final Fantasy XIII","Final Fantasy XIII-2"]

ff_main = ff[ff["Name"].isin(ff_list)]

ff_plot3 = ff_main.sort_values(by="Year").plot.bar(y="Global_Sales", x="Name")

#Seems that the order is not accurate

#This might be due to the different release dates of American version/Japanese version

#Manually controlling the order (according to the Wikipedia page of Final Fantasy)

ff_list2 = dict({"Final Fantasy":0, "Final Fantasy II":1,

                "Final Fantasy III":2, "Final Fantasy IV":3,

                "Final Fantasy V":4, "Final Fantasy VII":6,"Final Fantasy VIII":7,

                "Final Fantasy IX":8,"Final Fantasy X":9,

                "Final Fantasy X-2":10,"Final Fantasy XII":11,

                "Final Fantasy XIII":12,"Final Fantasy XIII-2":13})

ff_main2 = ff[ff["Name"].isin(ff_list2)]

ff_main2["key"] = ""

for index, row in ff_main2.iterrows():

    ff_main2.set_value(index, 'key', ff_list2[ff_main2.loc[index]['Name']]) 

ff_plot4 = ff_main2.sort_values(by="key").plot.bar(y="Global_Sales", x="Name")
#Pokemon

pk = tmp_df[tmp_df["Name"].str.contains("Pokemon")]

plt.figure(figsize=(8,6))

#Plot by individual sales

pk_plot1 = pk.sort_values(by="Year").plot.bar(y="Global_Sales", x="Name")

#Plot by year sales

pk_gb = pk.groupby(pk["Year"]).sum()

pk_plot2 = pk_gb.plot.bar(y="Global_Sales")

#Plot by main series sales

pk_list = ["Pokemon Red/Pokemon Blue",

                "Pokemon Yellow: Special Pikachu Edition",

                "Pokemon Gold/Pokemon Silver",

                "Pokemon Crystal Version",

                "Pokemon Ruby/Pokemon Sapphire",

                "Pokemon FireRed/Pokemon LeafGreen",

                "Pokemon Emerald Version",

                "Pokemon Diamond/Pokemon Pearl",

                "Pokemon Platinum Version",

                "Pokemon HeartGold/Pokemon SoulSilver",

                "Pokemon Black/Pokemon White",

                "Pokemon Black 2/Pokemon White 2",

                "Pokemon X/Pokemon Y",

                "Pokemon Omega Ruby/Pokemon Alpha Sapphire"]

pk_main = pk[pk["Name"].isin(pk_list)]

pk_plot3 = pk_main.sort_values(by="Year").plot.bar(y="Global_Sales", x="Name")

#Seems that the order is not accurate

#This might be due to the different release dates of American version/Japanese version

#Manually controlling the order (according to the Wikipedia page of Final Fantasy)

pk_list2 = dict({"Pokemon Red/Pokemon Blue":0,

                "Pokemon Yellow: Special Pikachu Edition":1,

                "Pokemon Gold/Pokemon Silver":2,

                "Pokemon Crystal Version":3,

                "Pokemon Ruby/Pokemon Sapphire":4,

                "Pokemon FireRed/Pokemon LeafGreen":5,

                "Pokemon Emerald Version":6,

                "Pokemon Diamond/Pokemon Pearl":7,

                "Pokemon Platinum Version":8,

                "Pokemon HeartGold/Pokemon SoulSilver":9,

                "Pokemon Black/Pokemon White":10,

                "Pokemon Black 2/Pokemon White 2":11,

                "Pokemon X/Pokemon Y":12,

                "Pokemon Omega Ruby/Pokemon Alpha Sapphire":13})

pk_main2 = pk[pk["Name"].isin(pk_list2)]

pk_main2["key"] = ""

for index, row in pk_main2.iterrows():

    pk_main2.set_value(index, 'key', pk_list2[pk_main2.loc[index]['Name']]) 

pk_plot4 = pk_main2.sort_values(by="key").plot.bar(y="Global_Sales", x="Name")
#Zelda

z = tmp_df[tmp_df["Name"].str.contains("Zelda")]

plt.figure(figsize=(8,6))

#Plot by individual sales

z_plot1 = z.sort_values(by="Year").plot.bar(y="Global_Sales", x="Name")

#Plot by year sales

z_gb = z.groupby(z["Year"]).sum()

z_plot2 = z_gb.plot.bar(y="Global_Sales")

#Plot by main series sales

z_main = z[z.index < 1000]

z_plot3 = z_main.sort_values(by="Year").plot.bar(y="Global_Sales", x="Name")