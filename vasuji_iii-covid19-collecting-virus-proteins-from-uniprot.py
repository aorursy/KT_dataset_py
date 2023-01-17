import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

file_path = "../input/corona-virus-proteins-from-uniprot-database/corona.csv"

df = pd.read_csv(file_path)
df.head(5)
df.shape
df_organism = pd.DataFrame(df.groupby("Organism").count()['Entry'])

df_organism = df_organism.sort_values(by = "Entry", ascending = False)

df_organism[0:20].plot.barh(figsize = [15,10], fontsize =20)

plt.gca().invert_yaxis()
df_organism[0:20]
df['Virus hosts'] = df['Virus hosts'].apply(lambda x: str(x)[0:50] )

df_host = pd.DataFrame(df.groupby("Virus hosts").count()['Entry'])

df_host = df_host.sort_values(by = "Entry", ascending = False)

df_host[1:20].plot.barh(figsize = [15,10], fontsize =20)

plt.gca().invert_yaxis()
df_host[1:20]
def filter(line):

    proteins = set()

    line = str(line)

    line = line.lower()

    

    '''for lines without () or [] terms'''

    if "(" not in line or "[" not in line:

        proteins.add(line.strip().replace(' ', '_'))

        

        

    '''for line including () terms'''    

    if '(' in line:

        start = 0

        open_in = line.find('(')

        tmp = line[start:open_in].strip().replace(' ', '_')

        proteins.add(tmp)

        while open_in >=0:

            start = open_in+1

            end = line.find(')', start)

            proteins.add(line[start:end].strip().replace(' ', '_'))

            open_in = line.find('(', end)

     

    '''for lines including [] trems'''

    if '[' in line:

        raw = line[line.find('['):line.find(']')]

        #print("THIS IS RAW:", raw[15:-1])

        raw = raw[15:-1]

        lraw = raw.split("; ")

        for item in lraw:

            #print(item)

            if '(' in item:

                start = 0

                open_in = item.find('(')

                tmp = item[start:open_in].strip().replace(' ', '_')

                proteins.add(tmp)

            else:

                proteins.add(item.strip().replace(' ', '_'))

    return proteins
allProteins = []

i = 0

for u,p in zip(df['Entry'],df['Protein names']):

    print(u,"|",p)

    print("------------")

    print(u,"|",filter(p))

    print("===================================================")

    i += 1

    if i>4:

        break
allProteins = []

for u,p in zip(df['Entry'],df['Protein names']):

    allProteins.append({"id":u, "names":list(filter(p))})
allProteins[0:5]
import json

with open("virus-proteins.json", 'w') as fn:

    json.dump(allProteins,fn)