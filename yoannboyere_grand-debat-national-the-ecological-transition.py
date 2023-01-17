import pandas as pd

pd.options.display.max_columns = None

import geopandas as gpd

import numpy as np

import matplotlib.pyplot as plt

from wordcloud import WordCloud, ImageColorGenerator
def graph_pie(labels, values, colors, title, explode, name_export):

    fig1, ax1 = plt.subplots()

    ax1.pie(values, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)

    ax1.axis('equal')

    plt.title(title)

    plt.tight_layout()

    plt.show()

    

def graph_bar(labels, values, title, name_export):

    fig1, ax1 = plt.subplots()

    y_pos = np.arange(len(labels))

    plt.xticks(y_pos, labels)

    ax1.bar(y_pos, values, color='#99ff99')

    plt.title(title)

    plt.show()

    

stop_words = ["de","la","c'est","du","pour","ce","les","lié","des","ou","et","tout","est","un","sont","tous","ci","dans","le",

             "il","faut","qui","que","n'y","pas","ces","ne","peut","par","toute","donc","mai","mais","leur","non","comme","une",

             "plus","important","trop","se","sur","fait","ceraines","problème","problèmes", "en", "place","commun","exemple",

             "France","avec","au","niveau",'aussi',"etc","doit","etre","cela","soit","bien","autre","même","nottament","si",

             "alors","car","autres","mettre","je","fais","ecologique","aide","solution","citoyen","elle","san","beaucoup","notamment",

             "surtout","faire","n'est","écologique","pourquoi","an","doivent","on","encore","avoir","rien","sans","moi","me","déjà",

             "nous","l'état","faudrait","temp","celle","françai","ex","peu","dont","monde","déjà","ma","être","serait","possible",

             "aujourd'hui","également","ainsi","français","toujour","pourrait","nécessaire","quand","possible","voir","ça",

             "possible","voir","personne","semble","ceux","ont","ils","liés","cette","certain","très","point","semble",

             "sui","mes","peux","ans","chose","km","qu'il","ca","j'ai","aucune","toutes","sais","no","vou","pourrai","toujours",

             "chacun","cas","nos","somme","avons","afin","qu","vous"]



def word_cloud(text, name_export):

    wordcloud = WordCloud(stopwords = stop_words, max_font_size=50, background_color="white", width=800, height=400).generate(text)

    plt.figure(figsize=[20,10])

    plt.imshow(wordcloud, interpolation='bilinear')

    plt.axis("off")

    plt.show()
democratie_citoyennete = pd.read_csv('/kaggle/input/granddebat/DEMOCRATIE_ET_CITOYENNETE.csv', dtype=str)

fiscalite_dp = pd.read_csv('/kaggle/input/granddebat/LA_FISCALITE_ET_LES_DEPENSES_PUBLIQUES.csv', dtype=str)

transition_ecologique = pd.read_csv('/kaggle/input/granddebat/LA_TRANSITION_ECOLOGIQUE.csv', dtype=str)

organisation_etat_sv = pd.read_csv('/kaggle/input/granddebat/ORGANISATION_DE_LETAT_ET_DES_SERVICES_PUBLICS.csv', dtype=str)
size_democratie_citoyennete = democratie_citoyennete.count()['id']

size_fiscalite_dp = fiscalite_dp.count()['id']

size_transition_ecologique = transition_ecologique.count()['id']

size_organisation_etat_sv = organisation_etat_sv.count()['id']



total_row = size_democratie_citoyennete+size_fiscalite_dp+size_transition_ecologique+size_organisation_etat_sv



print("Nombre de contributions pour :")

print("- Democratie et citoyenneté : "+str(size_democratie_citoyennete))

print("- Fiscalité et dépenses publiques : "+str(size_fiscalite_dp))

print("- Transition écologique : "+str(size_transition_ecologique))

print("- Organisation de l'état et des services publics : "+str(size_organisation_etat_sv))
labels = ['Democratie et Citoyenneté','La fiscalisté et les dépenses publiques',

          'La transition écologique', "Organisation de l'état"]

values = [size_democratie_citoyennete/total_row,size_fiscalite_dp/total_row,

             size_transition_ecologique/total_row,size_organisation_etat_sv/total_row]

explode = (0, 0, 0.1, 0)

colors = ['#ff9999','#66b3ff','#99ff99','#ffcc99']

title = 'Repartition du nombre de contributions entre les 4 thèmes'

name_export="repartition_themes"

graph_pie(labels, values, colors, title, explode, name_export)
list_df = [transition_ecologique,organisation_etat_sv,fiscalite_dp,democratie_citoyennete]

list_name_df=['transition_ecologique','organisation_etat_sv','fiscalite_dp','democratie_citoyennete']

list_count_df = []

i=0

for df in list_df:

    df['code_dep'] = [str(code_postal)[:2] for code_postal in df.authorZipCode]

    count_dep = df.groupby('code_dep').count()['id'].rename(list_name_df[i])

    list_count_df.append(count_dep)

    i+=1



df_count_dep = pd.concat(list_count_df, axis=1, sort=False)



df_count_dep['percentage'] = [((row['transition_ecologique'])/(row['organisation_etat_sv']+row['fiscalite_dp']+row['democratie_citoyennete']+row['transition_ecologique']))*100 for index, row in df_count_dep.iterrows()]



df_count_dep = df_count_dep.drop(['-1','0','3','4','5','6','7','8','-9','na'])

df_count_dep.loc['69D'] = df_count_dep.loc['69']

df_count_dep.loc['69M'] = df_count_dep.loc['69']



df_count_dep.head()
# Data recovery of french department shape

map_df = gpd.read_file('/kaggle/input/shape-department-france/dep_fr.shp')

merged = map_df.set_index('code_insee').join(df_count_dep)
variable = 'percentage'



vmin, vmax = merged[variable].min(), merged[variable].max()



fig, ax = plt.subplots(1, figsize=(10, 6))



ax.axis([-5,10,40,52.5])

ax.axis('off')

ax.set_title('Percentage of contributors in the ecological transition by department', fontdict={'fontsize': '25', 'fontweight' : '3'})

merged.plot(column=variable, cmap='Greens', linewidth=0.8, ax=ax, edgecolor='0.8')



sm = plt.cm.ScalarMappable(cmap='Greens', norm=plt.Normalize(vmin=vmin, vmax=vmax))

sm._A = []

cbar = fig.colorbar(sm)
transition_ecologique.head()
data_clean = transition_ecologique.drop(['id','reference','createdAt','publishedAt','updatedAt','trashed','trashedStatus','authorId','authorType','authorZipCode'], axis=1)

data_clean.columns=['title',

                    "Quel est aujourd'hui pour vous le problème concret le plus important dans le domaine de l'environnement ?",

                    "Que faudrait-il faire selon vous pour apporter des réponses à ce problème ?",

                    "Diriez-vous que votre vie quotidienne est aujourd'hui touchée par le changement climatique ?",

                    "Si oui, de quelle manière votre vie quotidienne est-elle touchée par le changement climatique ?",

                    "À titre personnel, pensez-vous pouvoir contribuer à protéger l'environnement ?",

                    "Si oui, que faites-vous aujourd'hui pour protéger l'environnement et/ou que pourriez-vous faire ?",

                    "Qu'est-ce qui pourrait vous inciter à changer vos comportements comme par exemple mieux entretenir et régler votre chauffage, modifier votre manière de conduire ou renoncer à prendre votre véhicule pour de très petites distances ?",

                    "Quelles seraient pour vous les solutions les plus simples et les plus supportables sur un plan financier pour vous inciter à changer vos comportements ?",

                    "Par rapport à votre mode de chauffage actuel, pensez-vous qu'il existe des solutions alternatives plus écologiques ?",

                    "Si oui, que faudrait-il faire pour vous convaincre ou vous aider à changer de mode de chauffage ?",

                    "Avez-vous pour vos déplacements quotidiens la possibilité de recourir à des solutions de mobilité alternatives à la voiture individuelle comme les transports en commun, le covoiturage, l'auto-partage, le transport à la demande, le vélo, etc. ?",

                    "Si oui, que faudrait-il faire pour vous convaincre ou vous aider à utiliser ces solutions alternatives ?",

                    "Si non, quelles sont les solutions de mobilité alternatives que vous souhaiteriez pouvoir utiliser ?",

                    "Et qui doit selon vous se charger de vous proposer ce type de solutions alternatives ?",

                    "Que pourrait faire la France pour faire partager ses choix en matière d'environnement au niveau européen et international ?",

                    "Y a-t-il d'autres points sur la transition écologique sur lesquels vous souhaiteriez vous exprimer ?",

                   "code_dep"]

size_file = data_clean.count()['title']

data_clean = data_clean.fillna('')

data_clean.head()
list_q1_possible_rep = ["La pollution de l'air","Les dérèglements climatiques (crue, sécheresse)","L'érosion du littoral","La biodiversité et la disparition de certaines espèces"] 

q1 = "Quel est aujourd'hui pour vous le problème concret le plus important dans le domaine de l'environnement ?"

data_q1 = data_clean[data_clean[q1]!=""]

nb_rep_q1 = data_q1.count()['title']

values = []



for possible_rep in list_q1_possible_rep:

    size_rep = data_q1[data_q1[q1]==possible_rep].count()['title']

    pourcentage_rep = (size_rep/nb_rep_q1)*100

    values.append(pourcentage_rep)

    

values.append(100-sum(values))



labels = ["La pollution de l'air","Les dérèglements\n climatiques\n (crue, sécheresse)",

         "L'érosion du littoral","La biodiversité et \nla disparition \nde certaines espèces",

         "Autres"]

colors = ['#ff9999','#66b3ff','#99ff99','#ffcc99','#98ff56']

title = "The most concrete problem in the environment"

explode = (0, 0, 0, 0, 0)

name_export = "problemes_plus_concret"

graph_pie(labels, values, colors, title, explode, name_export)
data_q1_autres = data_clean[~data_clean[q1].isin(list_q1_possible_rep)]



text = " ".join(review for review in data_q1_autres[q1])

word_cloud(text,'autre_theme')
q="Diriez-vous que votre vie quotidienne est aujourd'hui touchée par le changement climatique ?"

yes = (data_clean[data_clean[q]=="Oui"].count()['title']/size_file)*100

no = (data_clean[data_clean[q]=="Non"].count()['title']/size_file)*100

no_answer = (data_clean[data_clean[q]==""].count()['title']/size_file)*100

graph_bar(['Oui','Non','NP'],[yes,no,no_answer],"Votre vie quotidienne est-elle aujourd'hui touchée par le changement climatique ?",'vie_quotidienne_touchée_oui_non')
yes_dep = data_clean[data_clean[q]=="Oui"].groupby("code_dep").count()['title']

other_dep = data_clean[data_clean[q]!="Oui"].groupby("code_dep").count()['title']



yes_dep = yes_dep.drop(['-1','0','4','5','na'])

yes_dep.loc['69D'] = yes_dep.loc['69']

yes_dep.loc['69M'] = yes_dep.loc['69']

other_dep.loc['69D'] = other_dep.loc['69']

other_dep.loc['69M'] = other_dep.loc['69']



for index, value in yes_dep.items():

    yes_dep[index] = (value/(other_dep[index]+value))*100

    

df_yes_dep = pd.Series(yes_dep,name="pourcentage")    

df_yes_dep.head()
merged = map_df.set_index('code_insee').join(df_yes_dep)

# Variable utilisé pour les valeurs de la map

variable = 'pourcentage'



# Min et max pour la légende

vmin, vmax = merged[variable].min(), merged[variable].max()



# Création du graphique

fig, ax = plt.subplots(1, figsize=(10, 6))

# Zoom sur la France métropolitaine

ax.axis([-5,10,40,52.5])

ax.axis('off')

ax.set_title('Percentage of people affected by climate change (by department)', fontdict={'fontsize': '25', 'fontweight' : '3'})

merged.plot(column=variable, cmap='Greens', linewidth=0.8, ax=ax, edgecolor='0.8')



# Ajout de la légende

sm = plt.cm.ScalarMappable(cmap='Greens', norm=plt.Normalize(vmin=vmin, vmax=vmax))

sm._A = []

cbar = fig.colorbar(sm)
q="À titre personnel, pensez-vous pouvoir contribuer à protéger l'environnement ?"

yes = (data_clean[data_clean[q]=="Oui"].count()['title']/size_file)*100

no = (data_clean[data_clean[q]=="Non"].count()['title']/size_file)*100

no_answer = (data_clean[data_clean[q]==""].count()['title']/size_file)*100

print(yes,no,no_answer)



graph_bar(['Oui','Non','NP'],[yes,no,no_answer],"Pensez-vous pouvoir contribuer à protéger l'environnement ?",'contribution_oui_non')
q="Si oui, que faites-vous aujourd'hui pour protéger l'environnement et/ou que pourriez-vous faire ?"

text = " ".join(review for review in data_clean[q])

word_cloud(text, 'action_word_cloud')
q="Qu'est-ce qui pourrait vous inciter à changer vos comportements comme par exemple mieux entretenir et régler votre chauffage, modifier votre manière de conduire ou renoncer à prendre votre véhicule pour de très petites distances ?"

text = " ".join(review for review in data_clean[q])

word_cloud(text, 'incitation_word_cloud')
q="Quelles seraient pour vous les solutions les plus simples et les plus supportables sur un plan financier pour vous inciter à changer vos comportements ?"

text = " ".join(review for review in data_clean[q])

word_cloud(text, "solutions_supportables_word_cloud")