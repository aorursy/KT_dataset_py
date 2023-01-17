import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
def remCols(string):

    cols = df.columns.values

    for col in cols:

        if col.endswith(string):

            df.drop(col,axis=1,inplace=True)

            

def discoverData(column):

    print ("Total no. of unique values: {0} for {1} column". format(df[column].nunique(),column))

    return df[[column]].apply(pd.Series.value_counts)



def checkDuplicates(*col):

    print ("There are {0} duplicated rows in {1} column".format(df.duplicated(col).sum(), col) )
df=pd.read_table('../input/en.openfoodfacts.org.products.tsv',sep='\t', low_memory=False)
df.info()
df.head()
pd.options.display.max_columns = 108

df.describe()
pd.options.display.max_columns = 107

df.describe(exclude=[np.number])
df.drop(df[df.product_name.isnull()].index, inplace=True)
cols_to_drop = ['no_nutriments','nutrition_grade_uk','chlorophyl_100g','glycemic-index_100g','water-hardness_100g',

               'image_url','image_small_url','url','ingredients_from_palm_oil','ingredients_that_may_be_from_palm_oil']

df.drop(cols_to_drop,axis=1,inplace=True)
remCols('acid_100g')
df.loc[df.duplicated(),:]
checkDuplicates('product_name','brands')
df.drop_duplicates(['product_name','brands'],inplace=True)
m = df.astype(str).apply(lambda x: x.str.contains("palm|palme", case=False, na=False)).any() 

c = df.columns[m]

print (c)
frames = []



for col in c:

    mask = df[col].str.lower().str.contains('palm oil|palme',na=False)

    frames.append(df[mask])



palm_brands = pd.concat(frames)
palm_brands[['ingredients_from_palm_oil_n','ingredients_that_may_be_from_palm_oil_n']].dropna(how='any',axis=0)
palm_brands.shape
palm_brands.drop_duplicates(inplace=True)
palm_brands.shape
palm_brands['labels_en'].value_counts()
palm_brands.loc[palm_brands.labels_en.str.lower().str.contains('sustainable palm oil',na=False),'labels_en'] = 'Sustainable Palm Oil'

palm_brands.loc[palm_brands.labels_en.str.lower().str.contains('palm-oil',na=False),'labels_en'] = 'Palm-oil'

palm_brands.loc[palm_brands.labels_en.str.lower().str.contains('palm oil free',na=False),'labels_en'] = 'Palm oil free'

palm_brands.loc[palm_brands.labels_en.str.lower().str.contains('kosher|halal',na=False),'labels_en'] = 'Kosher,Halal'

palm_brands.loc[palm_brands.labels_en.str.lower().str.contains('organic',na=False),'labels_en'] = 'Organic'

palm_brands.loc[palm_brands.labels_en.str.lower().str.contains('vegetarian|green dot|vega',na=False),'labels_en'] = 'Vegetarian,Vegan'

palm_brands.loc[palm_brands.labels_en.str.lower().str.contains('gluten-free|gluten free',na=False),'labels_en'] = 'Gluten Free'
palm_brands['labels_en'].value_counts().head(5)
palm_brands['brands'].isnull().sum()
palm_brands['brands'].fillna(palm_brands.product_name.str.split(' ').str[:2].str.join(','),inplace=True)
palm_brands['brands']=palm_brands['brands'].str.replace('Tastykake,  Tasty Baking Company','Tasty Baking Company')

palm_brands['brands']=palm_brands['brands'].str.replace('Tastykake','Tasty Baking Company')

palm_brands['brands']=palm_brands['brands'].str.replace('Cornetto','Unilever')

palm_brands['brands']=palm_brands['brands'].str.replace('Weis Quality','Weis')

palm_brands['brands']=palm_brands['brands'].str.replace('Weis,  Weis Markets  Inc.','Weis')

palm_brands['brands']=palm_brands['brands'].str.replace('Hill & Valley', 'Hill & Valley Inc.')

palm_brands['brands']=palm_brands['brands'].str.replace('Gluten Free Bake Shop Inc', 'Gluten Free Bake Shoppe Inc.')

palm_brands['brands']=palm_brands['brands'].str.replace('Jessie Lord Bakery', 'Jessie Lord Bakery Llc')

palm_brands['brands']=palm_brands['brands'].str.replace('Carrfour', 'Carrefour')

palm_brands['brands']=palm_brands['brands'].str.replace('Giant, Fresh & Easy', 'Giant')

palm_brands['brands']=palm_brands['brands'].str.replace('Deep, Giant', 'Giant')

palm_brands['brands']=palm_brands['brands'].str.replace('Tarallinii,olio', 'Tarallini,')

palm_brands['brands']=palm_brands['brands'].str.replace('Weight,Watchers', 'Weight Watchers')

palm_brands['brands']=palm_brands['brands'].str.replace('Whole earth', 'Whole Earth')

palm_brands['brands']=palm_brands['brands'].str.replace('Vaseline,rosy', 'Unilever')

palm_brands['brands']=palm_brands['brands'].str.replace('Acme Fresh Market', 'Acme')

palm_brands['brands']=palm_brands['brands'].str.replace('Absolutely Gluten Free', 'Absolutely')

palm_brands['brands']=palm_brands['brands'].str.replace('Rita', 'Poppies')

palm_brands['brands']=palm_brands['brands'].str.replace('Coles Patisserie,Coles', 'Coles')

palm_brands['brands']=palm_brands['brands'].str.replace('Edward and Sons, Edward & Sons', 'Edward & Sons')

palm_brands['brands']=palm_brands['brands'].str.replace('7days', '7 Days')

palm_brands['brands']=palm_brands['brands'].str.replace('A Couple Of Squares', 'A Couple Of Squares Inc.')

palm_brands['brands']=palm_brands['brands'].str.replace('Aladin', 'Aladdin')
palm_brands.loc[palm_brands.brands.str.lower().str.contains("snyder's|lance",na=False),'brands'] = "Snyder's Lance"

palm_brands.loc[palm_brands.brands.str.lower().str.contains('hy-vee',na=False),'brands'] = 'Hy-Vee Inc'

palm_brands.loc[palm_brands.brands.str.lower().str.contains('meijer',na=False),'brands'] = 'Meijer'

palm_brands.loc[palm_brands.brands.str.lower().str.contains('market pantry|target',na=False),'brands'] = 'Market Pantry'

palm_brands.loc[palm_brands.brands.str.lower().str.contains('dawn food',na=False),'brands'] = 'Dawn Foods'

palm_brands.loc[palm_brands.brands.str.lower().str.contains('the bakery',na=False),'brands'] = 'The Bakery by Walmart'

palm_brands.loc[palm_brands.brands.str.lower().str.contains('wal-mart|walmart',na=False),'brands'] = 'Walmart'

palm_brands.loc[palm_brands.brands.str.lower().str.contains('lofthouse',na=False),'brands'] = 'Lofthouse Foods'

palm_brands.loc[palm_brands.brands.str.lower().str.contains('elmer',na=False),'brands'] = 'Elmer Corporation'

palm_brands.loc[palm_brands.brands.str.lower().str.contains('unilever',na=False),'brands'] = 'Unilever'

palm_brands.loc[palm_brands.brands.str.lower().str.contains('decorated',na=False),'brands'] = 'The Decorated Cookie Company'

palm_brands.loc[palm_brands.brands.str.lower().str.contains('clover valley',na=False),'brands'] = 'Clover Valley'

palm_brands.loc[palm_brands.brands.str.lower().str.contains('hill valley',na=False),'brands'] = 'Hill & Valley Inc.'

palm_brands.loc[palm_brands.brands.str.lower().str.contains('sainsbury',na=False),'brands'] = "Sainsbury's"

palm_brands.loc[palm_brands.brands.str.lower().str.contains('casino',na=False),'brands'] = "Casino"

palm_brands.loc[palm_brands.brands.str.lower().str.contains('carrefour|jury',na=False),'brands'] = "Carrefour"

palm_brands.loc[palm_brands.brands.str.lower().str.contains('ferrero|nutella',na=False),'brands'] = "Ferrero"

palm_brands.loc[palm_brands.brands.str.lower().str.contains('dean',na=False),'brands'] = "Dean Foods"

palm_brands.loc[palm_brands.brands.str.lower().str.contains('aldi',na=False),'brands'] = 'Aldi-Benner Company'

palm_brands.loc[palm_brands.brands.str.lower().str.contains('skippy',na=False),'brands'] = 'Skippy,Hormel Foods'

palm_brands.loc[palm_brands.brands.str.lower().str.contains('kroger|fresh food|bakery fresh',na=False),'brands'] = 'Kroger'

palm_brands.loc[palm_brands.brands.str.lower().str.contains('vico',na=False),'brands'] = 'Vico'

palm_brands.loc[palm_brands.brands.str.lower().str.contains('lipton|lay|dorito',na=False),'brands'] = 'PepsiCo'

palm_brands.loc[palm_brands.brands.str.lower().str.contains('lindt',na=False),'brands'] = 'Lindt & Sprungli Gmbh'

palm_brands.loc[palm_brands.brands.str.lower().str.contains('barilla|harry',na=False),'brands'] = 'Barilla'

palm_brands.loc[palm_brands.brands.str.lower().str.contains('so delicious',na=False),'brands'] = 'So Delicious Dairy Free'

palm_brands.loc[palm_brands.brands.str.lower().str.contains('findus',na=False),'brands'] = 'Findus'

palm_brands.loc[palm_brands.brands.str.lower().str.contains('western family',na=False),'brands'] = 'Western Family'

palm_brands.loc[palm_brands.brands.str.lower().str.contains('ibakefood',na=False),'brands'] = 'ibakefoods'

palm_brands.loc[palm_brands.brands.str.lower().str.contains('lotus',na=False),'brands'] = 'Lotus'

palm_brands.loc[palm_brands.brands.str.lower().str.contains('schnucks',na=False),'brands'] = 'Schnucks'

palm_brands.loc[palm_brands.brands.str.lower().str.contains('pillsbury',na=False),'brands'] = 'Pillsbury'

palm_brands.loc[palm_brands.brands.str.lower().str.contains('food club',na=False),'brands'] = 'Food Club'

palm_brands.loc[palm_brands.brands.str.lower().str.contains('entenmann',na=False),'brands'] = "Entenmann's"

palm_brands.loc[palm_brands.brands.str.lower().str.contains('wilton',na=False),'brands'] = "Wilton"

palm_brands.loc[palm_brands.brands.str.lower().str.contains("freshley's",na=False),'brands'] = "Mrs. Freshley's, Flowers Foods Inc."

palm_brands.loc[palm_brands.brands.str.lower().str.contains('fresh & easy',na=False),'brands'] = "Fresh & Easy"

palm_brands.loc[palm_brands.brands.str.lower().str.contains('hubert',na=False),'brands'] = "St Hubert"

palm_brands.loc[palm_brands.brands.str.lower().str.contains('betty',na=False),'brands'] = "Betty Crocker"

palm_brands.loc[palm_brands.brands.str.lower().str.contains('auchan',na=False),'brands'] = "Auchan"

palm_brands.loc[palm_brands.brands.str.lower().str.contains('tesco',na=False),'brands'] = "Tesco"

palm_brands.loc[palm_brands.brands.str.lower().str.contains('walkers',na=False),'brands'] = "Walkers"

palm_brands.loc[palm_brands.brands.str.lower().str.contains('heinz|kraft',na=False),'brands'] = "Kraft Heinz Co"

palm_brands.loc[palm_brands.brands.str.lower().str.contains('arnott|tim tam',na=False),'brands'] = "Arnott's"

palm_brands.loc[palm_brands.brands.str.lower().str.contains('marks|m&s',na=False),'brands'] = 'Marks & Spencer'

palm_brands.loc[palm_brands.brands.str.lower().str.contains('maltesers|mars|snickers|twix|skittles',na=False),'brands'] = 'Mars'

palm_brands.loc[palm_brands.brands.str.lower().str.contains('ghirardelli',na=False),'brands'] = 'Ghirardelli Chocolate Company'

palm_brands.loc[palm_brands.brands.str.lower().str.contains('endangered',na=False),'brands'] = 'Endangered Species Chocolate Co'

palm_brands.loc[palm_brands.brands.str.lower().str.contains('u bio',na=False),'brands'] = 'U'

palm_brands.loc[palm_brands.brands.str.lower().str.contains('soleil',na=False),'brands'] = 'Bio Soleil'

palm_brands.loc[palm_brands.brands.str.lower().str.contains('jardin',na=False),'brands'] = 'Jardin Bio'

palm_brands.loc[palm_brands.brands.str.lower().str.contains('power crunch',na=False),'brands'] = 'Power Crunch'

palm_brands.loc[palm_brands.brands.str.lower().str.contains('power bar',na=False),'brands'] = 'Power Bar'

palm_brands.loc[palm_brands.brands.str.lower().str.contains('big y',na=False),'brands'] = 'Big Y Foods Inc.'

palm_brands.loc[palm_brands.brands.str.lower().str.contains('little debbie|mckee|sunbelt|heartland|drake',na=False),'brands'] = 'McKee Foods'

palm_brands.loc[palm_brands.brands.str.lower().str.contains('dansk',na=False),'brands'] = 'Royal Dansk'

palm_brands.loc[palm_brands.brands.str.lower().str.contains('wild oats',na=False),'brands'] = 'Wild Oats Marketplace'

palm_brands.loc[palm_brands.brands.str.lower().str.contains('terres et céréales',na=False),'brands'] = 'Terres et Céréales Bio'

palm_brands.loc[palm_brands.brands.str.lower().str.contains('food lion|ahold|giant|hannaford',na=False),'brands'] = 'Ahold Delhaize'

palm_brands.loc[palm_brands.brands.str.lower().str.contains('supervalu',na=False),'brands'] = 'Essential Everyday,Supervalu'

palm_brands.loc[palm_brands.brands.str.lower().str.contains('harris|ht trader',na=False),'brands'] = 'Harris Teeter'

palm_brands.loc[palm_brands.brands.str.lower().str.contains('trader joe',na=False),'brands'] = "Trader Joe's"

palm_brands.loc[palm_brands.brands.str.lower().str.contains('mousquetaires',na=False),'brands'] = "Mousquetaires"

palm_brands.loc[palm_brands.brands.str.lower().str.contains('nestle|nestlé|maggi|buitoni|nescafe|herta|milo',na=False),'brands'] = "Mousquetaires"

palm_brands.loc[palm_brands.brands.str.lower().str.contains("migros sélection|farmer|anna's|m-classic|m-budget|cornatur|migros",na=False),'brands'] = "Elsa Miforma"

palm_brands.loc[palm_brands.brands.str.lower().str.contains("365|whole food",na=False),'brands'] = "Whole Foods Market"

palm_brands.loc[palm_brands.brands.str.lower().str.contains("coop",na=False),'brands'] = "Coop Group"

palm_brands.loc[palm_brands.brands.str.lower().str.contains("lactel|primevère|primevere",na=False),'brands'] = "Lactalis Intl"

palm_brands.loc[palm_brands.brands.str.lower().str.contains("marque repère|marque repere|leclerc",na=False),'brands'] = "Marque Repère,Leclerc"

palm_brands.loc[palm_brands.brands.str.lower().str.contains("belvita|cadbury|cote d'or|figaro|freia|marabou|mikado|milka|prince|oreo|ritz|toblerone|lu,",na=False),'brands'] = "Mondelez Intl"

palm_brands.loc[palm_brands.brands.str.lower().str.contains("brossard|jacquet",na=False),'brands'] = "Jacquet,Brossard"

palm_brands.loc[palm_brands.brands.str.contains("Cora",na=False),'brands'] = "Louis Delhaize"

palm_brands.loc[palm_brands.brands.str.contains("LU",na=False),'brands'] = "Mondelez Intl"

palm_brands.loc[palm_brands.brands.str.lower().str.contains("kellogg",na=False),'brands'] = "Kellogg's"

palm_brands.loc[palm_brands.brands.str.lower().str.contains("claude leger|claude léger|chabrior",na=False),'brands'] = "Mousquetaires"

palm_brands.loc[palm_brands.brands.str.lower().str.contains("monique",na=False),'brands'] = "Monique Ranou"

palm_brands.loc[palm_brands.brands.str.lower().str.contains("ivoria",na=False),'brands'] = "Ivoria Chocolate"

palm_brands.loc[palm_brands.brands.str.lower().str.contains("intermarché|intermarche",na=False),'brands'] = "Intermarché"

palm_brands.loc[palm_brands.brands.str.lower().str.contains("danone",na=False),'brands'] = "Danone"

palm_brands.loc[palm_brands.brands.str.lower().str.contains("united biscuits|bn|mcvitie|godiva",na=False),'brands'] = "Yildiz Holdings"

palm_brands.loc[palm_brands.brands.str.lower().str.contains("ültje|ultje",na=False),'brands'] = "Ültje"

palm_brands.loc[palm_brands.brands.str.lower().str.contains("tous",na=False),'brands'] = "Tous les jours,CJ Foodville"

palm_brands.loc[palm_brands.brands.str.lower().str.contains("3 toque",na=False),'brands'] = "3 Toques"

palm_brands.loc[palm_brands.brands.str.lower().str.contains("toque",na=False),'brands'] = "Toque du chef"

palm_brands.loc[palm_brands.brands.str.lower().str.contains("knorr",na=False),'brands'] = "Unilever"

palm_brands.loc[palm_brands.brands.str.lower().str.contains("nissin",na=False),'brands'] = "Nissin Foods"

palm_brands.loc[palm_brands.brands.str.lower().str.contains('7-eleven|7 select',na=False),'brands'] = "7-Eleven"

palm_brands.loc[palm_brands.brands.str.lower().str.contains('les delice|les délice',na=False),'brands'] = 'Les Delices des 7 vallees'

palm_brands.loc[palm_brands.brands.str.lower().str.contains("505",na=False),'brands'] = "505 Southwestern"

palm_brands.loc[palm_brands.brands.str.lower().str.contains('alnatura|altanatura',na=False),'brands'] = "Alnatura"

palm_brands.loc[palm_brands.brands.str.lower().str.contains('poppies|macaron|berlidon|delizza',na=False),'brands'] = "Poppies-Berlidon"

palm_brands.loc[palm_brands.brands.str.lower().str.contains('britannia',na=False),'brands'] = "Britannia Industries"

palm_brands.loc[palm_brands.brands.str.lower().str.contains('antoinette',na=False),'brands'] = "Antoinette Patisserie"

palm_brands.loc[palm_brands.brands.str.lower().str.contains('180',na=False),'brands'] = "180 Snacks Inc."

palm_brands.loc[palm_brands.brands.str.lower().str.contains('market square',na=False),'brands'] = "Market Square Bakery"

palm_brands.loc[palm_brands.brands.str.lower().str.contains('monoprix',na=False),'brands'] = "Monoprix"

palm_brands.loc[palm_brands.brands.str.lower().str.contains('gagnant|leader',na=False),'brands'] = "Leader Price"

palm_brands.loc[palm_brands.brands.str.lower().str.contains('st michel|saint michel',na=False),'brands'] = "St Michel"

palm_brands.loc[palm_brands.brands.str.lower().str.contains('révillon|revillon',na=False),'brands'] = "Révillon Chocolatier"

palm_brands.loc[palm_brands.brands.str.lower().str.contains('gerblé|gerble',na=False),'brands'] = "Gerblé"

palm_brands.loc[palm_brands.brands.str.lower().str.contains('m&m',na=False),'brands'] = "Mars"

palm_brands.loc[palm_brands.brands.str.lower().str.contains('mes p',na=False),'brands'] = "Mes P'tits Secrets"

palm_brands.loc[palm_brands.brands.str.lower().str.contains('pasquier',na=False),'brands'] = "Pasquier"

palm_brands.loc[palm_brands.brands.str.lower().str.contains('valu time',na=False),'brands'] = 'Valu Time'

palm_brands.loc[palm_brands.brands.str.lower().str.contains('pepperidge|campbell|royal dansk|arnott',na=False),'brands'] = 'Kelsen'

palm_brands.loc[palm_brands.brands.str.lower().str.contains('planta fin',na=False),'brands'] = 'Unilever'

palm_brands.loc[palm_brands.brands.str.lower().str.contains("planta fin|carte d'or|miko|fruit d'or|hellmann",na=False),'brands'] = 'Unilever'
#I cant simply search for string LU/Lu/lu as this will return many other unwanted rows. Therefore, I needed to specify that I am

#looking for a 2-lenght string and replace them with the correct company name.



filter = palm_brands.brands.str.len() == 2

palm_brands.loc[filter,'brands'] = "Mondelez Intl"
plt.figure(figsize=(10,8))

ax=palm_brands['brands'].value_counts()[:20].plot(kind='barh',color='teal')

ax.invert_yaxis()

ax.set_title("Number of products by Brand in Dataset",{'fontsize':14})

plt.show()
ing_palm_oil = palm_brands[(palm_brands['ingredients_from_palm_oil_n'] > 0.0) | (palm_brands['ingredients_that_may_be_from_palm_oil_n'] > 0.0)]
ing_palm_oil.shape
palm_brands['ingredients_from_palm_oil_n'].value_counts()
palm_brands['ingredients_that_may_be_from_palm_oil_n'].value_counts()
grouped = ing_palm_oil.groupby('brands').size().reset_index(name='count').sort_values(by='count',ascending=False).head(20)
ax = grouped.plot(x='brands',kind='barh',color='palevioletred',figsize=(10,8))

ax.invert_yaxis()

ax.legend().set_visible(False)

ax.set_title('Top 20 Brands producing products with Ingredients that (may) contain palm oil',{'fontsize':20})

plt.tight_layout()

plt.show()
plt.figure(figsize=(10,8))

ing_palm_oil[ing_palm_oil['labels_en'] == 'Sustainable Palm Oil']['brands'].value_counts().head(20).plot(kind='bar',color='skyblue')

plt.title('Brands producing with Sustainable Palm Oil',fontsize=14)

plt.show()
palm = ing_palm_oil['ingredients_from_palm_oil_tags']
palm_dict = {'huile-de-palme':'Palm Oil','huile-de-palme,e304-palmitate-d-ascorbyle':'Palm Oil,e304','oleine-de-palme':'Palm Olein',

             'e304-palmitate-d-ascorbyle':'e304','mono-et-diglycerides-d-acides-gras-de-palme':'e471',

             'huile-de-palme,stearine-de-palme':'Palm Oil,Palm Sterin','e304-palmitate-d-ascorbyle,huile-de-palme':'e304,Palm Oil',

             'huile-de-palme,oleine-de-palme':'Palm Oil,Palm Olein','oleine-de-palme,e304-palmitate-d-ascorbyle':'Palm Olein,e304',

             'stearine-de-palme':'Palm Sterin','huile-de-palme,mono-et-diglycerides-d-acides-gras-de-palme':'Palm Oil,e471',

             'oleine-de-palme,huile-de-palme':'Palm Oil,Palm Olein','stearine-de-palme,huile-de-palme':'Palm Oil,Palm Sterin'}



palm_updated = palm.map(palm_dict,na_action=None)
palm_updated = palm_updated.str.split(',',expand=True)
palm_count = palm_updated.apply(pd.value_counts)
palm_count['total'] = palm_count.apply(lambda row:row[0] + row[1],axis=1).astype(int)
enum = pd.DataFrame(ing_palm_oil['ingredients_that_may_be_from_palm_oil_tags'].str.extractall('([e]\d\d\d\w?)'))
enum.dropna(inplace=True)
enum_count = enum.apply(pd.value_counts)
fig = plt.figure(figsize=(14,8))

ax1 = fig.add_subplot(1,2,1)



x1_cor = np.arange(5)

ax1.bar(x1_cor,palm_count.total,color='tomato')

ax1.set_xticks(x1_cor)

ax1.set_xticklabels(palm_count.index)#,rotation='vertical')

ax1.set_title('Food Products containing Palm Oil and its variants')



x2_cor = np.arange(14)

ax2 = fig.add_subplot(1,2,2)

ax2.bar(x2_cor,enum_count[0],color='lightcoral')

ax2.set_xticks(x2_cor)

ax2.set_xticklabels(enum_count.index,rotation='vertical',fontsize=14)

ax2.set_title('Emulsifiers,Stabilisers,Thickeners and Coloring (E160a) Used')

plt.show()
plt.rcParams["figure.figsize"] = [10,6]

fig, (ax,ax2) = plt.subplots(ncols=2)

fig.subplots_adjust(wspace=0.5)

sns.heatmap(palm_brands[palm_brands['ingredients_from_palm_oil_n'] > 0.0].groupby('ingredients_from_palm_oil_n').mean()[['trans-fat_100g','saturated-fat_100g','fat_100g','cholesterol_100g']],annot=True,linewidth=.5, cmap="rocket", ax=ax, cbar=False)

fig.colorbar(ax.collections[0], ax=ax,location="left", use_gridspec=False, pad=0.2)

plt.title('Mean of Fats vs Number of Ingredients from Palm Oil',fontsize=14)

sns.heatmap(palm_brands[palm_brands['ingredients_that_may_be_from_palm_oil_n'] > 0.0].groupby('ingredients_that_may_be_from_palm_oil_n').mean()[['trans-fat_100g','saturated-fat_100g','fat_100g','cholesterol_100g']],annot=True,linewidth=.5, cmap="icefire", ax=ax2, cbar=False)

fig.colorbar(ax2.collections[0], ax=ax2,location="right", use_gridspec=False, pad=0.2)

#ax2.set_title('Mean of Fats vs Number of Ingredients from Palm Oil')

plt.show()
plt.figure(figsize=(12,10))

df[df['additives_n'] > 0.0]['additives_n'].value_counts().plot(kind='bar',color='cadetblue')

plt.title('Number of additives used in food products',fontsize=14)

plt.show()
additives = (df['additives_en'].str.extractall("(?P<Count>[E]\d\d\d\w?)"))
additives_count = additives.apply(pd.value_counts).head(30)
additives_count['Enum'] = additives_count.index
additives_count.reset_index(drop=True,inplace=True)
additives_mapping = {'E330': 'orange','E322':'purple','E322i':'purple','E101':'blue','E375':'orange','E101i':'blue',

                    'E300':'cyan','E415':'purple','E412':'orange','E500':'orange','E471':'purple','E203':'forestgreen','E407':'purple',

                    'E440':'purple','E250':'forestgreen','E150a':'blue','E450':'orange','E500i':'blue','E331':'orange',

                     'E129':'orange','E339':'orange','E440i':'purple','E160a':'blue','E270':'orange','E102':'blue',

                     'E410':'purple','E133':'blue','E341':'orange','E428':'purple','E621':'orange','E202':'blue'}



additives_count['Colors'] = additives_count['Enum'].map(additives_mapping)
import matplotlib.patches as mpatches
ax = additives_count.plot(x='Enum',y='Count',kind='barh',color=additives_count['Colors'],figsize=(12,10))

ax.invert_yaxis()

ax.legend().set_visible(False)

ax.set_title('30 most commonly used additives (categorized)',{'fontsize':20})



colors = mpatches.Patch(color='blue', label='Colors')

others = mpatches.Patch(color='orange', label='Others')

emulsifiers = mpatches.Patch(color='purple', label='Emulsifiers')

sweetners = mpatches.Patch(color='orchid', label='Sweetners')

antioxidant = mpatches.Patch(color='cyan', label='Antioxidants')

preservatives = mpatches.Patch(color='forestgreen', label='Preservatives')



plt.legend(handles=[colors,others,emulsifiers,sweetners,antioxidant,preservatives])

plt.show()
ing_df = df.dropna(subset=['ingredients_text'])
ing_df = ing_df['ingredients_text'].str.replace('(\W)',',').str.replace(',,,',',').str.replace(',,',',')
from collections import Counter

results = ing_df.str.split(',').values.tolist()

flat_results = [item for sublist in results for item in sublist]

count = Counter(flat_results)
count.most_common(20)
del count['de']

del count['']

del count['and']

del count['organic']

del count['d']

del count['natural']
count['salt'] = count.pop('sel',183655)

count['sugar'] = count.pop('sucre',135054)
top10 = count.most_common(10)
labels,count = zip(*top10)

xs = np.arange(len(labels)) 

plt.bar(xs,count,color='darkseagreen')

plt.xticks(xs, labels)

plt.title("10 Most commonly use ingredients")

plt.show()
df[['created_datetime','last_modified_datetime']].isnull().sum()
df.dropna(subset=['created_datetime'],axis=0,inplace=True)
converted_created_datetime = pd.to_datetime(df['created_datetime'],errors='coerce')
converted_last_modified = pd.to_datetime(df['last_modified_datetime'],errors='coerce')
ave_delta = ((converted_last_modified-converted_created_datetime)/2).astype('m8[D]').astype(np.float32)
ave_delta.hist(bins=5)

plt.title("Mean Difference between 'Last Modified' and 'First Created'")

plt.ylabel("Days")

plt.xlabel("Products")

plt.show()
df['converted_created_datetime'] = pd.to_datetime(df['created_datetime'],errors='coerce')
time = (df.set_index('converted_created_datetime').resample('M')['product_name'].count())/30
plt.figure(figsize=(12,10))

x_corr = np.arange(69)

x_time = ['Jan12','Feb12','Mar12','Apr12','May12','Jun12','Jul12','Aug12','Sep12','Oct12','Nov12','Dec12','Jan13','Feb13',

          'Mar13','Apr13','May13','Jun13','Jul13','Aug13','Sep13','Oct13','Nov13','Dec13','Jan14','Feb14','Mar14','Apr14',

          'May14','Jun14','Jul14','Aug14','Sep14','Oct14','Nov14','Dec14','Jan15','Feb15','Mar15','Apr15','May15','Jun15',

          'Jul15','Aug15','Sep15','Oct15','Nov15','Dec15','Jan16','Feb16','Mar16','Apr16','May16','Jun16','Jul16','Aug16',

          'Sep16','Oct16','Nov16','Dec16','Jan17','Feb17','Mar17','Apr17','May17','Jun17','Jul17','Aug17','Sep17']

frequency=4



plt.bar(x_corr,time)

plt.xticks(x_corr[::frequency],x_time[::frequency],rotation=90)

plt.title('Montly Mean of Created Items',{'fontsize':14})

plt.show()

nutrition_score = df[['nutrition-score-fr_100g','energy_100g','saturated-fat_100g','sugars_100g','sodium_100g','proteins_100g']]
nutrition_score.dropna(inplace=True)
nutrition_score_corr = nutrition_score.corr()
nutrition_score_corr
plt.figure(figsize=(10,8))

sns.heatmap(nutrition_score_corr,annot=True,linewidth=0.5)

plt.title('Correlation of elements that contribute to Nutrition Score',fontsize=14)

plt.show()