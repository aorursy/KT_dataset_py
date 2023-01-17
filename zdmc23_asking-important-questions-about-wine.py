import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import warnings
warnings.filterwarnings('ignore')
%matplotlib inline
df = pd.read_csv("../input/winemag-data-130k-v2.csv", index_col=0)
df.head()
df.shape
# Use BeautifulSoup to parse: https://en.wikipedia.org/wiki/List_of_grape_varieties
from urllib.request import urlopen
from bs4 import BeautifulSoup
 
wikiUrl = "https://en.wikipedia.org/wiki/List_of_grape_varieties"
wikiPage = urlopen(wikiUrl)

soup = BeautifulSoup(wikiPage, "lxml")
color_tables = soup.find_all("table", attrs={"class": "wikitable"})

reds = []
whites = []
roses = []
unknowns = []

for color_table in color_tables:
    table_variety = color_table.find_previous_sibling("h4").find("span").text.lower()
    tds = color_table.find_all("td")
    for td in tds:
        links = td.find_all("a", href=lambda xx: xx and '/wiki/' in xx)
        for link in links:
            if "red grapes" == table_variety:
                reds.append(link.text.lower())
            elif "white grapes" == table_variety:
                whites.append(link.text.lower())
            elif "rose grapes" == table_variety:
                #roses.append(link.text.lower())
                pass
            else:
                unknowns.append(link.text.lower())

# unlinked
reds.append("Borba".lower())
reds.append("Buket".lower())
reds.append("Caberinta".lower())
reds.append("Caino Bravo".lower())
reds.append("Caricagiola".lower())
roses.append("Agdam Gyzyl Uzumu".lower())
roses.append("Chardonnay Rose".lower())
roses.append("Barbera Rose".lower())
roses.append("Chablais Blanc".lower())
roses.append("Chablais Rose".lower())
roses.append("Chablis Blanc".lower())
roses.append("Chablis Rose".lower())
roses.append("Pink Chablis".lower())
roses.append("Rielsing Rose".lower())

print("# REDS: ", len(reds))
print("# WHITES: ", len(whites))
print("# ROSES: ", len(roses))
print("# UNKNOWNS: ", len(unknowns))
df.info()
#df.dtypes
print("Total number of examples: ", df.shape[0])
dupes_series = df.duplicated(df.columns)
print("Number of DUPLICATE examples: ", df[dupes_series].shape[0])
#print("Number of examples with the same title and description: ", df[df.duplicated(['description','title'])].shape[0])
df.isnull().sum()
print(df.nunique())
df["variety"].values
favorites_list = ["bordeaux","cabernet","malbec","tempranillo"]
varieties_str = ' '.join(df["variety"].astype(str).str.lower().values)
any(variety in varieties_str for variety in favorites_list)
favorites_filter = df["variety"].astype(str).str.lower().str.contains("bordeaux|cabernet|malbec|tempranillo")
df[favorites_filter]["variety"].unique()
df.describe()
df.describe(include="all").T
df['points'].plot.hist()
sns.distplot(df["points"])#, kde=False)
print("Skewness: %f" % df["points"].skew())
print("Kurtosis: %f" % df["points"].kurt())
# we want to ignore the NaNs for our assessment; later we will decide how to impute
sns.distplot(df[np.isfinite(df["price"])]["price"])
print("Skewness: %f" % df["price"].skew())
print("Kurtosis: %f" % df["price"].kurt())
df=df.drop_duplicates()
df=df.reset_index(drop=True)
dupes_series = df.duplicated(df.columns)
assert(df[dupes_series].shape[0]==0)
df.dropna(subset=["variety"], inplace=True)
assert(df["variety"].isnull().sum()==0)
df["price"] = df["price"].transform(lambda xx: xx.fillna(xx.mean()))
assert(df["price"].isnull().sum()==0)
sns.pairplot(df[["points","price"]])
def remove_outliers_by_col(df, col, min_zero=True, inplace=False):
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    # for more on IQR, see: https://en.wikipedia.org/wiki/Interquartile_range
    IQR = Q3 - Q1
    # calculate the maximum value and minimum values according to the Tukey rule
    max_value = Q3 + 1.5 * IQR
    min_value = Q1 - 1.5 * IQR
    if min_zero is True:
        # set a floor at zero
        min_value = max(min_value,0)
    # filter the data for price values that are greater than max_value or less than min_value (inclusive)
    filtered = (df[col] <= max_value) & (df[col] >= min_value)
    if inplace is True:
        df = df[filtered]
        return df
    else:
        tmp_df = df.copy()
        tmp_df = df[filtered]
        return tmp_df
prior_count = df.shape[0]
df = remove_outliers_by_col(df,"price")
curr_count = df.shape[0]
if curr_count < prior_count:
    print(prior_count - curr_count, "outliers removed")
print("Skewness: %f" % df["price"].skew())
print("Kurtosis: %f" % df["price"].kurt())
df["points"].describe()
def standardize_by_col(df, col, inplace=False):
    standardized = (df[col] - np.mean(df[col])) / np.std(df[col])
    if inplace is True:
        df[col] = standardized
        return df
    else:
        tmp_df = df.copy()
        tmp_df[str(col+"_standardized")] = standardized
        return tmp_df
df = standardize_by_col(df, "points")
df["points_standardized"].describe()
df = standardize_by_col(df, "price")
df[["points","points_standardized","price","price_standardized"]].head()
# varities, used to map values in Kaggle dataset which do not match Wiki varieties per color OR are present in combination varieties
white_varieties = ["chardonnay","riesling","sauvignon blanc","sauvignon","prosecco","vermentino","turbiana","assyrtico","carricante","insolia","zibibbo","loureiro","tokaji","muskat ottonel","xarel-lo","antão vaz","siria","vidal","vignoles","pinot gris","pedro ximénez","arinto","fernão pires","müller-thurgau","loin de l'oeil","traminette"]
red_varieties = ["pinot noir","cabernet sauvignon","merlot","tempranillo",'barbera', 'port','grenache', 'tempranillo blend', 'garnacha',"pinot nero","pinotage","cabernet sauvignon-merlot","cabernet sauvignon-syrah","malbec-merlot","tinto fino","tinta fina","tinta de toro","carmenère","g-s-m","mencía","cabernet","alicante bouschet","prugnolo gentile","cinsault","lemberger","claret","petite verdot","nero di troia","plavac mali","negrette","norton","carignan"]
# keywords, used when parsing "description"
sparkling_keywords = ["sparkling","champagne","brut"]
rose_keywords = ["rose","rosé","rosato","rosado"]
white_keywords = ["blanc","blanco","blanca","white","champagne","sherry"]
red_keywords = ["red","tinto","tinta","cabernet","bordeaux","malbec","tempranillo","grenache","syrah","shiraz","dark","cherry","blackberry"]
def map_val(val):
    if isinstance(val, str):
        val = val.lower()
        if any(keyword in val for keyword in sparkling_keywords):
            return "Sparkling"
        if any(keyword in val for keyword in rose_keywords):
            return "Rose"
        if any(keyword in val for keyword in white_keywords):
            return "White"
        if any(keyword in val for keyword in red_keywords):
            return "Red"
        return "Unknown"
    return "Unknown"
def map_unknown_varieties(variety, title, description):
    if any(_variety in variety for _variety in white_varieties):
        return "White"
    if any(_variety in variety for _variety in red_varieties):
        return "Red"
    if any(keyword in variety for keyword in sparkling_keywords):
        return "Sparkling"
    if any(keyword in variety for keyword in rose_keywords):
        return "Rose"
    if any(keyword in variety for keyword in white_keywords):
        return "White"
    if any(keyword in variety for keyword in red_keywords):
        return "Red"
    title_mapping = map_val(title)
    if title_mapping is not "Unknown":
        return title_mapping
    description_mapping = map_val(description)
    if description_mapping is not "Unknown":
        return description_mapping
    return "Unknown"
def determine_color(variety, title, description):
    #is_red = False
    #is_white = False
    #is_rose = False
    #is_sparkling = False
    is_red = is_white = is_rose = is_sparkling = None
    variety = variety.lower()
    for rose in roses:                
        if rose in variety or variety in rose:
            is_rose = True
    for white in whites:
        if white in variety or variety in white:
            is_white = True
    for red in reds:               
        if red in variety or variety in red:
            is_red = True
    # variety should only match a single "color", but with the keywords there is sometimes overlap (so invoke "map_unknown_varieties")
    if is_red and not is_white and not is_rose and not is_sparkling:
        return "Red"
    elif not is_red and is_white and not is_rose and not is_sparkling:
        return "White"
    elif not is_red and not is_white and is_rose and not is_sparkling:
        return "Rose"
    elif not is_red and not is_white and not is_rose and is_sparkling:
        return "Sparkling"
    else:
        return map_unknown_varieties(variety, title, description)
df["type"] = df.apply(lambda row: determine_color(row.variety, row.title, row.description), axis=1)
df.head()
print("# REDS: ", df[df["type"]=="Red"].shape[0])
print("# WHITE: ", df[df["type"]=="White"].shape[0])
print("# ROSE: ", df[df["type"]=="Rose"].shape[0])
print("# SPARKLING: ", df[df["type"]=="Sparkling"].shape[0])
print("# UNKNOWN: ", df[df["type"]=="Unknown"].shape[0])
df[df["type"]=="Unknown"].head()
unknowns = df[df["type"]=="Unknown"]["variety"].values
df_unknowns = pd.DataFrame(np.array(unknowns), columns=["variety"])
df_unknowns = df_unknowns.groupby(['variety']).size().reset_index(name='count')
df_unknowns = df_unknowns.sort_values(by="count", ascending=False, inplace=False, kind='quicksort', na_position='last')
print("# unknown: ", df_unknowns.shape[0])
print("# unknown repeated more than 5 times: ", df_unknowns[df_unknowns["count"] > 5].shape[0])
df_unknowns.head()
def determine_style(variety, title, description):
    dry = " dry "
    sweet = " sweet "
    if dry in variety.lower() or dry in title.lower() or dry in description.lower():
        return "Dry"
    elif sweet in variety.lower() or sweet in title.lower() or sweet in description.lower():
        return "Sweet"
    else:
        return "Unknown"
df_style = df.copy()
df_style["style"] = df.apply(lambda row: determine_style(row.variety, row.title, row.description), axis=1)
df_style.head()
print("# DRY: ", df_style[df_style["style"]=="Dry"].shape[0])
print("# SWEET: ", df_style[df_style["style"]=="Sweet"].shape[0])
print("# UNKNOWN: ", df_style[df_style["style"]=="Unknown"].shape[0])
favorites_filter = df["variety"].astype(str).str.lower().str.contains("bordeaux|cabernet|malbec|tempranillo")
df[favorites_filter]["variety"].unique()
df_varieties = df.groupby(['variety']).size().reset_index(name='count')
df_varieties = df_varieties.sort_values(by="count", ascending=False, inplace=False, kind='quicksort', na_position='last')
df_varieties.head()
def merge_variety(variety):
    variety = variety.lower()
    if "pinot" in variety:
        return "Pinot"
    elif "chardonnay" in variety:
        return "Chardonnay"
    elif "blend" in variety:
        return "Blend"
    elif "cabernet" in variety:
        return "Cabernet"
    elif "bordeaux" in variety:
        return "Bordeaux"
    elif any(_variety == variety for _variety in ["tinto fino", "tinta de toro", "tempranillo", "tempranillo blend"]):
        return "Tempranillo"
    elif "sangiovese" in variety:
        return "Sangiovese"
    else:
        return None
df_merged = df.copy()
df_merged["variety_merged"] = df.apply(lambda row: merge_variety(row.variety), axis=1)
df_merged.head()
df_merged_grouped = df_merged.groupby(["variety_merged"]).size().reset_index(name='count')
df_merged_grouped = df_merged_grouped.sort_values(by="count", ascending=False, inplace=False, kind='quicksort', na_position='last')
df_merged_grouped.head()
df_merged["variety_merged"].isnull().sum()
df["variety_merged"] = df_merged["variety_merged"]
df_by_variety = df.groupby(["variety"]).size().reset_index(name="count")
df_by_variety.head()
df_by_variety = df_by_variety[df_by_variety["count"]>250]
df_by_variety.shape
tmp_df = df[df["variety"].isin(df_by_variety["variety"].values)]
tmp_df.head()
tmp_df_avg = tmp_df.groupby(['variety'])["points"].mean().reset_index(name='avg_points')
tmp_df_avg_sorted = tmp_df_avg.sort_values(by="avg_points", ascending=False, inplace=False, kind='quicksort', na_position='last')
tmp_df_avg_sorted.head()
# how to interpret box plots: https://www.wellbeingatschool.org.nz/sites/default/files/W@S_boxplot-labels.png
plt.figure(figsize=(16,6))
sns.boxplot(x = tmp_df["variety"], y = tmp_df["points"], order=tmp_df_avg_sorted["variety"].values)
plt.title("Variety-wise boxplot of points")
plt.xticks(rotation=90);
df[df["variety"]=="Nebbiolo"]["price"].mean()
plt.figure(figsize=(16,6))
sns.boxplot(x = df[df["variety"]=="Nebbiolo"]["price"], y = df[df["variety"]=="Nebbiolo"]["points"])
plt.title("Price-wise boxplot of points")
plt.xticks(rotation=90);
plt.figure(figsize=(16,6))
sns.boxplot(x = df["variety_merged"], y = df["points"])
plt.title("MergedVariety-wise boxplot of points")
plt.xticks(rotation=90);
plt.figure(figsize=(16,6))
sns.boxplot(x = df["type"], y = df["points"])
plt.title("Type-wise boxplot of points")
plt.xticks(rotation=90);
plt.figure(figsize=(16,6))
sns.boxplot(x = df["country"], y = df["points"])
plt.title("Country-wise boxplot of points")
plt.xticks(rotation=90);
df_mex = df[df["country"]=="Mexico"]
df_mex.shape
plt.figure(figsize=(16,6))
sns.boxplot(x = df_mex["type"], y = df_mex["points"])
plt.title("Type-wise boxplot of points (for Mexico)")
plt.xticks(rotation=90);
def plot_scatter(df, col_x, col_y):
    N = df.shape[0]
    colors = np.random.rand(N)
    #area = (30 * np.random.rand(N))**2  # 0 to 15 point radii
    plt.scatter(df[col_x], df[col_y], c=colors, alpha=0.5)
    plt.xlabel(col_x)
    plt.ylabel(col_y)
    plt.show()
plot_scatter(df, "price", "points")
sns.regplot(x="price", y="points", x_estimator=np.mean, data=df)
sns.regplot(x="price", y="points", x_estimator=np.mean, data=df[(df["price"] <= 35) & (df["price"] >= 15)])
sns.regplot(x="price", y="points", x_estimator=np.mean, data=df[(df["price"] <= 60) & (df["price"] >= 20)])
df.shape
winery_filter = df["winery"].astype(str).str.lower().str.contains("cvne")
df_cvne = df[winery_filter]
df_cvne.shape
df_sans_cvne = pd.concat([df, df_cvne, df_cvne]).drop_duplicates(keep=False)
df_sans_cvne.shape
df_cvne["points"].mean()
df_sans_cvne["points"].mean()
df_cvne_reds = df_cvne[df_cvne["type"]=="Red"]
df_cvne_reds.shape
df_sans_cvne_reds = df_sans_cvne[df_sans_cvne["type"]=="Red"]
df_sans_cvne_reds.shape
df_cvne_reds["points"].mean()
df_sans_cvne_reds["points"].mean()
df_cvne["is_cvne"] = True
df_sans_cvne["is_cvne"] = False
df_concat = pd.concat([df_cvne, df_sans_cvne])
plt.figure(figsize=(16,6))
colors = ["#acdffe","#97020e"]
sns.boxplot(x=df_concat["type"], y = df_concat["points"], hue=df_concat["is_cvne"], palette=sns.color_palette(colors))
plt.title("Type-wise boxplot of points (for All Wineries vs CVNE)")
plt.xticks(rotation=90);
df_by_winery = df.groupby(["winery"]).size().reset_index(name="count")
# again, let's require a certain amount of entries: 100
df_by_winery = df_by_winery[df_by_winery["count"]>100]
df_by_winery.shape
tmp_df = df[df["winery"].isin(df_by_winery["winery"].values)]
tmp_df_avg = tmp_df.groupby(["winery"])["points"].mean().reset_index(name="avg_points")
tmp_df_avg_sorted = tmp_df_avg.sort_values(by="avg_points", ascending=False, inplace=False, kind='quicksort', na_position='last')
plt.figure(figsize=(16,6))
sns.boxplot(x = tmp_df["winery"], y = tmp_df["points"], order=tmp_df_avg_sorted["winery"].values)
plt.title("Winery-wise boxplot of points")
plt.xticks(rotation=90);
sns.jointplot(x='price', y='points', data=df, kind='hex', gridsize=20)
df_country = df.groupby(["country"]).mean()["price"].sort_values(ascending=False).to_frame()
plt.figure(figsize=(16,6))
sns.pointplot(x = df_country["price"] , y = df_country.index ,color="#97020e",orient='h',markers='o')
plt.title('Country-wise average wine price')
plt.xlabel("price")
plt.ylabel("country");
df_country.head()
df_price = df[df["price"] <= 50]
tmp_df = df_price[df_price['country'].isin(['US','France', 'Canada', 'Spain'])]
g = sns.FacetGrid(tmp_df, col = "country", col_wrap = 2)
g.map(sns.kdeplot, "price")
!pip install catboost
df["variety_merged"].replace(np.NaN, 'Unknown', inplace=True)
from sklearn.model_selection import train_test_split
from catboost import CatBoostRegressor, Pool

X=df.drop(["points","points_standardized","price_standardized"], axis=1)
X=X.fillna(-1)
categorical_features_indices = categorical_features_indices = np.where(X.dtypes != np.float)[0]
y=df['points']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=52)
def perform_model(X_train, y_train,X_valid, y_valid,X_test, y_test):
    model = CatBoostRegressor(
        random_seed = 400,
        loss_function = 'RMSE',
        iterations=400,
    )
    
    model.fit(
        X_train, y_train,
        cat_features = categorical_features_indices,
        eval_set=(X_valid, y_valid),
        verbose=False,
        #plot=True
    )
    
    print("RMSE on training data: "+ model.score(X_train, y_train).astype(str))
    print("RMSE on test data: "+ model.score(X_test, y_test).astype(str))
    
    return model
model = perform_model(X_train, y_train, X_valid, y_valid, X_test, y_test)
feature_score = pd.DataFrame(list(zip(X.dtypes.index, model.get_feature_importance(Pool(X, label=y, cat_features=categorical_features_indices)))),
                columns=['Feature','Score'])

feature_score = feature_score.sort_values(by='Score', ascending=False, inplace=False, kind='quicksort', na_position='last')
plt.rcParams["figure.figsize"] = (12,7)
ax = feature_score.plot('Feature', 'Score', kind='bar', color='c')
ax.set_title("Catboost Feature Importance Ranking", fontsize = 14)
ax.set_xlabel('')

rects = ax.patches

labels = feature_score['Score'].round(2)

for rect, label in zip(rects, labels):
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width()/2, height + 0.35, label, ha='center', va='bottom')

plt.show()
df["taster_twitter_handle"].nunique()
df.groupby(["taster_twitter_handle"]).size().reset_index(name="count")
df.groupby(["taster_twitter_handle"]).mean()
plt.figure(figsize=(16,6))
sns.boxplot(x = df["taster_twitter_handle"], y = df["points"])
plt.title("TasterTwitterHandle-wise boxplot of points")
plt.xticks(rotation=90);
