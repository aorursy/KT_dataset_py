#basic libraries
import pandas as pd
import numpy as np
import warnings
!pip install ppscore
warnings.filterwarnings("ignore")
import ppscore as pps
import ast
from tqdm.notebook import tqdm
import math
from collections import Counter

#visualization
import seaborn as sns
from matplotlib import pyplot as plt
from wordcloud import WordCloud
plt.style.use("ggplot")

#statistical analysis & machine learning
from sklearn.cluster import KMeans as KM
from sklearn.metrics import silhouette_score as score
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.model_selection import train_test_split as splitter
from sklearn.model_selection import cross_val_score as validator
from statsmodels.stats import power as sms
from scipy.stats import pearsonr, shapiro, ttest_ind, f_oneway, levene

#text preprocessing
import nltk
from collections import Counter
import string
from nltk.corpus import stopwords
#string representation of list -> list
def str2list(x):
    try:
        return ast.literal_eval(x)
    except:
        return np.nan

#accuracy metric
def accuracy(true, pred):
    return sum(true == pred)/len(pred)

#calculate accuracy of knn model (imputing algorithm 1)
def knn_score(k, X_known, y_known):
    model = KNN(n_neighbors = k)
    return validator(model, X_known, y_known, cv=5)

#calculate accuracy of many models
def many_knn_score(start_n, end_n, X_known, y_known):
    scores = []
    for i in range(start_n, end_n):
        scores.append(knn_score(i, X_known, y_known).mean())
    progress = pd.Series(scores, index = np.arange(start_n, end_n))
    fig = plt.figure(figsize = (10, 10))
    ax = fig.subplots()
    progress.plot(ax=ax, kind="line")
    ax.set_ylabel("accuracy")
    ax.set_xlabel("n_neighbors")
    plt.show()

#cosine similarity
def cosine_similarity(x1, x2):
    return np.dot(x1, x2)/(np.linalg.norm(x1)*np.linalg.norm(x2))

#imputing algorithm 2
def similarity_algorithm(X_train):
    similarity_lists = []
    for x in tqdm(range(len(X_train))):
        similarity_list = []
        for genre in range(len(genre_profile)):
            similarity = cosine_similarity(X_train.iloc[x], genre_profile.iloc[genre])
            similarity_list.append(similarity)
        similarity_lists.append(similarity_list)
    df = pd.DataFrame(similarity_lists, columns=popular_genres) 
    return df.transpose()

#silhouette score
def calc_silhouette(X, y):
    return score(X, y)

#determine clusters and find out how good model is using silhouette score, t-test and/or anova
def k_means(data, n_clusters, n_components=2):
    model = KM(n_clusters=n_clusters)
    model.fit(df_std)
    preds = model.predict(data)
    decomposer = PCA(n_components=n_components)
    decomposer.fit(df_std)
    data_for_plot = decomposer.transform(data)
    cols = ["x"+str(x+1) for x in range(n_components)]
    df_genre_profile = pd.DataFrame(data_for_plot, index=data.index, columns=cols)
    df_genre_profile["cluster"] = preds
    cluster_score = calc_silhouette(data, preds)
    tester = Test(alpha=0.05)
    anova_score = tester.anova(*[df_genre_profile[df_genre_profile["cluster"] == x]["x1"].values for x in range(n_clusters)])
    anova_score = anova_score["anova_stat"] if anova_score["test_is_accepted"] else None
    ttest_score = None
    if n_clusters == 2:
        ttest_score = tester.ttest(df_genre_profile[df_genre_profile["cluster"] == 0]["x1"].values,
                                   df_genre_profile[df_genre_profile["cluster"] == 1]["x1"].values)
        ttest_score = ttest_score["ttest_stat"] if ttest_score["test_is_accepted"] else None
    return df_genre_profile, cluster_score, anova_score, ttest_score

#plot a kmeans model
def plot_k_means(data, n_clusters):
    df_genre_profile, cluster_score, anova_score, ttest_score = k_means(data=data, n_clusters=n_clusters)
    rgb_colormap = np.random.randint(0, 255, size=(n_clusters, 3))/255
    rgb_values = rgb_colormap[df_genre_profile["cluster"]]
    
    fig = plt.figure(figsize = (10, 10))
    ax = fig.subplots()
    df_genre_profile.plot(ax=ax, x="x1", y="x2", kind = "scatter", c = rgb_values)
    string = f"n_clusters: {n_clusters}, silhouette: {cluster_score:.4f}"
    if anova_score:
        string += f", t-test: {list(test_score)[0]:.4f}"
    if ttest_score:
        string += f", anova: {list(anova_score)[0]:.4f}"
    ax.set_title(string)
    return ax

#plot many k-means models based on different n_clusters
def k_many_clusters(data, start_n, end_n):
    for i in range(start_n, end_n):
        plot_k_means(data, i)
        plt.show()
        
#Test class includes both ttest and anova tests
class Test():
    def __init__(self, alpha, power=0.90, only_result=True, ind_limit=0.20):
        """alpha and power required for identifying the min. sample size
        and ind_limit that defines the dependence using correlation coefficient"""
        self.alpha = alpha
        self.power = power
        self.only_result = only_result
        self.ind_limit = ind_limit
    def ttest(self, a, b):
        """min. sample size, shapiro, pearsonr and ttest, and their corresponding p-values"""
        only_result = self.only_result
        power, alpha = self.power, self.alpha
        p_a = a.mean()
        p_b = b.mean()
        n_a = len(a)
        n_b = len(b)
        effect_size = (p_b-p_a)/a.std()
        n_req = int(sms.TTestPower().solve_power(effect_size=effect_size, power=power, alpha=alpha))
        if len(a) > len(b):
            a = a[:len(b)]
        elif len(a) < len(b):
            b = b[:len(a)]
        stat1, p1 = shapiro(a)
        stat2, p2 = shapiro(b)
        stat3, p3 = pearsonr(a, b)
        stat4, p4 = ttest_ind(b, a)
        
        result_dict = {"power": power, "alpha": alpha, "n_req": n_req,
                       "n_control": n_a, "n_test": n_b, "shapiro_control_stat": stat1,
                       "shapiro_control_p": p1, "shapiro_test_stat": stat2, "shapiro_test_p": p2,
                       "pearsonr_stat": stat3, "pearsonr_p": p3, "ttest_stat": stat4,
                       "ttest_p": p4, "ind_limit": self.ind_limit, "very_low_number": n_req > n_a or n_req > n_b,
                       "control_is_normal": alpha < p1, "test_is_normal": alpha < p2,
                       "very_low_correlation": self.ind_limit > abs(stat3), "very_high_dependence": p3 < alpha,
                       "no_difference": p4 > alpha, "test_is_bigger": stat4 > 0, "control_is_bigger": stat4 < 0}
        
        accepted = all([
            not(result_dict["very_low_number"]),
            (result_dict["control_is_normal"] and result_dict["test_is_normal"]),
            (not(result_dict["very_high_dependence"]) or result_dict["very_low_correlation"])
        ])
        
        result_dict.update({"test_is_accepted": accepted})
        result_dict = {key: result_dict[key] for key in ["ttest_stat", "ttest_p", "test_is_accepted"]} if only_result else result_dict
        return result_dict
    def anova(self, *args):
        """shapiro, levene, one-way anova and their corresponding p-values"""
        only_result = self.only_result
        alpha = self.alpha
        normality = [shapiro(x) for x in args]
        every_group_is_normal = True if all([x[1] > alpha for x in normality]) else False
        stat1, p1 = levene(*args)
        equal_variance = False if(p1 < alpha) else True
        stat2, p2 = f_oneway(*args)
        equal_means = False if (p2 < alpha) else True
        accepted = all([every_group_is_normal, equal_variance, not(equal_means)])
        result_dict = {"alpha": alpha, "normality":every_group_is_normal, "levene_stat":stat1, "levene_p": p1,
                       "homogenity": equal_variance, "anova_stat": stat2, "anova_p": p2, 
                       "groups_are_different": not(equal_means), "test_is_accepted": accepted}
        result_dict = {key: result_dict[key] for key in ["anova_stat", "anova_p", "test_is_accepted"]} if only_result else result_dict
        return result_dict
    
#predictions over new data
def predict_cluster(sample):
    sample = sample.copy()
    for i in range(len(in_cols)):
        col = in_cols[i]
        sample[col] = (sample[col]-genre_means[i])/genre_stds[i]
    df_genre_profile = k_means(sample, 4)[0]
    return df_genre_profile["cluster"].to_dict()

#generate fake name for clusters
def create_cluster_name(artists_clusters):
    counts = []
    artists_data = [artists_clusters.query("cluster == "+str(x)).index for x in range(4)]
    for data in artists_data:
        data_string = " ".join(data).lower()
        tokens = nltk.word_tokenize(data_string)
        stopset = set(stopwords.words('english') + list(string.punctuation) + ["orchestra", "band", "symphony"])
        data = [token for token in tokens if token not in stopset and len(token) > 2]
        count = pd.Series(dict(Counter(data)))
        counts.append(count.sort_values(ascending=False)[:3].to_dict())
    name_dict = {}
    for i, count in enumerate(counts):
        indices = np.random.permutation(len(count))
        count = np.array(list(count.keys()))
        count = count[indices]
        genre_name = " ".join(count)
        name_dict.update({"Cluster "+str(i): genre_name})
    return name_dict
df = pd.read_csv("/kaggle/input/spotify-dataset-19212020-160k-tracks/data_by_genres.csv")
df_2 = pd.read_csv("/kaggle/input/spotify-dataset-19212020-160k-tracks/data_w_genres.csv")
out_cols = ["genres", "artists", "mode", "count", "key"]
in_cols = [x for x in df.columns if x not in out_cols] 

df = df.set_index("genres")[in_cols].drop("[]", 0)
df #genre data
#fill nan values by 0
df_2.set_index("artists", inplace=True)
df_2["genres"][df_2["genres"] == "[]"] = np.nan
df_2["genres"] = df_2["genres"].fillna(0)
df_2
#standardize data
df_2_std = df_2.copy()
for col in in_cols:
    df_2_std[col] = (df_2[col]-df_2[col].mean())/df_2[col].std()
       
#extract individual genres from genre lists
df_2_std.reset_index(inplace = True)
collist = list(df_2_std.columns)
new_rows = []
for index in tqdm(range(len(df_2_std))):
    row = df_2_std.iloc[index]
    genre_list = str2list(row["genres"])
    row = pd.DataFrame(row).transpose()
    if(not(isinstance(genre_list, list) and len(genre_list) != 0)):
        pass
    else:
        if(len(genre_list) == 1):
            row["genres"] = genre_list[0]
            new_rows.append(list(row.values[0]))
        else:
            row = pd.concat([row for i in range(len(genre_list))], 0)
            row["genres"] = genre_list
            for i in range(len(genre_list)):
                new_rows.append(list(row.values[i]))
                
df_known = pd.DataFrame(new_rows, columns = collist)
#export
df_known.to_csv("data_each_genres.csv")
df_known
X_known = df_known[in_cols]
y_known = df_known["genres"]
#missing data
df_unknown = df_2_std[df_2_std["genres"] == 0]
df_unknown
X_unknown = df_unknown[in_cols]
y_unknown = df_unknown["genres"]
correlations = pps.matrix(df_known.reset_index()[["artists", "genres"]])

fig = plt.figure(figsize=(5, 5))
ax = fig.subplots()
sns.heatmap(pd.DataFrame(correlations["ppscore"].values.reshape(2, 2),
                         columns = ["artists", "genres"], index = ["artists", "genres"]),
                         cmap = "Wistia", axes = ax)
ax.set_title("Predictive Power Score of Artists and Genres")
plt.show()
y_known.value_counts()[:25].to_dict()
fig = plt.figure(figsize = (10, 10))
ax = fig.subplots()
y_known.value_counts()[:25].plot(ax=ax, kind = "pie")
ax.set_ylabel("")
ax.set_title("Top 25 most popular genres")
plt.show()
max_words = 400
wordcloud = WordCloud(width = 800, height = 800, 
                background_color ='white', max_words = max_words, colormap="viridis",
                min_font_size = 10).generate(" ".join(df.index))

plt.figure(figsize=(10, 10))
plt.imshow(wordcloud)
plt.axis("off") 
plt.tight_layout(pad = 0)
plt.title(f"The most {max_words} frequent words in genres")
plt.show()
fig = plt.figure(figsize=(10, 10))
ax = fig.subplots()
ax.set_title("Top 25 artists having the most genres")
ax.set_ylabel("Genres")
ax.set_xlabel("Artists")
df_known["artists"].value_counts()[:25].plot(ax=ax, kind="bar")
plt.show()
many_knn_score(1, 6, X_known, y_known)
popular_genres = list(y_known.value_counts()[:25].index)
df_known_w_populars = df_known[df_known["genres"].isin(popular_genres)]
X_known_w_populars = df_known_w_populars[in_cols]
y_known_w_populars = df_known_w_populars["genres"]

many_knn_score(1, 26, X_known_w_populars, y_known_w_populars)
many_knn_score(250, 260, X_known_w_populars, y_known_w_populars)
genre_profile = df_known_w_populars[["genres", *in_cols]].groupby("genres").mean()
similarity_matrix=similarity_algorithm(X_known)

preds=list(similarity_matrix.index[similarity_matrix.values.argmax(axis=0)])

accuracy(preds, y_known)
genre_means, genre_stds = [], []
df_std = df.copy()
for col in in_cols:
    mean = df_std[col].mean()
    std = df_std[col].std()
    genre_means.append(mean)
    genre_stds.append(std)
    df_std[col] = (df_std[col] - mean) / std
df_std
k_many_clusters(df_std, 2, 10)
i = 2
tester = Test(alpha = 0.05, only_result=False)
df_genre_profile = k_means(df_std, 2)[0]
ttest_result = tester.ttest(df_genre_profile[df_genre_profile["cluster"] == 0]["x1"].values, df_genre_profile[df_genre_profile["cluster"] == 1]["x1"].values)

ttest_result
anova_results = []
for i in range(2, 9):
    tester = Test(alpha=0.05, only_result=False)
    df_genre_profile = k_means(df_std, i)[0]
    test_data = [df_genre_profile[df_genre_profile["cluster"] == x]["x1"] for x in range(i)]
    anova_results.append(tester.anova(*test_data))
    
pd.DataFrame(anova_results)
#predictions over new samples
#e.g. predict_cluster(df.iloc[:350])

#prediction over all data
{k: v for i, (k, v) in enumerate(predict_cluster(df).items()) if i%100==0}
pred = predict_cluster(df)

df_known_new = df_known.copy()
df_known_new["cluster"] = df_known_new["genres"].map(lambda x: pred[x])
#artists - clusters (group by artists and find the most frequent clusters)
artists_clusters = df_known_new[["cluster", "artists"]].groupby("artists").agg(lambda x: x.value_counts().index[0])
artists_clusters.value_counts(normalize=True)
fig = plt.figure(figsize = (10, 10))
ax = fig.subplots()
ax.set_title("The distribution of clusters w.r.t. artists")
ax.set_xlabel("Clusters")
artists_clusters.value_counts().plot(ax=ax, kind="pie", ylabel="Percentage", legend=True)
plt.show()
create_cluster_name(artists_clusters)
