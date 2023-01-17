%matplotlib inline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv("../input/AppleStore.csv", index_col = 0)
df.head()
print(df.columns)
df = df.rename(columns = {
    "track_name": "name",
    "rating_count_tot": "ratings_count",
    "rating_count_ver": "ratings_count_vers",
    "user_rating_ver": "user_rating_vers",
    "ver": "version",
    "prime_genre": "genre",
    "sup_devices.num": "sup_devices",
    "ipadSc_urls.num": "screenshots",
    "lang.num": "lang_num",
    "vpp_lic": "vpp_license"
})

print(df.columns)
df.info()
df = df.drop("id", axis = "columns")
print(df.currency.unique())
df = df.drop("currency", axis = "columns")
unique_names = df.name.unique()
print("DataFrame rows:", df.shape[0])
print("Unique names count:", unique_names.size)
df[df.duplicated("name", keep = False)]
df["is_free"] = df.price.apply(lambda price: True if price == 0 else False)
paid_apps = df[~df.is_free]
def ecdf(data):
    """Compute ECDF for a one-dimensional array of measurements."""
    x = np.sort(data)
    y = np.arange(1, len(data) + 1) / len(data)
    return x, y
x_price, y_price = ecdf(paid_apps.price)
plt.plot(x_price, y_price, marker = ".", linestyle = "none")
plt.xticks(range(0, 301, 10), rotation = "vertical")
plt.xlabel("price")
plt.ylabel("ECDF")
plt.show()
paid_apps[paid_apps.price > 100]
prices_under_100 = paid_apps.price < 100
x_price, y_price = ecdf(paid_apps[prices_under_100].price)
plt.plot(x_price, y_price, marker = ".", linestyle = "none")
plt.xticks(range(0, 101, 10), rotation = "vertical")
plt.xlabel("price")
plt.ylabel("ECDF")
plt.show()
plt.figure(figsize = (20, 3))
sns.boxplot(data = paid_apps[prices_under_100], x = "price")
plt.xticks(range(0, 102, 2))
plt.show()
percentage = len(paid_apps[paid_apps.price > 8.99]) / len(paid_apps) * 100
print("The percentage of the apps with price grather than $8.99 is {:.2f}%".format(percentage))
percentage = len(paid_apps[paid_apps.price > 9.99]) / len(paid_apps) * 100
print("The percentage of the apps with price grather than $9.99 is {:.2f}%".format(percentage))
selected_columns = ["name", "price", "ratings_count", "user_rating", "genre", "sup_devices", "lang_num"]
price_outliers = paid_apps[paid_apps.price > 9.99].sort_values(by = "price")
price_outliers[selected_columns]
print(np.percentile(paid_apps.price, [25, 50, 75]))
print(np.percentile(paid_apps.price, [85, 90, 95]))
print(np.percentile(paid_apps.price, [97.5, 99]))
df_original = df.copy()
paid_apps_original = df_original[~df_original.is_free]
free_apps_original = df_original[df_original.is_free]
df = df[df.price <= 9.99] # Remove suspected outliers
paid_apps = df[df.price != 0]
free_apps = df[df.price == 0]
print("Free apps percentage: {:.2f}%".format(len(free_apps) / len(df) * 100))
print("Paid apps percentage: {:.2f}%".format(len(paid_apps) / len(df) * 100))
unique_prices = paid_apps.price.unique()
print("We have {0} unique prices in the iOS store.".format(unique_prices.size))
print(sorted(unique_prices))
sns.countplot(paid_apps.price)
plt.xlabel("price ($)")
plt.show()
plt.figure(figsize = (20, 5))
sns.countplot(paid_apps_original.price)
plt.xticks(rotation = "vertical")
plt.show()
def describe_free_vs_paid(column):
    free_apps_ratings = pd.Series(free_apps[column].describe(), name = column + "_free")
    paid_apps_ratings = pd.Series(paid_apps[column].describe(), name = column + "_paid")
    return pd.concat([free_apps_ratings, paid_apps_ratings], axis = 1)
pd.set_option("display.float_format", lambda x: "%.2f" % x)
describe_free_vs_paid("ratings_count")
for count in range(2, 11):
    filtered_apps = df[df.ratings_count >= count]
    percentage = len(filtered_apps) / len(df) * 100
    print(count, len(filtered_apps), percentage)
len(df[df.ratings_count < 5]) / len(df) * 100
df = df[df.ratings_count >= 5]
free_apps = df[df.is_free]
paid_apps = df[~df.is_free]
df_original = df_original[df_original.ratings_count >= 5]
free_apps_original = df_original[df_original.is_free]
paid_apps_original = df_original[~df_original.is_free]
print(sorted(free_apps.user_rating.unique()))
print(sorted(paid_apps.user_rating.unique()))
describe_free_vs_paid("ratings_count_vers")
print(sorted(free_apps.user_rating_vers.unique()))
print(sorted(paid_apps.user_rating_vers.unique()))
df.cont_rating.unique()
sns.countplot(data = df, x = "cont_rating")
plt.show()
# Percentages
df.groupby("cont_rating").size().sort_values(ascending = False) / len(df) * 100
def print_unique_values_and_their_size(series):
    unique_values = series.genre.unique()
    print(unique_values.size)
    print(sorted(unique_values))
print_unique_values_and_their_size(free_apps)
print_unique_values_and_their_size(paid_apps)
print("Free: ", sorted(free_apps.sup_devices.unique()))
print("Paid: ", sorted(paid_apps.sup_devices.unique()))
free_paid = df.price.apply(lambda x: "Free" if x == 0 else "Paid")
sns.countplot(data = df, x = "sup_devices", hue = free_paid, hue_order = ["Free", "Paid"])
plt.xlabel("supported devices")
plt.show()
len(df[df.sup_devices.isin([37, 38, 40])]) / len(df)
print(sorted(free_apps.screenshots.unique()))
print(sorted(paid_apps.screenshots.unique()))
sns.countplot(data = df, x = "screenshots", hue = free_paid, hue_order = ["Free", "Paid"])
plt.show()
describe_free_vs_paid("screenshots")
print("Free:", sorted(free_apps.lang_num.unique()))
print("Paid:", sorted(paid_apps.lang_num.unique()))
def countplot_lang_num(data):
    plt.figure(figsize = (15, 4))
    sns.countplot(data = data, x = "lang_num", hue = free_paid, hue_order = ["Free", "Paid"])
    plt.xticks(rotation = "vertical")
    plt.legend(loc = "upper right")
    plt.show()
countplot_lang_num(df)
print(len(free_apps[free_apps.lang_num == 1]) / len(free_apps) * 100)
print(len(paid_apps[paid_apps.lang_num == 1]) / len(paid_apps) * 100)
selected_apps = df[(df.lang_num > 1) & (df.lang_num <= 20)]
countplot_lang_num(selected_apps)
selected_apps = df[df.lang_num > 20]
countplot_lang_num(selected_apps)
sns.countplot(data = df[df.lang_num >= 35], x = "genre")
plt.xticks(rotation = "vertical")
plt.show()
selected_columns = ["name", "lang_num", "genre", "is_free", "price", "user_rating", "ratings_count"]
df.sort_values(by = ["lang_num", "ratings_count"], ascending = False).head(10)[selected_columns]
df_google = df[df.name.str.contains("Google", case = False)][selected_columns]
df_google.sort_values(by = ["lang_num", "ratings_count"], ascending = False)
df.vpp_license.unique()
df.vpp_license = df.vpp_license.astype(np.bool)
print(df.vpp_license.dtype)
print(df.vpp_license.unique())
sns.countplot(data = df, x = "vpp_license", hue = "is_free")
plt.show()
selected_apps = df[~df.vpp_license]
sns.countplot(data = selected_apps, x = "vpp_license", hue = "is_free")
plt.show()
df.size_bytes.describe()
ax = sns.distplot(df.size_bytes)
ax.set(xscale = "log")
plt.show()
df["size_mb"] = df.size_bytes / (1024 * 1024)
df = df.drop("size_bytes", axis = "columns")
free_apps = df[df.is_free]
paid_apps = df[~df.is_free]
df_original["size_mb"] = df_original.size_bytes / (1024 * 1024)
df_original = df_original.drop("size_bytes", axis = "columns")
free_apps_original = df_original[df_original.is_free]
paid_apps_original = df_original[~df_original.is_free]
df.size_mb.describe()
plt.figure(figsize = (20, 5))
sns.distplot(free_apps.size_mb, kde = False, label = "free")
sns.distplot(paid_apps.size_mb, kde = False, label = "paid")
plt.xticks(np.arange(0, max(df.size_mb) + 1, 100), rotation = "vertical")
plt.legend()
plt.show()
print("Free apps results:")
print(np.percentile(free_apps.size_mb, [25, 50, 75]))
print(np.percentile(free_apps.size_mb, [85, 90, 95]))
print(np.percentile(free_apps.size_mb, [97.5, 99]))

print()

print("Paid apps results:")
print(np.percentile(paid_apps.size_mb, [25, 50, 75]))
print(np.percentile(paid_apps.size_mb, [85, 90, 95]))
print(np.percentile(paid_apps.size_mb, [97.5, 99]))
free_big_size_apps = free_apps[free_apps.size_mb > np.percentile(free_apps.size_mb, 99)]
print(len(free_big_size_apps))
free_big_size_apps
paid_big_size_apps = paid_apps[paid_apps.size_mb > np.percentile(paid_apps.size_mb, 99)]
print(len(paid_apps[paid_apps.size_mb > np.percentile(paid_apps.size_mb, 99)]))
paid_big_size_apps
selected_columns = ["name", "size_mb", "price", "ratings_count", "user_rating", "genre", 
                    "sup_devices", "screenshots", "lang_num"]
expensive_apps = paid_apps_original[paid_apps_original.price > 9.99]
expensive_apps[selected_columns].nlargest(10, columns = "price")
paid_apps[selected_columns].nlargest(10, columns = ["price", "ratings_count"])
paid_apps[selected_columns].nlargest(10, ["user_rating", "ratings_count"])
free_apps[selected_columns].nlargest(10, ["user_rating", "ratings_count"])
def show_apps_count_by_genre(data, title):
    genres_order = data.groupby('genre').size().sort_values(ascending = False).index
    plt.figure(figsize = (15, 3))
    sns.countplot(data = data, x = "genre", order = genres_order)
    plt.title(title)
    plt.xticks(rotation = 75)
    plt.show()
show_apps_count_by_genre(free_apps, "Free Apps per Genre")
show_apps_count_by_genre(paid_apps, "Paids Apps per Genre")
def show_apps_count_by_genre_with_hue(data):
    genres_order = data.groupby('genre').size().sort_values(ascending = False).index
    free_paid = data.price.apply(lambda x: "Free" if x == 0 else "Paid")
    plt.figure(figsize = (15, 5))
    plt.xticks(rotation = 75)
    sns.countplot(x = "genre", hue = free_paid, hue_order = ["Free", "Paid"], order = genres_order, data = df)
    plt.show()
show_apps_count_by_genre_with_hue(df)
show_apps_count_by_genre_with_hue(df[df.genre != "Games"])
def plot_correlation_matrix(data, title):
    correlation_matrix = data.corr()
    plt.figure(figsize = (10, 6))
    sns.heatmap(correlation_matrix, annot = True)
    plt.title(title)
    plt.show()
plot_correlation_matrix(df, "All Apps: Correlation Matrix")
correlation_matrix_all = df.corr()
correlation_matrix_all["user_rating"].sort_values(ascending = False)
correlation_matrix_all["price"].sort_values(ascending = False)
correlation_matrix_free = free_apps.corr()
correlation_matrix_paid = paid_apps.corr()

plt.figure(figsize = (20, 6))

ax = plt.subplot2grid((1, 2), (0, 0))
sns.heatmap(correlation_matrix_free, annot = True)
plt.title("Free Apps: Correlation Matrix")

ax = plt.subplot2grid((1, 2), (0, 1))
sns.heatmap(correlation_matrix_paid, annot = True)
plt.title("Paid Apps: Correlation Matrix")

plt.show()
correlation_matrix_paid["price"].sort_values(ascending = False)
correlation_matrix_free["user_rating"].sort_values(ascending = False)
correlation_matrix_paid["user_rating"].sort_values(ascending = False)
def get_free_vs_paid_corr(column, correlation_matrix_free, correlation_matrix_paid):
    corr_user_rating = pd.concat([correlation_matrix_free[column], correlation_matrix_paid[column]], axis = 1)
    corr_user_rating.columns = [column + "_corr_free", column + "_corr_paid"]
    return corr_user_rating
free_vs_paid_corr = get_free_vs_paid_corr("user_rating", correlation_matrix_free, correlation_matrix_paid)
free_vs_paid_corr
def plot_mean_of_column_vs_grouped_column(data, mean_column, group_by, title):
    grouped = data.groupby(group_by)[mean_column].mean()
    grouped = grouped.sort_values(ascending = False)
    grouped = grouped.reset_index()
    grouped.columns = [group_by, "mean_" + mean_column]

    plt.figure(figsize = (20, 5))
    ax = sns.barplot(data = grouped, x = group_by, y = "mean_" + mean_column)
    plt.title(title)
    plt.xlabel(str.title(" ".join(group_by.split("_"))))
    plt.ylabel("Mean " + str.title(" ".join(mean_column.split("_"))))
    plt.xticks(rotation = 90)
    plt.show()
plot_mean_of_column_vs_grouped_column(data = free_apps, mean_column = "user_rating", group_by = "genre",
                                      title = "Free Apps: Mean User Rating Per Genre")
plot_mean_of_column_vs_grouped_column(data = paid_apps, mean_column = "user_rating", group_by = "genre",
                                      title = "Paid Apps: Mean User Rating Per Genre")
plot_mean_of_column_vs_grouped_column(data = free_apps_original, mean_column = "user_rating", group_by = "genre",
                                      title = "Free Apps: Mean User Rating Per Genre (With Outliers)")
plot_mean_of_column_vs_grouped_column(data = paid_apps_original, mean_column = "user_rating", group_by = "genre",
                                      title = "Paid Apps: Mean User Rating Per Genre (With Outliers)")
genre_order = df.groupby("genre").size().sort_values(ascending = False).index
df.groupby(["genre", "is_free"]).size().unstack(level = -1).reindex(genre_order)
def plot_mean_of_column_per_genre_with_hue(data, column, title):
    grouped = data.groupby(["genre", "is_free"])[column].mean()
    grouped = grouped.reset_index()
    grouped.columns = ["genre", "is_free", "mean_" + column]

    genres_order = list(data.groupby("genre").size().sort_values(ascending = False).index)
    free_paid = grouped.is_free.apply(lambda x: "Free" if x else "Paid")
    plt.figure(figsize = (20, 4))
    ax = sns.barplot(data = grouped, x = "genre", y = "mean_" + column, 
                     hue = free_paid, hue_order = ["Free", "Paid"], order = genres_order)
    ax.legend_.set_title("Price")
    plt.legend(loc = "lower center")
    plt.title(title)
    plt.xlabel("Genre")
    plt.ylabel("Mean " + str.title(" ".join(column.split("_"))))
    plt.xticks(rotation = 90)
    plt.show()
without_shopping = df[df.genre != "Shopping"]
plot_mean_of_column_per_genre_with_hue(without_shopping, "user_rating", "Mean User Rating Per Genre")
without_shopping = df_original[df_original.genre != "Shopping"]
plot_mean_of_column_per_genre_with_hue(without_shopping, "user_rating", "Mean User Rating Per Genre")
plt.figure(figsize = (20, 5))
ax = sns.stripplot(data = df, y = "user_rating", x = "genre", jitter = True)
ax.set_xticklabels(ax.get_xticklabels(), rotation = 90)
plt.show()
plot_mean_of_column_vs_grouped_column(data = free_apps, mean_column = "ratings_count", group_by = "genre",
                                      title = "Free Apps: Mean Rating Count Per Genre")
plot_mean_of_column_vs_grouped_column(data = paid_apps, mean_column = "ratings_count", group_by = "genre",
                                      title = "Paid Apps: Mean Rating Count Per Genre")
plot_mean_of_column_per_genre_with_hue(df, "ratings_count", "Mean Ratings Count Per Genre")
plt.figure(figsize = (20, 5))
sns.boxplot(data = df, x = "sup_devices")
plt.show()
plot_mean_of_column_vs_grouped_column(data = free_apps, mean_column = "user_rating", group_by = "sup_devices",
                                     title = "Free Apps: Mean User Rating vs Supported Devices Count")
plot_mean_of_column_vs_grouped_column(data = paid_apps, mean_column = "user_rating", group_by = "sup_devices",
                                     title = "Paid Apps: Mean User Rating vs Supported Devices Count")
plt.figure(figsize = (20, 5))
free_paid = df.price.apply(lambda x: "Free" if x == 0 else "Paid")
sns.countplot(data = df, x = "sup_devices", hue = free_paid, hue_order = ["Free", "Paid"])
plt.xlabel("supported devices")
plt.show()
famous_supported_devices_counts = [37, 38, 40]
len(df[df.sup_devices.isin(famous_supported_devices_counts)]) / len(df) * 100
data = free_apps[free_apps.sup_devices.isin(famous_supported_devices_counts)]
plot_mean_of_column_vs_grouped_column(data = data, mean_column = "user_rating", group_by = "sup_devices",
                                      title = "Free Apps: Mean User Rating vs Supported Devices Count")

data = paid_apps[paid_apps.sup_devices.isin(famous_supported_devices_counts)]
plot_mean_of_column_vs_grouped_column(data = data, mean_column = "user_rating", group_by = "sup_devices",
                                     title = "Paid Apps: Mean User Rating vs Supported Devices Count")
def box_plot(data, x, y, title):
    plt.figure(figsize = (20, 5))
    sns.boxplot(data = data, x = x, y = y)
    plt.title(title)
    plt.show()
box_plot(data = free_apps, x = "sup_devices", y = "user_rating", title = "Free Apps: Supported Devices Count vs User Rating")
box_plot(data = paid_apps, x = "sup_devices", y = "user_rating", title = "Paid Apps: Supported Devices Count vs User Rating")
def violin_plot(data, x, y, title):
    plt.figure(figsize = (20, 5))
    sns.violinplot(data = data, x = x, y = y)
    plt.title(title)
    plt.show()
violin_plot(data = free_apps, x = "sup_devices", y = "user_rating", title = "Free Apps: Supported Devices Count vs User Rating")
violin_plot(data = paid_apps, x = "sup_devices", y = "user_rating", title = "Paid Apps: Supported Devices Count vs User Rating")
data = free_apps[free_apps.sup_devices.isin(famous_supported_devices_counts)]
violin_plot(data = data, x = "sup_devices", y = "user_rating", title = "Free Apps: Supported Devices Count vs User Rating")

data = paid_apps[paid_apps.sup_devices.isin(famous_supported_devices_counts)]
violin_plot(data = data, x = "sup_devices", y = "user_rating", title = "Paid Apps: Supported Devices Count vs User Rating")
def stripplot_sup_devices_vs_user_rating(data, title):
    plt.figure(figsize = (20, 5))
    sns.stripplot(data = data, x = "sup_devices", y = "user_rating", jitter = True)
    plt.title(title)
    plt.xlabel("Supported Devices Count")
    plt.ylabel("User Rating")
    plt.show()
stripplot_sup_devices_vs_user_rating(free_apps, "Free apps: User Rating vs Supported Devices")
stripplot_sup_devices_vs_user_rating(paid_apps, "Paid apps: User Rating vs Supported Devices")
plt.figure(figsize = (20, 5))
plt.scatter(df.lang_num, df.user_rating)
plt.title("User Rating vs Supported Languages Count")
plt.xlabel("Supported Languages Count")
plt.ylabel("User Rating")
plt.show()
plot_mean_of_column_vs_grouped_column(data = free_apps, mean_column = "user_rating", group_by = "lang_num",
                                     title = "Free Apps: Mean User Rating vs Supported Languages Count")
plot_mean_of_column_vs_grouped_column(data = paid_apps, mean_column = "user_rating", group_by = "lang_num",
                                     title = "Paid Apps: Mean User Rating vs Supported Languages Count")
box_plot(free_apps, x = "lang_num", y = "user_rating", title = "Free Apps: User Rating vs Supported Languges Count")
box_plot(paid_apps, x = "lang_num", y = "user_rating", title = "Free Apps: User Rating vs Supported Languges Count")
plt.figure(figsize = (20, 5))
sns.countplot(data = df, x = "lang_num")
plt.title("Count plot based on supported languages count")
plt.show()
len(df[df.lang_num <= 18]) / len(df) * 100
violin_plot(data = free_apps[free_apps.lang_num <= 18], x = "lang_num", y = "user_rating",
           title = "Free Apps: User Rating vs Supported Languages Count")

violin_plot(data = paid_apps[paid_apps.lang_num <= 18], x = "lang_num", y = "user_rating",
           title = "Paid Apps: User Rating vs Supported Languages Count")
plt.figure(figsize = (17, 8))
sns.violinplot(data = df, y = "lang_num", x = "user_rating")
plt.show()
def stripplot_lang_count_vs_user_rating(data, title):
    plt.figure(figsize = (20, 5))
    sns.stripplot(data = data, x = "lang_num", y = "user_rating", jitter = True)
    plt.title(title)
    plt.xlabel("Supported Languages Count")
    plt.ylabel("User Rating")
    plt.show()
stripplot_lang_count_vs_user_rating(free_apps, "Free apps: User Rating vs Supported Languages")
stripplot_lang_count_vs_user_rating(paid_apps, "Paid apps: User Rating vs Supported Languages")
apps_with_0_languages = df[df.lang_num == 0]
print(len(df[df.lang_num == 0]))
apps_with_0_languages.head()
sns.lmplot(data = df, x = "size_mb", y = "user_rating", col = "is_free", fit_reg = False)
plt.show()
sns.lmplot(data = paid_apps, x = "price", y = "user_rating", fit_reg = False)
plt.show()
sns.lmplot(data = paid_apps, x = "price", y = "user_rating", col = "cont_rating", fit_reg = False)
plt.show()
df.groupby(["cont_rating", "is_free"]).size()
plot_mean_of_column_vs_grouped_column(data = paid_apps, mean_column = "price", group_by = "genre",
                                     title = "Paid Apps: Mean Price Per Genre")
paid_apps[paid_apps.genre == "Catalogs"]
data = paid_apps[paid_apps.genre != "Catalogs"]
plot_mean_of_column_vs_grouped_column(data = data, mean_column = "price", group_by = "genre",
                                     title = "Paid Apps: Mean Price Per Genre")