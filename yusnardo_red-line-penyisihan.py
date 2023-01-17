import numpy

import scipy

import matplotlib

import sklearn

import xgboost

import imblearn
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import random as rnd

from scipy import stats

import copy 

import warnings

warnings.filterwarnings('ignore')



# visualization

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



# machine-learning

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, RandomizedSearchCV

from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score

from sklearn.feature_selection import RFE, RFECV

from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, VotingClassifier, GradientBoostingClassifier

from xgboost import XGBClassifier

from imblearn.over_sampling import SMOTE

from sklearn.ensemble import BaggingClassifier



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        

flight = pd.read_csv('../input/datavidia2019/flight.csv')

df_test = pd.read_csv('../input/datavidia2019/test.csv')

hotel = pd.read_csv('../input/datavidia2019/hotel.csv')
# Melihat ukuran dari data

print('Ukuran data flight adalah', flight.shape)

print('Ukuran data test adalah', df_test.shape)
print('Pada dataset flight terdapat beberapa feature, yaitu: ')

print('\n')

num_features = flight.select_dtypes(['float64', 'int64']).columns.tolist()

cat_features = flight.select_dtypes(['object']).columns.tolist()

print('{} numerical features:\n{} \n{} categorical features:\n{}'.format(len(num_features), num_features, len(cat_features), cat_features))

print('\n')

print('\n')

print('Sedangkan, pada dataset test terdapat beberapa feature, yaitu: ')

print('\n')

num_features = df_test.select_dtypes(['float64', 'int64']).columns.tolist()

cat_features = df_test.select_dtypes(['object']).columns.tolist()

print('{} numerical features:\n{} \n{} categorical features:\n{}'.format(len(num_features), num_features, len(cat_features), cat_features))
flight.head()
flight.tail()
flight.describe()
df_test.describe()
for col in flight.columns:

    print('Nilai unik pada feature', col, 'adalah')

    print(flight[col].value_counts())

    print('\n')
for col in df_test.columns:

    print('Nilai unik pada feature', col, 'adalah')

    print(df_test[col].value_counts())

    print('\n')
# Membuat feature cross_sell berdasarkan ada atau tidak adanya hotel id

var_target = []

for i in range(0, len(flight.hotel_id)):

    val = flight.hotel_id[i]

    if val != 'None':

        var_target.append(1)

    else:

        var_target.append(0)



flight['cross_sell'] = pd.Series(np.array(var_target))

flight.drop(['hotel_id'], axis=1, inplace = True)
used_library = [

    numpy, pd, scipy, sns, matplotlib, sklearn, xgboost, imblearn

]

print('Library yang digunakan pada kernel ini adalah sebagai berikut.\n')

for library in used_library:

    print('Library', library.__name__, 'dengan versi', library.__version__)
# Code untuk mengecek baris data mana yang terdapat missing value pada variabel gender

missing_index = []

for i in range(0, len(flight.gender)):

    val = flight.gender[i]

    if val != 'M' and val != 'F':

        missing_index.append(i)

        #print(i)

    else:

        continue



missing_index_no_cs = []

missing_index_with_cs = []

for i in missing_index:

    if flight.iloc[i].cross_sell == 0:

        missing_index_no_cs.append(i)

        print('data flight index {} yang terdapat missing value tidak terjadi cross selling'.format(i))

    else:

        missing_index_with_cs.append(i)

        print('data flight index {} yang terdapat missing value terjadi cross selling'.format(i))





# Menghapus data yang missing value pada gender dan tidak terjadi cross selling

flight = flight.drop(missing_index_no_cs, axis = 0)

flight.reset_index(drop=True, inplace=True)



# Memasukan nilai modus pada missing value gender yang terjadi cross selling. 

# Dilakukan secara manual karena hanya terdapat 2 baris yang memiliki kondisi tersebut.

flight.gender[65919, 106169] = 'M'
def ubah_visited_city(dataset):

    """ Docstring

    Fungsi ini digunakan untuk mengubah feature visited_city menjadi bentuk yang lebih berguna.

    Pada fungsi ini juga dilakukan pembuatan array yang mengembalikan nilai apakah pelanggan pernah

    berkunjung di Semarang, Jogja, Surabaya, Aceh, atau Manado.

    

    Fungsi ini hanya berguna pada dataset yang digunakan dalam penyisihan Datavidia 2019, menerima 

    input parameter dataset dalam hal ini flight dan dataset test.

    """

    visit_semarang = []

    visit_jogja = []

    visit_sby = []

    visit_aceh = []

    visit_manado = []

    for i in range(0, len(dataset.visited_city)):

        val = dataset.visited_city[i]

        val = val.replace('[', '').replace(']', '').replace("'", '').split(',')

        if 'Semarang' in val:

            visit_semarang.append(1)

        else:

            visit_semarang.append(0)

        if 'Jogjakarta' in val:

            visit_jogja.append(1)

        else:

            visit_jogja.append(0)

        if 'Surabaya' in val:

            visit_sby.append(1)

        else:

            visit_sby.append(0)

        if 'Aceh' in val:

            visit_aceh.append(1)

        else:

            visit_aceh.append(0)

        if 'Manado' in val:

            visit_manado.append(1)

        else:

            visit_manado.append(0)

    return visit_semarang, visit_jogja, visit_sby, visit_aceh, visit_manado



def ubah_log_transaction(dataset):

    """ Docstring

    Fungsi ini digunakan untuk mengubah feature log_transaction menjadi bentuk yang lebih berguna.

    Pada fungsi ini juga dilakukan pembuatan array yang mengembalikan nilai jumlah, total, dan rata-rata

    transaksi. Selain itu, fungsi ini juga mengembalikan nilai binary apakah pelanggan telah 

    bertransaksi lebih dari 10 juta, 50 juta, atau bahkan 100 juta.

    

    Fungsi ini hanya berguna pada dataset yang digunakan dalam penyisihan Datavidia 2019, menerima 

    input parameter dataset dalam hal ini flight dan dataset test.

    """

    var_jumlah = []

    var_total = []

    var_mean = []

    var_have_spend_morethan_10m = []

    var_have_spend_morethan_50m = []

    var_have_spend_morethan_100m = []

    for i in range(0, len(dataset.log_transaction)):

        val = dataset.log_transaction[i]

        val = val.replace('[', '').replace(']', '').replace("'", '').split(',')

        tot_transaksi = sum([float(x) for x in val])

        mean = round(tot_transaksi/len(val), 2)

        if tot_transaksi >= 10000000:

            var_have_spend_morethan_10m.append(1)

        else:

            var_have_spend_morethan_10m.append(0)

        if tot_transaksi >= 50000000:

            var_have_spend_morethan_50m.append(1)

        else:

            var_have_spend_morethan_50m.append(0)

        if tot_transaksi >= 100000000:

            var_have_spend_morethan_100m.append(1)

        else:

            var_have_spend_morethan_100m.append(0)

        var_jumlah.append(len(val))

        var_total.append(tot_transaksi)

        var_mean.append(mean)

        

    return var_jumlah, var_total, var_mean, var_have_spend_morethan_10m, var_have_spend_morethan_50m, var_have_spend_morethan_100m
# Menggunakan fungsi ubah_visited_city untuk membuat feature baru.

visit_semarang, visit_jogja, visit_sby, visit_aceh, visit_manado = ubah_visited_city(flight)

flight['have_visit_srg'] = pd.Series(np.array(visit_semarang))

flight['have_visit_jogc'] = pd.Series(np.array(visit_jogja))

flight['have_visit_sby'] = pd.Series(np.array(visit_sby))

flight['have_visit_aceh'] = pd.Series(np.array(visit_aceh))

flight['have_visit_mdc'] = pd.Series(np.array(visit_manado))



visit_semarang, visit_jogja, visit_sby, visit_aceh, visit_manado = ubah_visited_city(df_test)        

df_test['have_visit_srg'] = pd.Series(np.array(visit_semarang))

df_test['have_visit_jogc'] = pd.Series(np.array(visit_jogja))

df_test['have_visit_sby'] = pd.Series(np.array(visit_sby))

df_test['have_visit_aceh'] = pd.Series(np.array(visit_aceh))

df_test['have_visit_mdc'] = pd.Series(np.array(visit_manado))



# Menggunakan fungsi ubah_log_transaction untuk membuat feature baru.

jumlah, total, mean, have_spend_morethan_10m, have_spend_morethan_50m, have_spend_morethan_100m = ubah_log_transaction(flight)

flight['jumlah_transaksi'] = pd.Series(np.array(jumlah))

flight['mean_transaksi'] = pd.Series(np.array(mean))

flight['total_transaksi'] = pd.Series(np.array(total))

flight['have_spend_10m'] = pd.Series(np.array(have_spend_morethan_10m))

flight['have_spend_50m'] = pd.Series(np.array(have_spend_morethan_50m))

flight['have_spend_100m'] = pd.Series(np.array(have_spend_morethan_100m))



jumlah, total, mean, have_spend_morethan_10m, have_spend_morethan_50m, have_spend_morethan_100m = ubah_log_transaction(df_test)

df_test['jumlah_transaksi'] = pd.Series(np.array(jumlah))

df_test['mean_transaksi'] = pd.Series(np.array(mean))

df_test['total_transaksi'] = pd.Series(np.array(total))

df_test['have_spend_10m'] = pd.Series(np.array(have_spend_morethan_10m))

df_test['have_spend_50m'] = pd.Series(np.array(have_spend_morethan_50m))

df_test['have_spend_100m'] = pd.Series(np.array(have_spend_morethan_100m))
def categorical_to_numerical(dataset, feature):

    """ Docstring

    Fungsi ini digunakan untuk mengubah feature categorical menjadi numerik. Contohnya seperti feature service class 

    yang memiliki 2 nilai 'ECONOMY' dan 'BUSINESS' yang akan dirubah menjadi nilai 0 dan 1. Fungsi ini dibuat agar code 

    menjadi lebih bersih elegan.

    

    Parameter yang dibutuhkan ada 2, yaitu dataset (flight dan data test) dan feature (berupa categorical feature yang

    ingin diubah menjadi numerik)

    """

    dictionary = {}

    for value in dataset[feature].unique():

        index = np.where(dataset[feature].unique() == value)

        dictionary[value] = index[0]

    dataset[feature] = dataset[feature].map(dictionary).astype(int)
# Menggunakan fungsi categorical_to_numerical untuk mengubah seluruh feature categorical menjadi numerik.

dataset = [flight, df_test]

cat_feature = ['gender', 'trip', 'service_class', 'is_tx_promo', 'airlines_name', 'visited_city']

for data in dataset:

    for feature in cat_feature:

        categorical_to_numerical(data, feature)
# Membuat fungsi untuk membantu dalam visualisasi countplot dan annotationnya

def count_plot_with_annotation(variabel, data, ax_X, ax_Y):

    """ Docstring

    Fungsi ini digunakan untuk melakukan plotting countplot dan memberikan annotasi berupa nilai count pada masing-masing bar

    

    Parameter yang dibutuhkan ada 4, yaitu variabel, data, dan nilai axes x dan y.

    """

    ax = sns.countplot(variabel,data=data,ax=axes[ax_X, ax_Y])

    for p in ax.patches:

        ax.annotate(format(p.get_height()), (p.get_x() + p.get_width() / 2., p.get_height()), 

                       ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')

        

def prob_box_plot_with_annotation(x, y, ax_X, ax_Y):

    """ Docstring

    Fungsi ini digunakan untuk melakukan plotting boxplot dan memberikan annotasi berupa nilai probability pada masing-masing bar

    

    Parameter yang dibutuhkan ada 4, yaitu x, y, dan nilai axes x dan y. x dan y merupakan input parameter barplot dari seaborn 

    yang dapat dibaca pada dokumentasinya.

    """

    ax = sns.barplot(x, y, ax = axes[ax_X, ax_Y])

    ax.set_title('Probabilitas terjadi cross selling berdasarkan {}'.format(x.name))

    for p in ax.patches:

        ax.annotate(np.round(p.get_height(),decimals= 3), (p.get_x() + p.get_width() / 2., p.get_height()), 

                       ha = 'center', va = 'center', xytext = (0, 25), textcoords = 'offset points')
# Membuat subplot dan mengatur ukuran figur

fig, axes = plt.subplots(2, 4, figsize=(18, 14))



# Menggunakan fungsi yang telah dibuat untuk membuat countplot dengan mudah

count_plot_with_annotation('route', flight, 0, 0) # Countplot pada variabel route

count_plot_with_annotation('gender', flight, 0, 1) # Countplot pada variabel gender

count_plot_with_annotation('trip', flight, 0, 2) # Countplot pada variabel trip

count_plot_with_annotation('airlines_name', flight, 0, 3) # Countplot pada variabel airlines_name

count_plot_with_annotation('service_class', flight, 1, 0) # Countplot pada variabel service_class

count_plot_with_annotation('is_tx_promo', flight, 1, 1) # Countplot pada variabel is_tx_promo  

count_plot_with_annotation('visited_city', flight, 1, 2) # Countplot pada variabel visited_city

count_plot_with_annotation('cross_sell', flight, 1, 3) # Countplot pada variabel cross_sell
# Membuat subplot dan mengatur ukuran figur

fig, axes = plt.subplots(2, 3, figsize=(18, 14))



# Menggunakan fungsi yang telah dibuat untuk membuat boxplot dengan mudah

prob_box_plot_with_annotation(flight['gender'], flight['cross_sell'], 0, 0)

prob_box_plot_with_annotation(flight['trip'], flight['cross_sell'], 0, 1)

prob_box_plot_with_annotation(flight['airlines_name'], flight['cross_sell'], 0, 2)

prob_box_plot_with_annotation(flight['service_class'], flight['cross_sell'], 1, 0)

prob_box_plot_with_annotation(flight['is_tx_promo'], flight['cross_sell'], 1, 1)

prob_box_plot_with_annotation(flight['visited_city'], flight['cross_sell'], 1, 2)
sns.factorplot('gender', col = 'trip', data = flight, kind = 'count')
sns.factorplot('airlines_name', col = 'gender', data = flight, kind = 'count')
sns.factorplot('gender', col = 'service_class', data = flight, kind = 'count')
sns.factorplot('gender', col = 'is_tx_promo', data = flight, kind = 'count')
sns.factorplot('visited_city', col = 'gender', data = flight, kind = 'count')
sns.factorplot('airlines_name', col = 'trip', data = flight, kind = 'count')
sns.factorplot('trip', col = 'service_class', data = flight, kind = 'count')
sns.factorplot('trip', col = 'is_tx_promo', data = flight, kind = 'count')
sns.factorplot('visited_city', col = 'trip', data = flight, kind = 'count')
sns.factorplot('airlines_name', col = 'service_class', data = flight, kind = 'count')
sns.factorplot('airlines_name', col = 'is_tx_promo', data = flight, kind = 'count')
sns.factorplot('airlines_name', col = 'visited_city', data = flight, kind = 'count')
sns.factorplot('service_class', col = 'is_tx_promo', data = flight, kind = 'count')
sns.factorplot('visited_city', col = 'service_class', data = flight, kind = 'count')
sns.factorplot('visited_city', col = 'is_tx_promo', data = flight, kind = 'count')
figbi, axesbi = plt.subplots(1, 2, figsize=(16, 10))

sns.boxplot(x="cross_sell", y="member_duration_days", data=flight,ax=axesbi[0])

sns.boxplot(x="cross_sell", y="price", data=flight,ax=axesbi[1])
ax = sns.countplot(x = 'no_of_seats',data = flight) # Countplot pada variabel no_of_seats



for p in ax.patches:

    ax.annotate(format(p.get_height()), (p.get_x() + p.get_width() / 2., p.get_height()), 

                   ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')

sns.boxplot(x="cross_sell", y="no_of_seats", data=flight)
sns.jointplot("price", "member_duration_days", data=flight, kind="reg")
sns.jointplot("price", "no_of_seats", data=flight, kind="reg")
sns.jointplot("no_of_seats", "member_duration_days", data=flight, kind="reg")
# Mengubah bentuk dari member_duration_days kedalam bentuk tahunan (menggunakan standar 1 tahun = 365 hari)

flight['member_duration_years'] = flight['member_duration_days'] // 365 

df_test['member_duration_years'] = df_test['member_duration_days'] // 365



# Mengubah bentuk dari member_duration_days kedalam bentuk bulanan (menggunakan standar 1 tahun = 12 bulan)

flight['member_duration_months'] = (flight['member_duration_days'] % 365) // 12

df_test['member_duration_months'] = (df_test['member_duration_days'] % 365) // 12



# Menambahkan feature sisa hari, contohnya 1 tahun, 4 bulan, 20 hari. 20 hari tersebut menjadi nilai pada feature ini.

flight['member_duration_days_left'] = (flight['member_duration_days'] % 365) % 12

df_test['member_duration_days_left'] = (df_test['member_duration_days'] % 365) % 12
# Mengubah bentuk price kedalam jutaan

flight['price_in_million'] = flight['price'] // 1000000 

df_test['price_in_million'] = df_test['price'] // 1000000



# Mendapatkan nilai price per jumlah seat yang dibeli

flight['price_avg'] = flight['price'] / flight['no_of_seats']

df_test['price_avg'] = df_test['price'] / df_test['no_of_seats']
# Code dibawah ini membuat feature baru, apakah pelanggan tersebut naik pesawat secara individu atau berkelompok. 

# Dibuktikan dengan nilai no_of_seats

is_alone = []



for i in flight["no_of_seats"]:

    if i > 1:

        is_alone.append(0)

    else:

        is_alone.append(1)



flight["is_alone"] = pd.Series(np.array(is_alone))



is_alone = []



for i in df_test["no_of_seats"]:

    if i > 1:

        is_alone.append(0)

    else:

        is_alone.append(1)



df_test["is_alone"] = pd.Series(np.array(is_alone))
# Fungsi untuk mendapatkan index dari list 2 dimensi

def index_2d(myList, v):

    for i, x in enumerate(myList):

        if v in x:

            return i
# Code dibawah ini untuk melabeli pelanggan menggunakan numerik.

# acc_index = 74791 digunakan untuk data test karena nilai maskimal pada data flight adalah 74791.

# Adapun terdapat akun yang sama pada data test dan data flight sehingga dapat meningkatkan keakuratan hasil prediksi.



# flight['account'] = pd.factorize(flight.account_id)[0]

# acc_index = 74791

# account = []

# for i in range(0, len(df_test.account_id)):

#     val1 = df_test.account_id[i]

#     index = index_2d(flight.account_id, val1)

#     if index == None:

#         acc_index += 1

#         account.append(acc_index)

#     else:

#         account.append(flight.account[index])



# df_test['account'] = pd.Series(np.array(account))





# Code dibawah ini untuk membuat feature apakah pelanggan tersebut pernah melakukan cross selling sekaligus 

# mendapatkan berapa kali melakukan cross_selling dalam 2018.

# have_cross = []

# n_cross = []

# for i in range(0, len(flight.account_id)):

#     account = flight.account[i]

#     if len(flight[flight.account == account][flight.cross_sell == 1]) == 0:

#         have_cross.append(0)

#         n_cross.append(0)

#     else:

#         have_cross.append(1)

#         n_cross.append(len(flight[flight.account == account][flight.cross_sell == 1]))

               

# flight['have_cross'] = pd.Series(np.array(have_cross))

# flight['n_cross'] = pd.Series(np.array(n_cross))



# have_cross = []

# for i in range(0, len(df_test.account_id)):

#     account = df_test.account[i]

#     if len(flight[flight.account == account][flight.cross_sell == 1]) == 0:

#         have_cross.append(0)

#         n_cross.append(0)

#     else:

#         have_cross.append(1)

#         n_cross.append(len(flight[flight.account == account][flight.cross_sell == 1]))

                

# df_test['have_cross'] = pd.Series(np.array(have_cross))

# df_test['n_cross'] = pd.Series(np.array(n_cross))
# Code dibawah ini digunakan untuk menghapus feature yang tidak berguna berdasarkan hasil analisa pada tahap sebelumnya



# flight = flight.drop(['order_id', 'account_id', 'route', 'log_transaction'], axis=1)



# Menyimpan order_id dari data test (digunakan untuk submission)

# df_test_order_id = df_test.order_id

# df_test = df_test.drop(['order_id', 'account_id', 'route', 'log_transaction'], axis=1)
# Code dibawah ini digunakan untuk mengubah categorical menjadi dummy/indikator.



# cat_features = ['trip', 'airlines_name', 'visited_city']



# for feature in cat_features:

#     a = pd.get_dummies(flight[feature], prefix = feature)

#     frames = [flight, a]

#     flight = pd.concat(frames, axis = 1)

    

#     b = pd.get_dummies(df_test[feature], prefix = feature)

#     frames = [df_test, b]

#     df_test = pd.concat(frames, axis = 1)

    

# flight.drop(cat_features, axis = 1, inplace=True)

# df_test.drop(cat_features, axis = 1, inplace=True)

# flight
# flight.corr()["cross_sell"].sort_values(ascending = False)
# flight.to_csv('cleaned_flight.csv', index=False)

# df_test.to_csv('cleaned_df_test.csv', index=False)
# flight = pd.read_csv('cleaned_flight.csv')

# df_test = pd.read_csv('cleaned_df_test.csv')
# Berikut adalah feature yang dipilih berdasarkan korelasi

# columns = ['account', 'member_duration_days', 'member_duration_years',

#        'total_transaksi', 'have_spend_10m', 'have_spend_100m', 'no_of_seats',

#        'price', 'price_avg', 'have_visit_srg', 'have_cross',

#        'n_cross', 'have_visit_jogc', 'have_visit_sby', 'have_visit_mdc',

#        'gender', 'trip_0', 'is_alone', 'trip_1', 'trip_2', 'is_tx_promo',

#        'airlines_name_1', 'airlines_name_2',

#        'airlines_name_3', 'airlines_name_4', 'airlines_name_5',

#        'visited_city_0', 'visited_city_1', 'visited_city_2', 'visited_city_3',

#        'visited_city_4', 'visited_city_5', 'visited_city_7', 'cross_sell']



# flight = flight[columns]

# del columns[-1]

# df_test = df_test[columns]
# Membuat variabel train dan variabel target(cross_sell)

# train = flight.drop("cross_sell", axis=1)

# target = flight["cross_sell"]



# Melakukan metode oversampling untuk mengatasi imbalanced data

# smote = SMOTE(sampling_strategy='minority')

# train, target = smote.fit_sample(train, target)



# Melakukan normalisasi pada data, hal ini dilakukan karena terdapat banyak data yang memiliki nilai tidak seimbang

# seperti nilai price terhadap member_duration_days, dan juga kami berharap masalah outlier dapat teratasi dengan normalisasi

# scaler = StandardScaler()

# train = scaler.fit_transform(train)

# scaled_df_test = scaler.fit_transform(df_test)



# melakukan pembagian data menjadi data latih (80%) dan data validasi (20%)

# X_train, X_test, y_train, y_test = train_test_split(train, target, test_size=0.2, random_state = 123)
# print('Ukuran X_train = ', X_train.shape, 

#       '\nUkuran X_test  = ', X_test.shape, 

#       '\nUkuran y_train = ', y_train.shape,

#       '\nUkuran y_test  = ', y_test.shape

#      )
# pd.DataFrame(X_train)
# Code dibawah ini melakukan pelatihan pada data training dan validasi pada data test hasil split data flight

# Metode yang digunakan pada tahapan ini adalah xgboost, knn, random forest, decision tree, dan gradient boosting.

# parameter awal yang digunakan adalah parameter default



# xgboost_model = XGBClassifier()

# knn_model = KNeighborsClassifier() 

# rfc_model = RandomForestClassifier()

# dtc_model = DecisionTreeClassifier()

# gbc_model = GradientBoostingClassifier()



# model_score = []

# all_model = [xgboost_model, knn_model, rfc_model, dtc_model, gbc_model]

# for model in all_model:

#     model.fit(X_train, y_train)

#     y_pred = model.predict(X_test)

#     f_score = f1_score(y_test, y_pred, average='macro')

#     accuracy = accuracy_score(y_test, y_pred)

#     model_score.append([model.__class__.__name__, f_score, accuracy])

#     print("Metode {}".format(model.__class__.__name__))

#     print('Score F1-Macro = ', f_score, ', Accuracy = ', accuracy)

#     tn, fp, fn, tp = confusion_matrix(y_true=y_test, y_pred=y_pred).ravel()

#     print('TN:', tn, '\nFP:',fp, '\nFN:', fn, '\nTP:', tp)

#     print('\n')
# df_score = pd.DataFrame(model_score)

# df_score.columns = ['metode', 'f1-score', 'akurasi']

# df_score
# param_grid = {'learning_rate': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],

#               'n_estimators': [25, 50, 75, 100, 125, 150, 200],

#               'max_depth' : [25, 50, 75, 100, 125, 150, 200]

#              }

# grid = RandomizedSearchCV(XGBClassifier(), param_grid,

#                           cv=5, verbose=1, scoring='f1_macro', n_jobs = -1)



# grid.fit(X_train, y_train)

# grid.best_params_
# param_grid = {'n_neighbors': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}

# grid = RandomizedSearchCV(KNeighborsClassifier(), param_grid,

#                           cv=5, verbose=1, scoring='f1_macro', n_jobs = -1)



# grid.fit(X_train, y_train)

# grid.best_params_
# param_grid = {

#               'n_estimators': [25, 50, 75, 100, 125, 150, 200],

#               'max_depth' : [5, 10, 25, 50, 75, 100, 125, 150, 200]

#              }

# grid = RandomizedSearchCV(RandomForestClassifier(), param_grid,

#                           cv=5, verbose=1, scoring='f1_macro', n_jobs = -1)



# grid.fit(X_train, y_train)

# grid.best_params_
# param_grid = {

#               'max_depth' : [5, 10, 25, 50, 75, 100, 125, 150, 200]

#              }

# grid = RandomizedSearchCV(DecisionTreeClassifier(), param_grid,

#                           cv=5, verbose=1, scoring='f1_macro', n_jobs = -1)



# grid.fit(X_train, y_train)

# grid.best_params_
# param_grid = {'learning_rate': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],

#               'n_estimators': [25, 50, 75, 100, 125, 150, 200],

#               'max_depth' : [25, 50, 75, 100, 125, 150, 200]

#              }

# grid = RandomizedSearchCV(GradientBoostingClassifier(), param_grid,

#                           cv=5, verbose=1, scoring='f1_macro', n_jobs = -1)



# grid.fit(X_train, y_train)

# grid.best_params_
# Code dibawah ini melakukan pelatihan pada data training dan validasi pada data test hasil split data flight

# Metode yang digunakan pada tahapan ini adalah xgboost, knn, random forest, decision tree, dan gradient boosting.

# parameter yang digunakan adalah parameter hasil optimasi tahap sebelumnya.



# from sklearn.metrics import f1_score

# xgboost_model = XGBClassifier(learning_rate= 0.2, n_estimators= 150, max_depth=50, n_jobs= -1)

# knn_model = KNeighborsClassifier(n_neighbors=1) 

# rfc_model = RandomForestClassifier(random_state=0, n_estimators = 100, max_depth=50)

# dtc_model = DecisionTreeClassifier(max_depth=50)

# gbc_model = GradientBoostingClassifier(n_estimators=120, learning_rate=0.70, max_depth=50)



# model_score = []

# all_model = [xgboost_model, knn_model, rfc_model, dtc_model, gbc_model]

# for model in all_model:

#     print("Metode {}".format(model.__class__.__name__))

#     model.fit(X_train, y_train)

#     y_pred = model.predict(X_test)

#     f_score = f1_score(y_test, y_pred, average='macro')

#     accuracy = accuracy_score(y_test, y_pred)

#     model_score.append([model.__class__.__name__, f_score, accuracy]) 

#     print('Score F1-Macro = ', f_score, ', Accuracy = ', accuracy)

#     tn, fp, fn, tp = confusion_matrix(y_true=y_test, y_pred=y_pred).ravel()

#     print('TN:', tn, '\nFP:',fp, '\nFN:', fn, '\nTP:', tp)

#     print('\n')
# df_score = pd.DataFrame(model_score)

# df_score.columns = ['metode', 'f1-score', 'akurasi']

# df_score
# clf1 = RandomForestClassifier(random_state=0, n_estimators = 100, max_depth=50)

# clf2 = XGBClassifier(learning_rate= 0.2, n_estimators= 150, max_depth=50, n_jobs= -1)

# clf3 = KNeighborsClassifier(n_neighbors=1) 

# clf4 = DecisionTreeClassifier(max_depth=50)

# clf5 = GradientBoostingClassifier(n_estimators=120, learning_rate=0.72, max_depth=50)

# eclf = VotingClassifier(estimators=[('rf', clf1), ('xgboost', clf2), ('knn', clf3), 

#                                     ('dt', clf3), ('gbc', clf3)], 

#                         voting='hard')

# eclf = eclf.fit(X_train, y_train)

# y_pred = eclf.predict(X_test)

# print(f1_score(y_test, y_pred, average='macro'), accuracy_score(y_test, y_pred))

# tn, fp, fn, tp = confusion_matrix(y_true=y_test, y_pred=y_pred).ravel()

# print('TN:', tn, '\n FP:',fp, '\n FN:', fn, '\n TP:', tp)
# Code dibawah ini digunakan untuk melihat row mana saja terjadi kesalahan prediksi

# Setelah dilakuakn analisa ini, apabila diperlukan pengerjaan dapat 

# kembali dilakukan pada tahap-tahap sebelumnya (praproses, eda, dan feature engineering) 



# for idx, prediction, label in zip(enumerate(X_test), y_pred, y_test):

#     if prediction != label:

#         print("Sample", idx, ', has been classified as', prediction, 'and should be', label) 
# Berikut adalah code untuk membuat file submission

# Tidak dijalankan pada kernel kaggle karena telah dilakukan pada saat code masih di local



# Y_pred = knn_model.predict(cleaned_df_test)

# prediction = []

# for pred in Y_pred:

#     if pred == 0:

#         prediction.append("no")

#     elif pred == 1:

#         prediction.append("yes")

# prediction = np.array(prediction)

# submission = pd.DataFrame({

#         "order_id": df_test_order_id,

#         "is_cross_sell": prediction

#     })

# submission.to_csv("submission.csv", index=False)