from pandas import Series
import pandas as pd
import pandas_profiling
import numpy as np
import datetime as dt

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_selection import f_classif, mutual_info_classif
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.model_selection import cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import auc, roc_auc_score, roc_curve
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import plot_precision_recall_curve
from sklearn.impute import KNNImputer

from mlxtend.feature_selection import ExhaustiveFeatureSelector as EFS

import os
RANDOM_SEED = 42
!pip freeze > requirements.txt
CURRENT_DATE = pd.to_datetime('14/09/2020')

pd.options.display.float_format = '{:.4f}'.format
pd.options.mode.chained_assignment = None
def plot_confusion_matrix(y_true, y_pred, font_scale, classes,
                          normalize=False,
                          title=None):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """

    list_of_labels = [['TP', 'FP'], ['FN', 'TN']]

    if not title:
        if normalize:
            title = 'Нормализованная матрица ошибок'
        else:
            title = 'Матрица ошибок без нормализации'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    cm[0, 0], cm[1, 1] = cm[1, 1], cm[0, 0]

    # # Only use the labels that appear in the data
    # classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.style.use('seaborn-paper')
    cmap = plt.cm.Blues
    color_text = plt.get_cmap('PuBu')(0.85)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.grid(False)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries

           title=title)
    ax.title.set_fontsize(15)
    ax.set_ylabel('Предсказанные значения', fontsize=14, color=color_text)
    ax.set_xlabel('Целевая переменная', fontsize=14, color=color_text)
    ax.set_xticklabels(classes, fontsize=12, color='black')
    ax.set_yticklabels(classes, fontsize=12, color='black')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, list_of_labels[i][j] + '\n' + format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax


def confusion_matrix_f(columns, d_y, d_y_pred, font_scale=1, normalize=False):
    class_names = np.array(columns, dtype='U10')
    # Plot non-normalized confusion matrix
    plot_confusion_matrix(d_y, d_y_pred, font_scale, classes=class_names,
                          title='Матрица ошибок без нормализации')

    # Plot normalized confusion matrix
    if normalize:
        plot_confusion_matrix(d_y, d_y_pred, font_scale, classes=class_names, normalize=True,
                              title='Нормализованная матрица ошибок')

    plt.show()
    return


def PR_curve_with_area(d_y_true, d_y_pred_prob, font_scale=1):
    plt.style.use('seaborn-paper')
    sns.set(font_scale=font_scale)
    # sns.set_color_codes("muted")

    plt.figure(figsize=(8, 6))
    precision, recall, thresholds = precision_recall_curve(d_y_true, d_y_pred_prob, pos_label=1)
    prc_auc_score_f = auc(recall, precision)
    plt.plot(precision, recall, lw=3, label='площадь под PR кривой = %0.3f)' % prc_auc_score_f)

    plt.xlim([-.05, 1.0])
    plt.ylim([-.05, 1.05])
    plt.xlabel('Точность \n Precision = TP/(TP+FP)')
    plt.ylabel('Полнота \n Recall = TP/P')
    plt.title('Precision-Recall кривая')
    plt.legend(loc="upper right")
    plt.show()
    return


def vis_cross_val_score(d_name_metric, d_vec, d_value_metric, font_scale):
    num_folds = len(d_vec['train_score'])
    avg_metric_train, std_metric_train = d_vec['train_score'].mean(), d_vec['train_score'].std()
    avg_metric_test, std_metric_test = d_vec['test_score'].mean(), d_vec['test_score'].std()

    plt.style.use('seaborn-paper')
    sns.set(font_scale=font_scale)
    color_text = plt.get_cmap('PuBu')(0.85)

    plt.figure(figsize=(12, 6))
    plt.plot(d_vec['train_score'], label='тренировочные значения', marker='.', color='darkblue')
    plt.plot([0, num_folds - 1], [avg_metric_train, avg_metric_train], color='blue',
             label='среднее трен. значений ', marker='.', lw=2, ls='--')

    plt.plot(d_vec['test_score'], label='тестовые значения', marker='.', color='red')
    plt.plot([0, num_folds - 1], [avg_metric_test, avg_metric_test], color='lightcoral',
             label='среднее тест. значений ', marker='.', lw=2, ls='--')

    plt.plot([0, num_folds - 1], [d_value_metric, d_value_metric], color='grey',
             label='значение метрики до CV', marker='.', lw=3)

    # plt.xlim([1, num_folds])
    y_max = max(avg_metric_train, avg_metric_test) + 1.5 * max(std_metric_train, std_metric_test)
    y_min = min(avg_metric_train, avg_metric_test) - 3 * max(std_metric_train, std_metric_test)
    plt.ylim([y_min, y_max])
    plt.xlabel('номер фолда', fontsize=15, color=color_text)
    plt.ylabel(d_name_metric, fontsize=15, color=color_text)
    plt.title(f'Кросс-валидация по метрике {d_name_metric} на {num_folds} фолдах',
              color=color_text, fontsize=17)
    plt.legend(loc="lower right", fontsize=11)
    y_min_text = y_min + 0.5 * max(std_metric_train, std_metric_test)
    plt.text(0, y_min_text,
             f'{d_name_metric} на трейне = {round(avg_metric_train, 3)} +/- '
             f'{round(std_metric_train, 3)} \n{d_name_metric} на тесте    = {round(avg_metric_test, 3)} +/- '
             f'{round(std_metric_test, 3)} \n{d_name_metric} до CV        = {round(d_value_metric, 3)}',
             fontsize=15)
    plt.show()
    return


def model_coef(d_columns, d_model_coef_0):
    temp_dict = {}
    temp_dict['имя признака'] = d_columns
    temp_dict['коэффициент модели'] = d_model_coef_0
    temp_dict['модуль коэф'] = abs(temp_dict['коэффициент модели'])
    temp_df = pd.DataFrame.from_dict(temp_dict, orient='columns')
    temp_df = temp_df.sort_values(by='модуль коэф', ascending=False)
    temp_df.reset_index(drop=True, inplace=True)

    return temp_df.loc[:, ['имя признака', 'коэффициент модели']]

def print_metrics():
    print('Accuracy: %.4f' % accuracy_score(y_test, y_pred))
    print('Precision: %.4f' % precision_score(y_test, y_pred))
    print('Recall: %.4f' % recall_score(y_test, y_pred))
    print('F1: %.4f' % f1_score(y_test, y_pred))

def ROC_curve_with_area(d_y_true, d_y_pred_prob, font_scale):
    roc_auc_score_f = roc_auc_score(d_y_true, d_y_pred_prob)

    plt.style.use('seaborn-paper')
    sns.set(font_scale=font_scale)
    # sns.set_color_codes("muted")

    plt.figure(figsize=(8, 6))
    fpr, tpr, thresholds = roc_curve(d_y_true, d_y_pred_prob, pos_label=1)

    plt.plot(fpr, tpr, lw=3, label='площадь под ROC кривой = %0.3f)' % roc_auc_score_f)
    plt.plot([0, 1], [0, 1], color='grey')
    plt.xlim([-.05, 1.0])
    plt.ylim([-.05, 1.05])
    plt.xlabel('Ложно классифицированные \n False Positive Rate (FPR)')
    plt.ylabel('Верно классифицированные \n True Positive Rate (TPR)')
    plt.title('ROC кривая')
    plt.legend(loc="lower right")
    plt.show()
    return
def viz_counter_bar(df, column, title, max_values=0, sort_by_index=False):
    """
    Визуализация количества значений в ДФ, Value_counts, 
    горизонтальные столбцы
    df - датафрейм, column - столбец, title - подпись
    max_values - максимальное количество значений для отображения, 0 - все
    sort_by_index - сортирует по индексам, а не значен#иям
    dollars - отображает подписи с $$ для price_range
    """
    if max_values > 0:
        col_values = df[column].value_counts().nlargest(
            max_values).sort_values(ascending=True)
    else:
        col_values = df[column].value_counts(ascending=True)
    
    if sort_by_index: 
        col_values = col_values.sort_index()
    
    fig, ax = plt.subplots()
    plt.rcParams['figure.figsize'] = (6, 6)
    plt.figure 
    ax = col_values.plot(kind='bar', title=title)
    for i, v in enumerate(col_values):
        plt.text(i, v+(col_values.max()/100), ""+str(v), ha='center', rotation = 'horizontal')
    
    plt.xticks(rotation=0)
    
    plt.show()

    
def viz_counter_barh(df, column, title, max_values=0):
    """
    Визуализация количества значений в ДФ, Value_counts, 
    горизонтальные столбцы
    df - датафрейм, column - столбец, title - подпись
    max_values - максимальное количество значений для отображения, 0 - все
    """
    if max_values > 0:
        col_values = df[column].value_counts().nlargest(
            max_values).sort_values(ascending=True)
    else:
        col_values = df[column].value_counts(ascending=True)
    
    fig, ax = plt.subplots()
    plt.rcParams['figure.figsize'] = (8, 10)
    plt.figure 
    ax = col_values.plot(kind='barh', title=title)
    ax.set_xlim(0, col_values.max()*1.15)
    
    for i, v in enumerate(col_values):
        plt.text(v, i, " "+str(v), va='center')
    
    plt.show()

def iqr_test(column, lim_a, lim_b, bins, hist=True):
    """
    Функция определения медианы, квантилей 25%/75% и 
    границы выбросов для данного сталбцы 
    lim_a lim_b - пределы для построенния гистограммы
    """
    median = df[column].median()
    IQR = df[column].quantile(0.75) - df[column].quantile(0.25)
    perc25 = df[column].quantile(0.25)
    perc75 = df[column].quantile(0.75)
    print('25-й перцентиль: {},'.format(perc25), 
          '75-й перцентиль: {},'.format(perc75), 
          "IQR: {}, ".format(IQR),
          "Границы выбросов: [{f}, {l}]."
          .format(f=perc25 - 1.5*IQR, l=perc75 + 1.5*IQR))
    if hist:
        df[column].loc[df[column].between(perc25 - 1.5*IQR, perc75 + 1.5*IQR)].\
        hist(bins=bins, range=(lim_a, lim_b), label='IQR')
        
def plot_roc():
    plt.figure()
    plt.plot([0, 1], label='Baseline', linestyle='--')
    plt.plot(fpr, tpr, label = 'Regression')
    plt.title('Logistic Regression ROC AUC = %0.4f' % roc_auc)
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.legend(loc = 'lower right')
    plt.show()
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
PATH_to_file = '/kaggle/input/sf-dst-scoring/'
df_train = pd.read_csv(PATH_to_file+'train.csv')
df_test = pd.read_csv(PATH_to_file+'test.csv')
pd.set_option('display.max_columns', None)
print('Размерность тренировочного датасета: ', df_train.shape)
display(df_train.head(2))
print('Размерность тестового датасета: ', df_test.shape)
display(df_test.head(2))
df_train['Train'] = 1  # тренировочный
df_test['Train'] = 0 # тестовый

df = df_train.append(df_test, sort=False).reset_index(drop=True) 
df.describe(include='all').T
df.isna().sum()
viz_counter_barh(df, 'education', 'уровень образования')
train_knn = df[['education', 'good_work', 'age', 'home_address', 'work_address']]
imputer = KNNImputer(n_neighbors = 9)

# создаем словарь из списка типов образования (0 -самое популярной, школьное и т.д.)
edu = df.education.value_counts().index.to_list() 
edu_dict = {}
for counter, edu_word in enumerate(edu):
    edu_dict[edu_word] = counter

train_knn['education'].replace(edu_dict, inplace=True)
df_fill = imputer.fit_transform(train_knn)
df.loc[:, 'education'] = np.round(df_fill.T[0],0)
viz_counter_bar(df, 'sex', 'пол заемщика')
iqr_test('age', 21, 72, 51)
df['age'] = np.log(df['age'] + 1)
df['age'].plot.hist(bins=16);
viz_counter_bar(df, 'car', 'наличие автомобиля')
viz_counter_bar(df, 'car_type', 'наличие автомобиля')
viz_counter_barh(df, 'decline_app_cnt', 'количество отказов')
# Логарифмируем 
df['decline_app_cnt'] = np.log(df['decline_app_cnt'] + 1)
viz_counter_bar(df, 'good_work', 'наличие хорошей работы')
viz_counter_barh(df, 'bki_request_cnt', 'количество запросов в БКИ')
iqr_test('bki_request_cnt', 0, 55, 50, hist=False)
df['bki_request_cnt'] = np.log(df['bki_request_cnt'] + 1)
viz_counter_bar(df, 'home_address', 'домашний адрес', sort_by_index=True)
viz_counter_bar(df, 'work_address', 'место работы', sort_by_index=True)
iqr_test('income', 0, 100000, 20)
incomes = df.groupby(['home_address', 'work_address'])['income'].mean()
sns.heatmap(incomes.unstack(), annot=True, fmt='g');
cars = df[df.car=='Y'].groupby(['home_address', 'work_address'])['car'].count()
sns.heatmap(cars.unstack(), annot=True, fmt='g');
# логарифмируем income
df['income'] = np.log(df['income'] + 1)
viz_counter_bar(df, 'foreign_passport', 'наличие загранпаспорта')
viz_counter_bar(df, 'sna', 'связь заемщика с клиентами банка', sort_by_index=True)
viz_counter_bar(df, 'first_time', 'давность наличия информации о заемщике', sort_by_index=True)
iqr_test('score_bki', -4, 2, 50)
viz_counter_bar(df, 'region_rating', 'рейтинг региона', sort_by_index=True)
iqr_test('region_rating', 10, 90, 50, hist=False)
# Преобразуем формат признака
df.app_date = pd.to_datetime(df.app_date, format='%d%b%Y')
# Выясняем начало и конец периода нашего датасета - это 1 января и 30 апреля 2014 года
start = df.app_date.min()
end = df.app_date.max()
start,end
# Количество дней от старта заявки 
df['td'] = df.app_date - df.app_date.min()
df['td'] = df['td'].apply(lambda x: str(x).split()[0])
df['td'] = df['td'].astype(int)
iqr_test('td', 0 , 130, 10)
sns.scatterplot(x='client_id',y='td',data=df);
# удаляем дату выдачи заявки, вместо нне время со дня подачи - td
df = df.drop(columns = ['app_date'])
viz_counter_bar(df[df.Train==1], 'default', 'Клиенты совершившие дефолт')
num_cols = ['age', 'decline_app_cnt', 'bki_request_cnt', 'income', 'score_bki', 'td']
sns.heatmap(df[df.Train==1][num_cols].corr().abs(), annot=True, vmin=0, vmax=1, cmap='Blues');
df.loc[df.car=='N', 'car'] = 0
df.loc[df.car=='Y', 'car'] = 1
df.loc[df.car_type=='Y', 'car'] = 2
df.drop(columns=['car_type'], inplace=True)
df.loc[df.work_address==1, 'work_address'] = 'w1'
df.loc[df.work_address==2, 'work_address'] = 'w2'
df.loc[df.work_address==3, 'work_address'] = 'w3'

df.loc[df.home_address==1, 'home_address'] = 'h1'
df.loc[df.home_address==2, 'home_address'] = 'h2'
df.loc[df.home_address==3, 'home_address'] = 'h3'

df['work_home'] = df['work_address'] + df['home_address']

df.drop(columns=['home_address', 'work_address'], inplace=True)
bin_cols = ['sex', 'foreign_passport', 'good_work']
cat_cols = ['education', 'work_home', 'sna', 'first_time', 'region_rating', 'car']
num_cols = ['age', 'decline_app_cnt', 'bki_request_cnt', 'income', 'score_bki', 'td']
df_tr = df[df.Train==1]
imp_num = Series(f_classif(df_tr[num_cols], df_tr['default'])[0], index = num_cols)
imp_num.sort_values(inplace = True)
imp_num.plot(kind = 'barh', title='Значимость непрерывных переменных по ANOVA F test');
# преобразуем в int
df.education = df.education.astype(int)

# Для бинарных признаков мы будем использовать LabelEncoder
label_encoder = LabelEncoder()

for column in bin_cols:
    df[column] = label_encoder.fit_transform(df[column])

df['work_home'] = label_encoder.fit_transform(df['work_home'])
df.head(3)
df_tr = df[df.Train==1]
imp_cat = Series(mutual_info_classif(df_tr[bin_cols + cat_cols], df_tr['default'],
                                     discrete_features =True), index = bin_cols + cat_cols)
imp_cat.sort_values(inplace = True)
imp_cat.plot(kind = 'barh', title = 'Значимость бин. и категор. переменных по Mutual information test');
# StandardScaler отдельно для Train и Test для числовых значений
df.loc[df.Train==1, num_cols] = StandardScaler().fit_transform(df.query('Train==1')[num_cols].values)
df.loc[df.Train==0, num_cols] = StandardScaler().fit_transform(df.query('Train==0')[num_cols].values)
# Get_dummies для категорийных 
df=pd.get_dummies(df, prefix=cat_cols, columns=cat_cols)
df.head(3)
train = df.query('Train == 1').drop(['Train', 'client_id'], axis=1)
test = df.query('Train == 0').drop(['Train', 'client_id'], axis=1)

X = train.drop(['default'], axis=1)
y = train.default.values

# Воспользуемся специальной функцие train_test_split для разбивки тестовых данных
# выделим 20% данных на валидацию (параметр test_size)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED)

# проверяем
test.shape, train.shape, X.shape, X_train.shape, X_test.shape
model = LogisticRegression(random_state=RANDOM_SEED, max_iter=100, solver='liblinear')

model.fit(X_train, y_train)

y_pred_prob = model.predict_proba(X_test)[:,1]
y_pred = model.predict(X_test)
print_metrics()
confusion_matrix_f(['Дефолтный','Не дефолтный'], y_test, y_pred, 1, normalize=False)
ROC_curve_with_area(y_test, y_pred_prob, 1.1)
PR_curve_with_area(y_test, y_pred, 1.1)
temp_vec = cross_validate(model, X_test, y_test, cv=10, scoring='roc_auc', return_train_score=True);
vis_cross_val_score('ROC-AUC', temp_vec, 0.745, 1.1)
model = LogisticRegression(random_state=RANDOM_SEED)

iter_ = 50
epsilon_stop = 1e-3

param_grid = [
    {'penalty': ['l1'], 
     'solver': ['liblinear', 'lbfgs', 'saga'], 
     'class_weight':['none', 'balanced'], 
     'multi_class': ['auto','ovr'], 
     'max_iter':[iter_],
     'tol':[epsilon_stop]},
   # {'penalty': ['l2'], 
   #  'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'], 
   #  'class_weight':['none', 'balanced'], 
   #  'multi_class': ['auto','ovr'], 
   #  'max_iter':[iter_],
   #  'tol':[epsilon_stop]},
    {'penalty': ['none'], 
     'solver': ['newton-cg', 'saga'], 
     'class_weight':['none', 'balanced'], 
     'multi_class': ['auto','ovr'], 
     'max_iter':[iter_],
     'tol':[epsilon_stop]},
]
gridsearch = GridSearchCV(model, param_grid, scoring='f1', n_jobs=-1, cv=5)
gridsearch.fit(X_train, y_train)
model = gridsearch.best_estimator_
##печатаем параметры
best_parameters = model.get_params()
for param_name in sorted(best_parameters.keys()):
        print('\t%s: %r' % (param_name, best_parameters[param_name]))
    ##печатаем метрики
preds = model.predict(X_test)

y_pred_prob = model.predict_proba(X_test)[:,1]
y_pred = model.predict(X_test)

print('Accuracy: %.4f' % accuracy_score(y_test, y_pred))
print('Precision: %.4f' % precision_score(y_test, y_pred))
print('Recall: %.4f' % recall_score(y_test, y_pred))
print('F1: %.4f' % f1_score(y_test, y_pred))
model = LogisticRegression(random_state=RANDOM_SEED, 
                           C=1, 
                           class_weight= 'balanced', 
                           dual= False, 
                           fit_intercept= True, 
                           intercept_scaling= 1, 
                           l1_ratio= None, 
                           multi_class= 'auto', 
                           n_jobs= -1, 
                           penalty= 'none', 
                           solver = 'newton-cg', 
                           verbose= 0, 
                           warm_start= False)

model.fit(X_train, y_train)

y_pred_prob = model.predict_proba(X_test)[:,1]
y_pred = model.predict(X_test)
print_metrics()
confusion_matrix_f(['Дефолтный','Не дефолтный'], y_test, y_pred, 1, normalize=False)
ROC_curve_with_area(y_test, y_pred_prob, 1.1)
PR_curve_with_area(y_test, y_pred, 1.1)
temp_vec = cross_validate(model, X_test, y_test, cv=10, scoring='roc_auc', return_train_score=True);
vis_cross_val_score('ROC-AUC', temp_vec, 0.745, 1.1)
model = LogisticRegression(random_state=RANDOM_SEED, 
                           C=1, 
                           class_weight= 'balanced', 
                           dual= False, 
                           fit_intercept= True, 
                           intercept_scaling= 1, 
                           l1_ratio= None, 
                           multi_class= 'auto', 
                           n_jobs= None, 
                           penalty= 'l1', 
                           solver = 'liblinear', 
                           verbose= 0, 
                           warm_start= False)

model.fit(X_train, y_train)

display(model_coef(X_train.columns, model.coef_[0]))
drop_list1 = ['first_time_1', 'sna_2', 'work_home_8', 'education_2', 'car_0', 'car_1', 
              'income', 'region_rating_50', 'age', 'sna_3']
train = df.query('Train == 1').drop(['Train', 'client_id']+drop_list1, axis=1)
test = df.query('Train == 0').drop(['Train', 'client_id']+drop_list1, axis=1)

X = train.drop(['default'], axis=1)
y = train.default.values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED)
model = LogisticRegression(random_state=RANDOM_SEED, 
                           C=1, 
                           class_weight= 'balanced', 
                           dual= False, 
                           fit_intercept= True, 
                           intercept_scaling= 1, 
                           l1_ratio= None, 
                           multi_class= 'auto', 
                           n_jobs= -1, 
                           penalty= 'none', 
                           solver = 'newton-cg', 
                           verbose= 0, 
                           warm_start= False)

model.fit(X_train, y_train)

y_pred_prob = model.predict_proba(X_test)[:,1]
y_pred = model.predict(X_test)

print('Accuracy: %.4f' % accuracy_score(y_test, y_pred))
print('Precision: %.4f' % precision_score(y_test, y_pred))
print('Recall: %.4f' % recall_score(y_test, y_pred))
print('F1: %.4f' % f1_score(y_test, y_pred))
temp_vec = cross_validate(model, X_test, y_test, cv=10, scoring='roc_auc', return_train_score=True)
vis_cross_val_score('ROC-AUC', temp_vec, 0.744262, 1.1)
# блок поиска параметров закомментирован потому что он 
#выполняется очень долго, ниже модель использует оптимальные параметры
# l2 оегуляризация хуже работает для логистической регресси, убираем из расчетов

#model = LogisticRegression(multi_class = 'ovr', class_weight='balanced', random_state=RANDOM_SEED)

#param_grid = [
#    {'penalty': ['l1'], 'C':[0.02, 0.1, 0.2, 0.5, 1, 3, 10], 'max_iter':[1000],'tol':[1e-5], 'solver':['liblinear', 'saga']},
#    #{'penalty': ['l2'], 'C':[0.02, 0.1, 0.2, 0.5, 1, 3, 10], 'max_iter':[1000],'tol':[1e-5], 'solver':['newton-cg', 'lbfgs', 'sag', 'saga']},
#    {'penalty': ['none'], 'max_iter':[1000],'tol':[1e-5], 'solver':['lbfgs', 'newton-cg']},
#]
#gridsearch = GridSearchCV(model, param_grid, scoring='roc_auc', n_jobs=-1, cv=5)
#gridsearch.fit(X_train, y_train)
#model = gridsearch.best_estimator_



##печатаем параметры
#best_parameters = model.get_params()
#for param_name in sorted(best_parameters.keys()):
#        print('\t%s: %r' % (param_name, best_parameters[param_name]))

##печатаем метрики
#y_pred_prob = model.predict_proba(X_test)[:,1]
#y_pred = model.predict(X_test)
#preds = model.predict(X_test)
#print('Accuracy: %.4f' % accuracy_score(y_test, preds))
#print('Precision: %.4f' % precision_score(y_test, preds))
#print('Recall: %.4f' % recall_score(y_test, preds))
#print('F1: %.4f' % f1_score(y_test, preds))


# третья модель
model = LogisticRegression(C=2.15, 
                           penalty='l1', 
                           multi_class = 'ovr', 
                           class_weight='balanced', 
                           solver='saga', 
                           random_state=RANDOM_SEED,
                           max_iter = 1000)

model.fit(X_train, y_train)

y_pred_prob = model.predict_proba(X_test)[:,1]
y_pred = model.predict(X_test)

preds = model.predict(X_test)
print('Accuracy: %.4f' % accuracy_score(y_test, preds))
print('Precision: %.4f' % precision_score(y_test, preds))
print('Recall: %.4f' % recall_score(y_test, preds))
print('F1: %.4f' % f1_score(y_test, preds))
confusion_matrix_f(['Дефолтный','Не дефолтный'], y_test, y_pred, 1, normalize=False)
ROC_curve_with_area(y_test, y_pred_prob, 1.1)
PR_curve_with_area(y_test, y_pred, 1.1)
temp_vec = cross_validate(model, X_test, y_test, cv=10, scoring='roc_auc', return_train_score=True)
vis_cross_val_score('ROC-AUC', temp_vec, 0.744262, 1.1)
train = df.query('Train == 1').drop(['Train', 'client_id']+drop_list1, axis=1)
test = df.query('Train == 0').drop(['Train', 'client_id']+drop_list1, axis=1)
X_train=train.drop(['default'], axis=1)
y_train = train.default.values
X_test = test.drop(['default'], axis=1)

# проверяем
test.shape, train.shape, X_train.shape, y_train.shape, X_test.shape
LogisticRegression(C=2.15, 
                           penalty='l1', 
                           multi_class = 'ovr', 
                           class_weight='balanced', 
                           solver='saga', 
                           random_state=RANDOM_SEED,
                           max_iter = 2000)

model.fit(X_train, y_train)

y_pred_prob = model.predict_proba(X_test)[:,1]
submit = pd.DataFrame(df_test.client_id)
submit['default']=y_pred_prob
submit.to_csv('submission.csv', index=False)
