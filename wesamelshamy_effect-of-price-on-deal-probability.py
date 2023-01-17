"""Copy Keras pre-trained model files to work directory from:
https://www.kaggle.com/gaborfodor/keras-pretrained-models

Code from: https://www.kaggle.com/classtag/extract-avito-image-features-via-keras-vgg16/notebook
"""
import os

cache_dir = os.path.expanduser(os.path.join('~', '.keras'))
if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)

# Create symbolic links for trained models.
# Thanks to Lem Lordje Ko for the idea
# https://www.kaggle.com/lemonkoala/pretrained-keras-models-symlinked-not-copied
models_symlink = os.path.join(cache_dir, 'models')
if not os.path.exists(models_symlink):
    os.symlink('/kaggle/input/keras-pretrained-models/', models_symlink)

images_dir = os.path.expanduser(os.path.join('~', 'avito_images'))
if not os.path.exists(images_dir):
    os.makedirs(images_dir)
"""Extract images from Avito's advertisement image zip archive.

Code adapted from: https://www.kaggle.com/classtag/extract-avito-image-features-via-keras-vgg16/notebook
"""
import zipfile

NUM_IMAGES_TO_EXTRACT = 100

with zipfile.ZipFile('../input/avito-demand-prediction/train_jpg.zip', 'r') as train_zip:
    files_in_zip = sorted(train_zip.namelist())
    for idx, file in enumerate(files_in_zip[:NUM_IMAGES_TO_EXTRACT]):
        if file.endswith('.jpg'):
            train_zip.extract(file, path=file.split('/')[3])

!mv *.jpg/data/competition_files/train_jpg/* ~/avito_images
!rm -rf *.jpg
def get_avito_image(image_id):
    """Read image file without extracting it from Avito's training images archive file and return it.
    
    Args:
        image_id (str): Id of the image to read.
        
    Returns:
       file: Image file.  Can be read with plt.imread() 
    """
    with zipfile.ZipFile('../input/avito-demand-prediction/train_jpg.zip', 'r') as train_zip:
        archive_member = os.path.join('data/competition_files/train_jpg', image_id + '.jpg')
        member = train_zip.open(archive_member)
    return member

def is_image_extracted(image_id):
    """Check if extracted copy of the image exists.
    
    Args:
        image_id (str): Image id
        
    Returns:
        bool: True if extracted copy of the image exists.
    """
    dir_path = os.path.expanduser('~/avito_images')
    return os.path.exists(os.path.join(dir_path, 'data/competition_files/train_jpg' , image_id + '.jpg'))


def extract_avito_images(image_ids):
    """Extract images from Avito training images and return extracted images path.
    
    If extracted copies of all images exist, returns their path instead.
    Args:
        image_ids (list): Image ids to extract.
    
    Returns:
        list: Extracted images path.
    """
    ex = []
    if all([is_image_extracted(x) for x in image_ids]):
        dir_path = os.path.expanduser('~/avito_images')
        for image_id in image_ids:
            ex.append(os.path.join(dir_path, 'data/competition_files/train_jpg' , image_id + '.jpg'))
        return ex
    
    with zipfile.ZipFile('../input/avito-demand-prediction/train_jpg.zip', 'r') as train_zip:
        for image_id in image_ids:
            archive_member = os.path.join('data/competition_files/train_jpg', image_id + '.jpg')
            dest = os.path.expanduser('~/avito_images')
            if os.path.exists(os.path.join(dest, archive_member)):
                e = os.path.join(dest, archive_member)
            else:        
                e = train_zip.extract(archive_member, dest)
            ex.append(e)
    return ex


def show_avito_images(image_ids):
    """Plot images from Avito's image data set.
    
    Args:
        image_id (list): Ids of the images to plot.
        
    Returns:
        matplotlib axis
    """
    ncols = 3
    nrows = len(image_ids) // ncols + 1 - (len(image_ids) % ncols == 0)
    images_path = extract_avito_images(image_ids)
    fig, axes = plt.subplots(nrows, ncols, figsize=(20, 20))
    for i, (image_id, image_path) in enumerate(zip(image_ids, images_path)):
        image = Image.open(image_path)
        ax = axes[i//ncols, i%ncols]
        ax.imshow(image)
        ax.set_title(image_id)
        ax.axis('off')
    plt.show()
    return ax
# Dictionary of category names (Russian to English)
parent_cat_rus_eng = {
    'Личные вещи': 'Personal items',
    'Для дома и дачи': 'For home and cottages',
    'Бытовая электроника': 'Consumer electronics',
    'Недвижимость': 'Real estate',
    'Хобби и отдых': 'Hobbies and recreation',
    'Транспорт': 'Transportation',
    'Услуги': 'Services',
    'Животные':'Animals',
    'Для бизнеса': 'For business'
}

# Dictionary of parent category names (Russian to English)
cat_rus_en = {
    'Автомобили': 'Cars',
     'Аквариум': 'Aquarium',
     'Аудио и видео': 'Audio and video',
     'Билеты и путешествия': 'Tickets and travel',
     'Бытовая техника': 'Household appliances',
     'Велосипеды': 'Bicycles',
     'Водный транспорт': 'Water transport',
     'Гаражи и машиноместа': 'Garages and Parking spaces',
     'Готовый бизнес': 'Ready business',
     'Грузовики и спецтехника': 'Trucks and machinery',
     'Детская одежда и обувь': "Children's clothing and footwear",
     'Дома, дачи, коттеджи': 'Houses, cottages, cottages',
     'Другие животные': 'Other animals',
     'Земельные участки': 'Land plots',
     'Игры, приставки и программы': 'Games, consoles and programs',
     'Квартиры': 'Apartments',
     'Книги и журналы': 'Books and magazines',
     'Коллекционирование': 'Collecting',
     'Коммерческая недвижимость': 'Commercial real estate',
     'Комнаты': 'Rooms',
     'Кошки': 'Cats',
     'Красота и здоровье': 'Health and beauty',
     'Мебель и интерьер': 'Furniture and interior',
     'Мотоциклы и мототехника': 'Motorcycles and motor vehicles',
     'Музыкальные инструменты': 'Musical instruments',
     'Настольные компьютеры': 'Desktop computers',
     'Недвижимость за рубежом': 'Real estate abroad',
     'Ноутбуки': 'Laptops',
     'Оборудование для бизнеса': 'Business equipment',
     'Одежда, обувь, аксессуары': 'Clothes, shoes, accessories',
     'Оргтехника и расходники': 'Office equipment and consumables',
     'Охота и рыбалка': 'Hunting and fishing',
     'Планшеты и электронные книги': 'Tablets and e-books',
     'Посуда и товары для кухни': 'Kitchen utensils and goods',
     'Предложение услуг': 'Offer of services',
     'Продукты питания': 'Food',
     'Птицы': 'Birds',
     'Растения': 'Plants',
     'Ремонт и строительство': 'Repair and construction',
     'Собаки': 'Dogs',
     'Спорт и отдых': 'Sport and recreation',
     'Телефоны': 'Phones',
     'Товары для детей и игрушки': 'Goods for children and toys',
     'Товары для животных': 'Animal products',
     'Товары для компьютера': 'Goods for computer',
     'Фототехника': 'Photo equipment',
     'Часы и украшения': 'Watches and jewelry'
}
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
sns.set()
plt.rc('font', size=16)

%matplotlib inline
usecols = ['item_id', 'parent_category_name', 'category_name', 'price', 'image']
train_df = pd.read_csv('../input/avito-demand-prediction/train.csv', usecols=usecols)
test_df = pd.read_csv('../input/avito-demand-prediction/test.csv', usecols=usecols)
df = pd.concat([train_df, test_df], axis=0)

df.category_name = df.category_name.map(lambda x: cat_rus_en[x])
df.parent_category_name = df.parent_category_name.map(lambda x: parent_cat_rus_eng[x])
df.head()
cheap = df[df.price < 10].category_name.value_counts().map(lambda x: '{:.2f}%'.format(x/df.shape[0] * 100))
print('Categories of items < 10 \u20BD (top 10)')
cheap.head(10)
expensive = df[df.price > 1000000].category_name.value_counts().map(lambda x: '{:.2f}%'.format(x/df.shape[0] * 100))
print('Categories of items > 1M \u20BD (top 10)')
expensive.head(10)
na_count = df.price.isna().sum()
zero_count = df[df.price < 10].price.count()
mill_count = df[df.price > 1000000].price.count()
total_count = df.shape[0]
print('Ads with missing price:\t{:,}\t({:.2f}%)'.format(na_count, na_count/total_count*100))
print('Ads with price < 10 \u20BD:\t{:,}\t({:.2f}%)'.format(zero_count, zero_count/total_count*100))
print('Ads with price > 1M \u20BD:\t{:,}\t({:.2f}%)'.format(mill_count, mill_count/total_count*100))
total_dropped = na_count + zero_count + mill_count
print('---------------------------------------')
print('Total dropped:\t\t{:,}\t({:.2f}%)'.format(total_dropped, total_dropped/total_count*100))
dff = df[(df.price > 10) & (df.price < 1000000)] # Drop missing price and < 10 Ruble
price_log = np.log(dff.price)
out, bins = pd.cut(price_log, bins=100, retbins=True, labels=False)

plt.figure(figsize=(16, 5))
ax = sns.distplot(price_log, axlabel='Log(price)')
plt.title('Histogram of the Logarithmic value of the price.  Prices under 10 \u20BD and above 1M \u20BD are dropped.\nVertical lines show bin boundaries.')
plt.vlines(bins, color='g', ymin=0, ymax=0.1, alpha=0.3)
plt.show()
cats = dff.category_name.unique()
dfa = dff[dff.category_name.isin(cats[0: 23])][['category_name', 'price']]
category_name = dfa.category_name
price_log = np.log(dfa.price)

plt.figure(figsize=(20, 8))
ax = sns.violinplot(x=category_name, y=price_log, scale='width', palette='Set3')
plt.ylim([1, 15])
plt.xticks(rotation=40, fontsize=14)
plt.ylabel('Log(price) \u20BD')
plt.title('Price variance within categories (1 of 2)')
plt.show()
dfb = dff[dff.category_name.isin(cats[23: 47])][['category_name', 'price']]
category_name = dfb.category_name
price_log = np.log(dfb.price)
plt.figure(figsize=(20, 8))

ax = sns.violinplot(x=category_name, y=price_log, scale='width', palette='Set3')
plt.ylim([1, 15])
plt.xticks(rotation=40, fontsize=14)
plt.title('Price variance within categories (2 of 2)')
plt.ylabel('Log(price) \u20BD')
plt.show()
parent_category = dff.parent_category_name
category = dff.category_name
price_log = np.log(dff.price)

plt.figure(figsize=(20, 8))
ax = sns.violinplot(x=parent_category, y=price_log, scale='width', palette='Set3')
plt.ylim([1, 15])
plt.xticks(rotation=40, fontsize=14)
plt.title('Price variance within parent categories')
plt.ylabel('Log(price) \u20BD')
plt.show()
print('Coefficient of variation (CV) for prices in different categories (category_name).')
dffd = dff.groupby('category_name')['price'].apply(lambda x: np.std(x)/np.mean(x)).sort_values(ascending=False)
dffd
train_image_labels = pd.read_csv('../input/avito-images-recognized/train_image_labels.csv', index_col='image_id')
test_image_labels = pd.read_csv('../input/avito-images-recognized/test_image_labels.csv', index_col='image_id')
all_image_labels = pd.concat([train_image_labels, test_image_labels], axis=0)
from PIL import Image

dir_iter = os.scandir(os.path.expanduser('~/avito_images'))
fig, axes = plt.subplots(5, 5, figsize=(20, 20))
for i in range(25):
    e = next(dir_iter)
    img = Image.open(e.path)
    img = img.resize((360, 360))
    ax = axes[i//5, i%5]
    ax.imshow(img)
    im_id = e.name.split('.')[0]
    im_label = train_image_labels.loc[im_id]['image_label']
    ax.set_title(im_label, fontsize=24)
    ax.axis('off')
plt.show()
print('Coefficient of variation (CV) for prices in different recognized image categories.')
dfl = dff.merge(all_image_labels, left_on='image', right_index=True, how='left')
dfd = dfl.groupby('image_label')['price'].apply(lambda x: np.std(x)/np.mean(x)).sort_values(ascending=False)
dfd.head(10)
dfd2 = dfl.groupby('image_label')['price'].apply(lambda x: np.std(np.log(x))/np.mean(np.log(x)))
dffd2 = dfl.groupby('category_name')['price'].apply(lambda x: np.std(np.log(x))/np.mean(np.log(x)))
plt.figure(figsize=(16, 5))
sns.set()
ax = sns.distplot(dfd2.values, color='darkred', hist_kws={'alpha': 0.3})
ax = sns.distplot(dffd2.values, color='darkgreen', hist_kws={'alpha': 0.3}, ax=ax)
ax.set_title('CV of image_lable and category_name')
ax.set_xlabel('CV of $Log(price)$')
ax.legend(['image label', 'category_name'])
plt.show()
train_all = pd.read_csv('../input/avito-demand-prediction/train.csv', usecols=['param_1', 'param_2', 'param_3', 'price', 'deal_probability'])
param2v = train_all[(train_all.price > 0) & (train_all.price < 1000000)].groupby('param_2')['price'].apply(lambda x: np.std(np.log(x))/np.mean(np.log(x)))

plt.figure(figsize=(16, 5))
sns.set()
ax = sns.distplot(dfd2.values, color='darkred', hist_kws={'alpha': 0.3})
ax = sns.distplot(dffd2.values, color='darkgreen', hist_kws={'alpha': 0.3}, ax=ax)
ax = sns.distplot(param2v.values, color='navy', hist_kws={'alpha': 0.3}, ax=ax)

ax.set_title('CV of image_label, category_name, and param_2')
ax.set_xlabel('CV of $Log(price)$')
ax.legend(['image label', 'category_name', 'param_2'])
plt.show()
shoes = train_all[train_all.param_2 == 'Обувь'][['price', 'deal_probability']]
outer = train_all[train_all.param_2 == 'Верхняя одежда'][['price', 'deal_probability']]
dress = train_all[train_all.param_2 == 'Платья и юбки'][['price', 'deal_probability']]
knit = train_all[train_all.param_2 == 'Трикотаж'][['price', 'deal_probability']]

plt.figure(figsize=(16, 5))
fontsize=18

ax1 = plt.subplot(121)
plt.scatter(shoes.price, shoes.deal_probability, s=3)
x = plt.setp(ax1.get_yticklabels(), fontsize=fontsize)
plt.xlim([0, 6000])
plt.xlabel('Price \u20BD', fontsize=fontsize)
plt.xticks(fontsize=fontsize)
plt.yticks(fontsize=fontsize)
plt.ylabel('Deal probability', fontsize=fontsize)
ax = plt.title('Shoes', fontsize=fontsize)

ax2 = plt.subplot(122)
plt.scatter(outer.price, outer.deal_probability, s=3)
plt.setp(ax2.get_yticklabels(), visible=False)
plt.xlim([0, 7000])
plt.xlabel('Price \u20BD', fontsize=fontsize)
plt.xticks(fontsize=fontsize)
ax = plt.title('Outerwear', fontsize=fontsize)

# ax3 = plt.subplot(133)
# plt.scatter(dress.price, dress.deal_probability, s=3)
# x = plt.setp(ax3.get_yticklabels(), visible=False)
# plt.xlim([0, 6000])
# plt.xlabel('Price \u20BD')
# ax = plt.title('Dresses')

# ax4 = plt.subplot(144, sharex=ax1, sharey=ax1)
# plt.scatter(knit.price, knit.deal_probability)
# x = plt.setp(ax4.get_yticklabels(), visible=False)
# plt.xlim([0, 2000])
# plt.xlabel('Price \u20BD')
# ax = plt.title('Knitwear')

plt.tight_layout()