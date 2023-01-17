import numpy as np

import pandas as pd 

import os

print(os.listdir("../input"))
breeds = pd.read_csv('../input/breed_labels.csv')

colors = pd.read_csv('../input/color_labels.csv')

states = pd.read_csv('../input/state_labels.csv')

train = pd.read_csv('../input/train/train.csv')

test = pd.read_csv('../input/test/test.csv')



train.head()
import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

from PIL import Image

import gc
print("train.csv shape is {}".format(train.shape))

print("test.csv shape is {}".format(test.shape))
fig, ax = plt.subplots(figsize=(6, 5))

ax.set_title("AdoptionSpeed count in train")

sns.countplot(x="AdoptionSpeed", data=train, ax=ax)
sns.distplot(train['Age'], bins = 100, kde=False)
train.Age.unique()
#Удаляю средний возраст больше 20 и заменяю средним значением

train['Age'] = train['Age'].fillna( method='pad')

train.loc[train['Age'] > 20, 'Age'] = train['Age'].mean()
sns.distplot(train['Age'], bins = 20, kde=False)
breed_dict = dict(zip(breeds['BreedID'], breeds['BreedName']))
animal_dict = {1: 'Dog', 2: 'Cat'}
train.head()
import warnings

warnings.filterwarnings("ignore", 'This pattern has match groups')



def make_features(train):

    train['Named'] = (~train['Name'].isnull() | train['Name'].str.contains('(No | Puppies | Kitty | Puppy | Kitten)')).astype(int)

    train['PureBreed'] = (train.Breed2==0 | ~train.Breed1.isin(breeds[breeds.BreedName.str.contains('Domestic' or 'Mixed')].BreedID.values)).astype(int)

    train = train.drop('Breed2', axis=1)



    train = train.select_dtypes(exclude=[object])

    return train



dealed_train = make_features(train)
dealed_train['Type'] = dealed_train['Type'].map(animal_dict)

dealed_train['Breed1_new'] = dealed_train['Breed1'].map(breed_dict)
#Почему-то после применения функции make_features у меня пропадает PetID

#Поэтому я пойду вот на таких костылях

train_2 =  dealed_train.combine_first(train)

train = train_2.drop('Breed1', axis=1)

train = train.drop('Breed2', axis=1)

train = train.drop('Named', axis=1)
#Делаю аналогичные преображения с тестовой выборкой

dealed_test = make_features(test)

dealed_test['Type'] = dealed_test['Type'].map(animal_dict)

dealed_test['Breed1_new'] = dealed_test['Breed1'].map(breed_dict)

test_2 =  dealed_test.combine_first(test)

test = test_2.drop('Breed1', axis=1)

test = test.drop('Breed2', axis=1)

test = test.drop('Named', axis=1)
train_dogs = train[dealed_train['Type'] == 'Dog']

train_cats = train[dealed_train['Type'] == 'Cat']
plt.figure(figsize=(28, 12));

sns.barplot(x='PhotoAmt', y='AdoptionSpeed', data=train)

plt.title('What about photos?')
#Создаем для кошек датафреймы для описания распределений скорости adoption в разрезах различных категориальных фич

ms_train_cats = pd.DataFrame(train_cats.groupby([ 'AdoptionSpeed', 'MaturitySize'])['PetID'].count()).reset_index()

fr_train_cats = pd.DataFrame(train_cats.groupby(['AdoptionSpeed', 'FurLength'])['PetID'].count()).reset_index()

vc_train_cats = pd.DataFrame(train_cats.groupby(['AdoptionSpeed', 'Vaccinated'])['PetID'].count()).reset_index()

dw_train_cats = pd.DataFrame(train_cats.groupby(['AdoptionSpeed', 'Dewormed'])['PetID'].count()).reset_index()

st_train_cats = pd.DataFrame(train_cats.groupby(['AdoptionSpeed', 'Sterilized'])['PetID'].count()).reset_index()

h_train_cats = pd.DataFrame(train_cats.groupby(['AdoptionSpeed', 'Health'])['PetID'].count()).reset_index()

age_train_cats = pd.DataFrame(train_cats.groupby(['AdoptionSpeed', 'Age'])['PetID'].count()).reset_index()
#Добавляем колонку в датафреймы с разрезами с долей питомцев от общего числа для каждого значения категориальной фичи

ms_train_cats['PetID_Share'] = ms_train_cats['PetID']/14993

fr_train_cats['PetID_Share'] = ms_train_cats['PetID']/14993

vc_train_cats['PetID_Share'] = ms_train_cats['PetID']/14993

dw_train_cats['PetID_Share'] = ms_train_cats['PetID']/14993

st_train_cats['PetID_Share'] = ms_train_cats['PetID']/14993

h_train_cats['PetID_Share'] = ms_train_cats['PetID']/14993

age_train_cats['PetID_Share'] = ms_train_cats['PetID']/14993

#Если честно, не понимаю, для чего были совершены эти действия и предлагаю это потом удалить ))
#распределение размеров по количеству питомцев

train_cats.groupby(['MaturitySize'])['PetID'].count()
sns.distplot(train_cats['MaturitySize'])
#распределение размеров питомцев в разрезе adoption скорости

train_cats.groupby(['MaturitySize', 'AdoptionSpeed'])['PetID'].count()
# Хитмап по размеру питомца

pv_ms_train_cats = ms_train_cats.pivot_table(values='PetID_Share', index='MaturitySize',   columns='AdoptionSpeed')

sns.heatmap(pv_ms_train_cats, cmap='inferno_r')
# Утащила у каких-то студентиков https://www.kaggle.com/tgibbons/student-project-looking-for-feedback

encodedColor1 = pd.get_dummies( train['Color1'], prefix="color" )



# Add the new dummy variables to the pet_train data frame

train = pd.concat([train, encodedColor1], axis='columns')

# Do the same thing to the submission data

encodedColor2 = pd.get_dummies( test['Color1'], prefix="color" )

test = pd.concat([test, encodedColor2], axis='columns')



# print out the current data

print ("Size of pet_train = ", train.shape)

print ("Size of pet_submit = ", test.shape)

train.head(5)
cat_columns = ['Breed1_new','FurLength','Dewormed']



# Create the dummy variables for the columns listed above

dfTemp = pd.get_dummies( train[cat_columns], columns=cat_columns )

train = pd.concat([train, dfTemp], axis='columns')



# Do the same to the submission data

dfSummit = pd.get_dummies( test[cat_columns], columns=cat_columns )

test = pd.concat([test, dfSummit], axis='columns')

# Get missing columns in the submission data

missing_cols = set( train.columns ) - set( test.columns )

# Add a missing column to the submission set with default value equal to 0

for c in missing_cols:

    test[c] = 0

# Ensure the order of column in the test set is in the same order than in train set

test = test[train.columns]









# print out the current data

print ("Size of pet_train = ", train.shape)

print ("Size of pet_submit = ", test.shape)

train.head(5)
import cv2

import pandas as pd

import numpy as np

import os

from tqdm import tqdm, tqdm_notebook

from keras.applications.densenet import preprocess_input, DenseNet121



img_size = 256

batch_size = 16
pet_ids = train['PetID'].values

n_batches = len(pet_ids) // batch_size + 1
# The following functions and the rest solution have been taken from Valentina's kernel

# Thank you <3

def confusion_matrix(rater_a, rater_b, min_rating=None, max_rating=None):

    """

    Returns the confusion matrix between rater's ratings

    """

    assert(len(rater_a) == len(rater_b))

    if min_rating is None:

        min_rating = min(rater_a + rater_b)

    if max_rating is None:

        max_rating = max(rater_a + rater_b)

    num_ratings = int(max_rating - min_rating + 1)

    conf_mat = [[0 for i in range(num_ratings)]

                for j in range(num_ratings)]

    for a, b in zip(rater_a, rater_b):

        conf_mat[a - min_rating][b - min_rating] += 1

    return conf_mat





def histogram(ratings, min_rating=None, max_rating=None):

    """

    Returns the counts of each type of rating that a rater made

    """

    if min_rating is None:

        min_rating = min(ratings)

    if max_rating is None:

        max_rating = max(ratings)

    num_ratings = int(max_rating - min_rating + 1)

    hist_ratings = [0 for x in range(num_ratings)]

    for r in ratings:

        hist_ratings[r - min_rating] += 1

    return hist_ratings





def quadratic_weighted_kappa(y, y_pred):

    """

    Calculates the quadratic weighted kappa

    axquadratic_weighted_kappa calculates the quadratic weighted kappa

    value, which is a measure of inter-rater agreement between two raters

    that provide discrete numeric ratings.  Potential values range from -1

    (representing complete disagreement) to 1 (representing complete

    agreement).  A kappa value of 0 is expected if all agreement is due to

    chance.

    quadratic_weighted_kappa(rater_a, rater_b), where rater_a and rater_b

    each correspond to a list of integer ratings.  These lists must have the

    same length.

    The ratings should be integers, and it is assumed that they contain

    the complete range of possible ratings.

    quadratic_weighted_kappa(X, min_rating, max_rating), where min_rating

    is the minimum possible rating, and max_rating is the maximum possible

    rating

    """

    rater_a = y

    rater_b = y_pred

    min_rating=None

    max_rating=None

    rater_a = np.array(rater_a, dtype=int)

    rater_b = np.array(rater_b, dtype=int)

    assert(len(rater_a) == len(rater_b))

    if min_rating is None:

        min_rating = min(min(rater_a), min(rater_b))

    if max_rating is None:

        max_rating = max(max(rater_a), max(rater_b))

    conf_mat = confusion_matrix(rater_a, rater_b,

                                min_rating, max_rating)

    num_ratings = len(conf_mat)

    num_scored_items = float(len(rater_a))



    hist_rater_a = histogram(rater_a, min_rating, max_rating)

    hist_rater_b = histogram(rater_b, min_rating, max_rating)



    numerator = 0.0

    denominator = 0.0



    for i in range(num_ratings):

        for j in range(num_ratings):

            expected_count = (hist_rater_a[i] * hist_rater_b[j]

                              / num_scored_items)

            d = pow(i - j, 2.0) / pow(num_ratings - 1, 2.0)

            numerator += d * conf_mat[i][j] / num_scored_items

            denominator += d * expected_count / num_scored_items



    return (1.0 - numerator / denominator)
# Кажется, это адский алгоритм для вычисления этой самой каппы

# Но я ни в чем не уверена

class OptimizedRounder(object):

    def __init__(self):

        self.coef_ = 0



    def _kappa_loss(self, coef, X, y):

        X_p = np.copy(X)

        for i, pred in enumerate(X_p):

            if pred < coef[0]:

                X_p[i] = 0

            elif pred >= coef[0] and pred < coef[1]:

                X_p[i] = 1

            elif pred >= coef[1] and pred < coef[2]:

                X_p[i] = 2

            elif pred >= coef[2] and pred < coef[3]:

                X_p[i] = 3

            else:

                X_p[i] = 4



        ll = quadratic_weighted_kappa(y, X_p)

        return -ll



    def fit(self, X, y):

        loss_partial = partial(self._kappa_loss, X=X, y=y)

        initial_coef = [0.5, 1.5, 2.5, 3.5]

        self.coef_ = sp.optimize.minimize(loss_partial, initial_coef, method='nelder-mead')



    def predict(self, X, coef):

        X_p = np.copy(X)

        for i, pred in enumerate(X_p):

            if pred < coef[0]:

                X_p[i] = 0

            elif pred >= coef[0] and pred < coef[1]:

                X_p[i] = 1

            elif pred >= coef[1] and pred < coef[2]:

                X_p[i] = 2

            elif pred >= coef[2] and pred < coef[3]:

                X_p[i] = 3

            else:

                X_p[i] = 4

        return X_p



    def coefficients(self):

        return self.coef_['x']

    

def rmse(actual, predicted):

    return sqrt(mean_squared_error(actual, predicted))
#Что здесь происходит?

#Мне страшно

from keras.models import Model

from keras.layers import GlobalAveragePooling2D, Input, Lambda, AveragePooling1D

import keras.backend as K

from keras.applications.densenet import DenseNet121

inp = Input((256,256,3))

backbone = DenseNet121(input_tensor = inp, 

                       weights=None, #Мне не удалось загрузить весы, которые были у Валентины в отдельном файле, поэтому пускай алгоритм сам учится

                       include_top = False)

x = backbone.output

x = GlobalAveragePooling2D()(x)

x = Lambda(lambda x: K.expand_dims(x,axis = -1))(x)

x = AveragePooling1D(4)(x)

out = Lambda(lambda x: x[:,:,0])(x)



m = Model(inp,out)

#Это функция, которая превращает изображения в квадратики?? Серьезно?

#Но зачем?

def resize_to_square(im):

    old_size = im.shape[:2] # old_size is in (height, width) format

    ratio = float(img_size)/max(old_size)

    new_size = tuple([int(x*ratio) for x in old_size])

    # new_size should be in (width, height) format

    im = cv2.resize(im, (new_size[1], new_size[0]))

    delta_w = img_size - new_size[1]

    delta_h = img_size - new_size[0]

    top, bottom = delta_h//2, delta_h-(delta_h//2)

    left, right = delta_w//2, delta_w-(delta_w//2)

    color = [0, 0, 0]

    new_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT,value=color)

    return new_im



def load_image(path, pet_id):

    image = cv2.imread(f'{path}{pet_id}-1.jpg')

    new_image = resize_to_square(image)

    new_image = preprocess_input(new_image)

    return new_image
features = {}

for b in tqdm_notebook(range(n_batches)):

    start = b*batch_size

    end = (b+1)*batch_size

    batch_pets = pet_ids[start:end]

    batch_images = np.zeros((len(batch_pets),img_size,img_size,3))

    for i,pet_id in enumerate(batch_pets):

        try:

            batch_images[i] = load_image("../input/train_images/", pet_id)

        except:

            pass

    batch_preds = m.predict(batch_images)

    for i,pet_id in enumerate(batch_pets):

        features[pet_id] = batch_preds[i]

#К сожалению, мне не удалось удалить дубли изображений, тк у макбуков какая-то проблема с CUDA

#Или у меня какая-то проблема с CUDA

#Поэтому все будет долго и мучительно