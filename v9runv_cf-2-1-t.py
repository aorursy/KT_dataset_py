from nltk.tokenize import RegexpTokenizer

from nltk.corpus import stopwords

import nltk

from nltk.stem import WordNetLemmatizer

from sklearn.base import TransformerMixin,BaseEstimator

import numpy as np

import pandas as pd

import os

import matplotlib.pyplot as plt

import seaborn as sns

import plotly as ply

import xlrd 

from scipy.stats import norm

import re

from collections import Counter

from sklearn.decomposition import PCA,TruncatedSVD

import warnings

warnings.simplefilter(action='ignore')



test_data = pd.read_csv(r'../input/consultation-csv/Consultation_Test.csv')

train_data = pd.read_csv(r'../input/consultation-csv-train/Consultation_Train.csv',engine='python')



tokenizer = RegexpTokenizer(r'\w+')

lemmatizer = WordNetLemmatizer()

stop_words = set(stopwords.words("english"))



#preprocessing

from scipy.sparse import csr_matrix

from sklearn.preprocessing import StandardScaler,OneHotEncoder



def clean_machine(x,d,n):

    try:

        word = x.split(d)[n]

        word = word.lower()

        word = word.replace(" ","")

        word = re.sub(r"[,.;@#?!&$-]+\ *","",word)

        return word

    except AttributeError:

        return -1

train_data['Exp'] =  train_data['Experience'].apply(lambda x: int(clean_machine(x,' ',0))).values.reshape(-1,1)



tokenizer = RegexpTokenizer(r'\w+')

lemmatizer = WordNetLemmatizer()

stop_words = set(stopwords.words("english"))

from scipy.stats import rankdata



#preprocessing

from scipy.sparse import csr_matrix

from sklearn.preprocessing import StandardScaler,OneHotEncoder



common_words = ['the','be','to','of','and','a','in',

                'that','have','i','it','for','not','on','with',

                'he','as','you','do','at','this','but','his','by',

                'from','they','we','say','her','she','or','an','will',

                'my','one','all','would','there','their','what','so',

                'up','out','if','about','who','get','which','go','me','when',

                'make','can','like','time','no','just','him','know','take','people',

                'into','year','your','good','some','could','them','see','other','than',

                'then','now','look','only','come','its','over','think','also','back','after',

                'use','two','how','our','work','first','well','way','even','new','want','because',

                'any','these','give','day','most','us','full','life','pain','ship','base','root',

               'afghanistan','albania','algeria','american samoa','andorra','angola','anguilla','antarctica',

                'antigua and barbuda','argentina','armenia','aruba','australia','austria','azerbaijan','bahamas',

                'bahrain','bangladesh','barbados','belarus','belgium','belize','benin','bermuda','bhutan','bolivia',

                'bonaire','bosnia and herzegovina','botswana','bouvet island','brazil','british indian ocean territory',

                'brunei darussalam','bulgaria','burkina faso','burundi','cambodia','cameroon','canada','cape verde','cayman islands',

                'central african republic','chad','chile','china','christmas island','cocos (keeling) islands','colombia','comoros','congo',

                'democratic republic of the congo','cook islands','costa rica','croatia','cuba','curacao','cyprus','czech republic','denmark',

                'djibouti','dominica','dominican republic','ecuador','egypt','el salvador','equatorial guinea','eritrea','estonia','ethiopia',

                'falkland islands (malvinas)','faroe islands','fiji','finland','france','french guiana','french polynesia','french southern territories',

                'gabon','gambia','georgia','germany','ghana','gibraltar','greece','greenland','grenada','guadeloupe','guam','guatemala','guernsey','guinea',

                'guinea-bissau','guyana','haiti','heard island and mcdonald islands','holy see (vatican city state)','honduras','hong kong','hungary','iceland',

                'india','indonesia','iran, islamic republic of','iraq','ireland','isle of man','israel','italy','jamaica','japan','jersey','jordan','kazakhstan',

                'kenya','kiribati','korea, republic of','kuwait','kyrgyzstan','latvia',

                'lebanon','lesotho','liberia','libya','liechtenstein','lithuania','luxembourg','macao','macedonia, the former yugoslav republic of','madagascar',

                'malawi','malaysia','maldives','mali','malta','marshall islands','martinique','mauritania','mauritius','mayotte','mexico',

                'micronesia, federated states of','moldova, republic of','monaco','mongolia','montenegro','montserrat','morocco','mozambique','myanmar',

                'namibia','nauru','nepal','netherlands','new caledonia','new zealand','nicaragua','niger','nigeria','niue','norfolk island',

                'northern mariana islands','norway','oman','pakistan','palau','palestine, state of','panama','papua new guinea','paraguay','peru',

                'philippines','pitcairn','poland','portugal','puerto rico','qatar','romania','russian federation','rwanda','reunion','saint barthelemy',

                'saint helena','saint kitts and nevis','saint lucia','saint martin (french part)','saint pierre and miquelon','saint vincent and the grenadines',

                'samoa','san marino','sao tome and principe','saudi arabia','senegal','serbia','seychelles','sierra leone','singapore','sint maarten (dutch part)',

                'slovakia','slovenia','solomon islands','somalia','south africa','south georgia and the south sandwich islands','south sudan','spain','sri lanka',

                'sudan','suriname','svalbard and jan mayen','swaziland','sweden','switzerland','syrian arab republic','taiwan','tajikistan','united republic of tanzania',

                'thailand','timor-leste','togo','tokelau','tonga','trinidad and tobago','tunisia','turkey','turkmenistan','turks and caicos islands','tuvalu',

                'uganda','ukraine','united arab emirates','united kingdom','united states','united states minor outlying islands','uruguay','uzbekistan','vanuatu',

                'venezuela','viet nam','british virgin islands','us virgin islands','wallis and futuna','western sahara','yemen','zambia','zimbabwe','af','al','dz',

                'as','ad','ao','ai','aq','ag','ar','am','aw','au','at','az','bs','bh','bd','bb','by','be','bz','bj','bm','bt','bo','bq','ba','bw','bv','br','io','bn',

                'bg','bf','bi','kh','cm','ca','cv','ky','cf','td','cl','cn','cx','cc','co','km','cg','cd','ck','cr','hr','cu','cw','cy','cz','ci','dk','dj','dm','do','ec',

                'eg','sv','gq','er','ee','et','fk','fo','fj','fi','fr','gf','pf','tf','ga','gm','ge','de','gh','gi','gr','gl','gd','gp','gu','gt','gg','gn','gw','gy','ht','hm',

                'va','hn','hk','hu','is','in','id','ir','iq','ie','im','il','it','jm','jp','je','jo','kz','ke','ki','kp','kr','kw','kg','la','lv','lb','ls','lr','ly','li','lt','lu','mo',

                'mk','mg','mw','my','mv','ml','mt','mh','mq','mr','mu','yt','mx','fm','md','mc','mn','me','ms','ma','mz','mm','na','nr','np','nl','nc','nz','ni','ne','ng','nu','nf','mp','no','om','pk','pw','ps','pa','pg','py','pe',

                'ph','pn','pl','pt','pr','qa','ro','ru','rw','re','bl','sh','kn','lc','mf','pm','vc','ws','sm','st','sa','sn','rs','sc','sl','sg','sx','sk','si','sb','so','za','gs','ss','es','lk','sd','sr','sj','sz','se','ch','sy','tw',

                'tj','tz','th','tl','tg','tk','to','tt','tn','tr','tm','tc','tv','ug','ua','ae','gb','us','um','uy','uz','vu','ve','vn','vg','vi','wf','eh','ye','zm','zw','afg','alb','dza','asm','and','ago','aia','ata','atg','arg','arm','abw',

                'aus','aut','aze','bhs','bhr','bgd','brb','blr','bel','blz','ben','bmu','btn','bol',

                'bes','bih','bwa','bvt','bra','iot','brn','bgr','bfa','bdi','khm','cmr','can','cpv','cym','caf','tcd','chl','chn','cxr','cck','col','com',

                'cog','cod','cok','cri','hrv','cub','cuw','cyp','cze','civ','dnk','dji','dma','dom','ecu','egy','slv','gnq','eri','est','eth','flk','fro',

                'fji','fin','fra','guf','pyf','atf','gab','gmb','geo','deu','gha','gib','grc','grl','grd','glp','gum','gtm','ggy','gin','gnb','guy','hti',

                'hmd','vat','hnd','hkg','hun','isl','ind','idn','irn','irq','irl','imn','isr','ita','jam','jpn','jey','jor','kaz','ken','kir','prk','kor',

                'kwt','kgz','lao','lva','lbn','lso','lbr','lby','lie','ltu','lux','mac','mkd','mdg','mwi','mys','mdv','mli','mlt','mhl','mtq','mrt','mus','myt',

                'mex','fsm','mda','mco','mng','mne','msr','mar','moz','mmr','nam','nru','npl','nld','ncl','nzl','nic','ner','nga','niu','nfk','mnp','nor','omn',

                'pak','plw','pse','pan','png','pry','per','phl','pcn','pol','prt','pri','qat','rou','rus','rwa','reu','blm','shn','kna','lca','maf','spm',

                'vct','wsm','smr','stp','sau','sen','srb','syc','sle','sgp','sxm','svk','svn','slb','som','zaf','sgs','ssd','esp','lka','sdn','sur','sjm',

                'swz','swe','che','syr','twn','tjk','tza','tha','tls','tgo','tkl','ton','tto','tun','tur','tkm','tca','tuv','uga','ukr','are','gbr','usa',

                'umi','ury','uzb','vut','ven','vnm','vgb','vir','wlf','esh','yem','zmb','zwe','uk','head','face','hair','ear','neck','forehead','beard',

                'eye','nose','mouth','chin','shoulder','elbow','arm','chest','armpit','forearm','wrist','back','navel','toes',

                'ankle','instep','toenail','waist','abdomen','buttock','hip','leg','thigh','knee','foot','hand','thumb','skin','lip','food','39','exam','loss','body','lift']



def clean_machine(x,d,n):

    try:

        word = x.split(d)[n]

        word = word.lower()

        word = word.replace(" ","")

        word = re.sub(r"[,.;@#?!&$-]+\ *","",word)

        return word

    except AttributeError:

        return -1



class qualifications(TransformerMixin,BaseEstimator):

    def __init__(self,columns=None):

        self.columns = columns

    def fit(self,X,y=None):

        return self

    def transform(self,X):

        buffer = []

        for lines in tokenizer.tokenize_sents(X[self.columns]):

            lines = [word.replace('diploma','dipl') if(word == 'diploma') else word.lower() for word in lines]

            filtered_lines = [word for word in lines if (word not in stop_words) 

                              and (word not in common_words) 

                              and ((len(word)<=4) & (len(word)>1))

                              ]

            self.filtered_lines = set(filtered_lines)

            buffer.append(Counter(filtered_lines))

        

        return np.array(buffer)

    

class qualifications_dict(TransformerMixin,BaseEstimator):

    def __init__(self,vocabulary_size=1000):

        self.vocabulary_size = vocabulary_size

    def fit(self,X,y=None):

        total_count=Counter()

        for each_dict in X:

            for keys,values in each_dict.items():

                total_count[keys] += min(values,100)

        most_common = total_count.most_common()[:self.vocabulary_size]

        self.most_common_ = most_common

        self.vocabulary_ = {word: index + 1 for index, (word, count) in enumerate(most_common)}

        return self

    def transform(self,X):

        rows = []

        cols = []

        data = []

        for row, value in enumerate(X):

            for word, count in value.items():

                rows.append(row)

                cols.append(self.vocabulary_.get(word, 0))

                data.append(count)

        return csr_matrix((data, (rows, cols)), shape=(len(X), self.vocabulary_size + 1))[:,1:]



class cat_prep(TransformerMixin,BaseEstimator):

    def __init__(self):

        self

    def fit(self,X,y=None):

        return self

        

    def transform(self,X):

        city = X.Place.apply(lambda x: clean_machine(x,',',-1))

        locality = X.Place.apply(lambda x: clean_machine(x,',',0))

        profile = X.Profile

        clean_data = np.c_[city,locality,profile]

        return clean_data.astype(str)

    

class num_prep(TransformerMixin,BaseEstimator):

    def __init__(self,columns=None):

        self.columns = columns        

    def fit(self,X,y=None):

        return self

    def transform(self,X):

        return X[self.columns].apply(lambda x: int(clean_machine(x,' ',0))).values.reshape(-1,1)

         

    

from sklearn.pipeline import Pipeline,FeatureUnion

from sklearn.compose import ColumnTransformer

#pipelines



num_pipe = Pipeline([

   ('num',num_prep(columns='Experience')),

   ('sts',StandardScaler()) 

])



cat_pipe = Pipeline([

   ('cat',cat_prep()),

   ('ohe',OneHotEncoder(handle_unknown='ignore')),

    ('sts',StandardScaler(with_mean=False))

#     ('sts',StandardScaler(with_mean=False))

])

quali_pipe = Pipeline([

   ('qual',qualifications(columns='Qualification')),

   ('qual_dict',qualifications_dict(vocabulary_size=150)),

    ('sts',StandardScaler(with_mean=False)),

#     ('pca',pca())

#    ('ohe',OneHotEncoder())

])



full_pipeline = FeatureUnion([

#         ("num2", num_pipe),

        ("cat2", cat_pipe)

    ])



full_pipeline2 = FeatureUnion([

        ("num2", num_pipe),

        ("cat2", cat_pipe),

        ("quali",quali_pipe),

#         ("tsvd",TruncatedSVD(n_components=900, n_iter=10, random_state=42,algorithm='arpack'))

    ])

# TSVD = TruncatedSVD(n_components=900, n_iter=10, random_state=42,algorithm='arpack')    

train_X = full_pipeline2.fit_transform(train_data)

test_train_X = full_pipeline2.transform(train_data.iloc[:10])

test_X = full_pipeline2.transform(test_data)

y = train_data['Fees']

print(train_X.shape,test_X.shape)
from sklearn.linear_model import SGDRegressor,LinearRegression,ElasticNet,Lasso,Ridge

from sklearn.model_selection import StratifiedShuffleSplit,cross_val_predict,cross_val_score

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestRegressor

from sklearn.svm import SVR

from sklearn.metrics import mean_squared_error



from sklearn.ensemble import GradientBoostingRegressor

from xgboost import XGBRegressor

warnings.simplefilter(action='ignore')

from sklearn.model_selection import StratifiedShuffleSplit,cross_val_predict,cross_val_score

from sklearn.metrics import confusion_matrix,mean_squared_error,mean_absolute_error,mean_squared_log_error,r2_score





seed = 7



models = []



# params = {'n_estimators': 500, 'max_depth': 4, 'min_samples_split': 2,

#           'learning_rate': 0.01, 'loss': 'ls'}



# models.append(('LR', LinearRegression()))

# models.append(('DTR', DecisionTreeRegressor()))

# models.append(('RFR', RandomForestRegressor()))

# models.append(('SGDR', SGDRegressor()))

# models.append(('SVM-L', SVR(kernel='linear')))

# models.append(('SVM-R', SVR(kernel='rbf')))

# models.append(('LASSO', Lasso()))

# models.append(('RIDGE', Ridge()))

# models.append(('E_NET', ElasticNet(l1_ratio=0.60,warm_start =True))) # Preferred after scaling everything

models.append(('XGB',XGBRegressor()))

# models.append(('MLPR',MLPRegressor()))

scores = []

names = []



# for name, model in models:

#     lin = 0

#     lin = mean_squared_error(model.fit(train_X,y).predict(test_train_X).reshape(-1,1),y[:10])

#     lin = np.sqrt(lin)

    

#     score = cross_val_score(model, train_X, y,scoring="neg_mean_squared_error", cv=20)

#     py = np.sqrt(-score)

# #     py = np.sqrt(-score)

#     scores.append(py)

#     names.append(name)

      

    

#     message = "%s: %f (%f)" % (name, py.mean(), py.std())

#     print(message)

#     print("{0}: {1}".format(name,lin))

    

#     #     print(y[:10])

# fig = plt.figure()

# fig.suptitle('Algorithm Comparison')

# ax = fig.add_subplot(111)

# plt.boxplot(scores)

# ax.set_xticklabels(names)

# plt.show() 
# from sklearn.metrics import mean_squared_error

# from sklearn.model_selection import train_test_split

# def plot_learning_curves(models,X, y):

#     X_train, X_val, y_train, y_val = train_test_split(train_X, y, test_size=0.2)

#     train_errors, val_errors = [], []

#     for name, model in models:

        

#         for m in range(1, X_train.shape[0]):

#             model.fit(X_train[:m], y_train[:m])

#             y_train_predict = model.predict(X_train[:m])

#             y_val_predict = model.predict(X_val)

#             train_errors.append(mean_squared_error(y_train_predict, y_train[:m]))

#             val_errors.append(mean_squared_error(y_val_predict, y_val))

            

#             plt.plot(np.sqrt(train_errors), "r-", linewidth=.3)

#             plt.plot(np.sqrt(val_errors), "b-", linewidth=.3)

#         plt.show()

#     fig = plt.figure()

# #     fig.suptitle('Algorithm Comparison')

#     ax = fig.add_subplot(111)

#     plt.boxplot(scores)

#     ax.set_xticklabels(names)

#     plt.show()  



# plot_learning_curves(models,train_X,y)
from sklearn.model_selection import GridSearchCV

from xgboost import XGBRegressor



def rmsle(predictions, dmat):

    labels = dmat.get_label()

    diffs = numpy.log(predictions + 1) - numpy.log(labels + 1)

    squared_diffs = numpy.square(diffs)

    avg = numpy.mean(squared_diffs)

    return ('RMSLE', numpy.sqrt(avg))



param = {}

    

#     'learning_rate': [0.05,0.1,0.15,0.2],

#     'reg_alpha':[1e-5, 1e-2, 0.1, 1, 100]

#     'gamma': [0,0.01,0.05,0.07,0.1,0.15,0.17,0.2],

#     'colsample_bytree':[i/100.0 for i in range(10,55,5)],

#     'reg_lambda': [0.1,0.5,0.7,1],

#     'scale_pos_weight': [i/10000 for i in range(1,10,5)]}

#     'eval_metric': ['rmse']

#     }

XGBR = XGBRegressor(cv=5,objective = 'reg:linear',

                                   seed=27,silent = 1,eval_metric = 'rmse',

                    min_child_weight = 2,max_depth = 8,

                    subsample =  0.8, gamma = 1,

                   learning_rate = 0.02,colsample_bytree = 1,

                    reg_lambda = 0.7,scale_pos_weight = 1

                   )

gsearch1 = GridSearchCV(estimator = XGBR,

                        param_grid = param,

                        scoring='neg_mean_squared_log_error',cv=10)

gsearch1.fit(train_X,y)

# gsearch1.cv_results_, gsearch1.best_params_,

(1-np.sqrt(-gsearch1.best_score_))*100
XGBR.fit(train_X,y)

pd.Series(XGBR.predict(test_X)).plot(kind='hist')
submission = pd.DataFrame({

        "Fees": XGBR.predict(test_X)})

submission.head()

submission.to_csv('submission.csv', index=False)

print("Done")