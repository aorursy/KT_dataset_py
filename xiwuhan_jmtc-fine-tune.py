import os
import numpy
from sklearn import metrics
import pandas as pd
from sklearn.metrics import roc_auc_score
wr = open('results.txt', "w", encoding='utf8')
mybest = pd.read_csv('/kaggle/input/mybest/sub9523.csv')
sub = pd.read_csv('/kaggle/input/jigsaw-multilingual-toxic-test-translated/jigsaw_miltilingual_test_translated.csv')
sub1 = pd.read_csv('/kaggle/input/mysub42/sub9458.csv')
sub['prd'] = sub1['toxic']

#Ranges here may be manually reset to save computing time.
#p = [1.30,0.60,0.80,0.50,0.60,0.60]
for p0 in numpy.arange(1.2, 1.3, 0.1):
    for p1 in numpy.arange(0.6, 0.7, 0.1):
        for p2 in numpy.arange(0.8, 0.9, 0.1):
            for p3 in numpy.arange(0.6, 0.7, 0.1):
                for p4 in numpy.arange(0.6, 0.7, 0.1):
                    for p5 in numpy.arange(0.6, 0.7, 0.1):
                        out = []
                        for _, row in sub.iterrows():
                            item = [row['id'], row['prd'], row['lang']]
                            if(item[2]=='es'):
                                if(item[1]<0.7):
                                    item[1] *= p0#1.250    99.9455    99.8233
                            elif(item[2]=='fr'):
                                if(item[1]<0.7):
                                    item[1] *= p1#0.950    99.9436    99.8294
                            elif(item[2]=='ru'):
                                if(item[1]<0.7):
                                    item[1] *= p2#0.900    99.9409    99.8379
                            elif(item[2]=='it'):
                                if(item[1]<0.7):
                                    item[1] *= p3#0.750    99.9480    99.8688
                            elif(item[2]=='tr'):
                                if(item[1]<0.7):
                                    item[1] *= p4#0.900    99.9486    99.8715
                            elif(item[2]=='pt'):
                                if(item[1]<0.7):
                                    item[1] *= p5#-3 0.9991403473465919 0.9981593

                            out.append(item)

                        of = pd.DataFrame(out, columns=['id', 'toxic', 'lang'])
                        score1 = roc_auc_score(mybest.toxic.round().astype(int), of.toxic.values)
                        score2 = roc_auc_score(of.toxic.round().astype(int), mybest.toxic.values)
                        line = '%1.2f,%1.2f,%1.2f,%1.2f,%1.2f,%1.2f\t%2.4f\t%2.4f'%(p0, p1,p2,p3,p4,p5, 100*score1, 100*score2)
                        print(line)
                        wr.write(line+'\n')
wr.close()
wr = open('results.txt', "w", encoding='utf8')
mybest = pd.read_csv('/kaggle/input/mybest/sub9523.csv')
sub = pd.read_csv('/kaggle/input/jigsaw-multilingual-toxic-test-translated/jigsaw_miltilingual_test_translated.csv')
sub1 = pd.read_csv('/kaggle/input/mysub42/sub9458.csv')
sub['prd'] = sub1['toxic']

dic = {}#Profanity dictionary.
oft = open('/kaggle/input/profanity/Profanity.txt', "r", encoding='utf8')
for l in oft:
    ele = l.strip().lower().split(':')
    dic[ele[0]] = ele[1]
oft.close()

#I set all weights as 1 and tune them one by one to save computing time.
len, les, lit, ltr, lfr, lru, lpt = 1., 1., 1., 1., 1., 1., 1.

#len 1052 1.4: 99.9771    99.9512
#les 101 1.00:    99.9771    99.9512
#lit 235 1.10:    99.9770    99.9515
#ltr 41 1.00:    99.9770    99.9515
#lfr 202 1.00:    99.9770    99.9515
#lru 1.30:    99.9680    99.7974
#lpt 66 1.30:    99.9771    99.9519

enpros = dic['en'].split(',')
for len in numpy.arange(1.3, 1.4, 0.1):
    out = []
    found = 0
    for _, row in sub.iterrows():
        if(row['lang']=='es'):
            lmd = les#99.4925    99.2043
        elif(row['lang']=='it'):
            lmd = lit
        elif(row['lang']=='tr'):
            lmd = ltr
        elif(row['lang']=='fr'):
            lmd = lfr
        elif(row['lang']=='ru'):
            lmd = lru
        else:
            lmd = lpt

        item = [row['id'], row['prd']]
        if(item[1]<0.5):
            for w in enpros:
                if(str(row['translated']).lower().find(w)>=0):
                    item[1] *= len
                    found += 1
                    break

            ws = dic[row['lang']].split(',')
            for w in ws:
                if(str(row['content']).lower().find(w)>=0):
                    item[1] *= lmd
                    #if(row['lang']=='pt'):
                        #found += 1
                    break
        out.append(item)


    of = pd.DataFrame(out, columns=['id', 'toxic'])
    
    score1 = roc_auc_score(mybest.toxic.round().astype(int), of.toxic.values)
    score2 = roc_auc_score(of.toxic.round().astype(int), mybest.toxic.values)
    l = '%1.2f:\t%2.4f\t%2.4f'%(len, 100*score1, 100*score2)
    print(found, l)
    wr.write(l+'\n')
wr.close()