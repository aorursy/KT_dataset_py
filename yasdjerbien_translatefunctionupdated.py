# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#Translate function
import numpy as np
import re 
from numpy import loadtxt
#import sys
#print(sys.maxsize)

def FindTrTranslationIDs(Synsets,KeNetSynsets):
    TrTranslationIDs=[]
    for x in range(0,len(Synsets)):
        for y in range(0,len(KeNetSynsets)):
            if Synsets[x] in KeNetSynsets[y] :
                print(KeNetSynsets[y])
                print(KenetSynset)
                print("Yes 2 , found in List : ",Synset.strip(), '  ', y+1)
                TrTranslationIDs.append(y+1)
    return TrTranslationIDs
    

with open("/kaggle/input/pwndata/PWNSynsets.txt") as f:
        PWNSynsets = [[cell.strip() for cell in row.rstrip('\n').split('\t')] for row in f]
IDPWN, SynsetsPWN = zip(*PWNSynsets)
#print(IDPWN[10])
#print(SynsetsPWN[10])

with open("/kaggle/input/kenetdata/KeNetSynsets.txt") as f:
        KENETSynsets = [[cell.strip() for cell in row.rstrip('\n').split('\t')] for row in f]
IDKENET, SynsetsKENET = zip(*KENETSynsets)
#print(IDKENET[10])
#print(SynsetsKENET[10])
KenetSynset=[]
for y in range(0,len(SynsetsKENET)):
    if len(SynsetsKENET[y].split(','))==1:
        KenetSynset.append([SynsetsKENET[y]])
    else:
        KenetSynset.append(SynsetsKENET[y].split(','))

print(KenetSynset[11])
print(KenetSynset[len(KenetSynset)-1])


print('PWNNeighborhoods')
with open("/kaggle/input/nearestpointpwn5k1/NearestPoint5KPWN1.txt") as f:
        PWNNeighborhoods = [[cell.strip() for cell in row.rstrip('\n').split(',')] for row in f]
print(PWNNeighborhoods[0][0])
print(PWNNeighborhoods[3][1])

print("TranslationEn2Tr")
Translation = [x.split('*')[2] for x in open("/kaggle/input/translationen2tr/TranslationEn2TR.txt").readlines()]
#with open("/kaggle/input/translationen2tr/TranslationEn2TR.txt") as f:
 #       TranslationEn2Tr = [[cell.strip() for cell in row.rstrip('\n').split('*')[2]] for row in f]
#IDTranslation, Translation = zip(*TranslationEn2Tr)
print(Translation[10])
#print(TranslationEn2Tr[10])




Result=[]
for line in range(0,2):
    print('line ',line)    
    SourceId=PWNNeighborhoods[line][0]
    #print(SourceId)
    SourceSynset=SynsetsPWN[int(SourceId)-1]
    #print('PWN Source Synset')
    #print(SourceId, ' ', SourceSynset,'\n')
    SourceSynsetTranslation=Translation[int(SourceId)-1]
    if SourceSynsetTranslation:
        #print('PWN Source Synsets Turkish Translation')
        #print(SourceSynsetTranslation,'\n')
        TrSourceTranslation=[string for string in SourceSynsetTranslation.split('|') if string != ""]
        #print(TrSourceTranslation)
        #print('Source Translation IDs in KENET')
        TrSourceTranslationIDs=FindTrTranslationIDs(TrSourceTranslation,KenetSynset)
        #print(list(dict.fromkeys(TrSourceTranslationIDs)),'\n')
        if TrSourceTranslationIDs:
            TrTargetIDs=[]
            for dest in range(1,len(PWNNeighborhoods[line])):
                TargetId=PWNNeighborhoods[line][dest].strip()
                #print(TargetId, '\n')
                TargetSynset=SynsetsPWN[int(TargetId)-1]    
                #print('PWN Target Synset')
                #print(TargetId, ' ', TargetSynset,'\n')

                TargetSynsetTranslation=Translation[int(TargetId)-1]

                if TargetSynsetTranslation:

                    #print('PWN Target Synsets Turkish Translation')
                    #print(TargetSynsetTranslaion,'\n')


                    TrTargetTranslation=[string for string in TargetSynsetTranslation.split('|') if string != ""]
                    #print(TrTargetTranslation)


                    TrTargetTranslationIDs=FindTrTranslationIDs(TrTargetTranslation,KenetSynset)
                    
                    if TrTargetTranslationIDs:
                        TrTargetIDs=TrTargetIDs + TrTargetTranslationIDs
            
            if TrTargetIDs:
                #print('Target Translation IDs in KENET')
                
                
                #print(list(dict.fromkeys(TrSourceTranslationIDs)),'\n')
                #print(sorted(list(dict.fromkeys(TrTargetIDs))),'\n')
                x=[line+1, sorted(list(dict.fromkeys(TrSourceTranslationIDs))),sorted(list(dict.fromkeys(TrTargetIDs)))]
                Result.append(x)

print(Result)
with open('NeareastIdsKENET.txt', 'w') as f:
    f.writelines("%s\n" % line for line in Result)

print('finish')