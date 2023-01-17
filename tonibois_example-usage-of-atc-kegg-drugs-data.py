import pandas as pd

keggfull= pd.read_csv('/kaggle/input/properties-of-atc-accepted-medicines/KEGG_DRUG_ATC_PROPERTIES_PED_FP.csv', delimiter=',')
keggfull.head()
drugname='Ibuprofen'
labsearch=' '+drugname
longname=len(labsearch)
keggmol=keggfull[keggfull['CompoundName'].str[:longname]==labsearch]
keggmol.head()
keggmol.drop_duplicates(['BigGroup_ATC_class'], keep='first')
mol_and_derivatives=keggmol.drop_duplicates(['SMILES'], keep='first')
keggmol.drop_duplicates(['SMILES'], keep='first')
urlfull=mol_and_derivatives['URL_KEGG'].values
print(urlfull[0],urlfull[1])
from IPython.display import IFrame
IFrame(src=urlfull[1], width=1000, height=300)
from PIL import Image
import requests
from io import BytesIO
iden=0
keggcode=mol_and_derivatives['KEGG_code'].values
url='https://www.genome.jp/Fig/drug/'+keggcode[iden]+'.gif'
response = requests.get(url)
img = Image.open(BytesIO(response.content))
img
iden=1
keggcode=mol_and_derivatives['KEGG_code'].values
url='https://www.genome.jp/Fig/drug/'+keggcode[iden]+'.gif'
response = requests.get(url)
img = Image.open(BytesIO(response.content))
img
from IPython.display import SVG, display

#mol_and_derivatives[['SMILES','KEGG_code']].to_csv('subset.smi', sep=' ', encoding='utf-8')
#!gawk "{print $2,$3}" subset.smi | sed '1d'  > subsetcur.smi
#!head -n 20 subsetcur.smi > subsetcur20.smi
#!obabel -ismi subsetcur20.smi -osvg -O subset.svg

display(SVG('/kaggle/input/compoundgroupsatc/subset.svg'))
atcsearch='J05AE'
lensearch=len(atcsearch)
keggsubset2=keggfull[keggfull['ATC_full_code'].str[:lensearch]==atcsearch]
keggsubset2=keggsubset2.drop_duplicates(['SMILES'], keep='first')
#keggsubset2[['SMILES','KEGG_code']].to_csv('subset2.smi', sep=' ', encoding='utf-8')
#!gawk "{print $2,$3}" subset2.smi | sed '1d'  > subsetcur2.smi
#! head -n 20 subsetcur2.smi > ProteaseinhibJ05AE.smi
#!obabel -ismi ProteaseinhibJ05AE.smi -osvg -O ProteaseinhibJ05AE.svg
display(SVG('/kaggle/input/compoundgroupsatc/ProteaseinhibJ05AE.svg'))
keggfull[['ATC_label_class','ATC_full_code','BigGroup_ATC_class']].drop_duplicates(['ATC_label_class'], keep='first').head(n=100)
# Lets search for antiseptics in category R02AA :https://www.whocc.no/atc_ddd_index/?code=R02AA&showdescription=yes
#atcsearch='R02AA'
#lensearch=len(atcsearch)
#keggsubset2=keggfull[keggfull['ATC_full_code'].str[:lensearch]==atcsearch]
#keggsubset2=keggsubset2.drop_duplicates(['SMILES'], keep='first')
#keggsubset2[['SMILES','KEGG_code']].to_csv('subset2.smi', sep=' ', encoding='utf-8')
#!gawk "{print $2,$3}" subset2.smi | sed '1d'  > subsetcur2.smi
#! head -n 20 subsetcur2.smi > R02AA.smi
#!obabel -ismi R02AA.smi -osvg -O R02AA.svg
display(SVG('/kaggle/input/compoundgroupsatc/R02AA.svg'))
search='D07208'
lensearch=len(search)
class_search='KEGG_code'
kegg_search_mol=keggfull[keggfull[class_search].str[:lensearch]==search]
kegg_search_mol.drop_duplicates(['SMILES'], keep='last')
import numpy as np
round(np.mean(keggfull),2).head(n=7)
import matplotlib.pyplot as plt

fig, axs = plt.subplots(2, 3, constrained_layout=True,figsize=(15,15))

axs[0, 0].hist(keggfull['MW']);
axs[0, 0].set_title('MW')
axs[0, 1].hist(keggfull['logP']);
axs[0, 1].set_title('logP')
axs[0, 2].hist(keggfull['HBA1']);
axs[0, 2].set_title('HBA1')
axs[1, 1].hist(keggfull['HBD']);
axs[1, 1].set_title('HBD')
axs[1, 0].hist(keggfull['TPSA']);
axs[1, 0].set_title('TPSA')
axs[1, 2].hist(keggfull['MR']);
axs[1, 2].set_title('MR')

for ax in axs.flat:
    ax.set(xlabel='', ylabel='')

plt.show()
drugname='Chloroquine'
labsearch=' '+drugname
longname=len(labsearch)
keggmol=keggfull[keggfull['CompoundName'].str[:longname]==labsearch]
keggmol.head()
test=keggmol.drop_duplicates(['SMILES'], keep='first')
test
iden=0
keggcode=test['KEGG_code'].values
url='https://www.genome.jp/Fig/drug/'+keggcode[iden]+'.gif'
response = requests.get(url)
img = Image.open(BytesIO(response.content))
img
drugname='Hydroxychloroquine'
labsearch=' '+drugname
longname=len(labsearch)
keggmol=keggfull[keggfull['CompoundName'].str[:longname]==labsearch]
keggmol.head()
test=keggmol.drop_duplicates(['SMILES'], keep='first')
test

iden=0
keggcode=test['KEGG_code'].values
url='https://www.genome.jp/Fig/drug/'+keggcode[iden]+'.gif'
response = requests.get(url)
img = Image.open(BytesIO(response.content))
img
drugname='Lopinavir'
labsearch=' '+drugname
longname=len(labsearch)
keggmol=keggfull[keggfull['CompoundName'].str[:longname]==labsearch]
keggmol.head()
keggmol.drop_duplicates(['SMILES'], keep='first')
test=keggmol.drop_duplicates(['SMILES'], keep='first')
test
iden=0
keggcode=test['KEGG_code'].values
url='https://www.genome.jp/Fig/drug/'+keggcode[iden]+'.gif'
response = requests.get(url)
img = Image.open(BytesIO(response.content))
img
drugname='Darunavir'
labsearch=' '+drugname
longname=len(labsearch)
keggmol=keggfull[keggfull['CompoundName'].str[:longname]==labsearch]
keggmol.head()
keggmol.drop_duplicates(['SMILES'], keep='first')
test=keggmol.drop_duplicates(['SMILES'], keep='first')
test
iden=0
keggcode=test['KEGG_code'].values
url='https://www.genome.jp/Fig/drug/'+keggcode[iden]+'.gif'
response = requests.get(url)
img = Image.open(BytesIO(response.content))
img
drugname='Fingolimod'
labsearch=' '+drugname
longname=len(labsearch)
keggmol=keggfull[keggfull['CompoundName'].str[:longname]==labsearch]
keggmol.head()
keggmol.drop_duplicates(['SMILES'], keep='first')
test=keggmol.drop_duplicates(['SMILES'], keep='first')
test
iden=0
keggcode=test['KEGG_code'].values
url='https://www.genome.jp/Fig/drug/'+keggcode[iden]+'.gif'
response = requests.get(url)
img = Image.open(BytesIO(response.content))
img
iden=1
keggcode=test['KEGG_code'].values
url='https://www.genome.jp/Fig/drug/'+keggcode[iden]+'.gif'
response = requests.get(url)
img = Image.open(BytesIO(response.content))
img
