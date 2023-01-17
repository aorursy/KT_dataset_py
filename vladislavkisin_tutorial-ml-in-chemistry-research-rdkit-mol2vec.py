#Here we install the package. For me it's been a nightmare to install rdkit into Kaggle's environment. 

#But wonderful Kaggle's technical support helped me to find the way.



!conda install -y -c rdkit rdkit;

!pip install pandas==0.23.0
%matplotlib inline



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings("ignore")
#Let's load the data and look at them

df= pd.read_csv('../input/mlchem/logP_dataset.csv', names=['smiles', 'logP'])

df.head()
#Importing Chem module

from rdkit import Chem 



#Method transforms smiles strings to mol rdkit object

df['mol'] = df['smiles'].apply(lambda x: Chem.MolFromSmiles(x)) 



#Now let's see what we've got

print(type(df['mol'][0]))
from rdkit.Chem import Draw

mols = df['mol'][:20]



#MolsToGridImage allows to paint a number of molecules at a time

Draw.MolsToGridImage(mols, molsPerRow=5, useSVG=True, legends=list(df['smiles'][:20].values))
# AddHs function adds H atoms to a MOL (as Hs in SMILES are usualy ignored)

# GetNumAtoms() method returns a general nubmer of all atoms in a molecule

# GetNumHeavyAtoms() method returns a nubmer of all atoms in a molecule with molecular weight > 1





df['mol'] = df['mol'].apply(lambda x: Chem.AddHs(x))

df['num_of_atoms'] = df['mol'].apply(lambda x: x.GetNumAtoms())

df['num_of_heavy_atoms'] = df['mol'].apply(lambda x: x.GetNumHeavyAtoms())
import seaborn as sns

sns.jointplot(df.num_of_atoms, df.logP)

plt.show()
# First we need to settle the pattern.

c_patt = Chem.MolFromSmiles('C')



# Now let's implement GetSubstructMatches() method

print(df['mol'][0].GetSubstructMatches(c_patt))
#We're going to settle the function that searches patterns and use it for a list of most common atoms only

def number_of_atoms(atom_list, df):

    for i in atom_list:

        df['num_of_{}_atoms'.format(i)] = df['mol'].apply(lambda x: len(x.GetSubstructMatches(Chem.MolFromSmiles(i))))



number_of_atoms(['C','O', 'N', 'Cl'], df)
sns.pairplot(df[['num_of_atoms','num_of_C_atoms','num_of_N_atoms', 'num_of_O_atoms', 'logP']], diag_kind='kde', kind='reg', markers='+')

plt.show()
from sklearn.linear_model import RidgeCV

from sklearn.model_selection import train_test_split



#Leave only features columns

train_df = df.drop(columns=['smiles', 'mol', 'logP'])

y = df['logP'].values



print(train_df.columns)



#Perform a train-test split. We'll use 10% of the data to evaluate the model while training on 90%



X_train, X_test, y_train, y_test = train_test_split(train_df, y, test_size=.1, random_state=1)

from sklearn.metrics import mean_absolute_error, mean_squared_error

def evaluation(model, X_test, y_test):

    prediction = model.predict(X_test)

    mae = mean_absolute_error(y_test, prediction)

    mse = mean_squared_error(y_test, prediction)

    

    plt.figure(figsize=(15, 10))

    plt.plot(prediction[:300], "red", label="prediction", linewidth=1.0)

    plt.plot(y_test[:300], 'green', label="actual", linewidth=1.0)

    plt.legend()

    plt.ylabel('logP')

    plt.title("MAE {}, MSE {}".format(round(mae, 4), round(mse, 4)))

    plt.show()

    

    print('MAE score:', round(mae, 4))

    print('MSE score:', round(mse,4))
#Train the model

ridge = RidgeCV(cv=5)

ridge.fit(X_train, y_train)

#Evaluate results

evaluation(ridge, X_test, y_test)
atp = Chem.MolFromSmiles('C1=NC2=C(C(=N1)N)N=CN2[C@H]3[C@@H]([C@@H]([C@H](O3)COP(=O)(O)OP(=O)(O)OP(=O)(O)O)O)O')



# Getting number of rings with specified number of backbones

print('Number of rings with 1 backbone:', atp.GetRingInfo().NumAtomRings(1))

print('Number of rings with 2 backbones:', atp.GetRingInfo().NumAtomRings(2))
m = Chem.MolFromSmiles('C(=O)C(=N)CCl')

#Iterating through atoms to get atom symbols and explicit valencies 

for atom in m.GetAtoms():

    print('Atom:', atom.GetSymbol(), 'Valence:', atom.GetExplicitValence())
from rdkit.Chem import Descriptors

df['tpsa'] = df['mol'].apply(lambda x: Descriptors.TPSA(x))

df['mol_w'] = df['mol'].apply(lambda x: Descriptors.ExactMolWt(x))

df['num_valence_electrons'] = df['mol'].apply(lambda x: Descriptors.NumValenceElectrons(x))

df['num_heteroatoms'] = df['mol'].apply(lambda x: Descriptors.NumHeteroatoms(x))
train_df = df.drop(columns=['smiles', 'mol', 'logP'])

y = df['logP'].values



print(train_df.columns)



#Perform a train-test split. We'll use 10% of the data to evaluate the model while training on 90%



X_train, X_test, y_train, y_test = train_test_split(train_df, y, test_size=.1, random_state=1)
#Train the model

ridge = RidgeCV(cv=5)

ridge.fit(X_train, y_train)

#Evaluate results and plot predictions

evaluation(ridge, X_test, y_test)
#Installing a package

!pip install git+https://github.com/samoturk/mol2vec;
#Load the dataset and extract target values

mdf= pd.read_csv('../input/mlchem/logP_dataset.csv', names=['smiles', 

                                           'target'])

target = mdf['target']

mdf.drop(columns='target',inplace=True)
#Transforming SMILES to MOL

mdf['mol'] = mdf['smiles'].apply(lambda x: Chem.MolFromSmiles(x))

#Loading pre-trained model via word2vec

from gensim.models import word2vec

model = word2vec.Word2Vec.load('../input/mlchem/model_300dim.pkl')
from mol2vec.features import mol2alt_sentence, mol2sentence, MolSentence, DfVec, sentences2vec

from gensim.models import word2vec

print('Molecular sentence:', mol2alt_sentence(mdf['mol'][1], radius=1))

print('\nMolSentence object:', MolSentence(mol2alt_sentence(mdf['mol'][1], radius=1)))

print('\nDfVec object:',DfVec(sentences2vec(MolSentence(mol2alt_sentence(mdf['mol'][1], radius=1)), model, unseen='UNK')))
#Constructing sentences

mdf['sentence'] = mdf.apply(lambda x: MolSentence(mol2alt_sentence(x['mol'], 1)), axis=1)



#Extracting embeddings to a numpy.array

#Note that we always should mark unseen='UNK' in sentence2vec() so that model is taught how to handle unknown substructures

mdf['mol2vec'] = [DfVec(x) for x in sentences2vec(mdf['sentence'], model, unseen='UNK')]

X = np.array([x.vec for x in mdf['mol2vec']])

y = target.values



X.shape
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.1, random_state=1)

ridge = RidgeCV(cv=5)

ridge.fit(X_train, y_train)

evaluation(ridge, X_test, y_test)
mdf = pd.DataFrame(X)

new_df = pd.concat((mdf, train_df), axis=1)
X_train, X_test, y_train, y_test = train_test_split(new_df, y, test_size=.1, random_state=1)

ridge = RidgeCV(cv=5)

ridge.fit(X_train, y_train)

evaluation(ridge, X_test, y_test)
import warnings

warnings.filterwarnings("ignore")

#Read the data

hiv = pd.read_csv('../input/mlchem/HIV.csv')

hiv.head()
#Let's look at the target values count

sns.countplot(data = hiv, x='HIV_active', orient='v')

plt.ylabel('HIM active')

plt.xlabel('Count of values')

plt.show()
#Transform SMILES to MOL

hiv['mol'] = hiv['smiles'].apply(lambda x: Chem.MolFromSmiles(x)) 



#Extract descriptors

hiv['tpsa'] = hiv['mol'].apply(lambda x: Descriptors.TPSA(x))

hiv['mol_w'] = hiv['mol'].apply(lambda x: Descriptors.ExactMolWt(x))

hiv['num_valence_electrons'] = hiv['mol'].apply(lambda x: Descriptors.NumValenceElectrons(x))

hiv['num_heteroatoms'] = hiv['mol'].apply(lambda x: Descriptors.NumHeteroatoms(x))
y = hiv.HIV_active.values

X = hiv.drop(columns=['smiles', 'activity','HIV_active', 'mol'])





X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.20, random_state=1)
from sklearn.metrics import auc, roc_curve

def evaluation_class(model, X_test, y_test):

    prediction = model.predict_proba(X_test)

    preds = model.predict_proba(X_test)[:,1]

    fpr, tpr, threshold = roc_curve(y_test, preds)

    roc_auc = auc(fpr, tpr)

    

    plt.title('ROC Curve')

    plt.plot(fpr, tpr, 'g', label = 'AUC = %0.3f' % roc_auc)

    plt.legend(loc = 'lower right')

    plt.plot([0, 1], [0, 1],'r--')

    plt.xlim([0, 1])

    plt.ylim([0, 1])

    plt.ylabel('True Positive Rate')

    plt.xlabel('False Positive Rate')

    plt.show()

    

    print('ROC AUC score:', round(roc_auc, 4))
from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import StandardScaler



X_train = StandardScaler().fit_transform(X_train)

X_test = StandardScaler().fit_transform(X_test)



lr = LogisticRegression()

lr.fit(X_train, y_train)



evaluation_class(lr, X_test, y_test)
#Constructing sentences

hiv['sentence'] = hiv.apply(lambda x: MolSentence(mol2alt_sentence(x['mol'], 1)), axis=1)



#Extracting embeddings to a numpy.array

#Note that we always should mark unseen='UNK' in sentence2vec() so that model is taught how to handle unknown substructures

hiv['mol2vec'] = [DfVec(x) for x in sentences2vec(hiv['sentence'], model, unseen='UNK')]

X_mol = np.array([x.vec for x in hiv['mol2vec']])

X_mol = pd.DataFrame(X_mol)



#Concatenating matrices of features

new_hiv = pd.concat((X, X_mol), axis=1)



X_train, X_test, y_train, y_test = train_test_split(new_hiv, y, test_size=.20, random_state=1)



X_train = StandardScaler().fit_transform(X_train)

X_test = StandardScaler().fit_transform(X_test)



lr = LogisticRegression()

lr.fit(X_train, y_train)





evaluation_class(lr, X_test, y_test)