import pandas as pd
pd.set_option('display.max_columns', 300)

clinvar_data = pd.read_csv('../input/clinvar-conflicting/clinvar_conflicting.csv')
from sklearn.base import BaseEstimator, TransformerMixin

class FixChromosome(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.comment_ = 'Transform \'CHROM\' feature to a numerical feature. Assign X = 23, MT = 24'
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        X_copy = X.copy()

        X_copy.loc[:, 'CHROM'].replace('X', 23, inplace=True)
        X_copy.loc[:, 'CHROM'].replace('MT', 24, inplace=True)
        X_copy.loc[:, 'CHROM'] = X_copy.CHROM.astype(int)

        return X_copy
import numpy as np
import functools
def conjunction(*conditions):
    return functools.reduce(np.logical_and, conditions)

class CountAlleles(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.comment_ = 'Calculate the length of REF, ALT, Allele features and mark Single Nucleotide Variants'
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        X_copy = X.copy()
        
        X_copy.loc[:, 'REF_length'] = X_copy.REF.str.len()
        X_copy.loc[:, 'ALT_length'] = X_copy.ALT.str.len()
        X_copy.loc[:, 'Allele_length'] = X_copy.Allele.str.len()
        
        ref_is_1 = X_copy.REF.str.len() == 1
        alt_is_1 = X_copy.ALT.str.len() == 1

        X_copy.loc[conjunction(ref_is_1, alt_is_1), 'SNV'] = 1
        X_copy.SNV.fillna(0, inplace=True)

        return X_copy
class ExtractPositions(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.comment_ = 'Extract tstart and stop positions for cDNA_position, CDS_position, Protein_position'
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        X_copy = X.copy()

        ### START
        X_copy['cDNA_position_start'] = X_copy.cDNA_position.str.split('-').str.get(0)
        X_copy['CDS_position_start'] = X_copy.CDS_position.str.split('-').str.get(0)
        X_copy['Protein_position_start'] = X_copy.Protein_position.str.split('-').str.get(0)

        X_copy['cDNA_position_start'].replace('?', np.NaN, inplace=True)
        X_copy['CDS_position_start'].replace('?', np.NaN, inplace=True)
        X_copy['Protein_position_start'].replace('?', np.NaN, inplace=True)

        X_copy['cDNA_position_start'] = X_copy['cDNA_position_start'].astype(float)
        X_copy['CDS_position_start'] = X_copy['CDS_position_start'].astype(float)
        X_copy['Protein_position_start'] = X_copy['Protein_position_start'].astype(float)

        ### STOP
        X_copy['cDNA_position_stop'] = X_copy.cDNA_position.str.split('-').str.get(1)
        X_copy['CDS_position_stop'] = X_copy.CDS_position.str.split('-').str.get(1)
        X_copy['Protein_position_stop'] = X_copy.Protein_position.str.split('-').str.get(1)

        X_copy['cDNA_position_stop'].replace('?', np.NaN, inplace=True)
        X_copy['CDS_position_stop'].replace('?', np.NaN, inplace=True)
        X_copy['Protein_position_stop'].replace('?', np.NaN, inplace=True)

        X_copy['cDNA_position_stop'] = X_copy['cDNA_position_stop'].astype(float)
        X_copy['CDS_position_stop'] = X_copy['CDS_position_stop'].astype(float)
        X_copy['Protein_position_stop'] = X_copy['Protein_position_stop'].astype(float)

        for field in ['cDNA_position', 'CDS_position', 'Protein_position']:            
            start_pos_exists = X_copy[field + '_start'].notnull()
            stop_pos_does_not_exist = X_copy[field + '_stop'].isnull()
            cn_filter = conjunction(start_pos_exists, stop_pos_does_not_exist)
            X_copy.loc[cn_filter, field + '_stop'] = X_copy.loc[cn_filter, field + '_start']

        for field in ['cDNA_position', 'CDS_position', 'Protein_position']:            
            start_pos_does_not_exist = X_copy[field + '_start'].isnull()
            stop_pos_exists = X_copy[field + '_stop'].notnull()
            cn_filter = conjunction(start_pos_does_not_exist, stop_pos_exists)
            X_copy.loc[cn_filter, field + '_start'] = X_copy.loc[cn_filter, field + '_stop']

        for field in ['cDNA_position', 'CDS_position', 'Protein_position']:            
            start_pos_does_not_exist = X_copy[field + '_start'].isnull()
            stop_pos_does_not_exist = X_copy[field + '_stop'].isnull()
            cn_filter = conjunction(start_pos_does_not_exist, stop_pos_does_not_exist)
            X_copy.loc[cn_filter, field + '_start'] = 0
            X_copy.loc[cn_filter, field + '_stop'] = 0

        return X_copy
class MarkIntronsAndExons(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.comment_ = 'Mark intron and exon variants'
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        X_copy = X.copy()

        X_copy['Is_Exon'] = X_copy.EXON.notnull().astype(int)
        X_copy['Is_Intron'] = X_copy.INTRON.notnull().astype(int)

        return X_copy
def disjunction(*conditions):
    return functools.reduce(np.logical_or, conditions)

class MarkNotSpecifiedCLNDN(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.comment_ = 'Create binary indicator for CLNDN: not-specified vs. rest'
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        X_copy = X.copy()

        c1 = X_copy.CLNDN == "not_specified"
        c2 = X_copy.CLNDN == "not_specified|not_provided"
        c3 = X_copy.CLNDN == "not_provided|not_specified"
        c4 = X_copy.CLNDN == "not_provided"

        X_copy.loc[disjunction(c1, c2, c3, c4), 'CLNDN_not_specified'] = 1
        X_copy.CLNDN_not_specified.fillna(0, inplace=True)

        return X_copy
class ExtractEXONPositionAndLength(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.comment_ = 'Extract the position and length of EXON'
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        X_copy = X.copy()

        ### Position
        X_copy['EXON_position'] = X_copy.EXON.str.split('/').str.get(0)
        X_copy['EXON_position'] = X_copy['EXON_position'].astype(float)
        
        ### Length
        X_copy['EXON_length'] = X_copy.EXON.str.split('/').str.get(1)
        X_copy['EXON_length'] = X_copy['EXON_length'].astype(float)
        
        exon_pos_does_not_exist = X_copy["EXON_position"].isnull()
        exon_length_does_not_exist = X_copy["EXON_length"].isnull()
        cn_filter = conjunction(exon_pos_does_not_exist, exon_length_does_not_exist)
        X_copy.loc[cn_filter, "EXON_position"] = 0
        X_copy.loc[cn_filter, 'EXON_length'] = 0
        
        return X_copy
from sklearn.feature_extraction.text import CountVectorizer

class AddPathways(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.comment_ = 'Get Pathway IDs from Reactome based on list of symbols from ClinVar data set'
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        X_copy = X.copy()
        
        # Symbol_Pathways.csv generated by https://reactome.org/PathwayBrowser/#TOOL=AT based on list_of_symbols.csv
        symbol_pathways = pd.read_csv('../input/clinvar-symbol-pathways/Symbol_Pathways.csv')

        vectorizer = CountVectorizer(binary=True)
        vec_symbols = vectorizer.fit_transform(symbol_pathways['Submitted entities found']).toarray()
        symb_df = pd.concat([symbol_pathways['Pathway identifier'], 
                             pd.DataFrame(vec_symbols, columns=vectorizer.get_feature_names())], axis=1)
        symb_df.set_index('Pathway identifier', inplace=True)
        
        pathways = []

        for column in symb_df:
            pathways.append(symb_df[symb_df[column] > 0].index.str.cat(sep=';'))

        pathways_series = pd.Series(pathways, index=symb_df.columns)
        pathways_series.head()
        
        symb_df = symb_df.append(pathways_series, ignore_index=True)
        symb_df_transposed = symb_df.tail(1).transpose()
        
        symb_df_transposed.reset_index(level=0, inplace=True)
        symb_df_transposed.rename(columns={'index':'SYMBOL', 140: 'Pathways'}, inplace=True)
        
        symb_df_transposed.SYMBOL = symb_df_transposed.SYMBOL.str.upper()
        
        return pd.merge(X_copy, symb_df_transposed, how='left', on=['SYMBOL'])
from sklearn.pipeline import Pipeline

# Basic Tranformation Pipeline
transformation_pipeline = Pipeline([
    ('fix_chromosome', FixChromosome()),
    ('count_alleles', CountAlleles()),
    ('extract_positions', ExtractPositions()),
    ('mark_introns_and_exons', MarkIntronsAndExons()),
    ('mark_not_specified_CLNDN', MarkNotSpecifiedCLNDN()),
    ('extract_EXON_position_and_length', ExtractEXONPositionAndLength()),
    ('add_pathways', AddPathways())
])

clinvar_transformed = transformation_pipeline.fit_transform(clinvar_data)
# Features used for the analysis and models training
numeric_features = ['POS', 'AF_ESP', 'AF_EXAC', 'AF_TGP', 'LoFtool', 'CADD_PHRED', 'CADD_RAW', 'REF_length', 'ALT_length', 'Allele_length',
                    'SNV', 'cDNA_position_start', 'CDS_position_start', 'Protein_position_start', 'cDNA_position_stop', 'CDS_position_stop',
                    'Protein_position_stop', 'Is_Exon', 'Is_Intron', 'CLNDN_not_specified', 'EXON_position', 'EXON_length']
                    
categorical_features = ['CHROM', 'IMPACT', 'STRAND', 'BAM_EDIT', 'SIFT', 'PolyPhen',
                        'BLOSUM62', 'Consequence', 'CLNVC', 'Pathways']

target_feature = ['CLASS']
clinvar_transformed[numeric_features + categorical_features + target_feature].head()
clinvar_transformed[numeric_features + categorical_features + target_feature].info()
clinvar_transformed.describe(include=['object'])
import matplotlib.pyplot as plt

clinvar_transformed.hist(figsize=(14,14))
plt.show()
import seaborn as sns

features = ['CLNVC', 'IMPACT', 'SIFT', 'PolyPhen']

for feature in features:
    sns.countplot(y=feature, data=clinvar_transformed)
    plt.show()
# Calculate correlations between numeric features
correlations = clinvar_transformed.corr()

# Change color scheme
sns. set_style ("white")
# Generate a mask for the upper triangle
mask = np.zeros_like(correlations, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
# Make the figsize 20 x 20
plt.figure(figsize=(15,13))
# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)
# Plot heatmap of correlations
sns.heatmap(correlations * 100, annot=True, fmt='.0f', 
            mask=mask, cbar=False, cmap=cmap)

plt.show()
from sklearn.model_selection import train_test_split

X = clinvar_transformed[numeric_features + categorical_features]
y = clinvar_transformed.CLASS

def train_val_test_split(X, y):
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, train_size=0.2, random_state=111, stratify=y)
    X_test, X_val, y_test, y_val = train_test_split(X_temp, y_temp, test_size=0.5, random_state=111, stratify=y_temp)
    return X_train, X_val, X_test, y_train, y_val, y_test

X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(X, y)
print(X_train.shape)
print(X_val.shape)
print(X_test.shape)
def balance_training_set(X_train, y_train):
    X_temp = pd.concat([X_train, y_train], axis=1)
    balanced = pd.concat([X_temp, X_temp[X_temp.CLASS == 1], X_temp[X_temp.CLASS == 1]])
    return balanced.drop('CLASS', axis=1), balanced['CLASS']

X_train_balanced , y_train_balanced = balance_training_set(X_train, y_train)

print('1/0 CLASS ratio:', sum((y_train_balanced == 1).astype(int)) / len(y_train_balanced))
# Definition of the CategoricalEncoder class, copied from PR #9151.
# https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/preprocessing/data.py

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_array
from sklearn.preprocessing import LabelEncoder
from scipy import sparse

class CategoricalEncoder(BaseEstimator, TransformerMixin):
    """Encode categorical features as a numeric array.
    The input to this transformer should be a matrix of integers or strings,
    denoting the values taken on by categorical (discrete) features.
    The features can be encoded using a one-hot aka one-of-K scheme
    (``encoding='onehot'``, the default) or converted to ordinal integers
    (``encoding='ordinal'``).
    This encoding is needed for feeding categorical data to many scikit-learn
    estimators, notably linear models and SVMs with the standard kernels.
    Read more in the :ref:`User Guide <preprocessing_categorical_features>`.
    Parameters
    ----------
    encoding : str, 'onehot', 'onehot-dense' or 'ordinal'
        The type of encoding to use (default is 'onehot'):
        - 'onehot': encode the features using a one-hot aka one-of-K scheme
          (or also called 'dummy' encoding). This creates a binary column for
          each category and returns a sparse matrix.
        - 'onehot-dense': the same as 'onehot' but returns a dense array
          instead of a sparse matrix.
        - 'ordinal': encode the features as ordinal integers. This results in
          a single column of integers (0 to n_categories - 1) per feature.
    categories : 'auto' or a list of lists/arrays of values.
        Categories (unique values) per feature:
        - 'auto' : Determine categories automatically from the training data.
        - list : ``categories[i]`` holds the categories expected in the ith
          column. The passed categories are sorted before encoding the data
          (used categories can be found in the ``categories_`` attribute).
    dtype : number type, default np.float64
        Desired dtype of output.
    handle_unknown : 'error' (default) or 'ignore'
        Whether to raise an error or ignore if a unknown categorical feature is
        present during transform (default is to raise). When this is parameter
        is set to 'ignore' and an unknown category is encountered during
        transform, the resulting one-hot encoded columns for this feature
        will be all zeros.
        Ignoring unknown categories is not supported for
        ``encoding='ordinal'``.
    Attributes
    ----------
    categories_ : list of arrays
        The categories of each feature determined during fitting. When
        categories were specified manually, this holds the sorted categories
        (in order corresponding with output of `transform`).
    Examples
    --------
    Given a dataset with three features and two samples, we let the encoder
    find the maximum value per feature and transform the data to a binary
    one-hot encoding.
    >>> from sklearn.preprocessing import CategoricalEncoder
    >>> enc = CategoricalEncoder(handle_unknown='ignore')
    >>> enc.fit([[0, 0, 3], [1, 1, 0], [0, 2, 1], [1, 0, 2]])
    ... # doctest: +ELLIPSIS
    CategoricalEncoder(categories='auto', dtype=<... 'numpy.float64'>,
              encoding='onehot', handle_unknown='ignore')
    >>> enc.transform([[0, 1, 1], [1, 0, 4]]).toarray()
    array([[ 1.,  0.,  0.,  1.,  0.,  0.,  1.,  0.,  0.],
           [ 0.,  1.,  1.,  0.,  0.,  0.,  0.,  0.,  0.]])
    See also
    --------
    sklearn.preprocessing.OneHotEncoder : performs a one-hot encoding of
      integer ordinal features. The ``OneHotEncoder assumes`` that input
      features take on values in the range ``[0, max(feature)]`` instead of
      using the unique values.
    sklearn.feature_extraction.DictVectorizer : performs a one-hot encoding of
      dictionary items (also handles string-valued features).
    sklearn.feature_extraction.FeatureHasher : performs an approximate one-hot
      encoding of dictionary items or strings.
    """

    def __init__(self, encoding='onehot', categories='auto', dtype=np.float64,
                 handle_unknown='error'):
        self.encoding = encoding
        self.categories = categories
        self.dtype = dtype
        self.handle_unknown = handle_unknown

    def fit(self, X, y=None):
        """Fit the CategoricalEncoder to X.
        Parameters
        ----------
        X : array-like, shape [n_samples, n_feature]
            The data to determine the categories of each feature.
        Returns
        -------
        self
        """

        if self.encoding not in ['onehot', 'onehot-dense', 'ordinal']:
            template = ("encoding should be either 'onehot', 'onehot-dense' "
                        "or 'ordinal', got %s")
            raise ValueError(template % self.handle_unknown)

        if self.handle_unknown not in ['error', 'ignore']:
            template = ("handle_unknown should be either 'error' or "
                        "'ignore', got %s")
            raise ValueError(template % self.handle_unknown)

        if self.encoding == 'ordinal' and self.handle_unknown == 'ignore':
            raise ValueError("handle_unknown='ignore' is not supported for"
                             " encoding='ordinal'")

        X = check_array(X, dtype=np.object, accept_sparse='csc', copy=True)
        n_samples, n_features = X.shape

        self._label_encoders_ = [LabelEncoder() for _ in range(n_features)]

        for i in range(n_features):
            le = self._label_encoders_[i]
            Xi = X[:, i]
            if self.categories == 'auto':
                le.fit(Xi)
            else:
                valid_mask = np.in1d(Xi, self.categories[i])
                if not np.all(valid_mask):
                    if self.handle_unknown == 'error':
                        diff = np.unique(Xi[~valid_mask])
                        msg = ("Found unknown categories {0} in column {1}"
                               " during fit".format(diff, i))
                        raise ValueError(msg)
                le.classes_ = np.array(np.sort(self.categories[i]))

        self.categories_ = [le.classes_ for le in self._label_encoders_]

        return self

    def transform(self, X):
        """Transform X using one-hot encoding.
        Parameters
        ----------
        X : array-like, shape [n_samples, n_features]
            The data to encode.
        Returns
        -------
        X_out : sparse matrix or a 2-d array
            Transformed input.
        """
        X = check_array(X, accept_sparse='csc', dtype=np.object, copy=True)
        n_samples, n_features = X.shape
        X_int = np.zeros_like(X, dtype=np.int)
        X_mask = np.ones_like(X, dtype=np.bool)

        for i in range(n_features):
            valid_mask = np.in1d(X[:, i], self.categories_[i])

            if not np.all(valid_mask):
                if self.handle_unknown == 'error':
                    diff = np.unique(X[~valid_mask, i])
                    msg = ("Found unknown categories {0} in column {1}"
                           " during transform".format(diff, i))
                    raise ValueError(msg)
                else:
                    # Set the problematic rows to an acceptable value and
                    # continue `The rows are marked `X_mask` and will be
                    # removed later.
                    X_mask[:, i] = valid_mask
                    X[:, i][~valid_mask] = self.categories_[i][0]
            X_int[:, i] = self._label_encoders_[i].transform(X[:, i])

        if self.encoding == 'ordinal':
            return X_int.astype(self.dtype, copy=False)

        mask = X_mask.ravel()
        n_values = [cats.shape[0] for cats in self.categories_]
        n_values = np.array([0] + n_values)
        indices = np.cumsum(n_values)

        column_indices = (X_int + indices[:-1]).ravel()[mask]
        row_indices = np.repeat(np.arange(n_samples, dtype=np.int32),
                                n_features)[mask]
        data = np.ones(n_samples * n_features)[mask]

        out = sparse.csc_matrix((data, (row_indices, column_indices)),
                                shape=(n_samples, indices[-1]),
                                dtype=self.dtype).tocsr()
        if self.encoding == 'onehot-dense':
            return out.toarray()
        else:
            return out
from sklearn.feature_extraction.text import CountVectorizer

class EncodeNonNumericValues(BaseEstimator, TransformerMixin):
    def __init__(self, consequence_list):
        self.comment_ = 'Get feature matrix from Consequence and use CategoricalEncoder for the rest'
        self.consequence_list = consequence_list
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        X_copy = X.copy()
        
        consequene_vectorizer = CountVectorizer(binary=True, vocabulary=self.consequence_list)
        consequene_matrix = consequene_vectorizer.fit_transform(X_copy['Consequence'])
        
        consequence_df = pd.DataFrame(consequene_matrix.toarray(), columns=consequene_vectorizer.get_feature_names())
        
        symbol_pathways = pd.read_csv('../input/clinvar-symbol-pathways/Symbol_Pathways.csv')
        pathways = symbol_pathways['Pathway identifier'].unique()
        
        pathways_vectorizer = CountVectorizer(binary=True, vocabulary=pathways)
        pathways_matrix = pathways_vectorizer.fit_transform(X_copy['Pathways'].values.astype('U'))
        pathways_df = pd.DataFrame(pathways_matrix.toarray(), columns=pathways_vectorizer.get_feature_names())
        pathways_df.fillna(0, inplace=True)
        
        categorical_encoder = CategoricalEncoder(encoding='onehot-dense')
        ordinal_encoder = CategoricalEncoder(encoding='ordinal')
        
        impact_encoded = pd.DataFrame(ordinal_encoder.fit_transform(X_copy.IMPACT.values.reshape(-1, 1)))
        X_copy = categorical_encoder.fit_transform(X_copy.drop(['Consequence', 'IMPACT', 'Pathways'], axis=1))
        
        return pd.concat([pd.DataFrame(X_copy), consequence_df, impact_encoded, pathways_df], axis=1)
class ImputeCategoricalValues(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.comment_ = 'Custom categorical imputer'
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        X_copy = X.copy() # needed to prevent the subsequent code from overriding original dataframe
        
        X_copy['Bam_edit_was_missing'] = X_copy.BAM_EDIT.isnull()
        X_copy['Sift_was_missing'] = X_copy.SIFT.isnull()
        X_copy['PolyPhen_was_missing'] = X_copy.PolyPhen.isnull()
        X_copy['BLOSUM62_was_missing'] = X_copy.BLOSUM62.isnull()
        
        # Since only 14 "strands" are imputed, 'Strand_was_missing' indicator won't be created
        # Impute with most frequent value: -1
        X_copy.STRAND.fillna(-1, inplace=True)
        
        # As half of the records does not contain BAM_EDIT, impute 'Unknown' value
        X_copy.BAM_EDIT.fillna('Unknown', inplace=True)
        
        # Impute 'Unknown'
        X_copy.SIFT.fillna('Unknown', inplace=True)
        
        # Impute 'Unknown'
        X_copy.PolyPhen.fillna('Unknown', inplace=True)
        
        # Impute 'Unknown'
        X_copy.BLOSUM62.fillna('Unknown', inplace=True)
        X_copy.BLOSUM62 = X_copy.BLOSUM62.astype(str)
        
        return X_copy
class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        return X[self.attribute_names]
categorical_pipeline = Pipeline([
    ('selector', DataFrameSelector(categorical_features)),
    ('impute_category', ImputeCategoricalValues()),
    ('cat_encoder', EncodeNonNumericValues(clinvar_transformed.Consequence.unique()))
])
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import StandardScaler

numeric_pipeline = Pipeline([
    ('selector', DataFrameSelector(numeric_features)),
    ('impute_numeric', Imputer(strategy='median')),
    ('std_scaler', StandardScaler())
])
from sklearn.pipeline import FeatureUnion

full_pipeline = FeatureUnion(transformer_list=[
    ('categorical_pipeline', categorical_pipeline),
    ('numeric_pipeline', numeric_pipeline)
])
X_train_prepared = full_pipeline.fit_transform(X_train_balanced)
X_val_prepared = full_pipeline.transform(X_val)
X_test_prepared = full_pipeline.transform(X_test)
print(X_train_prepared.shape)
print(X_val_prepared.shape)
print(X_test_prepared.shape)
# Define function to plot ROC-Curves
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

def plot_roc_curve(y_test, y_pred):
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)

    fig = plt.figure(figsize=(8,8))
    plt.title('Receiver Operating Characteristic')

    plt.plot(fpr, tpr, label='l1')
    plt.legend('lower right')

    plt.plot([0, 1], [0, 1], 'k--')

    plt.xlim([-0.1, 1.1])
    plt.ylim([-0.1, 1.1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    
    print('Area Under the Curve:', auc(fpr, tpr))
# Define function that reports results of 10-fold Cross-Validations
def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard Deviation:", scores.std())
# Import all metrics for model evaluation
from sklearn.metrics import *
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import NearestCentroid, KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.feature_selection import SelectFromModel

pipelines = {
    'rf': make_pipeline(SelectFromModel(RandomForestClassifier(random_state=42)), RandomForestClassifier(random_state=42)),
    'gb': make_pipeline(SelectFromModel(GradientBoostingClassifier(random_state=42)), GradientBoostingClassifier(random_state=42)),
    'decision_tree': make_pipeline(SelectFromModel(DecisionTreeClassifier(random_state=42)), DecisionTreeClassifier(random_state=42)),
}
rf_hyperparameters = {
    'randomforestclassifier__bootstrap': [True],
    'randomforestclassifier__max_depth': [4, 5],
    'randomforestclassifier__max_features': [4, 5],
    'randomforestclassifier__min_samples_leaf': [3, 4],
    'randomforestclassifier__min_samples_split': [8, 10],
    'randomforestclassifier__n_estimators': [100]
}

gb_hyperparameters = {
    'gradientboostingclassifier__max_depth': [4, 5],
    'gradientboostingclassifier__max_features': [4, 5],
    'gradientboostingclassifier__min_samples_leaf': [3, 4],
    'gradientboostingclassifier__min_samples_split': [8, 10],
    'gradientboostingclassifier__n_estimators': [100]
}

decision_tree_hyperparameters = {
    'decisiontreeclassifier__criterion': ['gini', 'entropy'],
    'decisiontreeclassifier__max_depth': [ 6, 7, 8],
    'decisiontreeclassifier__max_features': [6, 7, 8]
}
hyperparameters = {
    'rf': rf_hyperparameters,
    'gb': gb_hyperparameters,
    'decision_tree': decision_tree_hyperparameters
}
from sklearn.model_selection import GridSearchCV, StratifiedKFold

fitted_models = {}

cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

for name, pipeline in pipelines.items():
    model = GridSearchCV(pipeline, hyperparameters[name], cv=cv, scoring='roc_auc', 
                         verbose=50, n_jobs=-1)
    model.fit(X_train_prepared, y_train_balanced)
    
    fitted_models[name] = model
    print(name, 'has been fitted.')
# Best score for each model
for name, model in fitted_models.items():
    print( name, model.best_score_ )
fig = plt.figure(figsize=(12,12))
list_of_models = {'rf': 'Random Forest', 'gb': 'Gradient Boosting', 'decision_tree': 'Decision Tree'}

for name, model in fitted_models.items():
    y_pred = fitted_models[name].best_estimator_.predict(X_val_prepared)
    fpr, tpr, thresholds = roc_curve(y_val, y_pred)
    
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label='%s: %0.2f' % (list_of_models[name], roc_auc), linewidth=5)
    
plt.title('Receiver Operating Characteristic')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([-0.1, 1.1])
plt.ylim([-0.1, 1.1])
plt.xlabel('False Positive Rate', fontsize=22)
plt.ylabel('True Positive Rate', fontsize=22)
plt.legend(loc=2)

plt.savefig('roc_curves.jpg')
plt.show()
from sklearn.metrics import accuracy_score

for name, model in fitted_models.items():
    y_pred = fitted_models[name].best_estimator_.predict(X_val_prepared)
    print(name, '\n', classification_report(y_val, y_pred))
    print(accuracy_score(y_val, y_pred))
X_final_train = np.concatenate((X_train_prepared, X_val_prepared), axis=0)
y_final_train = pd.concat([y_train_balanced, y_val], axis=0)
print(X_final_train.shape)
print(y_final_train.shape)
# Best Parameters
fitted_models['gb'].best_params_
best_gb_params = {'gradientboostingclassifier__max_depth': [5],
                  'gradientboostingclassifier__max_features': [5],
                  'gradientboostingclassifier__min_samples_leaf': [4],
                  'gradientboostingclassifier__min_samples_split': [8],
                  'gradientboostingclassifier__n_estimators': [100]
                 }

final_gb_pipeline = make_pipeline(SelectFromModel(GradientBoostingClassifier()), 
                                  GradientBoostingClassifier(random_state=42))

best_gb = GridSearchCV(final_gb_pipeline, best_gb_params, cv=cv, scoring='roc_auc', 
                         verbose=50, n_jobs=-1)
best_gb.fit(X_train_prepared, y_train_balanced)
fig = plt.figure(figsize=(12,12))

y_pred = best_gb.predict(X_test_prepared)
fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, label='Gradient Boosting: %0.2f' % (roc_auc))
    
plt.title('Receiver Operating Characteristic')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([-0.1, 1.1])
plt.ylim([-0.1, 1.1])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc=2)

plt.show()
y_pred_test = best_gb.predict(X_test_prepared)
print(name, '\n', classification_report(y_test, y_pred_test))
# Confusion matrix for the test set
print(confusion_matrix(y_test, y_pred_test))
# END