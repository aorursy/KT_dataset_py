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

#Use data load function -> Load data into a Data Frame. 
#Df is a table that contains the contents for the data
usa_2016_presidential_election_by_county = pd.read_csv('/kaggle/input/us-elections-dataset/usa-2016-presidential-election-by-county.csv', sep=';')
print(f"There are {len(usa_2016_presidential_election_by_county)} Records in total")
usa_2016_presidential_election_by_county.head()
#Print out how many data records/rows in the dataset 
#.head() function takes the first five rows in the dataset
#The dataset indicates indiviaul county of the U.S 
for k in usa_2016_presidential_election_by_county.keys():
    print(k)
#prints out: all the attributes for each county
usa_2016_presidential_election_by_county[["County", "Republicans 2016"]]
#Look at the attributes
#[A1]: one attribute, [[county, A1]]: prints out 2 attributes e.g. county and the percentage that the citizens voted for republicans
"""
Drop the existing records that we are interested in
Like: how many votes HC got votes in 2016, republicans performance in etc...
whenever a row contains a missing value, the record will be dropped
"""
df = usa_2016_presidential_election_by_county.dropna(subset=[
    "Votes16 Clintonh", "Votes16 Trumpd",
    "Republicans 2016", "Democrats 2016",
    "Republicans 2012", "Republicans 2008",
    "Democrats 2012", "Democrats 2008", "Votes"])

#df["Votes16 Clintonh"].sum()
#[Ref initial entropy1]
n_dem = df["Votes16 Clintonh"].sum() #df[]: How many votes H.C received each county -> Sum
n_rep = df["Votes16 Trumpd"].sum()
p_dem = n_dem / (n_dem + n_rep) # Respective number/total votes -> to get the probability each D.T or H.C
p_rep = n_rep / (n_dem + n_rep) #Rep:D.T
print(f"Votes for DEM {n_dem}, probability {p_dem:.4f}")
print(f"Votes for REP {n_rep}, probability {p_rep:.4f}")

#Nearly have half/half chance of voting for each of the candidate
# -> that means almost have maximum uncertainty of his/her vote

#[Ref initial entropy2]
ent = - (p_dem * np.log2(p_dem) + p_rep * np.log2(p_rep)).sum()
print(f"Entropy: {ent:.4f}")
#Since the entropy is very high, we need some attributes/information to recude the uncertainty 
"""
Instead of looking at the entire population, select sub-population and see if things get more formative
"""
df[df["State"] == "California"] #if the record is within California, -> df
#since the computation is repeated, make a def 
def exam_votes(df_i):
    n_dem = df_i["Votes16 Clintonh"].sum() 
    n_rep = df_i["Votes16 Trumpd"].sum()
    p_dem = n_dem / (n_dem + n_rep)
    p_rep = n_rep / (n_dem + n_rep)
    print(f"2016 Vote Statistics {n_dem + n_rep} votes in {len(df_i)} counties")
    print(f"Votes for DEM {n_dem}, probability {p_dem:.4f}")
    print(f"Votes for REP {n_rep}, probability {p_rep:.4f}")
    ent = - (p_dem * np.log2(p_dem) + p_rep * np.log2(p_rep)).sum()
    print(f"Entropy: {ent:.4f}")
    return ent, p_dem, p_rep, n_dem, n_rep
#input = df[CA]
ent, p_dem, p_rep, n_dem, n_rep = exam_votes(df[df["State"] == "California"])

import plotly.express as px
fig = px.scatter_geo(df, lat="lat", lon="lon", color="Republicans 2016", hover_name="County", size="Votes")#, 
 
fig.show()
px.scatter(df, x="Republicans 2016", y="Democrats 2016", hover_name="County")
# Get the record of the county "District of Columbia, District of Columbia"
df[df["County"] == "District of Columbia, District of Columbia"] # only one record
# Let's to the entropy computation
_ = exam_votes(df[df["County"] == "District of Columbia, District of Columbia"])
df["Republicans Won 2016"] = df["Democrats 2016"] < df["Republicans 2016"]
# Check the 2016 results
df["Republicans Won 2016"].value_counts(normalize=True)  #First one: T or F, second: %
                                                         #True: Republican won
#[Ref draft3]
prob = df["Republicans Won 2016"].value_counts(normalize=True)
prob = np.array(prob)
print(f"Distribution of *repub won* w.r.t. county is [True (Rep Won), False (Dem Won)]={prob}")
ent = - (prob * np.log2(prob)).sum()
print(f"Entropy is {ent:.4f}")
#State

def exam_counties(df, verbose=True):
    prob = df["Republicans Won 2016"].value_counts(normalize=True)
    prob = np.array(prob)
    ent = - (prob * np.log2(np.maximum(prob, 1e-6))).sum()
    if verbose:
        print(f"Distribution of *repub won* w.r.t. county is [True (Rep Won), False (Dem Won)]={prob}")
        print(f"Entropy is {ent:.4f}")
    return ent
#[Ref draft1], split the data respect to county, the no. of individual county in each state
states = df["State"].value_counts()
states
#exam_counties(df[df["State"]=="Texas"], verbose=False)
#[Ref draft2]
total_ent = 0
num_counties = 0
for k, v in states.iteritems():
    ent = exam_counties(df[df["State"]==k], verbose=False) # in this particular state take sub-population in a certain county
    print(f"State {k} has {v} counties, result entropy {ent:.3f}")
    total_ent += v * ent
    num_counties += v
    
print(f"Weighted sum of entropies {total_ent/num_counties :.3f}")
ent0 = - (prob * np.log2(prob)).sum()
print(f"Entropy is {ent0:.3f}")
print(f"Info Gain: {ent0 - total_ent/num_counties:.3f}")
#compare to the original entropy 0.6252, it got reduced
#Education
# Examine the education information.
df[["Less Than High School Diploma", "At Least High School Diploma",
    "At Least Bachelors's Degree","Graduate Degree"]]
fig = px.scatter(df, x="At Least Bachelors's Degree", y="Democrats 2016", 
                 color="Republicans Won 2016", color_discrete_sequence=['red','blue'])
fig.show()
df["More Than 30p Bachelors"] = df["At Least Bachelors's Degree"] > 30
#whether more than 30% of residents received Bachelor's degree
#[Ref Edu]
total_ent = 0
num_counties = 0
attr = "More Than 30p Bachelors"
for k, v in df[attr].value_counts().iteritems():
    ent = exam_counties(df[df[attr]==k], verbose=False) # in this particular state
    print(f"there are {v} counties where {attr} is {k}, result entropy {ent:.3f}")
    total_ent += v * ent
    num_counties += v
    
print(f"Weighted sum of entropies {total_ent/num_counties :.3f}")

# recall that the original entropy is ... (copied from above)
prob = df["Republicans Won 2016"].value_counts(normalize=True)
prob = np.array(prob)
print(f"Distribution of *repub won* w.r.t. county is [True (Rep Won), False (Dem Won)]={prob}")
ent0 = - (prob * np.log2(prob)).sum()
print(f"Entropy is {ent0:.4f}")
print(f"Info Gain: {ent0 - total_ent/num_counties:.4f}")
#Population
fig = px.scatter(df, x="White (Not Latino) Population", y="Democrats 2016", color="Republicans Won 2016",
                color_discrete_sequence=['red','blue'])
fig.show()
#cut at 60
df["White (Not Latino) Population Is Greater Than 60p"] = df["White (Not Latino) Population"] > 60
def compute_weighted_sub_entropy(df, attr, verbose=True):
    total_ent = 0
    num_counties = 0
    for k, v in df[attr].value_counts().iteritems():
        ent = exam_counties(df[df[attr]==k], verbose=False) # in this particular sub-population
        if verbose:
            print(f"there are {v} counties where {attr} is {k}, result entropy {ent:.3f}")
        total_ent += v * ent
        num_counties += v
    
    weighted_ent = total_ent/num_counties
    if verbose:
        print(f"Weighted sum of entropies {weighted_ent:.3f}")
    return weighted_ent

weighted_ent = compute_weighted_sub_entropy(df, "White (Not Latino) Population Is Greater Than 60p")
print(f"Info Gain: {ent0 - weighted_ent:.4f}")
#[Ref attributes in C]
attributes = ["White (Not Latino) Population", 
    "African American Population",
    "Native American Population",
    "Asian American Population", 
    "Latino Population",
    "Less Than High School Diploma",
    "At Least High School Diploma",
    "At Least Bachelors's Degree",
    "Graduate Degree",
    "School Enrollment",
    "Median Earnings 2010",
    "Children Under 6 Living in Poverty",
    "Adults 65 and Older Living in Poverty",
    "Preschool.Enrollment.Ratio.enrolled.ages.3.and.4",
    "Poverty.Rate.below.federal.poverty.threshold",
    "Gini.Coefficient",
    "Child.Poverty.living.in.families.below.the.poverty.line",
    "Management.professional.and.related.occupations",
    "Service.occupations",
    "Sales.and.office.occupations",
    "Farming.fishing.and.forestry.occupations",
    "Construction.extraction.maintenance.and.repair.occupations",
    "Production.transportation.and.material.moving.occupations",
    "Median Age",
    "Poor.physical.health.days",
    "Poor.mental.health.days",
    "Low.birthweight",
    "Teen.births",
    "Children.in.single.parent.households",
    "Adult.smoking",
    "Adult.obesity",
    "Diabetes",
    "Sexually.transmitted.infections",
    "HIV.prevalence.rate",
    "Uninsured",
    "Unemployment",
    "Violent.crime",
    "Homicide.rate",
    "Injury.deaths",
    "Infant.mortality"]
new_attributes = []
for a in attributes:
    new_a = "Quant4." + a
    df[new_a] = pd.qcut(df[a], q=4, labels=["q1", "q2", "q3", "q4"])
    new_attributes.append(new_a)
#pd.qcut(df[ "At Least Bachelors's Degree"], q=4, labels=["q1", "q2", "q3", "q4"])
#[REF_TreeBuilding]
def compute_entropy(y):
    """
    :param y: The data samples of a discrete distribution
    """
    if len(y) < 2: #  a trivial case
        return 0
    freq = np.array( y.value_counts(normalize=True) )
    return -(freq * np.log2(freq + 1e-6)).sum() # the small eps for 
    # safe numerical computation 
    
def compute_info_gain(samples, attr, target):
    values = samples[attr].value_counts(normalize=True)
    split_ent = 0
    for v, fr in values.iteritems():
        index = samples[attr]==v
        sub_ent = compute_entropy(target[index])
        split_ent += fr * sub_ent
    
    ent = compute_entropy(target)
    return ent - split_ent

class TreeNode:
    """
    A recursively defined data structure to store a tree.
    Each node can contain other nodes as its children
    """
    def __init__(self, node_name="", min_sample_num=10, default_decision=None):
        self.children = {} # Sub nodes -- starts from here, and make it growing, chldrn=collection of more tree nodes
        # recursive, those elements of the same type (TreeNode)
        self.decision = None # Undecided
        self.split_feat_name = None # Splitting feature
        self.name = node_name
        self.default_decision = default_decision
        self.min_sample_num = min_sample_num

    def pretty_print(self, prefix=''):
        if self.split_feat_name is not None:
            for k, v in self.children.items():
                v.pretty_print(f"{prefix}:When {self.split_feat_name} is {k}")
                #v.pretty_print(f"{prefix}:{k}:")
        else:
            print(f"{prefix}:{self.decision}")

    def predict(self, sample):
        if self.decision is not None:
            # uncomment to get log information of code execution
            print("Decision:", self.decision)
            return self.decision
        else: 
            # this node is an internal one, further queries about an attribute 
            # of the data is needed.
            attr_val = sample[self.split_feat_name]
            child = self.children[attr_val]
            # uncomment to get log information of code execution
            print("Testing ", self.split_feat_name, "->", attr_val)

            # [Exercise]
            # Insert your code here
            return child.predict(sample)
 
#below: 
    def fit(self, X, y):
        if self.default_decision is None:
            self.default_decision = y.mode()[0]
            
            
        print(self.name, "received", len(X), "samples")
        if len(X) < self.min_sample_num:
            # If the data is empty when this node is arrived, 
            # we just make an arbitrary decision
            """
            if the sub-population is small enough, it's better not to split anymore ^^, 
            **OR**
            we have received unanimous conclusion that all counties in the sub-population
            will give one same decision
            """
            if len(X) == 0:
                self.decision = self.default_decision
                print("DECESION", self.decision)
            else:
                self.decision = y.mode()[0]
                print("DECESION", self.decision)
            return
        else: 
            unique_values = y.unique()
            if len(unique_values) == 1:
                self.decision = unique_values[0]
                print("DECESION", self.decision)
                return
            else:
                info_gain_max = 0    #From here, important: divide data using different keys like Qs
                for a in X.keys(): # Examine each attribute
                    aig = compute_info_gain(X, a, y)
                    if aig > info_gain_max:
                        info_gain_max = aig
                        self.split_feat_name = a
                print(f"Split by {self.split_feat_name}, IG: {info_gain_max:.2f}")
                self.children = {}
                for v in X[self.split_feat_name].unique():
                    index = X[self.split_feat_name] == v
                    self.children[v] = TreeNode( #Create Treenode #creat sub population using children
                        node_name=self.name + ":" + self.split_feat_name + "==" + str(v),
                        min_sample_num=self.min_sample_num,
                        default_decision=self.default_decision)
                    self.children[v].fit(X[index], y[index]) #split and fit the children of one child by another using sub-population
# Test tree building
data = df[new_attributes].dropna('columns', 'any')
target = df["Republicans Won 2016"]

t = TreeNode(min_sample_num=50)
t.fit(data, target)
        
#[Ref Show]
data = df[new_attributes].dropna('columns', 'any')
data.keys()
data[['Quant4.White (Not Latino) Population',
      'Quant4.African American Population']]
corr = 0
err_fp = 0 #false positive: actually false, but predicted true
err_fn = 0 #false negative: actually true, predicted false
for (i, ct), tgt in zip(data.iterrows(), target):
    a = t.predict(ct)
    if a and not tgt:
        err_fp += 1
    elif not a and tgt:
        err_fn += 1
    else:
        corr += 1
corr, err_fp, err_fn 

"""dataset = usa_2016_presidential_election_by_county
X, y = dataset["data"], dataset["target"]
ind = (y == 0) + (y == 1)
X = X[ind]
y = y[ind] # take two classes"""