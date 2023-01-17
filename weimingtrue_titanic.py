import numpy as np 
import pandas as pd
%matplotlib inline
from matplotlib import pyplot as plt
import seaborn as sns
sns.set(style="whitegrid")
import re
import pandas as pd
import numpy as np
from IPython.display import display, HTML

#For Kaggle 
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
#For win machine, csv files stored at the same folder as notebook
#train = pd.read_csv('train.csv')
#test = pd.read_csv('test.csv')

#The PassengerId col is made for indexing, especially when one plan to merge train and test for feature engineering 
train = train.set_index("PassengerId")
test = test.set_index("PassengerId")

train_sur = train['Survived']
#keep survival data, add back after preprocess
combined = pd.concat([train.drop("Survived",axis=1),test])
train.describe(include='all')
combined.describe(include='all')
test.describe(include='all')
combined.sample(10)
combined[combined["Embarked"].isna()]
combined[combined["Fare"].isna()]
sur =train[train["Survived"] == 1].copy()
dec = train[train["Survived"] == 0].copy()
#nbins  = 1 + log2(N) ~ 11
sur["Age" ].plot(kind='hist',alpha=0.5,color='red',bins=11)
dec["Age"].plot(kind='hist',alpha=0.5,color='blue',bins=11,title ="Death/Survival count by by age, na are filled as -0.5" )
plt.legend(['Survived','Died'])
plt.show()
sur_no_na= sur[sur["Age"]> 0]
dec_no_na = dec[dec["Age"]> 0] 
bins = [-5,5,15,30,45,60,100]
sur_no_na["Age" ].plot(kind='hist',alpha=0.5,color='red',bins=bins)
dec_no_na["Age"].plot(kind='hist',alpha=0.5,color='blue',bins=bins,title ="Death/Survival count by Age groups, w/O missing values" )
plt.xticks(bins,["NA","Infant","Child","Yound Adult","Adult","Middle Age","Senior"])
plt.legend(['Survived','Died'])
plt.show()
#nbins  = 3
from matplotlib.ticker import FormatStrFormatter
ss  = train.pivot_table(index=["Pclass","Sex"],values='Survived').copy()
ss.plot(kind='bar',alpha=0.5,color='red',title="Survival rate of each sex across different classes")
plt.show()
def process_na(df):
    """Fill na w. most probable values"""
    df["Embarked"] = df["Embarked"].fillna( df["Embarked"].mode().iloc[0])
    df["Cabin"] = df["Cabin"].fillna("Unknown")
    return df

def process_age(df):
    """20% of age info is missing, handle with care"""
    df["Age"] = df["Age"].fillna(-0.5)
    cut_pts = [-1, 0, 5, 15, 30, 45, 65, 100]
    #Given the accident took place in the early 2oth century, 
    #it make sense to category age by the standard then.
    #I should probably differentiate age groups between male and female,
    #but the improvement maybe marginal, will try next time
    age_labels = ["NA",          #[-1]
                  "Infant",      #[0-5) 
                  "Child",       #[5,15)
                  "Young Adult", #[15,30)
                  "Adult",       #[30,45)
                  "Middle Age",  #[45,65)
                  "Senior"]      #[65,100)
    df["Age_cat"] = pd.cut(df["Age"],cut_pts, labels=age_labels)
    return df
combined = process_na(combined)
combined = process_age(combined)
combined[combined["Fare"].isna()]
Fare_nas = combined.loc[combined["Fare"].isnull()]
Fare_pt = combined.pivot_table(index=["Pclass","Embarked"],values='Fare').copy()

for i in Fare_nas.index:
    p,e = test.at[i,"Pclass"], test.at[i,"Embarked"]
    if i in combined.index:
        combined.loc[i,"Fare"] = float(Fare_pt.loc[p,e])
        
g = sns.PairGrid(combined)
g.map_diag(plt.hist)
g.map_offdiag(plt.scatter);
cabin_cat = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'T', 'U']
ocp_info = dict()
# ocp_info is a two way dictionary that stores all ocupy info, it is more ideal to store in a matrix, but this will work

#Credit: https://github.com/matthagy/titanic-families
name_rgx = re.compile(r'''
^                 # Explicit start
       \s*
  ([^,]+)         # Last Name: 1
     , \s+
  ([^.]+) \.      # Title:2
       \s+
  ([^("]+)?       # Main name:3
       \s*
  (?:
    "([^"]+)"     # Nick name:4
  )?
       \s*
  (?:             # Other name:5
     \(
        ([^)]+)
      \)
   )?
''', re.VERBOSE)

def process_fare(df):
    """Process the fare using real fare, i.e. per person fare
    Apart 1st, 2nd and 3rd class, there were two luxury first class suites on titanic, which I marked as Luxury.
    I do not want to look up the price range but based my division on the fare distribution. 
    """       
    cut_points = [-1,1,10,25,60,1000]
    label_names = ["free",       # [0]Free ticket, ship crew 
                   "third",    # [1,10)Third class tickets
                   "second",   # [10,25)econd class tickets
                   "first",   # [25,60)First class tickets
                   "luxury"]     #[60,1000) Upper first class tickets
    df["Fare_cat"] = pd.cut(df["real_fare"],cut_points,labels=label_names)
    return df


def process_titles_lastname(df):
    """Extract and categorize the title and last name from the name column """
    for pid in df.index:
        name = df.at[pid,"Name"]
        m = name_rgx.match(name)
        if not m:
            raise ValueError('bad name %r' % (name,))
        df.loc[pid,"LastName"] = m.group(1).strip()
        df.loc[pid,"Title"] = m.group(2).strip()        
        if m.group(3) is not None:
            df.loc[pid,"MainName"] = m.group(3).strip()
        else:
            df.loc[pid,"MainName"] = ""
                    
        if m.group(4) is not None:
            df.loc[pid,"OtherName"] = m.group(4).strip()
        else:
            df.loc[pid,"OtherName"] = ""
                    
        if m.group(5) is not None:
            df.loc[pid,"NickName"] = m.group(5).strip()
        else:
            df.loc[pid,"NickName"] = ""
            
    titles = {
        "Mr" :         "Mr",
        "Mme":         "Mrs",
        "Ms":          "Mrs",
        "Mrs" :        "Mrs",
        "Master" :     "Master",
        "Mlle":        "Miss",
        "Miss" :       "Miss",
        #Will differentiate by pclass for all officer and royalty
        "Capt":        "Officer",        
        "Major":       "Officer",        
        "Col":         "Officer",       
        "Dr":          "Officer",
        "Rev":         "Officer",        
        "Jonkheer":    "Royalty",
        "Dona":        "Royalty",
        "Don":         "Royalty",
        "Sir" :        "Royalty",
        "Lady" :       "Royalty",        
        "the Countess": "Royalty"  
    }
    
    df["Title"] = df["Title"].map(titles)
    #Add embarked and ticket info to lastname can distinguish different families of the same last name.
    df["LastName"] = (df["LastName"]+ df["Embarked"] + df["Pclass"].apply(str) + 
                     df["Ticket_Prfx"] + df["Ticket_Range"])
    
    return df

def process_cabin(df):
    """
    This function generate a two-way dictionary, ocp_info, where each passenger's cabin info and 
    each cabin's passgener info is stored.

   
    """
    cabin_info = df["Cabin"].to_dict()
    
    for pid in cabin_info.keys():
        if pid not in ocp_info.keys():
            ocp_info[pid] = []
        
        for cab in cabin_info[pid].split():
            #df.loc[pid,"Cabin_"+cab[0]] += 1
            ocp_info[pid].append(cab)
            if cab in ocp_info.keys():
                ocp_info[cab].append(pid)
            else:
                ocp_info[cab] = [pid]
    
    return df
   
def create_dummies(df,column_name):
    """Create Dummy Columns (One Hot Encoding) from a single Column

    Usage
    ------

    train = create_dummies(train,"Age")
    """
    dummies = pd.get_dummies(df[column_name],prefix=column_name)
    df = pd.concat([df,dummies],axis=1)
    return df

def process_ticket(df):
    extracted_tickets  = df["Ticket"].str.extract('([0-9]+)$',expand=False)
    extracted_tk_pref  = df["Ticket"].str.extract('^(.+) [0-9]+$',expand=False)
    extracted_tickets = list(map(lambda x: 0 if x is np.nan else int(x), extracted_tickets))
    extracted_tk_range = list(map(lambda x: 0 if x is np.nan else str(int(x/100)), extracted_tickets))
    extracted_tk_pref = list(map(lambda x: "No." if x is np.nan else x, extracted_tk_pref))
    df["Ticket_No"] = extracted_tickets
    df["Ticket_Range"] = extracted_tk_range
    df["Ticket_Prfx"] = extracted_tk_pref
    return df

def preprocess(df):  
    df = process_cabin(df)
    df = process_ticket(df)
    
    df = process_titles_lastname(df)
    """
    for col in ["Sex","Age_cat"]:#"Pclass","Cabin_cat","Age"
        df = create_dummies(df,col)    """
    return df

def assign_Age_cat(age):
    """This function is used to assign age cat for those with no age info, based on an educated guess."""
    if 0 < age < 5:        cat = "Infant"
    elif 5 <= age < 15:    cat = "Child"
    elif 15 <= age < 30:   cat = "Young Adult"
    elif 30 <= age < 45:   cat = "Adult"
    elif 45 <= age < 65:   cat = "Middle Age"
    elif 65 <= age < 100:  cat = "Senior"
    else:                  cat = "NA"
    
    return cat

def assign_ac_by_title(title):
    """This function is used to assign age cat simply from title, this will result in a lot error"""

    if title == "Miss" or title == "Master":
        return "Child"
    
    if title == "Mr"  or  title == "Mrs":
        return "Young Adult"    
    
    if title == "Officer":
        return "Middle Age"
    
    if title == "Royalty":        
        return "Senior"
    
    return "NA"


combined["immediate_family"] = combined["Parch"] + combined["SibSp"]

combined = preprocess(combined)
#Assign three kinds of travel group 
#1 family, has same last name, same or successive ticket number
#2 passengers on same ticket or passengers share a cabin as acquaintance
#3 alone

families = list(set(combined["LastName"]))

combined["family_size"] = 1
combined["n_cabin_mates"] = 1
combined["n_ticket_holders"] = 1

combined["group_id"]= 0
combined["group_type"] = "0"
combined["group_size"] = 1

cp_id = 0
combined["cp_id"] = -1
combined["family_has_couple"] = 0
for pid in combined.index:
    if combined.loc[pid,"cp_id"] != -1:
        continue
    cp = combined[combined["MainName"] == combined.loc[pid,"MainName"]]
    if len(cp) == 2:       
        
        if ( set(combined.loc[cp.index,"Sex"]) == set(["female","male"])) and        (combined.loc[cp.index[0],"Age"] > 15 or combined.loc[cp.index[0],"Age"] < 0) and         (combined.loc[cp.index[1],"Age"] > 15 or combined.loc[cp.index[1],"Age"] < 0):
             
            
            #Needs fixing, a Dr. and Mrs will not be considered as couple
            combined.loc[cp.index,"cp_id"] = cp_id
            cp_id += 1    
f_dict = dict(zip(families, range(len(families))))
g_counter  =len(f_dict)
combined["family_has_children"] = 0
combined["family_has_senior"] = 0
combined["family_role"] = "no_role"
for ln in families:
    f_group = combined[combined["LastName"] == ln]
    fs = max(len(f_group.index),combined.loc[f_group.index,"immediate_family"].max())
    #Kaggle data doesn't contain all passengers, so parch + sibsp is more reliable than last name look up
    #Although, kaggle data itself contains a few errors.
    #if a passenger has a family, assign he/she to the family group
    combined.loc[f_group.index, "family_size"] = fs
    combined.loc[f_group.index, "group_id"] = f_dict[ln]
    n_children, n_seniors  = 0, 0
    for f_member in f_group.index:
        age, title = combined.loc[f_member,"Age"],combined.loc[f_member,"Title"]
        if 0< age <= 15 or title == "Miss" or title == "Master":
            combined.loc[f_member,"family_role"] = "Child"
            n_children += 1
            continue
        if age > 65:
            combined.loc[f_member,"family_role"] = "Senior"
            n_seniors += 1
            continue
        if combined.loc[f_member,"cp_id"] != -1:
            if combined.loc[f_member,"Sex"] == "male":
                combined.loc[f_member,"family_role"] = "Father"
            else:
                combined.loc[f_member,"family_role"] = "Mother"
        
            
    combined.loc[f_group.index,"family_has_children"] = n_children
    combined.loc[f_group.index,"family_has_senior"] = n_seniors

    if fs > 1:
        combined.loc[f_group.index, "group_type"] = "family"
        combined.loc[f_group.index, "group_size"] = fs


tk_lst = list(set(combined["Ticket"]))
for tk in tk_lst:
    t_group = combined[combined["Ticket"] ==  tk]
    combined.loc[t_group.index, "n_ticket_holders"] = len(t_group.index)
    n_t  = len(t_group.index)
    for tm in t_group.index:
        combined.loc[tm, "real_fare"] = combined.loc[tm, "Fare"] / n_t
combined = process_fare(combined)        

for pid in combined.index:
    t_group = combined[combined["Ticket"] == combined.loc[pid,"Ticket"]]
    
    if combined.loc[pid,"group_type"] != "0":
        continue
    ln = combined["LastName"].at[pid]

    if ocp_info[pid] != ['Unknown']:#Has cabin info
        cabin_lst =[x for x in ocp_info[pid]]
        #All the cabins in the ticket
        c_group  = list(set([traveler for cabin in cabin_lst for traveler in ocp_info[cabin] if len(cabin)>1]))
        combined.loc[c_group, "n_cabin_mates"] = len(c_group)
    else:
        #passenger with no cabin info are assumed to be alone.
        combined.loc[pid,"n_cabin_mates"] = 1
        c_group  = []

    co_travelers = pd.Index(sorted(set(list(t_group.index) +  c_group)))
    #index of fellow travelers
    dominant_ln = combined.loc[co_travelers,"LastName"].value_counts().index[0]
    dominant_fs = len(combined[combined["LastName"] == dominant_ln].index)
    dominant_fg = combined[combined["group_id"] == f_dict[dominant_ln]].index    
    #Assign group next:     
    
    #if a lone passenger shares cabin/ticket with a family, assign this passenger to the family
    if  dominant_ln != ln and dominant_fs > 1:
        co_travelers = pd.Index(sorted(set(list(t_group.index) +  c_group + list(dominant_fg))))
        for p in co_travelers:
            combined.loc[p, "group_type"] = "family"
            combined.loc[p, "group_size"] = len(co_travelers)
            combined.loc[p, "group_id"] = f_dict[dominant_ln]

    #else if a group of passenger travels together, assign as travel companions      
    elif dominant_fs == 1 and len(co_travelers)> 1:

        for p in co_travelers:
            combined.loc[p,"group_type"] = "acquaintance"
            combined.loc[p,"group_size"] = len(co_travelers)
            combined.loc[p,"group_id"] = g_counter       

        g_counter += 1
    else:
        combined.loc[pid, "group_type"] = "alone"
        combined.loc[pid, "group_size"] = 1
#Assign age cat for couples, if both NA, assume Middle age
for couple_id in range(cp_id):
    cp = combined[combined["cp_id"] == couple_id].index
    age_cats = set(combined.loc[cp,"Age_cat"])
    # mark has couple in the family 
    tag = combined[combined["group_id"] == combined.loc[cp[0],"group_id"] ]
    combined.loc[tag.index,"family_has_couple"] = combined.loc[cp[0], "cp_id"]
    if "NA" not in age_cats:
        continue
    
    if age_cats == set(["NA"]):
        combined.loc[cp,"Age_cat"] = "Middle Age"
    elif "NA" in age_cats:
        non_NA_cat = list(age_cats - set(["NA"]))[0]
        combined.loc[cp,"Age_cat"] = non_NA_cat
combined["isVIP"] = 0

for pid in combined.index:
    if (combined.loc[pid,"Fare_cat"] == "luxury") or (combined.loc[pid,"Pclass"] == "One" and (combined.loc[pid,"Title"] == "Officer"))  or (combined.loc[pid,"Title"] == "Royalty"):
        combined.loc[pid,"isVIP"] = 1
        if combined.loc[pid,"cp_id"] != -1: #"family":
            cp = combined[combined["cp_id"] == combined.loc[pid,"cp_id"] ]
            combined.loc[cp.index,"isVIP"] = 1
combined["isCrew"] = 0

staff = combined[combined["real_fare"] == 0].index
combined.loc[staff,"isCrew"] = 1
for pid in combined.index:
    if combined.loc[pid,"Age_cat"] != "NA":
        continue
        
    if combined.loc[pid,"group_type"] == "acquaintance":
        aq_group = combined[
            (combined["group_id"] == combined.loc[pid,"group_id"]) 
            & (combined["Age"] > 0)]
        aq_avg_age =aq_group["Age"].mean()
        combined.loc[pid,"Age_cat"] = assign_Age_cat(aq_avg_age)
        if combined.loc[pid,"Age_cat"] != "NA":        continue
        
    if combined.loc[pid,"group_type"] == "family":
        f_group = combined[combined["group_id"] == combined.loc[pid,"group_id"]]
        if combined.loc[pid,"family_has_couple"] != 0:
            cp = combined[combined["cp_id"] == combined.loc[pid,"family_has_couple"]]
            couple_age =  combined.loc[cp.index, "Age"]
            if combined.loc[pid,"Parch"] == 2:
                child_age = max(couple_age.min() - 18, 0.5)
                combined.loc[pid,"Age_cat"] = assign_Age_cat(child_age)                
                if combined.loc[pid,"Age_cat"] != "NA":        continue
        #group has couple, try to assign member as ch,par
    tmp_group = combined[
        (combined["Pclass"] == combined.loc[pid,"Pclass"]) &
        (combined["Title"] == combined.loc[pid,"Title"]) &
        (combined["Embarked"] == combined.loc[pid,"Embarked"]) &
        (combined["Age"] > 0)]
    avg_age = tmp_group["Age"].mean()        
    combined.loc[pid,"Age_cat"] = assign_Age_cat(avg_age)
    if combined.loc[pid,"Age"] == "NA":
        combined.loc[pid,"Age_cat"] = assign_ac_by_title(combined.loc[pid,"Title"])
        print(combined.loc[pid,"Age_cat"])
int_to_eng = dict({1:"First",
                  2:"Second",
                  3:"Third"})
combined["Pclass"] = combined["Pclass"].map(int_to_eng)
combined["Pclass_Sex"] =  combined["Sex"] +  combined["Pclass"]
#combined.drop("family_size",axis=1, inplace=True)
combined = create_dummies(combined,["Pclass_Sex","family_role","Age_cat"])#,"group_type"])  
combined.to_csv("combined.csv")
train = combined.loc[train.index].copy()
train["Survived"] = train_sur
test = combined.loc[test.index].copy()   
print("Done Preprocessing!\n")
family_in_trian = train[train["group_type"]=="family"]
f_gs = family_in_trian.pivot_table(index=["group_size"],values='Survived').copy()
f_gs.plot(kind="bar",label="Survival",color="Red",alpha=0.5, title="Survival rate VS family size")
 
f_gs = family_in_trian.pivot_table(index=["immediate_family"],values='Survived').copy()
f_gs.plot(kind="bar",label="Survival",color="Red",alpha=0.5, title="Survival rate VS immediate family size")
#Odd group size 1 family, investigate
#Seems necessary to differentiate small (2-4 members) families to larger ones(5-11)
#The different fate of famiies, maybe due to how many under age members there are 
#and how many adults are avaliable to look after them. 
fp_gs = family_in_trian.pivot_table(index=["Pclass"],values='Survived').copy()
fp_gs.plot(kind="bar",label="Survival",color="Red",alpha=0.5, title=" Family Survival rate in different class")
#Has 0 or more than 3 immediate family is an indicator of vulnerability, while 1-3 immediate family helps with survival 

acq_in_train =  train[train["group_type"]=="acquaintance"]
aq_gs = acq_in_train.pivot_table(index=["Pclass","group_size"],values='Survived').copy()
aq_gs.plot(kind="bar",color="Red",alpha=0.5
           , title="Survival rate of acquiantance groups of different sizes in 3 classes")
#acquaintance group have higher survival rate in first class. 
#Mid size groups fare better, but why no group larger than 3 in 3rd class?
f_alone = train[(train["group_type"] == "alone") &(train["Sex"] == "female")]
m_alone = train[(train["group_type"] == "alone") &(train["Sex"] == "male")]
f_no_imf = train[(train["group_type"] == "family") &(train["immediate_family"] == 0)]
m_no_imf = train[(train["group_type"] == "family") &(train["Sex"] == "female") &(train["immediate_family"] == 0)]
df1 = pd.crosstab(f_alone.Pclass, f_alone.Survived)
df2 = pd.crosstab(m_alone.Pclass, m_alone.Survived)
df3 = pd.crosstab(f_no_imf.Pclass, f_no_imf.Survived)
df4 = pd.crosstab(m_no_imf.Pclass, m_no_imf.Survived)
df1 = df1.style.set_caption("Female travels alone") 
display(df1)
df3 = df3.style.set_caption("Female without immediate family")
display(df3)

df2 = df2.style.set_caption("Male travels alone")
display(df2)
df4 = df4.style.set_caption("Male without immediate family")
display(df4)
bins = [-5,5,15,30,45,60,100]
figs, axes = plt.subplots(nrows=2, ncols=2 ,squeeze=True)
figs.set_figheight(15)
figs.set_figwidth(15)
female_alone_s = train[(train["group_type"] == "alone")&(train["Sex"]=="female")&(train["Survived"] == 1)].copy()
female_alone_d = train[(train["group_type"] == "alone")&(train["Sex"]=="female")&(train["Survived"] == 0)].copy()
male_alone_s = train[(train["group_type"] == "alone")&(train["Sex"]=="male")&(train["Survived"] == 1)].copy()
male_alone_d = train[(train["group_type"] == "alone")&(train["Sex"]=="male")&(train["Survived"] == 0)].copy()

lone_f_in_fg_s = train[(train["group_type"]=="family")&(train["immediate_family"]== 0)
                     &(train["Sex"]=="female")&(train["Survived"]==1)].copy()
lone_f_in_fg_d = train[(train["group_type"]=="family")&(train["immediate_family"]== 0)
                     &(train["Sex"]=="female")&(train["Survived"]==0)].copy()

lone_m_in_fg_s = train[(train["group_type"]=="family")&(train["immediate_family"]== 0)
                     &(train["Sex"]=="male")&(train["Survived"]==1)].copy()
lone_m_in_fg_d = train[(train["group_type"]=="family")&(train["immediate_family"]== 0)
                     &(train["Sex"]=="male")&(train["Survived"]==0)].copy()
female_alone_s['Age_cat'].value_counts(sort=False).plot(kind="bar",alpha=0.5,color='red',ax=axes[0,0])
female_alone_d['Age_cat'].value_counts(sort=False).plot(kind="bar",alpha=0.5,color='blue'
                ,title="Female lone traveler survival count by age",ax=axes[0,0])
#plt.legend(['Survived','Died'])

male_alone_s['Age_cat'].value_counts(sort=False).plot(kind="bar",alpha=0.5,color='red',ax=axes[0,1])
male_alone_d['Age_cat'].value_counts(sort=False).plot(kind="bar",alpha=0.5,color='blue'
                ,title="Male lone traveler survival count by age",ax=axes[0,1])
#plt.legend(['Survived','Died'])
#plt.show()

lone_f_in_fg_s['Age_cat'].value_counts(sort=False).plot(kind="bar",alpha=0.5,color='red',ax=axes[1,0])
lone_f_in_fg_d['Age_cat'].value_counts(sort=False).plot(kind="bar",alpha=0.5,color='blue'
                ,title="Female traveler with no immediate family survival count by age",ax=axes[1,0])
#plt.legend(['Survived','Died'])

lone_m_in_fg_s['Age_cat'].value_counts(sort=False).plot(kind="bar",alpha=0.5,color='red',ax=axes[1,1])
lone_m_in_fg_d['Age_cat'].value_counts(sort=False).plot(kind="bar",alpha=0.5,color='blue'
                ,title="Male traveler with no immediate family survival count by age",ax=axes[1,1])

axes[0,0].legend(['Survived','Died'])
axes[0,1].legend(['Survived','Died'])
axes[1,0].legend(['Survived','Died'])
axes[1,1].legend(['Survived','Died'])
plt.show()
#Survival rate for married couple
#Father and Mother survival rate by pclass
figs, axes = plt.subplots(nrows=2, ncols=2 ,squeeze=True)
figs.set_figheight(15)
figs.set_figwidth(15)
train["Pclass"] = train["Pclass"].map({
    "First":1,
    "Second":2,
    "Third":3
})
dad_s = train[(train["family_role"] == "Father")&(train["Survived"] == 1)].copy()
mom_s = train[(train["family_role"] == "Mother")&(train["Survived"] == 1)].copy()

dad_d = train[(train["family_role"] == "Father")&(train["Survived"] == 0)].copy()
mom_d = train[(train["family_role"] == "Mother")&(train["Survived"] == 0)].copy()

dad_s['Age_cat'].value_counts(sort=False).plot(kind="bar",alpha=0.5,color='red',ax=axes[0,0])
dad_d['Age_cat'].value_counts(sort=False).plot(kind="bar",alpha=0.5,color='blue'
                                           ,title="Fathers survival count by age",ax=axes[0,0])

mom_s['Age_cat'].value_counts(sort=False).plot(kind="bar",alpha=0.5,color='red',ax=axes[0,1])
mom_d['Age_cat'].value_counts(sort=False).plot(kind="bar",alpha=0.5,color='blue'
                ,title="Mothers survival count by age",ax=axes[0,1])

dad_s['Pclass'].value_counts(sort=False).plot(kind="bar",alpha=0.5,color='red',ax=axes[1,0])
dad_d['Pclass'].value_counts(sort=False).plot(kind="bar",alpha=0.5,color='blue'
                                          ,title="Fathers survival count by Pclass",ax=axes[1,0])

mom_d['Pclass'].value_counts(sort=False).plot(kind="bar",alpha=0.5,color='blue',ax=axes[1,1])
#no mother died in first class, if this is placed after wife_s, it won't show pclass =1 
mom_s['Pclass'].value_counts(sort=False).plot(kind="bar",alpha=0.5,color='red'
                ,title="Mothers survival count by Pclass",ax=axes[1,1])


axes[0,0].legend(['Survived','Died'])
axes[0,1].legend(['Survived','Died'])
axes[1,0].legend(['Survived','Died'])
axes[1,1].legend(['Died','Survived'])
plt.show()
train["Pclass"] = train["Pclass"].map({
    1:"First",
    2:"Second",
    3:"Third"
})
officer = train[train["Title"] == "Officer"]
royalty = train[train["Title"] == "Royalty"]
"""figs, axes = plt.subplots(nrows=2, ncols=2 ,squeeze=True)
figs.set_figheight(15)
figs.set_figwidth(15) """
op = officer.pivot_table(index=["Sex","Age_cat"],values='Survived').copy()
op.plot(kind="bar",color="Red",alpha=0.5
           , title="Survival rate of officers")
rp = royalty.pivot_table(index=["Sex","Age_cat"],values='Survived').copy()
rp.plot(kind="bar",color="Red",alpha=0.5
           , title="Survival rate of royalty")
#VIP female officer, royalty has high survival rate
#I think we have done enough work for data cleaning, let's start fitting data to model
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import GradientBoostingClassifier

def select_features(df,model):
    # Remove non-numeric columns, columns that have null values
    df = df.select_dtypes([np.number]).dropna(axis=1)
    all_X = df.drop(["Survived","Ticket_No","Fare","Age"],axis=1)
    all_y = df["Survived"]
    
    clf = eval(model)#RandomForestClassifier(random_state=1)
    selector = RFECV(clf,cv=10)
    selector.fit(all_X,all_y)
    
    best_columns = list(all_X.columns[selector.support_])
    print(model,"\nBest Columns \n"+"-"*12+"\n{}\n".format(best_columns))
    
    return best_columns

family_in_train = train[train["group_type"] == "family"].copy()
basic_cols = ['SibSp', 'Parch', 'Age_cat_NA', 'Age_cat_Infant', 'Age_cat_Child', 'Age_cat_Young Adult', 'Age_cat_Adult',
 'Age_cat_Middle Age', 'Age_cat_Senior', 'isVIP', 'isCrew','Pclass_Sex_femaleFirst', 'Pclass_Sex_femaleSecond', 'Pclass_Sex_femaleThird',
 'Pclass_Sex_maleFirst', 'Pclass_Sex_maleSecond', 'Pclass_Sex_maleThird']

family_cols = ["family_has_children","family_has_couple","family_has_senior",
               "family_role", "SibSp","Parch","OtherName","NickName","cp_id",
              "family_role_Father","family_role_Mother","family_role_Senior","family_role_Child",
               "family_role_no_role"
              ]
group_cols = ["group_type","group_size","group_id","n_cabin_mates","n_ticket_holders"]

acq_in_train= train[train["group_type"]  == "acquaintance"].copy()
acq_in_train = acq_in_train.drop(family_cols, axis=1) 

alone_in_train = train[train["group_type"]  == "alone"].copy()
alone_in_train =  alone_in_train.drop(family_cols + group_cols, axis=1)



LR_cols = [select_features(family_in_train, "LogisticRegression()"),
           select_features(acq_in_train, "LogisticRegression()"),
           select_features(alone_in_train, "LogisticRegression()")]
KNN_result =  [family_cols + group_cols + basic_cols,group_cols + basic_cols, basic_cols]
SVC_result = [family_cols + group_cols + basic_cols,group_cols + basic_cols, basic_cols]

RF_cols = [select_features(family_in_train, "RandomForestClassifier(random_state=1)"),
           select_features(acq_in_train, "RandomForestClassifier(random_state=1)"),
           select_features(alone_in_train, "RandomForestClassifier(random_state=1)")]

Adaboost_cols = [select_features(family_in_train, "AdaBoostClassifier(random_state=1)"),
                 select_features(acq_in_train, "AdaBoostClassifier(random_state=1)"),
                 select_features(alone_in_train, "AdaBoostClassifier(random_state=1)")]

GBT_cols = [select_features(family_in_train, "GradientBoostingClassifier(random_state=1)"),
            select_features(acq_in_train, "GradientBoostingClassifier(random_state=1)"),
            select_features(alone_in_train, "GradientBoostingClassifier(random_state=1)")]


Bagging_result = [family_cols + group_cols + basic_cols,group_cols + basic_cols, basic_cols]
train_by_type = [family_in_train, acq_in_train, alone_in_train]

def select_model(df,features,model_number):
    models_CV = [
        {
            "name": "LogisticRegression",
            "estimator": LogisticRegression(),
            "hyperparameters":
                {
                    "solver": ["newton-cg", "lbfgs", "liblinear"]
                }
        },
        {
            "name": "KNeighborsClassifier",
            "estimator": KNeighborsClassifier(),
            "hyperparameters":
                {
                    "n_neighbors": range(2,20),
                    "weights": ["distance", "uniform"],
                    "algorithm": ["ball_tree", "kd_tree", "brute"],
                    "p": [1,2]
                }
        },
        {
         "name":"SVC",
         "estimator": SVC(random_state=1),
         "hyperparameters":  
            {
                "C":range(1,10,2),
                "gamma":[0.1, 0.5, 1]# [0.001, 0.01, 0.1, 1],
            }
            
        },   
        {
            "name": "RandomForestClassifier",
            "estimator": RandomForestClassifier(random_state=1),
            "hyperparameters":
                {
                    "n_estimators": [4, 6, 9],
                    "criterion": ["entropy", "gini"],
                    "max_depth": [2, 5, 10],
                    "max_features": ["log2", "sqrt"],
                    "min_samples_leaf": [1, 5, 8],
                    "min_samples_split": [2, 3, 5]

                }
        },    
        {
        "name":"AdaBoost",
        "estimator": AdaBoostClassifier(random_state=1) ,
        "hyperparameters":  
            {
                "n_estimators":[50,100, 200],
                "learning_rate":[0.4,0.7,1]
                #[b/100 for b in range(10,101,10)]                
            }
            
        },    
        {
        "name":"GradienBoostTree",
        "estimator": GradientBoostingClassifier(random_state=1) ,
        "hyperparameters":  
            {
                "n_estimators":[50,100, 200],
                "learning_rate":[0.4,0.7,1],
                'max_depth':[1, 2, 3]
            }
            
        },
        {
        "name":"Bagging",
        "estimator": BaggingClassifier(random_state=1) ,
        "hyperparameters":  
            {
                "max_samples":[0.3,0.5,0.7],
                "max_features":[0.2,0.5,0.8]
            }
            
        }
    ]
    model = models_CV[model_number]
    all_X = df[features]
    all_y = df["Survived"]

    # List of dictionaries, each containing a model name,
    # it's estimator and a dict of hyperparameters
     
    print(model['name'])
    print('-'*len(model['name']))
    grid = GridSearchCV(model["estimator"],
                        param_grid=model["hyperparameters"],
                        cv=10)
    grid.fit(all_X,all_y)
    model["best_params"] = grid.best_params_
    model["best_score"] = grid.best_score_
    model["best_model"] = grid.best_estimator_
    
    print("Best Score: {}".format(model["best_score"]))
    print("Best Parameters: {}\n".format(model["best_params"]))

    return model

LR_result = [[],[], []]
KNN_result =  [[],[], []]
SVC_result = [[],[], []]
RF_result =  [[],[], []]#np.zeros([3,1])
Adaboost_result =  [[],[], []]
GBT_result =  [[],[], []]
Bagging_result =  [[],[], []]

for group_type in range(3):  
    LR_result[group_type] = select_model(train_by_type[group_type], LR_cols[group_type], 0)
    
    KNN_result[group_type] = select_model(train_by_type[group_type], LR_cols[group_type], 1)
    SVC_result[group_type] = select_model(train_by_type[group_type], LR_cols[group_type], 2)
    RF_result[group_type] = select_model(train_by_type[group_type], RF_cols[group_type],3)
    Adaboost_result[group_type] = select_model(train_by_type[group_type], Adaboost_cols[group_type], 4)
    GBT_result[group_type] = select_model(train_by_type[group_type], GBT_cols[group_type], 5)
    Bagging_result[group_type] = select_model(train_by_type[group_type], GBT_cols[group_type],6)
    

scores = {
    "Algorithms":["family","acquiantance","alone"],
    "LR":
    [
    LR_result[0]["best_score"], 
    LR_result[1]["best_score"], 
    LR_result[2]["best_score"]
          ],
    "KNN":
    [
        KNN_result[0]["best_score"], 
        KNN_result[1]["best_score"], 
        KNN_result[2]["best_score"]
    ],
    "SVC":
    [
        SVC_result[0]["best_score"], 
        SVC_result[1]["best_score"], 
        SVC_result[2]["best_score"]
    ],
    "RF":
    [
        RF_result[0]["best_score"],  
        RF_result[1]["best_score"], 
        RF_result[2]["best_score"]
    ] , 
    "Adaboost":
    [
        Adaboost_result[0]["best_score"], 
        Adaboost_result[1]["best_score"], 
        Adaboost_result[2]["best_score"]
    ], 
    "GBT":
    [
        GBT_result[0]["best_score"], 
        GBT_result[1]["best_score"], 
        GBT_result[2]["best_score"]
    ],
    "Bagging":
    [
        Bagging_result[0]["best_score"], 
        Bagging_result[1]["best_score"],
        Bagging_result[2]["best_score"]
    ]
}

 
n_algs = 4
score_df =pd.DataFrame(scores).set_index("Algorithms").T
display(score_df)
print("Best models for passengers with family:\n", score_df.nlargest(n_algs,"family")["family"],"\n")
print("Best models for passengers with acquiantance:\n", score_df.nlargest(n_algs,"acquiantance")["acquiantance"],"\n")
print("Best models for passengers alone:\n", score_df.nlargest(n_algs,"alone")["alone"],"\n")
fml_algs = list(score_df.nlargest(n_algs,"family").index)
acq_algs = list(score_df.nlargest(n_algs,"acquiantance").index)
alo_algs = list(score_df.nlargest(n_algs,"alone").index)
models = [[
    LR_result[0]["best_model"], 
    LR_result[1]["best_model"], 
    LR_result[2]["best_model"]
          ],
    [
        KNN_result[0]["best_model"], 
        KNN_result[1]["best_model"], 
        KNN_result[2]["best_model"]
    ],
    [
        SVC_result[0]["best_model"], 
        SVC_result[1]["best_model"], 
        SVC_result[2]["best_model"]
    ],
    [
        RF_result[0]["best_model"],  
        RF_result[1]["best_model"], 
        RF_result[2]["best_model"]
    ] , 
    [
        Adaboost_result[0]["best_model"], 
        Adaboost_result[1]["best_model"], 
        Adaboost_result[2]["best_model"]
    ], 
    [
        GBT_result[0]["best_model"], 
        GBT_result[1]["best_model"], 
        GBT_result[2]["best_model"]
    ],
    [
        Bagging_result[0]["best_model"], 
        Bagging_result[1]["best_model"],
        Bagging_result[2]["best_model"]
    ]
]
 
cols = [
    LR_cols,
    LR_cols,
    LR_cols,
    RF_cols,
    Adaboost_cols,
    GBT_cols,
    GBT_cols
]

holdout_ids = [
    test[test["group_type"] == "family"].index, 
    test[test["group_type"] == "acquaintance"].index, 
    test[test["group_type"] == "alone"].index
]

total_result= pd.DataFrame()
from collections import OrderedDict 
model_names = ["LR","KNN","SVC","RF","Adaboost","GBT","Bagging"]
for model_id in range(len(models)):
    
    model_result = pd.DataFrame()
    for group_type in range(len(holdout_ids)):      
        group_ids = holdout_ids[group_type]
        holdout_data = test.loc[group_ids, cols[model_id][group_type]]
        predictions = models[model_id][group_type].predict(holdout_data)
        group_result =pd.DataFrame({ "PassengerId": group_ids,
                        model_names[model_id]: predictions}).set_index("PassengerId")
        #
        model_result = model_result.append([group_result])
    #print(model_result)
    #model_result = pd.DataFrame(OrderedDict(sorted(model_result.items())))
    total_result = pd.concat([total_result, model_result.sort_index()], axis=1, join_axes=[model_result.index])
total_result.describe()
### total_result["Survived"] = 0
family_in_test = test[test["group_type"] == "family"].index#pd.Index(list(test[test["group_type"] == "family"].index ))
acq_in_test = test[test["group_type"] == "acquaintance"].index
alone_in_test = test[test["group_type"] == "alone"].index

"""
#total_result.loc[family_in_test, "Survived"]  =  total_result.loc[family_in_test,  "RF"] 
#+ total_result.loc[family_in_test,"GBT"] + total_result.loc[family_in_test]
total_result.loc[family_in_test, "Survived"]= (total_result.loc[family_in_test,  ["RF","GBT","Bagging","Adaboost"]].sum(axis=1) == 3).astype(int)
total_result.loc[acq_in_test, "Survived"] = (total_result.loc[acq_in_test,  ["LR","Bagging","GBT"]].sum(axis=1) == 3).astype(int)
total_result.loc[alone_in_test, "Survived"] = (total_result.loc[alone_in_test,  ["KNN","GBT","RF"]].sum(axis=1) == 3).astype(int)
"""
total_result.loc[family_in_test, "Survived"]= (total_result.loc[family_in_test,  fml_algs].sum(axis=1) >= n_algs -1).astype(int)
total_result.loc[acq_in_test, "Survived"] = (total_result.loc[acq_in_test,  acq_algs].sum(axis=1)  >= n_algs -1).astype(int)
total_result.loc[alone_in_test, "Survived"] = (total_result.loc[alone_in_test,alo_algs].sum(axis=1) >= n_algs - 1).astype(int)

submission = total_result.sort_index().reset_index().copy()
submission = submission[["PassengerId","Survived"]].astype(int)
submission.to_csv("Kaggle_final.csv",index=False)