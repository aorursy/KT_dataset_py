import pandas as pd
import numpy as np
import plotly.graph_objs as go
from plotly.offline import iplot,init_notebook_mode 
from plotly.tools import make_subplots
from lightgbm.sklearn import LGBMRegressor
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import StratifiedKFold, train_test_split
pd.options.display.max_columns = 100
init_notebook_mode(connected=True)
train_data  = pd.read_csv("../input/train/train.csv")
test_data = pd.read_csv("../input/test/test.csv")
breed_labels = pd.read_csv("../input/breed_labels.csv")
breed_names = {k:v for k, v in zip(list(breed_labels['BreedID']), list(breed_labels['BreedName']))}
breed_types = {k:v for k, v in zip(list(breed_labels['BreedID']), list(breed_labels['Type']))}
breed_names[0] = "NA"
breed_types[0] = "NA"
color_labels  = pd.read_csv("../input/color_labels.csv")
color_names = {k:v for k, v in zip(list(color_labels['ColorID']), list(color_labels['ColorName']))}
color_names[0] = "NA"
state_labels = pd.read_csv("../input/state_labels.csv")
state_names = {k:v for k, v in zip(list(state_labels['StateID']), list(state_labels['StateName']))}
train_data.columns
train_data["Breed1"] = train_data["Breed1"].apply(lambda x: breed_names[x])
train_data["Breed2"] = train_data["Breed2"].apply(lambda x: breed_names[x])
test_data["Breed1"] = test_data["Breed1"].apply(lambda x: breed_names[x])
test_data["Breed2"] = test_data["Breed2"].apply(lambda x: breed_names[x])
train_data["Color1"] = train_data["Color1"].apply(lambda x: color_names[x])
train_data["Color2"] = train_data["Color2"].apply(lambda x: color_names[x])
train_data["Color3"] = train_data["Color3"].apply(lambda x: color_names[x])
test_data["Color1"] = test_data["Color1"].apply(lambda x: color_names[x])
test_data["Color2"] = test_data["Color2"].apply(lambda x: color_names[x])
test_data["Color3"] = test_data["Color3"].apply(lambda x: color_names[x])
train_data["State"] = train_data["State"].apply(lambda x: state_names[x])
test_data["State"] = test_data["State"].apply(lambda x: state_names[x])
train_data["Type"] = train_data["Type"].apply(lambda x: "Dog" if x==1 else "Cat")
test_data["Type"] = test_data["Type"].apply(lambda x: "Dog" if x==1 else "Cat")
yes_no_dict = {1: "Yes", 2: "No", 3: "Not Sure"}
train_data["Vaccinated"] = train_data["Vaccinated"].apply(lambda x: yes_no_dict[x])
train_data["Dewormed"] = train_data["Dewormed"].apply(lambda x: yes_no_dict[x])
train_data["Sterilized"] = train_data["Sterilized"].apply(lambda x: yes_no_dict[x])
test_data["Vaccinated"] = test_data["Vaccinated"].apply(lambda x: yes_no_dict[x])
test_data["Dewormed"] = test_data["Dewormed"].apply(lambda x: yes_no_dict[x])
test_data["Sterilized"] = test_data["Sterilized"].apply(lambda x: yes_no_dict[x])
gender_dict = {1: "Male", 2: "Female", 3: "Mixed"}
train_data["Gender"] = train_data["Gender"].apply(lambda x: gender_dict[x])
test_data["Gender"] = test_data["Gender"].apply(lambda x: gender_dict[x])
health_dict = {1 : "Healthy", 2 : "Minor Injury", 3 : "Serious Injury", 0 : "Not Specified"}
train_data["Health"] = train_data["Health"].apply(lambda x: health_dict[x])
test_data["Health"] = test_data["Health"].apply(lambda x: health_dict[x])
size_dict = {1 : "Small", 2 : "Medium", 3 : "Large", 4 : "Extra Large", 0 : "Not Specified"}
train_data["MaturitySize"] = train_data["MaturitySize"].apply(lambda x: size_dict[x])
test_data["MaturitySize"] = test_data["MaturitySize"].apply(lambda x: size_dict[x])
fur_dict =  {1 : "Short", 2 : "Medium", 3 : "Long", 0 : "Not Specified"}
train_data["FurLength"] = train_data["FurLength"].apply(lambda x: fur_dict[x])
test_data["FurLength"] = test_data["FurLength"].apply(lambda x: fur_dict[x])
pd.DataFrame({"Columns Name": list(train_data.columns),
              "Number of unique values (train)": [train_data[c].unique().shape[0] for c in train_data.columns], 
              "Number of unique values (test)": 
              [0 if c=="AdoptionSpeed" else test_data[c].unique().shape[0] for c in train_data.columns]})
counts = dict(train_data["Type"].value_counts())
trace_train = go.Bar(x=list(counts.keys()), y=list(counts.values()), name="Training Data")

counts = dict(test_data["Type"].value_counts())
trace_test = go.Bar(x=list(counts.keys()), y=list(counts.values()), name="Testing Data")

fig  = make_subplots(rows=1, cols=2)

fig.append_trace(trace_train, 1, 1)
fig.append_trace(trace_test, 1, 2)

fig["layout"].update(title="Number of Pets by Type")
iplot(fig)
trace_train = go.Histogram(x=list(train_data.loc[train_data["Age"] < train_data["Age"].mean()+3*train_data["Age"].std(), "Age"]), 
                           opacity=0.75,
                           xbins=dict(size=1),
                           name="Training Data")
trace_test = go.Histogram(x=list(test_data.loc[test_data["Age"] < test_data["Age"].mean()+3*test_data["Age"].std(), "Age"]), 
                          opacity=0.75,
                          xbins=dict(size=1),
                          name="Testing Data")

layout = go.Layout(title="Number of pets by Age", barmode="overlay", xaxis=dict(title="Age (Months)"))

fig = go.Figure(data=[trace_train, trace_test], layout=layout)

iplot(fig)
VALUE_THRESHOLD = 10
counts = dict(train_data["Breed1"].value_counts())
counts = {k[:15]:v for k,v in counts.items() if v>=3*VALUE_THRESHOLD}
trace_train = go.Bar(x=list(counts.keys()), y=list(counts.values()), name="Training Data")

counts = dict(test_data["Breed1"].value_counts())
counts = {k[:15]:v for k,v in counts.items() if v>=VALUE_THRESHOLD}
trace_test = go.Bar(x=list(counts.keys()), y=list(counts.values()), name="Testing Data")

fig  = make_subplots(rows=2, cols=1, vertical_spacing = 0.2)

fig.append_trace(trace_train, 1, 1)
fig.append_trace(trace_test, 2, 1)

fig["layout"].update(title="Number of Pets by Breed 1", height=800, margin=go.layout.Margin(b=150))
iplot(fig)
counts = dict(train_data["Breed2"].value_counts())
counts = {k[:15]:v for k,v in counts.items() if v>=3*VALUE_THRESHOLD}
trace_train = go.Bar(x=list(counts.keys()), y=list(counts.values()), name="Training Data")

counts = dict(test_data["Breed2"].value_counts())
counts = {k[:15]:v for k,v in counts.items() if v>=VALUE_THRESHOLD}
trace_test = go.Bar(x=list(counts.keys()), y=list(counts.values()), name="Testing Data")

fig  = make_subplots(rows=2, cols=1)

fig.append_trace(trace_train, 1, 1)
fig.append_trace(trace_test, 2, 1)

fig["layout"].update(title="Number of Pets by Breed 2", height=800)
iplot(fig)
counts = dict(train_data["Color1"].value_counts())
trace_train = go.Bar(x=list(counts.keys()), y=list(counts.values()), name="Training Data")

counts = dict(test_data["Color1"].value_counts())
trace_test = go.Bar(x=list(counts.keys()), y=list(counts.values()), name="Testing Data")

fig  = make_subplots(rows=2, cols=1)

fig.append_trace(trace_train, 1, 1)
fig.append_trace(trace_test, 2, 1)

fig["layout"].update(title="Number of Pets by color 1", height=800)
iplot(fig)
counts = dict(train_data["Color2"].value_counts())
trace_train = go.Bar(x=list(counts.keys()), y=list(counts.values()), name="Training Data")

counts = dict(test_data["Color2"].value_counts())
trace_test = go.Bar(x=list(counts.keys()), y=list(counts.values()), name="Testing Data")

fig  = make_subplots(rows=2, cols=1)

fig.append_trace(trace_train, 1, 1)
fig.append_trace(trace_test, 2, 1)

fig["layout"].update(title="Number of Pets by color 2", height=800)
iplot(fig)
counts = dict(train_data["Color3"].value_counts())
trace_train = go.Bar(x=list(counts.keys()), y=list(counts.values()), name="Training Data")

counts = dict(test_data["Color3"].value_counts())
trace_test = go.Bar(x=list(counts.keys()), y=list(counts.values()), name="Testing Data")

fig  = make_subplots(rows=2, cols=1)

fig.append_trace(trace_train, 1, 1)
fig.append_trace(trace_test, 2, 1)

fig["layout"].update(title="Number of Pets by color 3", height=800)
iplot(fig)
counts = dict(train_data["MaturitySize"].value_counts())
trace_train = go.Bar(x=list(counts.keys()), y=list(counts.values()), name="Training Data")

counts = dict(test_data["MaturitySize"].value_counts())
trace_test = go.Bar(x=list(counts.keys()), y=list(counts.values()), name="Testing Data")

fig  = make_subplots(rows=1, cols=2)

fig.append_trace(trace_train, 1, 1)
fig.append_trace(trace_test, 1, 2)

fig["layout"].update(title="Number of Pets by Size", height=600)
iplot(fig)
counts = dict(train_data["FurLength"].value_counts())
trace_train = go.Bar(x=list(counts.keys()), y=list(counts.values()), name="Training Data")

counts = dict(test_data["FurLength"].value_counts())
trace_test = go.Bar(x=list(counts.keys()), y=list(counts.values()), name="Testing Data")

fig  = make_subplots(rows=1, cols=2)

fig.append_trace(trace_train, 1, 1)
fig.append_trace(trace_test, 1, 2)

fig["layout"].update(title="Number of Pets by Fur Length", height=600)
iplot(fig)
counts = dict(train_data["Vaccinated"].value_counts())
trace_train = go.Bar(x=list(counts.keys()), y=list(counts.values()), name="Training Data")

counts = dict(test_data["Vaccinated"].value_counts())
trace_test = go.Bar(x=list(counts.keys()), y=list(counts.values()), name="Testing Data")

fig  = make_subplots(rows=1, cols=2)

fig.append_trace(trace_train, 1, 1)
fig.append_trace(trace_test, 1, 2)

fig["layout"].update(title="Number of Pets by Vaccination", height=600)
iplot(fig)
counts = dict(train_data["Dewormed"].value_counts())
trace_train = go.Bar(x=list(counts.keys()), y=list(counts.values()), name="Training Data")

counts = dict(test_data["Dewormed"].value_counts())
trace_test = go.Bar(x=list(counts.keys()), y=list(counts.values()), name="Testing Data")

fig  = make_subplots(rows=1, cols=2)

fig.append_trace(trace_train, 1, 1)
fig.append_trace(trace_test, 1, 2)

fig["layout"].update(title="Number of Pets by Deworming", height=600)
iplot(fig)
counts = dict(train_data["Health"].value_counts())
trace_train = go.Bar(x=list(counts.keys()), y=list(counts.values()), name="Training Data")

counts = dict(test_data["Health"].value_counts())
trace_test = go.Bar(x=list(counts.keys()), y=list(counts.values()), name="Testing Data")

fig  = make_subplots(rows=1, cols=2)

fig.append_trace(trace_train, 1, 1)
fig.append_trace(trace_test, 1, 2)

fig["layout"].update(title="Number of Pets by Health", height=600)
iplot(fig)
counts = dict(train_data["State"].value_counts())
trace_train = go.Bar(x=list(counts.keys()), y=list(counts.values()), name="Training Data")

counts = dict(test_data["State"].value_counts())
trace_test = go.Bar(x=list(counts.keys()), y=list(counts.values()), name="Testing Data")

fig  = make_subplots(rows=2, cols=1)

fig.append_trace(trace_train, 1, 1)
fig.append_trace(trace_test, 2, 1)

fig["layout"].update(title="Number of Pets by State", height=800)
iplot(fig)
counts = {"No Fees (Zero)": (train_data["Fee"]==0).sum(), "With Fees (Non-Zero)":  (train_data["Fee"]!=0).sum()}
trace_train = go.Bar(x=list(counts.keys()), y=list(counts.values()), name="Training Data")

counts = {"No Fees (Zero)": (test_data["Fee"]==0).sum(), "With Fees (Non-Zero)":  (test_data["Fee"]!=0).sum()}

trace_test = go.Bar(x=list(counts.keys()), y=list(counts.values()), name="Testing Data")

fig  = make_subplots(rows=1, cols=2)

fig.append_trace(trace_train, 1, 1)
fig.append_trace(trace_test, 1, 2)

fig["layout"].update(title="Number of Pets by Fees", height=600)
iplot(fig)
trace_train = go.Histogram(x=list(train_data.loc[(train_data["Fee"] < (train_data["Fee"].mean() + 3*train_data["Fee"].std())) 
                                    & (train_data["Fee"] > 0) , "Fee"]), 
                           xbins=dict(size=10),
                           opacity=0.75, 
                           name="Training Data")
trace_test = go.Histogram(x=list(test_data.loc[(test_data["Fee"] < (test_data["Fee"].mean() + 3*test_data["Fee"].std())) 
                                    & (test_data["Fee"] > 0) , "Fee"]),
                          xbins=dict(size=10),
                          opacity=0.75, 
                          name="Testing Data")

layout = go.Layout(title="Number of pets by Fee (Non-Zero)", barmode="overlay", xaxis=dict(title="Fees"))

fig = go.Figure(data=[trace_train, trace_test], layout=layout)

iplot(fig)
counts = dict(train_data["VideoAmt"].value_counts())
trace_train = go.Bar(x=list(counts.keys()), y=list(counts.values()), name="Training Data")

counts = dict(test_data["VideoAmt"].value_counts())
trace_test = go.Bar(x=list(counts.keys()), y=list(counts.values()), name="Testing Data")

fig  = make_subplots(rows=2, cols=1)

fig.append_trace(trace_train, 1, 1)
fig.append_trace(trace_test, 2, 1)

fig["layout"].update(title="Number of Pets by Videos", height=800)
iplot(fig)
counts = dict(train_data["PhotoAmt"].value_counts())
trace_train = go.Bar(x=list(counts.keys()), 
                           y=list(counts.values()), 
                           name="Training Data")

counts = dict(test_data["PhotoAmt"].value_counts())
trace_test = go.Bar(x=list(counts.keys()), 
                          y=list(counts.values()), 
                          name="Testing Data")

fig  = make_subplots(rows=2, cols=1)

fig.append_trace(trace_train, 1, 1)
fig.append_trace(trace_test, 2, 1)

fig["layout"].update(title="Number of Pets by Photos", height=800)
iplot(fig)
train_data.loc[train_data["Description"].isnull(), "Description"] = ""
train_data["Descrpition_Length"]  = train_data["Description"].apply(lambda s: len(s))
test_data.loc[test_data["Description"].isnull(), "Description"] = ""
test_data["Descrpition_Length"]  = test_data["Description"].apply(lambda s: len(s))
trace_train = go.Histogram(x=list(train_data.loc[train_data["Descrpition_Length"] < train_data["Descrpition_Length"].mean()+3*train_data["Descrpition_Length"].std(), "Descrpition_Length"]), 
                           opacity=0.75,
                           xbins=dict(size=50),
                           name="Training Data")
trace_test = go.Histogram(x=list(test_data.loc[test_data["Descrpition_Length"] < test_data["Descrpition_Length"].mean()+3*test_data["Descrpition_Length"].std(), "Descrpition_Length"]), 
                          opacity=0.75,
                          xbins=dict(size=50),
                          name="Testing Data")

layout = go.Layout(title="Number of pets by Decription Length", barmode="overlay", xaxis=dict(title="Length in Characters"))

fig = go.Figure(data=[trace_train, trace_test], layout=layout)

iplot(fig)
counts = dict(train_data["AdoptionSpeed"].value_counts())
trace_train = go.Bar(x=list(counts.keys()), y=list(counts.values()), name="Training Data")

layout = go.Layout(title="Number of Pets by Adoption Speed")

fig = go.Figure(data=[trace_train], layout=layout)

iplot(fig)
train_data = pd.read_csv("../input/train/train.csv")
train_data
FOLDS = 5
catagorical_features = ["Type", "Breed1", "Breed2", "Color1", "Color2", "Color3", "MaturitySize", "FurLength", "Vaccinated", "Dewormed", "Sterilized", "Health", "State"]
non_features  = ["Name", "Description", "PetID", "RescuerID"]
label = "AdoptionSpeed"
kfold = StratifiedKFold(n_splits=5, random_state=22)
train_data["DescriptionLength"] = train_data["Description"].apply(lambda s: len(s) if isinstance(s, str) else 0)
features = [c for c in train_data.columns if c not in set(non_features+[label])]; features
def clip(x):
    if x < 0:
        return 0
    if x > 4:
        return 4
    return x
vclip = np.vectorize(clip)
models = []
predictions = []
ids = []
result= []
for train_indices, val_indices in kfold.split(X=np.arange(train_data.shape[0]), y=train_data["AdoptionSpeed"]):
    ids.append(train_data.iloc[train_indices]["PetID"])
    model = LGBMRegressor(colsample_bytree=0.9, subsample=0.9, n_estimators=1000, random_state=22, silent=True)
    model.fit(X=train_data.iloc[train_indices][features], y=train_data.iloc[train_indices][label], categorical_feature=catagorical_features, 
             eval_set=(train_data.iloc[val_indices][features], train_data.iloc[val_indices][label]), early_stopping_rounds=10)
    pred = np.round(model.predict(train_data.iloc[val_indices][features]))
    pred = vclip(pred)
    kappa = cohen_kappa_score(y1=pred, y2=np.array(train_data.iloc[val_indices][label]), weights="quadratic")
    print("Kappa score is ", kappa)
    predictions.append(pred)
    result.append(kappa)
    models.append(model)
np.mean(kappa)
trace = go.Bar(x=features, y=list(np.mean([models[i].feature_importances_ for i in range(FOLDS)], axis=0)), name="Training Data")

layout = go.Layout(title="Feature Importance")

fig = go.Figure(data=[trace], layout=layout)

iplot(fig)
iterations = np.mean([models[i].best_iteration_ for i in range(FOLDS)]); [models[i].best_iteration_ for i in range(FOLDS)]
test_data = pd.read_csv("../input/test/test.csv")
test_data["DescriptionLength"] = test_data["Description"].apply(lambda s: len(s) if isinstance(s, str) else 0)
predictions  = []
for i in range(FOLDS):
    pred = np.round(models[i].predict(test_data[features]))
    pred = vclip(pred)
    predictions.append(pred)
predictions = np.array(predictions, dtype=np.int64)
predictions[:, 0]
final_pred = []
for i in range(predictions.shape[1]):
    final_pred.append(np.argmax(np.bincount(predictions[:, i])))
submission = pd.DataFrame({"PetID": list(test_data["PetID"]), "AdoptionSpeed": final_pred})
submission.to_csv("submission.csv", index=None)