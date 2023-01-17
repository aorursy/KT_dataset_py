%reload_ext autoreload
%autoreload 2
%matplotlib inline
from fastai.tabular import *
os.makedirs("data/pet_finder", exist_ok=True)
path = Path('data/pet_finder'); path
train_df = pd.read_csv('../input/train/train.csv')
test_df = pd.read_csv('../input/test/test.csv')
dep_var = 'AdoptionSpeed'
cat_names = ['Type', 'Name', 'Breed1', 'Breed2', 'Gender', 'Color1', 'Color2', 'Color3', 'MaturitySize',
             'FurLength', 'State', 'RescuerID', 'PetID']
cont_names = ['Age', 'Fee', 'VideoAmt', 'PhotoAmt']
procs = [FillMissing, Categorify, Normalize]

# Not including 'Description'
test = TabularList.from_df(test_df, path=path, cat_names=cat_names, cont_names=cont_names)
data = (TabularList.from_df(train_df, path=path, cat_names=cat_names, cont_names=cont_names, procs=procs)
        .random_split_by_pct(0.2, seed=42)
        .label_from_df(cols=dep_var)
        .add_test(test)
        .databunch()
)
data.show_batch(rows=10)
learn = tabular_learner(data, layers=[200,100], metrics=accuracy)
learn.lr_find()
learn.recorder.plot()
learn.fit(1, 1e-2)
test_preds = np.argmax(learn.get_preds(DatasetType.Test)[0],axis=1)
test_preds
sub_df = pd.DataFrame(data={'PetID': pd.read_csv('../input/test/test.csv')['PetID'],
                            'AdoptionSpeed': test_preds})
sub_df.head()
sub_df.to_csv('submission.csv', index=False)