import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import datetime
import os
#print(os.listdir("../input"))
from ipywidgets import FloatProgress,FloatText
from IPython.display import display

import time
import pdb

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from trackml.dataset import load_event
from trackml.randomize import shuffle_hits
from trackml.score import score_event
from trackml.dataset import load_dataset
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from itertools import product
import gc
import cProfile
from tqdm import tqdm

%matplotlib inline
#make wider graphs
sns.set(rc={'figure.figsize':(12,5)})
plt.figure(figsize=(12,5))
#path = '../input/train_1/'
path='../input/unzippedcernsample/'
label_shift_M=1000000
def create_one_event_submission(event_id, hits, labels):
    sub_data = np.column_stack(([event_id]*len(hits), hits.hit_id.values, labels))
    submission = pd.DataFrame(data=sub_data, columns=["event_id", "hit_id", "track_id"]).astype(int)
    return submission
def hit_score(res,truth):
    tt=res.merge(truth[['hit_id','particle_id','weight']],on='hit_id',how='left')
    un,inv,count = np.unique(tt['track_id'],return_inverse=True, return_counts=True)
    tt['track_len']=count[inv]
    un,inv,count = np.unique(tt['particle_id'],return_inverse=True, return_counts=True)
    tt['real_track_len']=count[inv]
    gp=tt.groupby('track_id')
    gp=gp['particle_id'].value_counts().rename('par_freq').reset_index()
    tt=tt.merge(gp,on=['track_id','particle_id'],how='left')
    gp=gp.groupby('track_id').head(1)
    gp=gp.rename(index=str, columns={'particle_id': 'common_particle_id'})
    tt = tt.merge(gp.drop(['par_freq'],axis=1),on='track_id',how='left')
    tt['to_score']=(2*tt['par_freq']>tt['track_len']) & (2*tt['par_freq']>tt['real_track_len'])
    tt['score']=tt['weight']*tt['to_score']
    return tt
def calc_features(hits,hipos,phik,double_sided=False):
    
    if not 'rr' in list(hits.columns):
        hits['theta_']=np.arctan2(hits.y,hits.x)
        hits['rr']=np.sqrt(np.square(hits.x)+np.square(hits.y))
    ktrr=hits.rr*hipos.kt
    hits['dtheta']=np.where((np.abs(ktrr)<1),np.arcsin(ktrr,where=(np.abs(ktrr)<1) ),ktrr)
    hits['theta'] = hits.theta_+hits.dtheta
    hits['phi'] = np.arctan2((hits.z-hipos.z0) ,phik*hits.dtheta/hipos.kt)*2.0/np.pi
    hits['sint']=np.sin(hits['theta'])
    hits['cost']=np.cos(hits['theta'])
    hits['fault']=(np.abs(ktrr)>1).astype('int')
    if double_sided:
        hits['phi2'] = np.arctan2((hits.z-hipos.z0) ,phik*(np.pi-hits.dtheta)/hipos.kt)*2.0/np.pi
        hits['theta2'] = hits.theta_+np.pi-hits.dtheta
        hits['sint2']=np.sin(hits['theta2'])
        hits['cost2']=np.cos(hits['theta2'])
    return hits
def tag_bins(cat):
    un,inv,count = np.unique(cat,return_inverse=True, return_counts=True)
    bin_tag=inv
    bin_count=count[inv]
    return bin_tag,bin_count
def sparse_bin(features,bin_num,randomize=True,fault=None):
    err=np.random.rand(features.shape[1])*randomize
    cat=np.zeros(features.shape[0]).astype('int64')
    factore=1
    for i,feature in enumerate(features.columns):
        cat=cat+(features[feature]*bin_num._asdict()[feature]+err[i]).astype('int64')*factore
        factore=factore*(2*bin_num._asdict()[feature]+1)
    if not fault is None:
        cat=cat+(factore*features.index*fault).astype('int64')
    return tag_bins(cat)
    
def clustering(hits,stds,filters,phik=1.0,nu=500,weights=None,res=None,truth=None,history=None,pre_test_points=None):
    start = time.time()
    rest = hits.copy()
    if weights is None:
        weights={'phi':1, 'theta':0.15}
    calc_score = not truth is None
    if not history is None:
        hist_list=[]
    
    if calc_score:
        rest = rest.merge(truth[['hit_id','particle_id','weight']],on='hit_id',how='left')
        dum,rest['particle_track_len']=tag_bins(rest['particle_id'])
        score = 0 
        hit_num=0
        total_num=0
        frs=FloatText(value=0, description="full score:")
        display(frs)
        fs=FloatText(value=0, description="score:")
        display(fs)
        fss=FloatText(value=0, description="s rate:")
        display(fss)
        fsd=FloatText(value=0, description="add score:")
        display(fsd)
    
    ft = FloatText(value=rest.shape[0], description="Rest size:")
    display(ft)
    fg = FloatText(value=rest.shape[0], description="Group size:")
    display(fg)
    fgss = FloatText(description="filter:")
    display(fgss)
    
    if res is None:
        rest['track_len']=1
        rest['track_id']=-rest.index
        rest['kt']=1e-6
        rest['z0']=0
    else:
        rest=rest.merge(res[['hit_id','track_id','kt','z0']],on='hit_id',how='left')
        dum,rest['track_len']=tag_bins(rest['track_id'])

    res_list=[]
    rest['sensor']=rest.volume_id+rest.layer_id*100+100000*rest.module_id
    rest['layers']=rest.volume_id+rest.layer_id*100
    if pre_test_points is None:
        maxprog= filters.npoints.sum()
    else:
        maxprog = filters.shape[0]*pre_test_points.shape[0]
    pbar = tqdm(total=maxprog,mininterval=5.0)
    rest['pre_track_id']=rest['track_id']
    p=-1
    feature_cols=['theta','sint','cost','phi','rr','theta_','dtheta','fault']
    for filt in filters.itertuples():
        if pre_test_points is None:
            test_points=pd.DataFrame()
            for col in stds:
                test_points[col] = np.random.normal(scale=stds[col],size=filt.npoints)
        else:
            test_points=pre_test_points.sample(frac=filt.npoints).reset_index(drop=True)
        
        for row in test_points.itertuples():
            p=p+1
            pbar.update()
            calc_features(rest,row,phik)
            rest['new_track_id'],rest['new_track_len']=sparse_bin(rest[['phi','sint','cost']],filt,fault=rest.fault)
            rest['new_track_id']=rest['new_track_id']+(p+1)*label_shift_M
            better = (rest.new_track_len>rest.track_len) & (rest.new_track_len<19)
            rest['new_track_id']=rest['new_track_id'].where(better,rest.track_id)
            dum,rest['new_track_len']=tag_bins(rest['new_track_id'])
            better = (rest.new_track_len>rest.track_len) & (rest.new_track_len<19)
            rest['track_id']=rest['track_id'].where(~better,rest['new_track_id']) 
            rest['track_len']=rest['track_len'].where(~better,rest['new_track_len'])
            rest['kt']=rest['kt'].where(~better,row.kt)
            rest['z0']=rest['z0'].where(~better,row.z0)
            
            if (((row.Index+1)%nu == 0) or (row.Index + 1 == test_points.shape[0])):
                dum,rest['track_len']=tag_bins(rest['track_id'])
                calc_features(rest,rest[['kt','z0']],phik)
                gp = rest.groupby(['track_id']).agg({'phi': np.mean , 
                    'sint':np.mean, 'cost':np.mean}).rename(columns={ 'phi': 'mean_phi', 
                                'sint':'mean_sint', 'cost':'mean_cost'}).reset_index()
                cols_to_drop = rest.columns.intersection(gp.columns).drop('track_id')
                rest = rest.drop(cols_to_drop,axis=1).reset_index().merge(gp,on=['track_id'],how = 'left').set_index('index')
                rest['dist'] = weights['theta']*np.square(rest.sint-rest.mean_sint)+ weights['theta']*np.square(rest.cost-rest.mean_cost)+ weights['phi']*np.square(rest.phi-rest.mean_phi)
                rest=rest.sort_values('dist')
                rest['closest']=rest.groupby(['track_id','sensor'])['dist'].cumcount()
                rest['closest2']=rest.groupby(['track_id','layers'])['dist'].cumcount()
                select = (rest['closest']!=0) | (rest['closest2']>2)  
                rest['track_id']=rest['track_id'].where(~select,rest['pre_track_id'])
                dum,rest['track_len']=tag_bins(rest['track_id'])
                fgss.value=filt.phi
                fg.value=filt.min_group
                ft.value = rest[rest.track_len<=filt.min_group].shape[0]

                select = (rest['track_len']>filt.min_group)
                
                #The next lines are just for printing
                if calc_score:
                    tm=rest[select]                   
                    gp = tm.groupby(['track_id','particle_id'])['hit_id'].count().rename('par_count').reset_index()
                    tm=tm.merge(gp,on=['track_id','particle_id'],how='left')
                    gp = rest.groupby(['track_id','particle_id'])['hit_id'].count().rename('par_count').reset_index()
                    rs=rest.merge(gp,on=['track_id','particle_id'],how='left')
                    to_full_score=(rs.weight*((rs.par_count*2>rs.track_len) & (rs.par_count*2>rs.particle_track_len)))
                    frs.value=to_full_score.sum()+fs.value
                    to_score=(tm.weight*((tm.par_count*2>tm.track_len) & (tm.par_count*2>tm.particle_track_len)))
                    hit_num=hit_num+(to_score>0).sum()
                    total_num=total_num+tm.weight.sum()
                    fs.value=fs.value+to_score.sum()
                    fss.value=fs.value/total_num
                    fsd.value=to_score.sum()
                    gp = rest.groupby(['track_id','particle_id'])['hit_id'].count().rename('par_count').reset_index()
                    rs=rest.merge(gp,on=['track_id','particle_id'],how='left')
                    to_full_score=(rs.weight*((rs.par_count*2>rs.track_len) & (rs.par_count*2>rs.particle_track_len)))
                    frs.value=to_full_score.sum()+fs.value-to_score.sum()
                    if not history is None:
                        hist_list.append(pd.DataFrame({'P':p,'ftheta':filt.phi,'added_score':to_score.sum(),'min_group':filt.min_group,
                                                    'full_score':frs.value,'score':fsd.value,'correct':fss.value,
                                                    'clustered':tm.shape[0],'left':rest.shape[0]-tm.shape[0]}, index=[0]))

                #end of printing part 
                
                tm=rest[select][['hit_id','track_id','kt','z0']]
                res_list.append(tm)
                rest = rest[~select]
                dum,rest['track_len']=tag_bins(rest['track_id'])
                rest['pre_track_id']=rest['track_id']

    ft.value = rest.shape[0]
    res_list.append(rest[['hit_id','track_id','kt','z0']].copy())
    res = pd.concat(res_list, ignore_index=True)
    pbar.close()
    rest['track_id'],dum=tag_bins(rest['track_id'])
     
    if not history is None:
        history.append(pd.concat(hist_list,ignore_index=False))
    print ('took {:.5f} sec'.format(time.time()-start))
    return res 

def refine_hipos(res,hits,stds,nhipos,phik=3.3,weights=None): 
    cols=list(res.columns)
    if weights is None:
        weights={'theta':0.15, 'phi':1.0}

    groups = res.merge(hits,on='hit_id',how='left')
    if not groups.columns.contains('kt'):
        groups['kt']=0
        groups['z0']=0
        print("No kt's, calculating")
    calc_features(groups,groups[['kt','z0']],phik)

    gp=groups.groupby('track_id').agg({'phi': np.std , 'sint' : np.std,
            'cost' : np.std}).rename(columns={ 'phi': 'phi_std', 
            'sint' : 'sint_std', 'cost':'cost_std'}).reset_index()
    groups=groups.merge(gp,on='track_id',how='left')
    groups['theta_std']=np.sqrt(weights['theta']*np.square(groups.sint_std)+weights['theta']*np.square(groups.cost_std))
    hipos=pd.DataFrame()
    for col in stds:
        hipos[col]=np.random.normal(scale=stds[col],size=nhipos)

    for hipo in tqdm(hipos.itertuples(),total=nhipos):

        groups['kt_new']=groups['kt']+hipo.kt
        groups['z0_new']=groups['z0']+hipo.z0
        calc_features(groups,groups[['kt_new','z0_new']].rename(columns={"kt_new": "kt", "z0_new": "z0"}),phik)
        gp=groups.groupby('track_id').agg({'phi': np.std , 'sint' : np.std,
            'cost' : np.std}).rename(columns={ 'phi': 'new_phi_std', 
            'sint' : 'new_sint_std', 'cost':'new_cost_std'}).reset_index()
        groups=groups.merge(gp,on='track_id',how='left')
        groups['new_theta_std']=np.sqrt(weights['theta']*np.square(groups.new_sint_std)+weights['theta']*np.square(groups.new_cost_std))

        old_std=np.sqrt(np.square(groups.theta_std)+weights['phi']*np.square(groups.phi_std))
        new_std=np.sqrt(np.square(groups.new_theta_std)+np.square(groups.new_phi_std))
        cond=(old_std<=new_std) 
        groups['kt']=groups['kt'].where(cond,groups.kt_new)
        groups['z0']=groups['z0'].where(cond,groups.z0_new)
        groups['theta_std']=groups['theta_std'].where(cond,groups.new_theta_std)
        groups['sint_std']=groups['sint_std'].where(cond,groups.new_sint_std)
        groups['cost_std']=groups['cost_std'].where(cond,groups.new_cost_std)
        groups['phi_std']=groups['phi_std'].where(cond,groups.new_phi_std)
        groups=groups.drop(['new_theta_std','new_phi_std','new_sint_std','new_cost_std'],axis=1)

        #pdb.set_trace()
    to_return=groups[cols+['theta_std','phi_std','sint_std','cost_std']]
    return to_return
    

def expand_tracks(res,hits,min_track_len,max_track_len,max_expand,to_track_len,mstd=1.0,dstd=0.0,phik=3.3,max_dtheta=10,mstd_size=None,mstd_vol=None,drop=0,nhipo=1000,weights=None):
    if weights is None:
        weights={'theta':0.25, 'phi':1.0}

    if mstd_size is None:
        mstd_size=[0 for i in range(20)]
    if mstd_vol is None:
        mstd_vol={7:0,8:0,9:0,12:0,13:0,14:0,16:0,17:0,18:0}
    gp=res.groupby('track_id').first().reset_index()
    orig_hipo=gp[['track_id','kt','z0']]
    eres=res.copy()
    res_list=[]
    stds={'kt':7e-5,'z0':0.8}
    eres=refine_hipos(eres,hits,stds,nhipo,phik=phik,weights=weights)
    dum,eres['track_len']=tag_bins(eres['track_id'])
    eres['max_track_len']=np.clip(eres.track_len+max_expand,0,max_track_len) 
    eres['max_track_len']=2*(  eres['max_track_len']/2).astype('int')+1
    eres=eres.sort_values('track_len')
    eres = eres.merge(hits,on='hit_id',how='left')
    eres['sensor']=eres.volume_id+eres.layer_id*100+100000*eres.module_id
    group_sensors=eres.groupby('track_id').sensor.unique()
    groups=eres[eres.track_len>min_track_len].groupby('track_id').first().reset_index().copy()
    groups['order']=-groups.track_len 
    groups=groups.sort_values('order').reset_index(drop=True)
    groups=groups.head(int((1.0-drop)*groups.shape[0])).copy()
    select=eres.track_len<to_track_len
    grouped=eres[~select]
    regrouped=eres[select].copy()
    regrouped['min_dist']=100
    regrouped['new_track_len']=0
    regrouped['new_track_id']=regrouped['track_id']
    regrouped['new_kt']=regrouped['kt']
    regrouped['new_z0']=regrouped['z0']
    regrouped['new_max_size'] = max_track_len

    f = FloatProgress(min=0, max=groups.shape[0], description='calculating:') # instantiate the bar
    display(f) # display the bar

    for group_tul in tqdm(groups.itertuples(),total=groups.shape[0]):
        if group_tul.Index%20 ==0: f.value=group_tul.Index
        if group_tul.track_len>=max_track_len: continue
        group=eres[eres.track_id==group_tul.track_id].copy()
        calc_features(group,group[['kt','z0']],phik)
        group['abs_z']=np.abs(group.z)
        group['abs_theta']=np.abs(group.theta)
        phi_mean=group.phi.mean()
        sint_mean=group.sint.mean()            
        cost_mean=group.cost.mean()
        max_z=group.abs_z.max()
        max_theta=group.abs_theta.max()
        regrouped['abs_z']=np.abs(regrouped.z)
        calc_features(regrouped,group_tul,phik,double_sided=True)
        regrouped['dist'] =np.sqrt(weights['theta']*np.square(regrouped.sint-sint_mean)+weights['theta']*np.square(regrouped.cost-cost_mean)+weights['phi']*np.square(regrouped.phi-phi_mean))
        regrouped['dist2'] =np.sqrt(weights['theta']*np.square(regrouped.sint2-sint_mean)+weights['theta']*np.square(regrouped.cost2-cost_mean)+weights['phi']*np.square(regrouped.phi2-phi_mean))
        select = (regrouped.abs_z>max_z)  & (max_dtheta >max_dtheta) & (regrouped.dist2<regrouped.dist)
        regrouped['dist']=regrouped['dist'].where(~select,regrouped['dist2'])    
        cmstd=regrouped.volume_id.map(mstd_vol)+mstd_size[group_tul.track_len]+mstd
        if (dstd==0.0):
            sdstd==group.dstd
        else:
            sdstd=dstd
        better =( regrouped.dist<cmstd*sdstd) & ( regrouped.dist<regrouped.min_dist) & (~regrouped.sensor.isin(group_sensors.loc[group_tul.track_id]))
        regrouped['min_dist']=np.where(better,regrouped.dist,regrouped.min_dist)
        regrouped['new_track_id']=np.where(better,group_tul.track_id,regrouped.new_track_id)
        regrouped['new_z0']=np.where(better,group_tul.z0,regrouped.new_z0)
        regrouped['new_kt']=np.where(better,group_tul.kt,regrouped.new_kt)
        regrouped['new_track_len']=np.where(better,group_tul.track_len,regrouped.new_track_len)
        regrouped['new_max_size']=np.where(better,group_tul.max_track_len,regrouped.new_max_size)
    f.value=group_tul.Index
    regrouped=regrouped.sort_values('min_dist')
    regrouped['closest']=regrouped.groupby('new_track_id')['min_dist'].cumcount()
    better=regrouped.closest+regrouped.new_track_len>=regrouped.new_max_size
    regrouped['track_id']=regrouped['track_id'].where(better,regrouped['new_track_id'])
    res_list.append(grouped[['hit_id','track_id']])
    res_list.append(regrouped[['hit_id','track_id']])
    to_return=pd.concat(res_list)
    to_return=to_return.merge(orig_hipo,on='track_id',how='left')
    return to_return
        

# the following 2 functions are taken from outrunner's kernel: https://www.kaggle.com/outrunner/trackml-2-solution-example

def get_event(event):
    hits= pd.read_csv(path+'%s-hits.csv'%event)
    cells= pd.read_csv(path+'%s-cells.csv'%event)
    truth= pd.read_csv(path+'%s-truth.csv'%event)
    particles= pd.read_csv(path+'%s-particles.csv'%event)
    return hits, cells, particles, truth

def score_event_fast(truth, submission):
    truth = truth[["hit_id", "particle_id", "weight"]].merge(submission, how='left', on='hit_id')
    df = truth.groupby(['track_id', 'particle_id']).hit_id.count().to_frame('count_both').reset_index()
    truth = truth.merge(df, how='left', on=['track_id', 'particle_id'])
    
    df1 = df.groupby(['particle_id']).count_both.sum().to_frame('count_particle').reset_index()
    truth = truth.merge(df1, how='left', on='particle_id')
    df1 = df.groupby(['track_id']).count_both.sum().to_frame('count_track').reset_index()
    truth = truth.merge(df1, how='left', on='track_id')
    truth.count_both *= 2
    score = truth[(truth.count_both > truth.count_particle) & (truth.count_both > truth.count_track)].weight.sum()
    particles = truth[(truth.count_both > truth.count_particle) & (truth.count_both > truth.count_track)].particle_id.unique()

    return score
history=[]
event_num = 0
event_prefix = 'event00000100{}'.format(event_num)
#hits, cells, particles, truth = load_event(os.path.join(path, event_prefix))
hits, cells, particles, truth = get_event(event_prefix)
weights={'pi':1,'theta':0.15}
stds={'z0':7.5, 'kt':7.5e-4}
d =    {'sint':[225,110,110,110,110,110],
        'cost':[225,110,110,110,110,110],
          'phi':[550,260,260,260,260,260],
        'min_group':[11,11,10,9,8,7],
        #'npoints':[50,50,50,50,50,50]}  # half minute run for testing
        'npoints':[500,2000,1000,1000,500,500]}


filters=pd.DataFrame(d)
nu=500
resa=clustering(hits,stds,filters,phik=3.3,nu=nu,truth=truth,history=history)
resa["event_id"]=event_num
score = score_event_fast(truth, resa.rename(index=str, columns={"label": "track_id"}))
print("Your score: ", score)

mstd_vol={7:0,8:0,9:0,12:2,13:1,14:2,16:3,17:2,18:3}
mstd_size=[4,4,4,4,3,3,3,2,2,2,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
weights={'theta':0.1, 'phi':1}
nresa=expand_tracks(resa,hits,5,16,5,7,mstd=8,dstd=0.00085,phik=3.3,max_dtheta=0.9*np.pi/2,mstd_vol=mstd_vol,mstd_size=mstd_size,weights=weights,nhipo=100)
nresa['event_id']=0
score = score_event_fast(truth, nresa)
print("Your score: ", score)
import pickle
def load_obj(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)
    
df_train=load_obj('../input/trackml-validation-data-for-ml-pandas-df/df_train_v1.pkl')
df_test=load_obj('../input/trackml-validation-data-for-ml-pandas-df/df_test_v1.pkl')
y_train=df_train.target.values
y_test=df_test.target.values
print("The dataframe with all features:")
display(df_train.head())
print("Features for each track:",df_train.columns.values)
import lightgbm
s=time.time()
# choose which features of the tracks we want to use:
columns=['svolume','nclusters', 'nhitspercluster', 'xmax','ymax','zmax', 'xmin','ymin','zmin', 'zmean', 'xvar','yvar','zvar']
rounds=1000
round_early_stop=50
parameters = { 'subsample_for_bin':800, 'max_bin': 512, 'num_threads':8, 
               'application': 'binary','objective': 'binary','metric': 'auc','boosting': 'gbdt',
               'num_leaves': 128,'feature_fraction': 0.7,'learning_rate': 0.05,'verbose': 0}
train_data = lightgbm.Dataset(df_train[columns].values, label=y_train)
test_data = lightgbm.Dataset(df_test[columns].values, label=y_test)
model = lightgbm.train(parameters,train_data,valid_sets=test_data,num_boost_round=rounds,early_stopping_rounds=round_early_stop,verbose_eval=50)
print('took',time.time()-s,'seconds')
def precision_and_recall(y_true, y_pred,threshold=0.5):
    tp,fp,fn,tn=0,0,0,0

    for i in range(0,len(y_true)):
        if y_pred[i]>=threshold:
            if y_true[i]>0:
                tp+=1
            else:
                fp+=1
        elif y_true[i]==0:
            tn+=1
        else:
            fn+=1
    precision=tp/(tp+fp) if (tp+fp != 0) else 0
    recall=tp/(tp+fn) if (tp+fn != 0) else 0
    accuracy=(tp+tn)/(tp+tn+fp+fn)
    print('Threshold',threshold,' --- Precision: {:5.4f}, Recall: {:5.4f}, Accuracy: {:5.4f}'.format(precision,recall,accuracy))
    return precision, recall, accuracy

y_test_pred=model.predict(df_test[columns].values)
precision, recall, accuracy=precision_and_recall(y_test, y_test_pred,threshold=0.1)
precision, recall, accuracy=precision_and_recall(y_test, y_test_pred,threshold=0.5)
precision, recall, accuracy=precision_and_recall(y_test, y_test_pred,threshold=0.9)
### create one more submission
history=[]
resa2=clustering(hits,stds,filters,phik=3.3,nu=nu,truth=truth,history=history)
resa2["event_id"]=event_num
score = score_event_fast(truth, resa2.rename(index=str, columns={"label": "track_id"}))
print("Your score: ", score)
from tqdm import tqdm_notebook
from sklearn.cluster.dbscan_ import dbscan

def get_features(track_hits,cluster_size=10):    
    """
    Input: dataframe with hits of 1 track
    Output: array with features of track
    """
    nhits = len(track_hits)
    svolume=track_hits['volume_id'].values.min()
    X=np.column_stack([track_hits.x.values, track_hits.y.values, track_hits.z.values])
    _, labels = dbscan(X, eps=cluster_size, min_samples=1, algorithm='ball_tree', metric='euclidean')
    uniques = np.unique(labels)
    nclusters = len(uniques)
    nhitspercluster = nhits/nclusters
    xmax=track_hits['x'].values.max()
    xmin=track_hits['x'].values.min()
    xvar=track_hits['x'].values.var()
    ymax=track_hits['y'].values.max()
    ymin=track_hits['y'].values.min()
    yvar=track_hits['y'].values.var()
    zmax=track_hits['z'].values.max()
    zmin=track_hits['z'].values.min()
    zvar=track_hits['z'].values.var()
    zmean=track_hits['z'].values.mean()
    features=np.array([svolume,nclusters,nhitspercluster,xmax,ymax,zmax,xmin,ymin,zmin,zmean,xvar,yvar,zvar])
    return features

def get_predictions(sub,hits,model,min_length=4):  
    """
    Input: dataframe sub with track id for each hit, 
           dataframe hits with hit information, 
           model=ML model to get prediction
    Output: dataframe with predicted probability for each track
    """
    preds=pd.DataFrame()
    sub=sub.merge(hits,on='hit_id',how='left')
    trackids_long=[]
    trackids_short=[]
    features=[]
    
    trackids=np.unique(sub['track_id']).astype("int64")
    for track_id in tqdm_notebook(trackids):        
        track_hits=sub[sub['track_id']==track_id]
        if len(track_hits) < min_length:
            trackids_short.append(track_id)
        else:
            features.append(get_features(track_hits))
            trackids_long.append(track_id)

    probabilities_long=model.predict(np.array(features))
    probabilities_short=np.array([0]*len(trackids_short))
    
    preds['quality']=np.concatenate((probabilities_long,probabilities_short))
    preds['track_id']=np.concatenate((trackids_long,trackids_short))
    preds['quality']=preds['quality'].fillna(1)  # assume it is a good track, if no probability can be calculated
    return preds

preds={}
preds[0]=get_predictions(resa,hits,model)
preds[1]=get_predictions(resa2,hits,model)
def merge_with_probabilities(sub1,sub2,preds1,preds2,truth=None,length_factor=0,at_least_more=0):
    """
    Input:  sub1 and sub2 are two dataframes, which assign track_ids to hits
            preds1 and preds2 are dataframes, which assign a quality (probability to be correct) to each track_id
            truth: if given, then calculate score of the merge
            length_factor: merge not only by quality, but also by length
            at_least_more: ask new quality to be at_least_more than old, to overwrite
    
    Output: new dataframe with updated track_ids
    """
    un,inv,count = np.unique(sub1['track_id'],return_inverse=True, return_counts=True)
    sub1['group_size']=count[inv]
    un,inv,count = np.unique(sub2['track_id'],return_inverse=True, return_counts=True)
    sub2['group_size']=count[inv]
    sub1=sub1.merge(preds1,on='track_id',how='left')
    sub2=sub2.merge(preds2,on='track_id',how='left')
    
    sub1['quality']=sub1['quality']+length_factor*sub1['group_size']
    sub2['quality']=sub2['quality']+length_factor*sub2['group_size']
    
    sub=sub1.merge(sub2,on='hit_id',suffixes=('','_new'))
    mm=sub.track_id.max()+1
    sub['track_id_new']=sub['track_id_new']+mm
    
    sub['quality']=sub['quality']+at_least_more
    cond=(sub['quality']>=sub['quality_new'])
    for col in ['track_id','z0','kt']:
        sub[col]=sub[col].where(cond,sub[col+'_new'])
    
    sub=sub[['hit_id','track_id','event_id','kt','z0']]
    if not truth is None:
        print('Score',score_event(truth,sub))
    
    # calculate track_ids again to make them smaller
    un,inv,count = np.unique(sub['track_id'],return_inverse=True, return_counts=True)
    sub['track_id']=inv
    return sub

print('Merge submission 0 and 1 into sub01:')
sub01=merge_with_probabilities(resa,resa2,preds[0],preds[1],truth,length_factor=0.5)
# Expanding the merged submission of the two clustering solutions gives:
mstd_vol={7:0,8:0,9:0,12:2,13:1,14:2,16:3,17:2,18:3}
mstd_size=[4,4,4,4,3,3,3,2,2,2,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
weights={'theta':0.1, 'phi':1}
nresa2=expand_tracks(sub01,hits,5,16,5,7,mstd=8,dstd=0.00085,phik=3.3,max_dtheta=0.9*np.pi/2,mstd_vol=mstd_vol,mstd_size=mstd_size,weights=weights,nhipo=100)
nresa2['event_id']=0
score = score_event_fast(truth, nresa2)
print("Your score: ", score)
