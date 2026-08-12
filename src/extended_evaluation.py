"""Frozen cross-city confirmation, ablation, efficiency, collision and case audits."""
from __future__ import annotations
import argparse, json, time
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
import pandas as pd
from sid_downstream_experiment import cases, eval_model, make_features

METRICS=['Acc@1','Recall@10','NDCG@10','Infeasible@10','MeanDist@10']
NYC_STATIC=np.array([0,0,.1,0,0,0,0,0,0,0,0,0,0,0,0,0,0.],float)
NYC_FULL=np.array([0,0,2,.25,0,8,8,4,12,12,12,12,12,12,12,0,-12.],float)

def load_city(data_dir,prefix,sid_dir):
    tr=pd.read_csv(data_dir/f'{prefix}_train.csv'); va=pd.read_csv(data_dir/f'{prefix}_val.csv'); te=pd.read_csv(data_dir/f'{prefix}_test.csv')
    for d in (tr,va,te):
        d['local_time']=pd.to_datetime(d.local_time,utc=True,errors='coerce'); d.dropna(subset=['local_time'],inplace=True)
    sid=pd.read_csv(sid_dir/'static_semantic_ids.csv'); idx={p:i for i,p in enumerate(sid.POI_id)}
    codes=sid[['sid_1','sid_2','sid_3']].to_numpy(int); lat=sid.latitude.to_numpy(); lon=sid.longitude.to_numpy()
    hp=np.full((len(sid),24),.5); trans=defaultdict(Counter)
    for p,h in zip(tr.POI_id,tr.local_time.dt.hour):
        if p in idx: hp[idx[p],int(h)]+=1
    hp/=hp.max(axis=1,keepdims=True)
    for _,d in tr.sort_values('local_time').groupby('user_id'):
        seq=[idx[p] for p in d.POI_id if p in idx]
        for x,y in zip(seq,seq[1:]): trans[x][y]+=1
    cs=cases(te,pd.concat([tr,va]),idx)
    t0=time.perf_counter(); feats=make_features(cs,idx,codes,lat,lon,hp,trans,np.random.default_rng(20260807)); feature_seconds=time.perf_counter()-t0
    return tr,va,te,sid,idx,feats,feature_seconds

def weights():
    specs={'Static-GNPR-SID':NYC_STATIC.copy(),'DynaSID-v8':NYC_FULL.copy()}
    groups={
      'w/o SID history':[2], 'w/o user recurrence':[5,9,11,13],
      'w/o recent memory':[6,10,12,14], 'w/o POI transition':[7,15],
      'w/o multi-scale local':[9,10,11,12,13,14], 'w/o spatial mechanisms':[3,9,10,11,12,13,14],
      'w/o safety penalty':[8], 'w/o stay/leave':[16]}
    for name,ids in groups.items():
        w=NYC_FULL.copy(); w[ids]=0; specs[name]=w
    return specs

def paired(new,base,comparison,rng):
    merged=new.merge(base,on=['user','trajectory'],suffixes=('_new','_base')); out=[]
    for metric in METRICS:
        by=merged.groupby('user').apply(lambda z:float((z[f'{metric}_new']-z[f'{metric}_base']).mean()),include_groups=False).to_numpy()
        boots=np.array([rng.choice(by,len(by),replace=True).mean() for _ in range(5000)])
        out.append({'comparison':comparison,'metric':metric,'mean_delta':by.mean(),'ci95_low':np.quantile(boots,.025),'ci95_high':np.quantile(boots,.975),'n_users':len(by)})
    return out

def collision_audit(sid):
    z=sid.copy(); z['semantic_id']=z[['sid_1','sid_2','sid_3']].astype(str).agg('-'.join,axis=1)
    z['lat_bin']=pd.qcut(z.latitude,10,labels=False,duplicates='drop'); z['lon_bin']=pd.qcut(z.longitude,10,labels=False,duplicates='drop'); z['region']=z.lat_bin.astype(str)+'-'+z.lon_bin.astype(str)
    rows=[]; examples=[]
    for sem,g in z.groupby('semantic_id'):
        if len(g)<2: continue
        rows.append({'semantic_id':sem,'n_pois':len(g),'n_categories':g.category.nunique(),'n_regions':g.region.nunique(),
                     'category_pure':int(g.category.nunique()==1),'region_pure':int(g.region.nunique()==1)})
        if len(examples)<20:
            examples.append({'semantic_id':sem,'n_pois':len(g),'poi_ids':' | '.join(map(str,g.POI_id.head(5))),
                             'categories':' | '.join(map(str,g.category.head(5))),
                             'coordinates':' | '.join(f'{a:.4f},{b:.4f}' for a,b in zip(g.latitude.head(5),g.longitude.head(5)))})
    return pd.DataFrame(rows),pd.DataFrame(examples)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--data-dir',type=Path,required=True); ap.add_argument('--city-prefix',required=True)
    ap.add_argument('--sid-dir',type=Path,required=True); ap.add_argument('--out',type=Path,required=True); a=ap.parse_args(); a.out.mkdir(parents=True,exist_ok=True)
    tr,va,te,sid,idx,feats,feature_seconds=load_city(a.data_dir,a.city_prefix,a.sid_dir)
    summaries=[]; details={}; timings=[]
    for name,w in weights().items():
        t0=time.perf_counter(); m,d=eval_model(feats,w); seconds=time.perf_counter()-t0
        summaries.append({'city':a.city_prefix,'method':name,**m.to_dict()}); details[name]=d
        timings.append({'city':a.city_prefix,'method':name,'n_cases':len(d),'feature_seconds_shared':feature_seconds,'score_seconds':seconds,'ms_per_case':1000*seconds/len(d)})
    pd.DataFrame(summaries).to_csv(a.out/'extended_summary.csv',index=False)
    pd.DataFrame(timings).to_csv(a.out/'efficiency.csv',index=False)
    rng=np.random.default_rng(20260809); pair=[]
    pair+=paired(details['DynaSID-v8'],details['Static-GNPR-SID'],'DynaSID-v8 - Static-GNPR-SID',rng)
    pd.DataFrame(pair).to_csv(a.out/'paired_bootstrap.csv',index=False)
    # Most interpretable test cases: rank changes and feasibility changes.
    base=details['Static-GNPR-SID']; new=details['DynaSID-v8']; c=new.merge(base,on=['user','trajectory'],suffixes=('_dyn','_base'))
    c['rank_gain']=c.rank_base-c.rank_dyn; c['distance_reduction']=c['MeanDist@10_base']-c['MeanDist@10_dyn']
    c.to_csv(a.out/'paired_cases_all.csv',index=False)
    c.sort_values(['rank_gain','distance_reduction'],ascending=False).head(50).to_csv(a.out/'case_improvements.csv',index=False)
    c.sort_values(['rank_gain','distance_reduction'],ascending=True).head(50).to_csv(a.out/'case_regressions.csv',index=False)
    coll,ex=collision_audit(sid); coll.to_csv(a.out/'collision_groups.csv',index=False); ex.to_csv(a.out/'collision_examples.csv',index=False)
    audit={'city':a.city_prefix,'train_rows':len(tr),'val_rows':len(va),'test_rows':len(te),'train_users':int(tr.user_id.nunique()),
           'train_pois':int(tr.POI_id.nunique()),'test_cases':len(feats),'same_poi_test_rate':float(np.mean([x[2]==x[3] for x in feats])),
           'collision_groups':len(coll),'pois_in_collision_groups':int(coll.n_pois.sum()) if len(coll) else 0,
           'collision_category_purity':float(np.average(coll.category_pure,weights=coll.n_pois)) if len(coll) else 1,
           'collision_region_purity':float(np.average(coll.region_pure,weights=coll.n_pois)) if len(coll) else 1}
    (a.out/'audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    print(pd.DataFrame(summaries).to_string(index=False)); print(json.dumps(audit,indent=2))
if __name__=='__main__': main()
