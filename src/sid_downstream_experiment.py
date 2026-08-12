"""Downstream retrieval evaluation using actually trained GNPR-SID V2 codes."""
from __future__ import annotations
import argparse, json, math
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
import pandas as pd


def hav(lat,lon,lats,lons):
    r=6371.0088; p1=np.radians(lat); p2=np.radians(lats)
    a=np.sin((p2-p1)/2)**2+np.cos(p1)*np.cos(p2)*np.sin(np.radians(lons-lon)/2)**2
    return 2*r*np.arcsin(np.sqrt(np.clip(a,0,1)))
def ndcg(r): return 0 if r>10 else 1/math.log2(r+1)
def prefix_sim(code,codes):
    e0=codes[:,0]==code[0]; e1=e0&(codes[:,1]==code[1]); e2=e1&(codes[:,2]==code[2])
    return (e0.astype(float)+e1.astype(float)+e2.astype(float))/3
def cases(source,history,idx):
    hist=defaultdict(list)
    for r in history.sort_values('local_time').itertuples():
        if r.POI_id in idx: hist[str(r.user_id)].append((r.POI_id,r.local_time))
    out=[]
    for u,df in source.sort_values('local_time').groupby('user_id',sort=False):
        h=hist[str(u)].copy()
        for tid,t in df.groupby('trajectory_id',sort=False):
            t=t.sort_values('local_time'); target=t.iloc[-1]; pre=[(r.POI_id,r.local_time) for r in t.iloc[:-1].itertuples() if r.POI_id in idx]
            if target.POI_id in idx and h: out.append((str(u),str(tid),h.copy()+pre,idx[target.POI_id]))
            h.extend([(r.POI_id,r.local_time) for r in t.itertuples() if r.POI_id in idx])
    return out
def make_features(cs,idx,codes,lat,lon,hour_prof,trans,rng):
    random_codes=rng.integers(0,64,size=codes.shape); out=[]; two_step_cache={}
    for u,tid,h,target in cs:
        current=idx[h[-1][0]]; hour=int(h[-1][1].hour); dist=hav(lat[current],lon[current],lat,lon)
        sid_cur=prefix_sim(codes[current],codes); rand_cur=prefix_sim(random_codes[current],random_codes)
        sid_hist=np.zeros(len(codes)); userpoi=np.zeros(len(codes)); recent=np.zeros(len(codes))
        for age,(p,_) in enumerate(reversed(h[-20:])):
            j=idx[p]; sid_hist+=math.exp(-age/5)*prefix_sim(codes[j],codes); recent[j]+=math.exp(-age/5)
        for p,_ in h: userpoi[idx[p]]+=1
        for z in (sid_hist,userpoi,recent):
            if z.max()>0: z/=z.max()
        ptrans=np.zeros(len(codes)); total=sum(trans[current].values())
        if total:
            for j,v in trans[current].items(): ptrans[j]=v/total
        if current not in two_step_cache:
            p2=np.zeros(len(codes))
            if total:
                for mid,v1 in trans[current].items():
                    t2=sum(trans[mid].values())
                    if t2:
                        for dest,v2 in trans[mid].items(): p2[dest]+=(v1/total)*(v2/t2)
            if p2.max()>0: p2/=p2.max()
            two_step_cache[current]=p2
        ptrans2=two_step_cache[current]
        spatial=np.exp(-dist/5); temporal=hour_prof[:,hour]
        # Local recurrence disentangles "the user likes this POI" from
        # "the POI is reachable from the current context".
        local_user=userpoi*spatial; local_recent=recent*spatial
        near=np.exp(-dist/2); broad=np.exp(-dist/10)
        stay=np.zeros(len(codes)); stay[current]=1
        out.append((u,tid,current,target,rand_cur,sid_cur,sid_hist,spatial,temporal,userpoi,recent,ptrans,dist,local_user,local_recent,userpoi*near,recent*near,userpoi*broad,recent*broad,ptrans2,stay))
    return out
def eval_model(feats,w,detail=False):
    rows=[]
    for u,tid,current,target,*f in feats:
        R,C,H,S,T,U,E,V,D,LU,LE,LUN,LEN,LUB,LEB,V2,STAY=f
        score=w[0]*R+w[1]*C+w[2]*H+w[3]*S+w[4]*T+w[5]*U+w[6]*E+w[7]*V-w[8]*np.log1p(D)+w[9]*LU+w[10]*LE+w[11]*LUN+w[12]*LEN+w[13]*LUB+w[14]*LEB+w[15]*V2+w[16]*STAY
        score=score.copy(); target_score=score[target]
        # Deterministic total order prevents an all-tied score vector from
        # being incorrectly counted as rank 1 for every target.
        ids=np.arange(len(score)); rank=int(np.count_nonzero((score>target_score)|((score==target_score)&(ids<target)))+1)
        top=np.argpartition(-score,10)[:10]
        rows.append((u,tid,rank,float(np.mean(D[top]>10)),float(D[top].mean())))
    d=pd.DataFrame(rows,columns=['user','trajectory','rank','Infeasible@10','MeanDist@10']); d['Acc@1']=(d['rank']==1).astype(float); d['Recall@10']=(d['rank']<=10).astype(float); d['NDCG@10']=d['rank'].map(ndcg)
    return d.groupby('user')[['Acc@1','Recall@10','NDCG@10','Infeasible@10','MeanDist@10']].mean().mean(),d
def tune(val,active,base=None,extended=False,rounds=2,grid_override=None):
    w=np.zeros(17) if base is None else base.copy()
    # The first experiment hit the upper boundary (3.0) on recurrence and
    # transition features. Extend the validation-only grid before concluding.
    grid=(grid_override if grid_override is not None else ([0,.05,.1,.25,.5,.75,1,1.5,2,3,4,6,8] if extended
          else [0,.1,.25,.5,1,1.5,2,3]))
    for _ in range(rounds):
        for j in active:
            best=None
            for v in grid:
                q=w.copy(); q[j]=v; m,_=eval_model(val,q); obj=.55*m['Recall@10']+.45*m['NDCG@10']
                if best is None or obj>best[0]: best=(obj,v)
            w[j]=best[1]
    return w
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--data-dir',type=Path,required=True); ap.add_argument('--sid-dir',type=Path,required=True); ap.add_argument('--out',type=Path,default=Path('artifacts/sid_downstream')); ap.add_argument('--city-prefix',default='NYC'); ap.add_argument('--refine-only',action='store_true'); ap.add_argument('--baselines-only',action='store_true'); a=ap.parse_args(); a.out.mkdir(parents=True,exist_ok=True)
    tr=pd.read_csv(a.data_dir/f'{a.city_prefix}_train.csv'); va=pd.read_csv(a.data_dir/f'{a.city_prefix}_val.csv'); te=pd.read_csv(a.data_dir/f'{a.city_prefix}_test.csv')
    for d in (tr,va,te): d['local_time']=pd.to_datetime(d.local_time,utc=True,errors='coerce'); d.dropna(subset=['local_time'],inplace=True)
    sid=pd.read_csv(a.sid_dir/'static_semantic_ids.csv'); idx={p:i for i,p in enumerate(sid.POI_id)}; codes=sid[['sid_1','sid_2','sid_3']].to_numpy(int); lat=sid.latitude.to_numpy(); lon=sid.longitude.to_numpy()
    hp=np.full((len(sid),24),.5); trans=defaultdict(Counter)
    for p,h in zip(tr.POI_id,tr.local_time.dt.hour):
        if p in idx: hp[idx[p],int(h)]+=1
    hp/=hp.max(axis=1,keepdims=True)
    for _,d in tr.sort_values('local_time').groupby('user_id'):
        seq=[idx[p] for p in d.POI_id if p in idx]
        for x,y in zip(seq,seq[1:]): trans[x][y]+=1
    rng=np.random.default_rng(20260807); vf=make_features(cases(va,tr,idx),idx,codes,lat,lon,hp,trans,rng); tf=make_features(cases(te,pd.concat([tr,va]),idx),idx,codes,lat,lon,hp,trans,np.random.default_rng(20260807))
    baseline_specs={'Random-ID':([0],None),'GNPR-SID':([1,2],None),'GNPR-SID+Spatial':([1,2,3],None),'GNPR-SID+Temporal':([1,2,4],None)}
    specs=({'DynaSID':([16],None)} if a.refine_only else (baseline_specs if a.baselines_only else {**baseline_specs,'DynaSID':([1,2,3,4,5,6,7,9,10,11,12,13,14,15,16],None)})); rows=[]; raw=[]; raw_by_name={}; configs={}
    for name,(active,_) in specs.items():
        start=(np.array([0,0,2,.25,0,8,8,4,12,12,12,12,12,12,12,0,0.],dtype=float) if a.refine_only and name=='DynaSID' else None)
        focused=[-12,-8,-6,-4,-2,-1,0,1,2,4,6,8,12]
        w=tune(vf,active,base=start,extended=(name=='DynaSID'),rounds=(1 if a.refine_only else 2),grid_override=(focused if a.refine_only else None))
        if name=='DynaSID':
            scans=[]
            for p in [0,.1,.25,.5,.75,1,1.5,2,3,4,5,6,8,10,12,16,20,24,30]:
                q=w.copy();q[8]=p;m,_=eval_model(vf,q);scans.append((p,m))
            ok=[z for z in scans if z[1]['Infeasible@10']<=.005]
            pool=ok if ok else sorted(scans,key=lambda z:z[1]['Infeasible@10'])[:1]
            p,m=max(pool,key=lambda z:.55*z[1]['Recall@10']+.45*z[1]['NDCG@10']);w[8]=p
        m,d=eval_model(tf,w); rows.append({'method':name,**m.to_dict()}); raw_by_name[name]=d.copy(); d.insert(0,'method',name);raw.append(d);configs[name]=w.tolist()
    pd.DataFrame(rows).to_csv(a.out/'sid_downstream_summary.csv',index=False); pd.concat(raw).to_csv(a.out/'sid_downstream_cases.csv',index=False); (a.out/'sid_downstream_config.json').write_text(json.dumps(configs,indent=2),encoding='utf-8')
    if 'GNPR-SID' in raw_by_name and 'DynaSID' in raw_by_name:
        new=raw_by_name['DynaSID']; base=raw_by_name['GNPR-SID']; merged=new.merge(base,on=['user','trajectory'],suffixes=('_new','_base')); brng=np.random.default_rng(20260807); paired=[]
        for metric in ['Acc@1','Recall@10','NDCG@10','Infeasible@10','MeanDist@10']:
            by=merged.groupby('user').apply(lambda z:float((z[f'{metric}_new']-z[f'{metric}_base']).mean()),include_groups=False).to_numpy(); boots=np.array([brng.choice(by,len(by),replace=True).mean() for _ in range(5000)])
            paired.append({'comparison':'DynaSID - GNPR-SID','metric':metric,'mean_delta':by.mean(),'ci95_low':np.quantile(boots,.025),'ci95_high':np.quantile(boots,.975),'n_users':len(by)})
        pd.DataFrame(paired).to_csv(a.out/'dynasid_vs_gnpr_pairwise.csv',index=False)
    print(pd.DataFrame(rows).to_string(index=False))
if __name__=='__main__':main()
