"""Official GNPR-SID V2 CRQ-VAE training and SID audit on real city data."""
from __future__ import annotations
import argparse, copy, importlib, json, math, random, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from poi_features import build_poi_features


def load_crqvae(gnpr_root: Path):
    """Load CRQVAE from a separately cloned official GNPR-SID repository."""
    sid_root = gnpr_root.resolve() / "V2" / "SID"
    if not (sid_root / "CRQVAE" / "crqvae.py").is_file():
        raise FileNotFoundError(
            f"CRQVAE was not found under {sid_root}. Clone the official "
            "GNPR-SID repository and pass its root with --gnpr-root."
        )
    sys.path.insert(0, str(sid_root))
    return importlib.import_module("CRQVAE.crqvae").CRQVAE


def collision(codes: np.ndarray) -> float:
    return 1 - len(set(map(tuple, codes))) / len(codes)


@torch.no_grad()
def get_codes(model, x: torch.Tensor) -> tuple[torch.Tensor, np.ndarray]:
    model.eval(); q, idx = model.get_indices(x, use_sk=False)
    return q, idx.reshape(len(x), -1).numpy().astype(int)


def quality(model, x: torch.Tensor, poi: pd.DataFrame) -> tuple[dict, np.ndarray]:
    q, codes = get_codes(model, x)
    model.eval(); recon = model.decoder(q)
    m = {"reconstruction_mse": float(torch.mean((recon-x)**2).detach()), "unique_sids": len(set(map(tuple,codes))),
         "collision_rate": collision(codes)}
    for j in range(3):
        cnt=np.bincount(codes[:,j],minlength=64); p=cnt[cnt>0]/cnt.sum()
        m[f"level_{j+1}_usage"]=int((cnt>0).sum()); m[f"level_{j+1}_perplexity"]=float(np.exp(-(p*np.log(p)).sum()))
    cat=poi.category.to_numpy(); lat=pd.qcut(poi.latitude,10,labels=False,duplicates="drop").to_numpy(); lon=pd.qcut(poi.longitude,10,labels=False,duplicates="drop").to_numpy(); reg=lat*10+lon
    for d in range(1,4):
        groups={}
        for i,z in enumerate(codes): groups.setdefault(tuple(z[:d]),[]).append(i)
        cp=rp=n=0
        for ids in groups.values():
            for a,b in zip(ids[:99],ids[1:100]): n+=1; cp+=int(cat[a]==cat[b]); rp+=int(reg[a]==reg[b])
        m[f"prefix_{d}_category_coherence"]=cp/max(n,1); m[f"prefix_{d}_region_coherence"]=rp/max(n,1)
    return m,codes


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--data-dir",type=Path,required=True)
    ap.add_argument("--gnpr-root",type=Path,required=True,
                    help="Root of a separately cloned official GNPR-SID repository")
    ap.add_argument("--out",type=Path,default=Path("artifacts/full_sid_v2"))
    ap.add_argument("--city-prefix",default="NYC",help="File prefix, e.g. NYC or TKY")
    ap.add_argument("--epochs",type=int,default=200); ap.add_argument("--batch-size",type=int,default=256)
    ap.add_argument("--seed",type=int,default=20260807); ap.add_argument("--patience",type=int,default=60)
    args=ap.parse_args(); args.out.mkdir(parents=True,exist_ok=True)
    CRQVAE=load_crqvae(args.gnpr_root)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    tr=pd.read_csv(args.data_dir/f'{args.city_prefix}_train.csv'); tr['local_time']=pd.to_datetime(tr.local_time,utc=True,errors='coerce'); tr.dropna(subset=['local_time'],inplace=True)
    poi,feat,meta=build_poi_features(tr,f'{args.city_prefix}_train'); x=torch.from_numpy(feat)
    loader=DataLoader(TensorDataset(x),batch_size=args.batch_size,shuffle=True,generator=torch.Generator().manual_seed(args.seed))
    model=CRQVAE(in_dim=x.shape[1],num_emb_list=[64,64,64],e_dim=64,layers=[512,256,128],dropout_prob=.1,bn=False,
                 loss_type='mse',quant_loss_weight=.5,beta=.25,kmeans_init=True,kmeans_iters=100,
                 sk_epsilons=[.1,.1,.1],sk_iters=50,use_linear=0,use_ema=True,ema_decay=.95,ema_epsilon=1e-5)
    opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4)
    trace=[]; best_key=(math.inf,math.inf); best_model=None; best_epoch=None; stale=0
    for epoch in range(1,args.epochs+1):
        model.train(); tl=trc=tq=0.
        for (b,) in loader:
            opt.zero_grad(set_to_none=True); out,qloss,_=model(b,use_sk=False); loss,q,recon=model.compute_loss(qloss,out,xs=b)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.); opt.step()
            tl+=float(loss.detach()); tq+=float(q.detach()); trc+=float(recon.detach())
        if epoch==1 or epoch%10==0:
            _,codes=get_codes(model,x); c=collision(codes); rec={"epoch":epoch,"loss":tl/len(loader),"quantization_loss":tq/len(loader),"reconstruction_mse":trc/len(loader),"collision_rate":c}; trace.append(rec); print(json.dumps(rec))
            key=(c,rec['reconstruction_mse'])
            if key < best_key: best_key=key; best_model=copy.deepcopy(model).cpu(); best_epoch=epoch; stale=0
            else: stale+=10
        if stale>=args.patience: break
    model=best_model; metrics,codes=quality(model,x,poi); metrics.update(meta); metrics.update({"epochs_run":epoch,"selected_sid_epoch":best_epoch,"seed":args.seed,"official_module":"GNPR-SID V2 CRQ-VAE","codebook_sizes":[64,64,64],"ema":True})
    sid=poi.copy()
    for j in range(3): sid[f'sid_{j+1}']=codes[:,j]
    sid['semantic_id']=[f'<a_{z[0]}><b_{z[1]}><c_{z[2]}>' for z in codes]
    sid.to_csv(args.out/'static_semantic_ids.csv',index=False); pd.DataFrame(trace).to_csv(args.out/'crqvae_training_trace.csv',index=False)
    (args.out/'sid_quality_metrics.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8'); (args.out/'feature_audit.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    torch.save({'state_dict':model.state_dict(),'input_dim':x.shape[1],'meta':meta,'quantizers_initted':[True,True,True]},args.out/'crqvae_best.pt')
    print(json.dumps(metrics,indent=2))

if __name__=='__main__': main()
