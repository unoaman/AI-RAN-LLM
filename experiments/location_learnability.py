"""Position learnability study (docs/LOCATION_AWARE_HANDOVER.md §3), original simulator.

    PYTHONPATH=. python experiments/location_learnability.py
"""
import numpy as np, torch
from ai_ran_llm.config import ObsConfig
from ai_ran_llm.simulator import generate_episode, run_policy
from ai_ran_llm.dataset import _behaviour_policy
from ai_ran_llm.policies import label_decision
torch.manual_seed(0)
rng=np.random.default_rng(5); o=ObsConfig()
SIGMAS=[0,10,30,100]
F={k:[] for k in ['base']+[f'geo{s}' for s in SIGMAS]}; Y=[]; Yany=[]
for e in range(16):
    ep=generate_episode(32,600,rng)
    # positioning error: temporally correlated (AR(1), ~1 s memory), per UE
    errs={}
    for s in SIGMAS:
        z=np.zeros((ep.n_ue,ep.n_steps,2)); a=0.9; x=rng.normal(0,s,(ep.n_ue,2))
        for t in range(ep.n_steps):
            z[:,t]=x; x=a*x+np.sqrt(1-a*a)*rng.normal(0,s,(ep.n_ue,2))
        errs[s]=z
    _,_,seen=run_policy(ep,_behaviour_policy(rng,o),o,record=True)
    for t,obs in seen:
        if t<8 or t+10>=600: continue
        d=obs.nbr_hist-obs.serving_hist[:,None,:]
        base=np.concatenate([d[:,:,-1],d[:,:,-1]-d[:,:,0],(obs.serving_hist[:,-1]-obs.serving_hist[:,0])[:,None],
                             obs.serving_hist[:,-1:]/10+9,obs.sinr_db[:,None]/10,obs.speed_kmh[:,None]/100],1)
        F['base'].append(base)
        for s in SIGMAS:
            p_now=ep.pos[:,t]+errs[s][:,t]; p_old=ep.pos[:,t-8]+errs[s][:,t-8]
            cells=np.concatenate([obs.serving[:,None],obs.nbr_ids],1)          # (U,5)
            site=ep.sites[cells]                                                 # (U,5,2)
            dn=np.linalg.norm(p_now[:,None]-site,axis=-1)+1; do=np.linalg.norm(p_old[:,None]-site,axis=-1)+1
            geo=np.concatenate([np.log10(dn[:,:1]/dn[:,1:]), (dn-do)/0.8/30],1)  # distance ratio to nbrs, radial speed to each cell
            F[f'geo{s}'].append(np.concatenate([base,geo],1))
        tg,_=label_decision(ep,t,obs,o); Y.append(tg==obs.nbr_ids[:,0]); Yany.append(tg>=0)
def auc(s,y):
    o_=np.argsort(s); r=np.empty(len(s)); r[o_]=np.arange(len(s)); p=y.sum(); return (r[y].sum()-p*(p-1)/2)/(p*(len(y)-p))
for lab,YY in [('HO to strongest nbr',Y),('HO to any nbr',Yany)]:
    y=torch.tensor(np.concatenate(YY),dtype=torch.float32); n=len(y); tr=int(n*.8)
    print(f'== label: {lab}  (n={n}, positives {100*y.mean():.1f}%)')
    for k,v in F.items():
        X=torch.tensor(np.concatenate(v),dtype=torch.float32)
        m=torch.nn.Sequential(torch.nn.Linear(X.shape[1],128),torch.nn.ReLU(),torch.nn.Linear(128,64),torch.nn.ReLU(),torch.nn.Linear(64,1))
        opt=torch.optim.Adam(m.parameters(),3e-3)
        for i in range(600):
            idx=torch.randint(0,tr,(8192,))
            l=torch.nn.functional.binary_cross_entropy_with_logits(m(X[idx]).squeeze(1),y[idx]); opt.zero_grad(); l.backward(); opt.step()
        with torch.no_grad(): p=torch.sigmoid(m(X[tr:]).squeeze(1)).numpy()
        yt=y[tr:].numpy().astype(bool)
        # recall at the precision of the base model's operating point: report AUC and recall@precision 0.5
        o_=np.argsort(-p); tp=np.cumsum(yt[o_]); prec=tp/np.arange(1,len(p)+1); rec=tp/yt.sum()
        r50=rec[prec>=0.5].max() if (prec>=0.5).any() else 0
        name={'base':'report only (today)'}.get(k, f'+ position & heading, error σ={k[3:]} m' if k!='geo0' else '+ exact position & heading')
        print(f'   {name:42s} AUC {auc(p,yt):.3f}   recall@precision50 {r50:.3f}')
