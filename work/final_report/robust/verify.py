from pathlib import Path
import sys,json,hashlib
import numpy as np,pandas as pd
sys.path.insert(0,str(Path(__file__).parent));import study as s
import torch
from sklearn.metrics import mean_absolute_error
p=s.p;O=s.O;P=pd.read_parquet(O/'predictions.parquet');R=pd.read_csv(O/'results.csv');L=pd.read_parquet(s.ROOT/'out/features.parquet');te=L[L.split=='test'].reset_index(drop=True);sc=json.loads((O/'schema.json').read_text());S=json.loads((O/'selections.json').read_text());T=p.tensors(te,sc);checks=[];maxreload=0.;maxmetric=0.
assert len(P)==len(te)
for c in ['external_code','retail','forecast_date','category','t','y']:
 assert P[c].reset_index(drop=True).equals(te[c].reset_index(drop=True)),c
for r in R.itertuples():
 name=r.model;seed=None if pd.isna(r.seed) else int(r.seed);key=name if seed is None else f'{name}|{seed}';saved=P[key].to_numpy()
 if seed is None:pr=te.lag1.to_numpy() if name=='Naive last week' else te.mean_sofar.to_numpy()
 elif name.startswith('LightGBM'):
  m=s.lgb.Booster(model_file=str(O/(name.lower().replace(' ','_')+f'_{seed}.txt')));pr=m.predict(p.tree_frame(te,p.NUM+p.CAT,sc))
 elif name=='Tree availability check':
  fs=json.loads((O/'tree_availability_check.json').read_text())['features'];m=s.lgb.Booster(model_file=str(O/'tree_availability_check.txt'));pr=m.predict(p.tree_frame(te,fs,sc))
 elif name.startswith('Tree no'):
  fs=[f for f in p.NUM+p.CAT if f not in (p.DISC if name=='Tree no discount' else p.STOCK)];m=s.lgb.Booster(model_file=str(O/(name.lower().replace(' ','_')+'.txt')));pr=m.predict(p.tree_frame(te,fs,sc))
 else:
  if name=='Paired free control':
   cfg=S['Piecewise monotone']['config'];file=O/f'paired_free_{seed}.pt'
   if not file.exists():file=O/f'piecewise_free_{seed}.pt'
   m=s.Piecewise(sc,cfg)
  else:
   cfg=S[name]['config'];m=s.Piecewise(sc,cfg) if name.startswith('Piecewise') else p.MLP(sc,**cfg);file=O/(name.lower().replace(' ','_')+f'_{seed}.pt')
  m.load_state_dict(torch.load(file,map_location='cpu',weights_only=True));pr=p.nn_predict(m,T,'mse' if name=='MLP MSE' else 'poisson')
 delta=float(abs(pr-saved).max());yy=te.y.to_numpy(dtype=np.float64);pp=np.asarray(pr,dtype=np.float64);mae=float(mean_absolute_error(yy,pp));metrics=dict(WAPE=mae*len(yy)/yy.sum(),MAE=mae,Bias=float((pp.sum()-yy.sum())/yy.sum()));md=max(abs(float(metrics[k])-getattr(r,k)) for k in ['WAPE','MAE','Bias']);assert delta<1e-6 and md<1e-10,(name,seed,delta,md);maxreload=max(maxreload,delta);maxmetric=max(maxmetric,md);checks.append(dict(model=name,seed=seed,reload_difference=delta,metric_difference=md));print('VERIFIED',name,seed,flush=True)
# Direct checks of projection and monotonicity at random contexts and dense discount steps.
rng=np.random.default_rng(20261123);ix=np.sort(rng.choice(len(te),1000,replace=False));q=te.iloc[ix].copy();mon=[]
for seed in [0,1,2]:
 m=s.Piecewise(sc,S['Piecewise monotone']['config']);m.load_state_dict(torch.load(O/f'piecewise_monotone_{seed}.pt',map_location='cpu',weights_only=True));assert min(float(m.slopes.weight.min()),float(m.slopes.bias.min()))>=0;arr=[]
 for d in np.linspace(0,1,101):
  z=q.copy();z['disc']=d;z['disc_change']=d-z.disc_prev;z['disc_missing']=0;arr.append(p.nn_predict(m,p.tensors(z,sc),'poisson'))
 minimum=float(np.diff(np.array(arr),axis=0).min());assert minimum>=-1e-5;mon.append(dict(seed=seed,contexts=1000,discount_steps=101,minimum_step=minimum))
# Independently reconstruct all restricted refit scores from their saved weights.
refit_checks=[]
for row in json.loads((O/'refit_replicates.json').read_text()):
 rep=row['replicate'];rs=json.loads((O/f'refit_schema_{rep}.json').read_text());tm=s.lgb.Booster(model_file=str(O/f'refit_tree_{rep}.txt'));tp=tm.predict(p.tree_frame(te,p.NUM+p.CAT,rs));nm=p.MLP(rs,**row['selection']['MLP Poisson']);nm.load_state_dict(torch.load(O/f'refit_poisson_{rep}.pt',map_location='cpu',weights_only=True));npred=p.nn_predict(nm,p.tensors(te,rs),'poisson');dt=abs(p.wape(te.y,tp)-row['tree_WAPE']);dn=abs(p.wape(te.y,npred)-row['poisson_WAPE']);assert max(dt,dn)<1e-10;refit_checks.append(dict(replicate=rep,tree_difference=dt,poisson_difference=dn));print('VERIFIED REFIT',rep,flush=True)
p.save_json(O/'verification.json',dict(evaluations=len(checks),refit_evaluations=40,refit_checks=refit_checks,test_rows=len(te),max_reload_difference=maxreload,max_metric_difference=maxmetric,checks=checks,monotonicity=mon,source_hashes={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in O.glob('*.py')}));print('VERIFICATION COMPLETE',flush=True)
