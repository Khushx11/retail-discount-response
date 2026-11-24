from pathlib import Path
import sys,json,time,argparse
import numpy as np,pandas as pd
sys.path.insert(0,str(Path(__file__).parent));import study as s
import torch
p=s.p;O=s.O;parts=O/'refit_parts';parts.mkdir(exist_ok=True)
def refit(rep,L=None):
 path=parts/f'rep_{rep}.json'
 if path.exists():print('REUSE REFIT',rep,flush=True);return
 L=pd.read_parquet(s.ROOT/'out/features.parquet') if L is None else L;tr,va,te=[L[L.split==k].reset_index(drop=True) for k in ['train','val','test']];products,idx=np.unique(tr.external_code.to_numpy(),return_inverse=True);rng=np.random.default_rng(20261121+rep);count=np.bincount(rng.integers(0,len(products),len(products)),minlength=len(products));boot=tr.iloc[np.repeat(np.arange(len(tr)),count[idx])].reset_index(drop=True);sc=p.fit_schema(boot);Ts=[p.tensors(x,sc) for x in [boot,va,te]];p.save_json(O/f'refit_schema_{rep}.json',sc);testpred={};selected={};trial_records={};start=time.time()
 for family in ['LightGBM Tweedie','MLP Poisson']:
  configs=s.TREES[:2] if family=='LightGBM Tweedie' else [s.CONFIGS[4],s.CONFIGS[0]];best=None;trials=[]
  for cfg in configs:
   print('REFIT',rep,family,cfg,flush=True)
   if family=='LightGBM Tweedie':m,v=s.fit_tree(boot,va,sc,cfg,seed=rep)
   else:m,v,h,ep=s.fit_nn(Ts[0],Ts[1],sc,cfg,family,seed=rep)
   trials.append(dict(config=cfg,val_WAPE=v))
   if best is None or v<best[0]:best=(v,cfg,m)
  v,cfg,m=best
  if family=='LightGBM Tweedie':m.save_model(str(O/f'refit_tree_{rep}.txt'));pr=m.predict(p.tree_frame(te,p.NUM+p.CAT,sc))
  else:torch.save(m.state_dict(),O/f'refit_poisson_{rep}.pt');pr=p.nn_predict(m,Ts[2],'poisson')
  testpred[family]=pr;selected[family]=cfg;trial_records[family]=trials
 r=dict(replicate=rep,seed=20261121+rep,tree_WAPE=p.wape(te.y,testpred['LightGBM Tweedie']),poisson_WAPE=p.wape(te.y,testpred['MLP Poisson']),selection=selected,trials=trial_records,bootstrap_rows=len(boot),seconds=time.time()-start);r['gap_pp']=100*(r['poisson_WAPE']-r['tree_WAPE'])
 z=pd.DataFrame({'date':te.forecast_date,'y':te.y,'tree':abs(te.y-testpred['LightGBM Tweedie']),'nn':abs(te.y-testpred['MLP Poisson'])}).groupby('date').sum().to_numpy();n=len(z);mixture=[]
 for b in range(200):
  starts=rng.integers(n,size=int(np.ceil(n/8)));ix=((starts[:,None]+np.arange(8))%n).flatten()[:n];a=z[ix].sum(axis=0);mixture.append(dict(replicate=rep,gap_pp=float(100*(a[2]-a[1])/a[0])))
 pd.DataFrame(mixture).to_csv(parts/f'blocks_{rep}.csv',index=False);p.save_json(path,r);print('REFIT COMPLETE',rep,r['gap_pp'],flush=True)
def merge():
 rows=[json.loads((parts/f'rep_{i}.json').read_text()) for i in range(20)];blocks=pd.concat([pd.read_csv(parts/f'blocks_{i}.csv') for i in range(20)],ignore_index=True);p.save_json(O/'refit_replicates.json',rows);blocks.to_csv(O/'refit_calendar_mixture.csv',index=False);p.save_json(O/'robustness_completed.json',dict(folds=3,training_bootstraps=20,candidates_per_refit_family=2,seed_rule='20261121 plus replicate index',refit_preprocessing=True,limitation='Fixed validation observations; restricted two-candidate pools and twenty outer replicates; retrospective sensitivity, not definitive coverage or prospective assurance.'));print('ALL REFITS MERGED',flush=True)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--start',type=int,default=0);ap.add_argument('--stop',type=int,default=20);ap.add_argument('--merge',action='store_true');a=ap.parse_args()
 if a.merge:merge()
 else:
  L=pd.read_parquet(s.ROOT/'out/features.parquet')
  for rep in range(a.start,a.stop):refit(rep,L)
