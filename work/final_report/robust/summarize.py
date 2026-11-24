from pathlib import Path
import sys,json
import numpy as np,pandas as pd
from sklearn.metrics import f1_score,precision_score,recall_score,confusion_matrix,mean_squared_error,r2_score
sys.path.insert(0,str(Path(__file__).parent));import study as s
import torch
import subprocess
subprocess.run([sys.executable,str(Path(__file__).parent/'availability.py')],check=True)
p=s.p;O=s.O;L=pd.read_parquet(s.ROOT/'out/features.parquet');tr,va,te=[L[L.split==k].reset_index(drop=True) for k in ['train','val','test']];sc=json.loads((O/'schema.json').read_text());P=pd.read_parquet(O/'predictions.parquet');R=pd.read_csv(O/'results.csv');S=json.loads((O/'selections.json').read_text());summary=[];bycat=[];life=[]
for name,g in R.groupby('model',sort=False):
 summary.append(dict(model=name,WAPE=g.WAPE.mean(),SD=g.WAPE.std() if len(g)>1 else np.nan,MAE=g.MAE.mean(),Bias=g.Bias.mean(),val_WAPE=g.val_WAPE.mean(),runs=len(g)))
 for r in g.itertuples():
  key=name if pd.isna(r.seed) else name+'|'+str(int(r.seed));pr=P[key].to_numpy();bycat+=p.group_metrics(te,pr,name,r.seed,'category');life+=p.group_metrics(te,pr,name,r.seed,'t')
pd.DataFrame(summary).to_csv(O/'summary.csv',index=False);pd.DataFrame(bycat).to_csv(O/'by_category.csv',index=False);pd.DataFrame(life).to_csv(O/'by_life_week.csv',index=False)
# Domain error slices from the final selected seed-zero tree.
pr=P['LightGBM Tweedie|0'].to_numpy();actual=te.y.to_numpy();unknown=~te.retail.astype(str).isin(sc['vocab']['retail']).to_numpy();slices=[]
for name,mask in [('Zero actual sales',actual==0),('Positive actual sales',actual>0),('Nonpositive estimated stock',te.inv_start.to_numpy()<=0),('Positive estimated stock',te.inv_start.to_numpy()>0),('Discounted',te.disc.to_numpy()>0),('Full price',te.disc.to_numpy()==0),('Previously unseen store',unknown),('Known store',~unknown)]:
 yy=actual[mask];pp=pr[mask];slices.append(dict(group=name,rows=int(mask.sum()),actual_units=float(yy.sum()),MAE=float(abs(pp-yy).mean()),WAPE=p.wape(yy,pp),mean_signed_error=float((pp-yy).mean())))
pd.DataFrame(slices).to_csv(O/'error_slices.csv',index=False)
P[['y']+[f'LightGBM Tweedie|{i}' for i in range(3)]].to_csv(O/'report_prediction_columns.csv',index=False)
# Selected checkpoint loading is shared by verification, validation diagnostics and sweeps.
def load(name,seed):
 cfg=S[name]['config'];slug=name.lower().replace(' ','_')+f'_{seed}'
 if name.startswith('LightGBM'):return s.lgb.Booster(model_file=str(O/f'{slug}.txt'))
 m=s.Piecewise(sc,cfg) if name.startswith('Piecewise') else p.MLP(sc,**cfg);m.load_state_dict(torch.load(O/f'{slug}.pt',map_location='cpu',weights_only=True));return m

def predict(name,m,df):return m.predict(p.tree_frame(df,p.NUM+p.CAT,sc)) if name.startswith('LightGBM') else p.nn_predict(m,p.tensors(df,sc),'mse' if name=='MLP MSE' else 'poisson')
# Freeze an occurrence threshold using validation macro F1, before evaluating it on test.
thresholds=[];diag=[];reg=[];grid=np.arange(1,41)*.05
for name in S:
 vals=[]
 for seed in [0,1,2]:
  m=load(name,seed);vp=predict(name,m,va);scores=np.array([f1_score(va.y>0,vp>=t,average='macro',zero_division=0) for t in grid]);t=float(grid[np.argmax(scores)]);thresholds.append(dict(model=name,seed=seed,threshold=t,val_macro_F1=float(scores.max())));pr=P[name+'|'+str(seed)].to_numpy();a=te.y>0;b=pr>=t;cm=confusion_matrix(a,b,labels=[False,True]);diag.append(dict(model=name,seed=seed,threshold=t,precision=precision_score(a,b,zero_division=0),recall=recall_score(a,b,zero_division=0),sale_F1=f1_score(a,b),macro_F1=f1_score(a,b,average='macro'),TN=int(cm[0,0]),FP=int(cm[0,1]),FN=int(cm[1,0]),TP=int(cm[1,1])));reg.append(dict(model=name,seed=seed,RMSE=float(np.sqrt(mean_squared_error(te.y,pr))),R2=r2_score(te.y,pr)))
  print('DIAGNOSTIC',name,seed,t,flush=True)
pd.DataFrame(thresholds).to_csv(O/'thresholds.csv',index=False);pd.DataFrame(diag).to_csv(O/'occurrence_metrics.csv',index=False);pd.DataFrame(reg).to_csv(O/'regression_metrics.csv',index=False)
# Sampled sweeps share the support-study sample selection.
rng=np.random.default_rng(20261120);rng.choice(len(va),10000,replace=False);ix=np.sort(rng.choice(len(te),10000,replace=False));Q=te.iloc[ix].copy();Q[['external_code','retail','forecast_date']].to_csv(O/'scenario_sample.csv',index=False);curves=[];checks=[]
for name in ['LightGBM Tweedie','MLP Poisson','Piecewise monotone','Piecewise free']:
 for seed in [0,1,2]:
  m=load(name,seed);rows=[];preds=[]
  for d in np.arange(7)/10:
   z=Q.copy();z['disc']=d;z['disc_change']=d-z.disc_prev;z['disc_missing']=0;pr=predict(name,m,z);preds.append(pr)
   for cat,ii in z.groupby('category',observed=True).indices.items():rows.append(dict(model=name,seed=seed,category=cat,discount=float(d),mean=float(pr[ii].mean()),n=len(ii)))
  arr=np.array(preds);checks.append(dict(model=name,seed=seed,individual_monotone_share=float((np.diff(arr,axis=0)>=-1e-5).all(axis=0).mean()),minimum_step=float(np.diff(arr,axis=0).min())));curves+=rows;print('SWEEP',name,seed,checks[-1],flush=True)
pd.DataFrame(curves).to_csv(O/'sweeps.csv',index=False);p.save_json(O/'monotonicity_checks.json',checks)
# Conditional two-way sensitivity, not a model-selection-adjusted coverage claim.
a=P['MLP Poisson|0'].to_numpy();b=P['LightGBM Tweedie|0'].to_numpy();y=P.y.to_numpy();prod=pd.DataFrame({'product':P.external_code,'y':y,'tree':abs(y-b),'nn':abs(y-a)}).groupby('product').sum().to_numpy();weeks=pd.DataFrame({'date':P.forecast_date,'y':y,'tree':abs(y-b),'nn':abs(y-a)}).groupby('date').sum().to_numpy();rng=np.random.default_rng(20261122);unc=[]
for typ,v in [('product',prod),('calendar8',weeks)]:
 dif=[]
 for _ in range(2000):
  if typ=='product':ix=rng.integers(len(v),size=len(v))
  else:ix=((rng.integers(len(v),size=int(np.ceil(len(v)/8)))[:,None]+np.arange(8))%len(v)).flatten()[:len(v)]
  z=v[ix].sum(axis=0);dif.append(100*(z[2]-z[1])/z[0])
 unc.append(dict(method=typ,gap_pp=float(100*(abs(y-a).sum()-abs(y-b).sum())/y.sum()),ci95_pp=np.quantile(dif,[.025,.975]).tolist(),replicates=2000))
# Matched piecewise constraint comparison uses the same conditional resampling schemes.
a=P['Piecewise monotone|0'].to_numpy();b=P['Paired free control|0'].to_numpy();prod=pd.DataFrame({'product':P.external_code,'y':y,'free':abs(y-b),'mono':abs(y-a)}).groupby('product').sum().to_numpy();weeks=pd.DataFrame({'date':P.forecast_date,'y':y,'free':abs(y-b),'mono':abs(y-a)}).groupby('date').sum().to_numpy()
for typ,v in [('monotone_product',prod),('monotone_calendar8',weeks)]:
 dif=[]
 for _ in range(2000):
  if typ=='monotone_product':ix=rng.integers(len(v),size=len(v))
  else:ix=((rng.integers(len(v),size=int(np.ceil(len(v)/8)))[:,None]+np.arange(8))%len(v)).flatten()[:len(v)]
  z=v[ix].sum(axis=0);dif.append(100*(z[2]-z[1])/z[0])
 unc.append(dict(method=typ,gap_pp=float(100*(abs(y-a).sum()-abs(y-b).sum())/y.sum()),ci95_pp=np.quantile(dif,[.025,.975]).tolist(),replicates=2000))
p.save_json(O/'conditional_uncertainty.json',unc);p.save_json(O/'summary_completed.json',dict(test_rows=len(te),sweep_rows=len(Q),families=len(S)))
