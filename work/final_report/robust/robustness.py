from pathlib import Path
import sys,json,time
import numpy as np,pandas as pd
sys.path.insert(0,str(Path(__file__).parent));import study as s
import torch
p=s.p;O=s.O;start=time.time();L=pd.read_parquet(s.ROOT/'out/features.parquet');folds=json.loads((O/'rolling_folds.json').read_text()) if '--resume' in sys.argv and (O/'rolling_folds.json').exists() else [];fpreds=[pd.read_parquet(O/'rolling_predictions.parquet')] if '--resume' in sys.argv and (O/'rolling_predictions.parquet').exists() else []
for vstart,origin,end in [('2018-02-05','2018-05-07','2018-08-06'),('2018-08-06','2018-11-05','2019-02-04'),('2019-02-04','2019-05-06','2019-08-05')]:
 if sum(x['origin']==origin for x in folds)>=6:
  print('REUSE COMPLETED FOLD',origin,flush=True);continue
 tr=L[L.forecast_date<pd.Timestamp(vstart)].reset_index(drop=True);va=L[(L.forecast_date>=pd.Timestamp(vstart))&(L.forecast_date<pd.Timestamp(origin))].reset_index(drop=True);te=L[(L.forecast_date>=pd.Timestamp(origin))&(L.forecast_date<pd.Timestamp(end))].reset_index(drop=True)
 assert tr.label_end.max()<=va.forecast_date.min() and va.label_end.max()<=te.forecast_date.min()
 sc=p.fit_schema(tr);T=[p.tensors(x,sc) for x in [tr,va,te]];pred=te[['external_code','forecast_date','y']].copy();pred['origin']=origin
 for family in ['LightGBM Tweedie','MLP Poisson','Piecewise monotone']:
  best=None;trials=[];configs=s.TREES[:3] if family=='LightGBM Tweedie' else [s.CONFIGS[j] for j in [0,1,6]]
  for cfg in configs:
   print('FOLD',origin,family,cfg,flush=True)
   if family=='LightGBM Tweedie':m,v=s.fit_tree(tr,va,sc,cfg)
   else:m,v,h,ep=s.fit_nn(T[0],T[1],sc,cfg,family)
   trials.append(dict(config=cfg,val_WAPE=v))
   if best is None or v<best[0]:best=(v,cfg,m)
  v,cfg,m=best;pr=m.predict(p.tree_frame(te,p.NUM+p.CAT,sc)) if family=='LightGBM Tweedie' else p.nn_predict(m,T[2],'poisson');pred[family]=pr
  folds.append(dict(origin=origin,end=end,validation_start=vstart,model=family,train_rows=len(tr),val_rows=len(va),test_rows=len(te),WAPE=p.wape(te.y,pr),val_WAPE=v,selected=cfg,trials=trials));p.save_json(O/'rolling_folds.json',folds)
  if family=='Piecewise monotone':
   free,v,h,ep=s.fit_nn(T[0],T[1],sc,cfg,'Piecewise free');fr=p.nn_predict(free,T[2],'poisson');pred['Paired free control']=fr;folds.append(dict(origin=origin,end=end,model='Paired free control',train_rows=len(tr),val_rows=len(va),test_rows=len(te),WAPE=p.wape(te.y,fr),val_WAPE=v,selected=cfg))
 for n,pr in [('Naive last week',te.lag1),('Naive mean so far',te.mean_sofar)]:folds.append(dict(origin=origin,end=end,model=n,train_rows=len(tr),val_rows=len(va),test_rows=len(te),WAPE=p.wape(te.y,pr)));pred[n]=pr.to_numpy()
 fpreds.append(pred);p.save_json(O/'rolling_folds.json',folds);pd.concat(fpreds).to_parquet(O/'rolling_predictions.parquet',index=False)
 print('FOLD COMPLETE',origin,flush=True)
# Deterministic per-replicate workers can also run sequentially.
from refit_worker import refit,merge
for rep in range(20):refit(rep,L)
merge()
