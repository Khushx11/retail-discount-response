"""Reload final selected checkpoints on one historical context; no fitting."""
from pathlib import Path
import sys,argparse,json
import pandas as pd,numpy as np
sys.path.insert(0,str(Path(__file__).parent));import study as s
import torch
p=s.p;O=s.O;ap=argparse.ArgumentParser();ap.add_argument('--discount',type=float,default=.3);a=ap.parse_args();assert 0<=a.discount<=1,'Discount must be a fraction from zero to one'
L=pd.read_parquet(s.ROOT/'out/features.parquet');x=L[L.split=='test'].iloc[[0]].copy();observed_discount=float(x.disc.iloc[0]);actual=float(x.y.iloc[0]);sc=json.loads((O/'schema.json').read_text());sel=json.loads((O/'selections.json').read_text());x['disc']=a.discount;x['disc_change']=a.discount-x.disc_prev;x['disc_missing']=0
out={'forecast_date':str(x.forecast_date.iloc[0].date()),'planned_discount':a.discount,'historically_observed_discount':observed_discount,'status':'Retrospective saved-model inference; changed discount has no observed counterfactual','predictions':{}}
for name in ['LightGBM Tweedie','Piecewise monotone']:
 slug=name.lower().replace(' ','_')+'_0'
 if name.startswith('LightGBM'):m=s.lgb.Booster(model_file=str(O/f'{slug}.txt'));pred=m.predict(p.tree_frame(x,p.NUM+p.CAT,sc))[0]
 else:m=s.Piecewise(sc,sel[name]['config']);m.load_state_dict(torch.load(O/f'{slug}.pt',map_location='cpu',weights_only=True));pred=p.nn_predict(m,p.tensors(x,sc),'poisson')[0]
 out['predictions'][name]=float(pred)
if np.isclose(a.discount,observed_discount):out['actual_at_observed_discount']=actual
print(json.dumps(out,indent=2))
