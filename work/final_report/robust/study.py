from pathlib import Path
import sys,json,time,hashlib,copy
import numpy as np,pandas as pd
ROOT=Path('outputs/COMP5064_Demo_and_Results').resolve();sys.path[:0]=[str(ROOT/'vendor'),str(ROOT/'src')]
import pipeline as p
import lightgbm as lgb
import torch
from torch import nn
O=Path('work/final_report/robust');O.mkdir(exist_ok=True)
CONFIGS=[dict(hidden=[128,64],drop=.1),dict(hidden=[256,128,64],drop=.1),dict(hidden=[256,128,64],drop=.3),dict(hidden=[512,256,128],drop=.2),dict(hidden=[64,32],drop=.1),dict(hidden=[128,128],drop=.2),dict(hidden=[256,128],drop=.2),dict(hidden=[256,256,128],drop=.2)]
TREES=[dict(num_leaves=n,min_data_in_leaf=m,lambda_l2=l) for n,m,l in [(15,100,1),(31,100,1),(63,200,1),(127,200,1),(31,300,5),(63,100,5),(31,200,10),(63,300,10)]]
KNOTS=[0.,.15,.30,.45]
class Piecewise(nn.Module):
 def __init__(self,schema,cfg):
  super().__init__();base=p.MLP(schema,**cfg);self.embs=base.embs;self.body=base.body;self.mask=[schema['columns'].index(c) for c in ['disc','disc_change']];width=cfg['hidden'][-1];self.baseline=nn.Linear(width,1);self.slopes=nn.Linear(width,4)
  nn.init.constant_(self.slopes.weight,.001);nn.init.constant_(self.slopes.bias,.05)
 def forward(self,x,c,d):
  x=x.clone();x[:,self.mask]=0;h=self.body(torch.cat([x]+[e(c[:,i]) for i,e in enumerate(self.embs)],1));b=self.slopes(h);basis=torch.stack([(d-k).clamp(min=0,max=.15 if j<3 else 1.-k) for j,k in enumerate(KNOTS)],1);return self.baseline(h)[:,0]+(b*basis).sum(1),b
 def project(self):
  with torch.no_grad():self.slopes.weight.clamp_(min=0);self.slopes.bias.clamp_(min=0)

def fit_nn(Ttr,Tva,s,cfg,family,seed=0,weights=None):
 torch.manual_seed(seed);np.random.seed(seed);torch.use_deterministic_algorithms(True)
 piece=family.startswith('Piecewise');m=Piecewise(s,cfg) if piece else p.MLP(s,**cfg)
 head=m.baseline if piece else m.head
 loss='mse' if family=='MLP MSE' else 'poisson'
 if loss=='poisson':nn.init.constant_(head.bias,float(torch.log(Ttr[3].mean())))
 opt=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=1e-5);sched=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,factor=.5,patience=1)
 lf=nn.MSELoss(reduction='none') if loss=='mse' else nn.PoissonNLLLoss(log_input=True,reduction='none')
 best=float('inf');state=None;bad=0;hist=[];epoch=0
 for ep in range(20):
  m.train();ixs=torch.randperm(len(Ttr[0]));total=0
  for i in range(0,len(ixs),2048):
   ix=ixs[i:i+2048];v,_=m(*(x[ix] for x in Ttr[:3]));ell=lf(v,Ttr[3][ix]);ell=ell.mean() if weights is None else (ell*weights[ix]).mean();opt.zero_grad();ell.backward();opt.step()
   if family=='Piecewise monotone':m.project()
   total+=float(ell.detach())*len(ix)
  v=p.wape(Tva[3].numpy(),p.nn_predict(m,Tva,loss));sched.step(v);hist.append(dict(epoch=ep+1,loss=total/len(ixs),val_WAPE=v))
  if v<best-1e-4:best=v;state=copy.deepcopy(m.state_dict());bad=0;epoch=ep+1
  else:bad+=1
  if bad>=3:break
 m.load_state_dict(state);return m,best,hist,epoch

def fit_tree(tr,va,s,cfg,obj='tweedie',seed=0,features=None,weights=None):
 features=features or p.NUM+p.CAT
 params=dict(objective=obj,metric='None',learning_rate=.05,feature_fraction=.8,bagging_fraction=.8,bagging_freq=1,verbosity=-1,seed=seed,num_threads=2,deterministic=True,force_col_wise=True,**cfg)
 if obj=='tweedie':params['tweedie_variance_power']=1.2
 a=lgb.Dataset(p.tree_frame(tr,features,s),label=tr.y,weight=weights,categorical_feature=[c for c in p.CAT if c in features]);b=lgb.Dataset(p.tree_frame(va,features,s),label=va.y,reference=a)
 m=lgb.train(params,a,3000,valid_sets=[b],feval=lambda pr,d:('WAPE',p.wape(d.get_label(),pr),False),callbacks=[lgb.early_stopping(100,first_metric_only=True,verbose=False)])
 return m,p.wape(va.y,m.predict(p.tree_frame(va,features,s)))

def run():
 start=time.time();L=pd.read_parquet(ROOT/'out/features.parquet');tr,va,te=[L[L.split==k].reset_index(drop=True) for k in ['train','val','test']];s=p.fit_schema(tr);p.save_json(O/'schema.json',s);Ts=[p.tensors(z,s) for z in [tr,va,te]];records=[];tuning=[];pred=te[['external_code','retail','forecast_date','category','t','y','disc','inv_start']].copy();selections={}
 for family in ['LightGBM Tweedie','LightGBM L2','MLP Poisson','MLP MSE','Piecewise monotone','Piecewise free']:
  best=None;tree=family.startswith('LightGBM');configs=TREES if tree else CONFIGS;obj='regression' if family=='LightGBM L2' else 'tweedie'
  for j,cfg in enumerate(configs):
   print('SEARCH',family,j,flush=True)
   if tree:m,v=fit_tree(tr,va,s,cfg,obj);h=[];ep=m.best_iteration
   else:m,v,h,ep=fit_nn(Ts[0],Ts[1],s,cfg,family)
   tuning.append(dict(family=family,trial=j,config=cfg,val_WAPE=v,best_iteration=ep,history=h));p.save_json(O/'tuning.json',tuning)
   if best is None or v<best[0]:best=(v,cfg,m,ep)
  selections[family]=dict(config=best[1],val_WAPE=best[0]);p.save_json(O/'selections.json',selections)
  for seed in [0,1,2]:
   if seed==0:v,cfg,m,ep=best
   elif tree:m,v=fit_tree(tr,va,s,best[1],obj,seed);ep=m.best_iteration
   else:m,v,h,ep=fit_nn(Ts[0],Ts[1],s,best[1],family,seed)
   pr=m.predict(p.tree_frame(te,p.NUM+p.CAT,s)) if tree else p.nn_predict(m,Ts[2],'mse' if family=='MLP MSE' else 'poisson')
   r=p.metrics(te,pr,family,seed,v);r['best_iteration']=ep;records.append(r);slug=family.lower().replace(' ','_')+f'_{seed}';pred[f'{family}|{seed}']=pr
   if tree:m.save_model(str(O/f'{slug}.txt'))
   else:torch.save(m.state_dict(),O/f'{slug}.pt')
   p.save_json(O/f'{slug}.json',dict(family=family,seed=seed,config=best[1],val_WAPE=v,best_iteration=ep));pd.DataFrame(records).to_csv(O/'results.csv',index=False);pred.to_parquet(O/'predictions.parquet',index=False);print('RESULT',r,flush=True)
 # Same-configuration projected/free control, selected via monotone validation.
 cfg=selections['Piecewise monotone']['config'];freecfg=selections['Piecewise free']['config']
 for seed in [0,1,2]:
  if cfg==freecfg:pr=pred[f'Piecewise free|{seed}'].to_numpy();v=next(r['val_WAPE'] for r in records if r['model']=='Piecewise free' and r['seed']==seed)
  else:
   m,v,h,ep=fit_nn(Ts[0],Ts[1],s,cfg,'Piecewise free',seed);pr=p.nn_predict(m,Ts[2],'poisson');torch.save(m.state_dict(),O/f'paired_free_{seed}.pt')
  records.append(p.metrics(te,pr,'Paired free control',seed,v));pred[f'Paired free control|{seed}']=pr
 # Retrain ablations at the validation-selected Tweedie settings.
 for name,remove in [('Tree no discount',p.DISC),('Tree no stock',p.STOCK)]:
  fs=[f for f in p.NUM+p.CAT if f not in remove];m,v=fit_tree(tr,va,s,selections['LightGBM Tweedie']['config'],features=fs);pr=m.predict(p.tree_frame(te,fs,s));records.append(p.metrics(te,pr,name,0,v));pred[name+'|0']=pr;m.save_model(str(O/(name.lower().replace(' ','_')+'.txt')))
 for name,pr in [('Naive last week',te.lag1.to_numpy()),('Naive mean so far',te.mean_sofar.to_numpy())]:records.append(p.metrics(te,pr,name));pred[name]=pr
 pd.DataFrame(records).to_csv(O/'results.csv',index=False);pred.to_parquet(O/'predictions.parquet',index=False)
 p.save_json(O/'completed.json',dict(seconds=time.time()-start,trials=len(tuning),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),status='Retrospective exploratory extension; not fresh holdout'));print('MAIN COMPLETE',flush=True)
if __name__=='__main__':run()
