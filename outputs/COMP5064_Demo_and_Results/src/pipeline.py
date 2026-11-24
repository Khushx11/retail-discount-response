"""Reproducible retrospective one-week sales forecasting; no cross-store life-week aggregates."""
import argparse, hashlib, json, platform, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
# Load LightGBM before torch: this macOS runtime otherwise initializes incompatible OpenMP state.
import lightgbm as lgb
import torch
from torch import nn
from build_features import build

ROOT = Path(__file__).resolve().parents[1]
NUM = ['t','disc','disc_prev','disc_change','disc_max_prev','weeks_on_disc_prev',
       'lag1','lag2','lag3','mean_sofar','cum_sales','inv_start','deliv_prev','opening_stock',
       'stockout_start','price','rel_month','cal_woy','gtrend','disc_missing']
CAT = ['category','color','fabric','retail','season_type']
STOCK = ['inv_start','deliv_prev','opening_stock','stockout_start']
DISC = ['disc','disc_prev','disc_change','disc_max_prev','weeks_on_disc_prev','disc_missing']
LOG = ['lag1','lag2','lag3','mean_sofar','cum_sales','inv_start','deliv_prev','opening_stock']
LEVELS = [0.,.1,.2,.3,.4,.5,.6]
torch.set_num_threads(2)

def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, default=str, allow_nan=False))

def wape(y,p):
    d=np.asarray(y).sum()
    return float(np.abs(np.asarray(y)-np.asarray(p)).sum()/d) if d>0 else None

def metrics(df,p,name,seed=None,val=None):
    y=df.y.to_numpy();p=np.asarray(p);d=y.sum()
    r=dict(model=name,seed=seed,WAPE=wape(y,p),MAE=float(np.abs(y-p).mean()),Bias=float((p-y).sum()/d) if d else None,val_WAPE=val)
    for key, mask in [('discounted',df.disc>0),('full_price',df.disc==0),('positive_start_stock',df.inv_start>0),('nonpositive_start_stock',df.inv_start<=0)]:
        r['WAPE_'+key]=wape(y[mask],p[mask])
    return r

def group_metrics(df,p,name,seed,column):
    x=df[[column,'y']].copy();x['p']=p
    rows=[]
    for v,z in x.groupby(column,observed=True):
        rows.append(dict(model=name,seed=seed,group=str(v),n=len(z),actual_sum=float(z.y.sum()),WAPE=wape(z.y,z.p),MAE=float(np.abs(z.y-z.p).mean()),Bias=float((z.p-z.y).sum()/z.y.sum()) if z.y.sum() else None))
    return rows

def numeric(df):
    x=df[NUM].copy()
    for c in ['lag2','lag3']:x[c+'_miss']=x[c].isna().astype(float)
    x=x.fillna(0)
    for c in LOG:x[c]=np.sign(x[c])*np.log1p(np.abs(x[c]))
    return x

def fit_schema(tr):
    x=numeric(tr)
    return dict(num=NUM,cat=CAT,log=LOG,columns=x.columns.tolist(),mean=x.mean().to_dict(),std=x.std().replace(0,1).to_dict(),vocab={c:sorted(tr[c].astype(str).unique().tolist()) for c in CAT},forecast_assumption='Planned current discount known; only prior completed-week sales and delivery records used.',split_dates=['2018-11-05','2019-05-06'])

def transform(df,schema):
    x=numeric(df)[schema['columns']]
    x=(x-pd.Series(schema['mean']))/pd.Series(schema['std'])
    c=np.column_stack([df[k].astype(str).map({v:i+1 for i,v in enumerate(schema['vocab'][k])}).fillna(0).to_numpy(dtype=np.int64) for k in CAT])
    return x.to_numpy(dtype=np.float32),c

def tree_frame(df,features,schema):
    x=df[features].copy()
    for c in CAT:
        if c in x:x[c]=pd.Categorical(x[c].astype(str),categories=schema['vocab'][c])
    return x

class MLP(nn.Module):
    def __init__(self,schema,hidden=(256,128,64),drop=.1,elastic=False):
        super().__init__();self.elastic=elastic
        self.disc_idx=[schema['columns'].index(c) for c in ['disc','disc_change']]
        self.embs=nn.ModuleList([nn.Embedding(len(schema['vocab'][c])+1,min(16,(len(schema['vocab'][c])+2)//2),padding_idx=0) for c in CAT])
        width=len(schema['columns'])+sum(e.embedding_dim for e in self.embs);layers=[]
        for h in hidden:layers += [nn.Linear(width,h),nn.ReLU(),nn.Dropout(drop)];width=h
        self.body=nn.Sequential(*layers);self.head=nn.Linear(width,2 if elastic else 1)
    def forward(self,x,c,d):
        if self.elastic:x=x.clone();x[:,self.disc_idx]=0
        z=self.head(self.body(torch.cat([x]+[e(c[:,i]) for i,e in enumerate(self.embs)],1)))
        if self.elastic:
            beta=nn.functional.softplus(z[:,1]);return z[:,0]+beta*d,beta
        return z[:,0],None

def tensors(df,schema):
    x,c=transform(df,schema)
    return tuple(torch.tensor(v) for v in [x,c,df.disc.to_numpy(dtype=np.float32),df.y.to_numpy(dtype=np.float32)])

def nn_predict(model,T,loss):
    model.eval();out=[]
    with torch.no_grad():
        for start in range(0,len(T[0]),16384):
            raw,_=model(*(v[start:start+16384] for v in T[:3]));out.append(torch.exp(raw) if loss=='poisson' else raw.clamp(min=0))
    p=torch.cat(out).numpy();assert np.isfinite(p).all();return p

def train_nn(Ttr,Tva,schema,cfg,loss='poisson',elastic=False,seed=0):
    torch.manual_seed(seed);np.random.seed(seed);torch.use_deterministic_algorithms(True)
    model=MLP(schema,**cfg,elastic=elastic)
    if loss=='poisson':
        nn.init.constant_(model.head.bias,0);model.head.bias.data[0]=float(np.log(Ttr[3].mean()))
    opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=1e-5)
    sched=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,factor=.5,patience=1)
    lf=nn.PoissonNLLLoss(log_input=True) if loss=='poisson' else nn.MSELoss()
    best=float('inf');state=None;bad=0;hist=[];best_epoch=0
    for ep in range(20):
        model.train();perm=torch.randperm(len(Ttr[0]));total=0
        for i in range(0,len(perm),2048):
            ix=perm[i:i+2048];o,_=model(*(v[ix] for v in Ttr[:3]));l=lf(o,Ttr[3][ix]);opt.zero_grad();l.backward();opt.step();total+=float(l.detach())*len(ix)
        val=wape(Tva[3].numpy(),nn_predict(model,Tva,loss));sched.step(val)
        hist.append(dict(epoch=ep+1,train_loss=total/len(perm),val_WAPE=val))
        print(f'epoch {ep+1}: train loss {total/len(perm):.5f}, validation WAPE {val:.5f}',flush=True)
        if val<best-1e-4:
            best=val;state={k:v.detach().clone() for k,v in model.state_dict().items()};bad=0;best_epoch=ep+1
        else:bad+=1
        if bad>=3:break
    model.load_state_dict(state);return model,best,hist,best_epoch

def sweep(df,predict):
    rows=[]
    for d in LEVELS:
        x=df.copy();x['disc']=d;x['disc_change']=d-x.disc_prev;x['disc_missing']=0
        p=predict(x)
        for c,z in pd.DataFrame({'category':x.category.to_numpy(),'p':p}).groupby('category'):
            rows.append(dict(category=c,discount=d,mean_prediction=float(z.p.mean())))
    return pd.DataFrame(rows)

def run(data):
    import lightgbm as lgb
    for d in ['out','models','fig']:(ROOT/d).mkdir(exist_ok=True)
    data=Path(data).resolve();start=time.time()
    protocol=dict(task='One-week-ahead observed product-store sales',split='target-week calendar date; prior labels complete',train_before='2018-11-05',validation_before='2019-05-06',test_status='Retrospective corrected test: historical test data already inspected; not a fresh holdout',removed=['prod_lag1','prod_mean_sofar','target-week realized deliveries'],seeds=[0,1,2],config_selected_on='validation only',created_utc=pd.Timestamp.now(tz='UTC').isoformat())
    save_json(ROOT/'out/protocol.json',protocol)
    L=build(str(data));L.to_parquet(ROOT/'out/features.parquet',index=False)
    tr,va,te=(L[L.split==k].copy() for k in ['train','val','test'])
    schema=fit_schema(tr);save_json(ROOT/'models/preprocessing.json',schema)
    manifest=[]
    for k,x in [('train',tr),('val',va),('test',te)]:
        manifest.append(dict(split=k,rows=len(x),series=int(x.row.nunique()),products=int(x.external_code.nunique()),forecast_min=str(x.forecast_date.min().date()),forecast_max=str(x.forecast_date.max().date()),label_end_max=str(x.label_end.max().date()),mean_sales=float(x.y.mean()),discount_share=float((x.disc>0).mean()),nonpositive_start_stock_share=float((x.inv_start<=0).mean())))
    save_json(ROOT/'out/split_manifest.json',manifest)
    files=['sales.csv','price_discount_series.csv','restocks.csv','vis2_gtrends_data.csv']
    save_json(ROOT/'out/environment.json',dict(python=sys.version,platform=platform.platform(),pandas=pd.__version__,numpy=np.__version__,torch=torch.__version__,lightgbm=lgb.__version__,data_hashes={f:hashlib.sha256((data/f).read_bytes()).hexdigest() for f in files},source_hashes={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in (ROOT/'src').glob('*.py')}))
    print('SPLITS',manifest,flush=True)
    res=[];cats=[];weeks=[];depths=[];preds=te[['external_code','retail','t','forecast_date','category','y','disc','inv_start','stockout_start']].copy().reset_index(drop=True)
    te['discount_bin']=pd.cut(te.disc,[-.001,0,.15,.25,.35,.55,1],labels=['0%','(0,15]%','(15,25]%','(25,35]%','(35,55]%','(55,100]%']).astype(str)
    def record(name,p,seed=None,val=None,**extra):
        assert len(p)==len(te) and np.isfinite(p).all()
        r=metrics(te,p,name,seed,val);r.update(extra);res.append(r)
        cats.extend(group_metrics(te,p,name,seed,'category'));weeks.extend(group_metrics(te,p,name,seed,'t'));depths.extend(group_metrics(te,p,name,seed,'discount_bin'))
        preds[name+(f'|{seed}' if seed is not None else '')]=p
        pd.DataFrame(res).to_csv(ROOT/'out/results.csv',index=False)
        print('RESULT',json.dumps(r),flush=True)
    record('Naive last week',te.lag1.to_numpy());record('Naive mean so far',te.mean_sofar.to_numpy())
    configs=[('LightGBM Tweedie','tweedie',NUM+CAT,False),('LightGBM L2','regression',NUM+CAT,False),('LightGBM no discount','tweedie',[c for c in NUM+CAT if c not in DISC],False),('LightGBM no stock','tweedie',[c for c in NUM+CAT if c not in STOCK],False),('LightGBM monotone','tweedie',NUM+CAT,True)]
    for name,obj,features,mono in configs:
        t0=time.time();params=dict(objective=obj,metric='None',learning_rate=.05,num_leaves=63,min_data_in_leaf=200,feature_fraction=.8,bagging_fraction=.8,bagging_freq=1,lambda_l2=1.,verbosity=-1,seed=0,num_threads=2,deterministic=True,force_col_wise=True)
        if obj=='tweedie':params['tweedie_variance_power']=1.2
        if mono:params.update(monotone_constraints=[int(f in ['disc','disc_change']) for f in features],monotone_constraints_method='advanced')
        dtr=lgb.Dataset(tree_frame(tr,features,schema),label=tr.y,categorical_feature=[c for c in CAT if c in features]);dv=lgb.Dataset(tree_frame(va,features,schema),label=va.y,reference=dtr)
        hist={};m=lgb.train(params,dtr,3000,valid_sets=[dv],valid_names=['validation'],feval=lambda p,d:('WAPE',wape(d.get_label(),p),False),callbacks=[lgb.early_stopping(100,first_metric_only=True,verbose=False),lgb.record_evaluation(hist)])
        p=m.predict(tree_frame(te,features,schema));v=wape(va.y,m.predict(tree_frame(va,features,schema)))
        slug=name.lower().replace(' ','_');m.save_model(str(ROOT/f'models/{slug}.txt'));save_json(ROOT/f'models/{slug}.json',dict(name=name,features=features,params=params,best_iteration=m.best_iteration))
        save_json(ROOT/f'out/history_{slug}.json',hist);record(name,p,seed=0,val=v,iterations=m.best_iteration,secs=round(time.time()-t0,2))
        pd.DataFrame({'feature':features,'gain':m.feature_importance('gain')}).sort_values('gain',ascending=False).to_csv(ROOT/f'out/importance_{slug}.csv',index=False)
        if name!='LightGBM no discount':sweep(te,lambda x:m.predict(tree_frame(x,features,schema))).to_csv(ROOT/f'out/sweep_{slug}.csv',index=False)
    Ttr,Tva,Tte=(tensors(x,schema) for x in [tr,va,te]);tuning=[]
    for cfg in [dict(hidden=[128,64],drop=.1),dict(hidden=[256,128,64],drop=.1),dict(hidden=[256,128,64],drop=.3),dict(hidden=[512,256,128],drop=.2)]:
        print('TUNING',cfg,flush=True);m,b,h,ep=train_nn(Ttr,Tva,schema,cfg);tuning.append(dict(config=cfg,val_WAPE=b,best_epoch=ep,history=h))
    save_json(ROOT/'out/tuning.json',tuning);cfg=min(tuning,key=lambda x:x['val_WAPE'])['config'];save_json(ROOT/'out/nn_best.json',cfg)
    for name,loss,elastic in [('MLP MSE','mse',False),('MLP Poisson','poisson',False),('Constrained discount net','poisson',True)]:
        for seed in [0,1,2]:
            t0=time.time();print('TRAINING',name,seed,flush=True);m,b,h,ep=train_nn(Ttr,Tva,schema,cfg,loss,elastic,seed);p=nn_predict(m,Tte,loss);slug=name.lower().replace(' ','_')+f'_{seed}'
            torch.save(m.state_dict(),ROOT/f'models/{slug}.pt');save_json(ROOT/f'models/{slug}.json',dict(name=name,config=cfg,loss=loss,elastic=elastic,seed=seed,best_epoch=ep));save_json(ROOT/f'out/history_{slug}.json',h)
            record(name,p,seed,b,epochs=len(h),best_epoch=ep,secs=round(time.time()-t0,2));sweep(te,lambda x:nn_predict(m,tensors(x,schema),loss)).to_csv(ROOT/f'out/sweep_{slug}.csv',index=False)
            if elastic:
                with torch.no_grad():_,beta=m(*Tte[:3])
                z=te[['external_code','retail','t','forecast_date','category']].copy();z['beta']=beta.numpy();z.to_parquet(ROOT/f'out/beta_{seed}.parquet',index=False)
    pd.DataFrame(cats).to_csv(ROOT/'out/by_category.csv',index=False);pd.DataFrame(weeks).to_csv(ROOT/'out/by_life_week.csv',index=False);pd.DataFrame(depths).to_csv(ROOT/'out/by_discount.csv',index=False);preds.to_parquet(ROOT/'out/test_predictions.parquet',index=False)
    # Predeclare demo model selection via validation, never select a checkpoint on test error.
    eligible=[r for r in res if r['val_WAPE'] is not None];chosen=min(eligible,key=lambda r:r['val_WAPE']);save_json(ROOT/'models/demo_selection.json',chosen)
    save_json(ROOT/'out/completed.json',dict(seconds=round(time.time()-start),models=len(res),test_rows=len(te),finished_utc=pd.Timestamp.now(tz='UTC').isoformat()))
    print('COMPLETE',time.time()-start,flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--data',required=True);args=ap.parse_args();run(args.data)
