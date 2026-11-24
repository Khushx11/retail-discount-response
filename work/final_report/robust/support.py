from pathlib import Path
import sys,json
import numpy as np,pandas as pd
from scipy import sparse
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import pairwise_distances_argmin_min
ROOT=Path('outputs/COMP5064_Demo_and_Results').resolve();sys.path[:0]=[str(ROOT/'vendor'),str(ROOT/'src')]
import pipeline as p
O=Path('work/final_report/robust');L=pd.read_parquet(ROOT/'out/features.parquet');tr,va,te=[L[L.split==k].reset_index(drop=True) for k in ['train','val','test']];s=p.fit_schema(tr)
def labels(df):
 z=df[p.CAT].astype(str).copy()
 for c in p.CAT:z[c]=z[c].where(z[c].isin(s['vocab'][c]),'__UNSEEN__')
 return z
e=OneHotEncoder(categories=[s['vocab'][c]+['__UNSEEN__'] for c in p.CAT],handle_unknown='error',dtype=np.float32);ct=e.fit_transform(labels(tr))/np.sqrt(2)
def mat(df,cat=None):
 x,_=p.transform(df,s);return sparse.hstack([sparse.csr_matrix(x),e.transform(labels(df))/np.sqrt(2) if cat is None else cat],format='csr')
X=mat(tr,ct);groups=tr.groupby(['category','t'],observed=True).indices
rng=np.random.default_rng(20261120);v=va.iloc[np.sort(rng.choice(len(va),10000,replace=False))].reset_index(drop=True);t=te.iloc[np.sort(rng.choice(len(te),10000,replace=False))].reset_index(drop=True)
def distances(q,d=None):
 q=q.copy()
 if d is not None:q['disc']=d;q['disc_change']=d-q.disc_prev;q['disc_missing']=0
 Q=mat(q);out=np.full(len(q),np.inf);counts=np.zeros(len(q),int)
 for key,ii in q.groupby(['category','t'],observed=True).indices.items():
  jj=np.asarray(groups.get(key,[]));
  if not len(jj):continue
  # Split observed validation values into exact unique discount levels.
  vals=[d] if d is not None else q.iloc[ii].disc.unique()
  for level in vals:
   iq=ii if d is not None else ii[np.isclose(q.iloc[ii].disc.to_numpy(),level)]
   cand=jj[(tr.iloc[jj].disc.to_numpy()-level).__abs__()<=.025];counts[iq]=len(cand)
   if len(cand):_,dd=pairwise_distances_argmin_min(Q[iq],X[cand],metric='euclidean');out[iq]=dd
 return out,counts
vd,vc=distances(v);finite=vd[np.isfinite(vd)];cut=float(np.quantile(finite,.95));rows=[]
for level in np.arange(7)/10:
 dd,n=distances(t,float(level));rows.append(dict(discount=float(level),sample_rows=len(t),no_candidates_share=float((n==0).mean()),outside_validation_distance_share=float((dd>cut).mean()),median_finite_distance=float(np.median(dd[np.isfinite(dd)]))));print('SUPPORT',rows[-1],flush=True)
pd.DataFrame(rows).to_csv(O/'joint_support.csv',index=False);p.save_json(O/'joint_support_protocol.json',dict(seed=20261120,validation_rows=len(v),test_sample=len(t),validation_no_candidates=int((vc==0).sum()),validation_q95_distance=cut,numeric_columns=s['columns'],categorical=p.CAT,metric='Euclidean after train-only numeric standardization; one-hot categories scaled so one mismatch contributes squared distance one',conditioning='exact category and life week, discount within .025; all numerical and categorical model inputs',limitation='Distance is a heuristic coverage diagnostic; validation-calibrated threshold is not a counterfactual validity guarantee. Finite validation distances define q95; unavailable candidates are outside support.'))
