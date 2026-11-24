from pathlib import Path
import json
import pandas as pd,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
O=Path('work/final_report/robust');R=pd.read_csv(O/'summary.csv').set_index('model');C=pd.read_csv(O/'by_category.csv');P=pd.read_parquet(O/'predictions.parquet');tuning=json.loads((O/'tuning.json').read_text());sel=json.loads((O/'selections.json').read_text());plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'axes.titlesize':12.5,'axes.labelsize':12,'legend.fontsize':11.5,'xtick.labelsize':11.5,'ytick.labelsize':11.5,'axes.spines.top':False,'axes.spines.right':False})
colors={'LightGBM Tweedie':'#dc6632','MLP Poisson':'#276bb0','Piecewise monotone':'#13936e','Piecewise free':'#8756a0'}
fig,axs=plt.subplots(1,2,figsize=(8,2.8))
for name,col in [('MLP Poisson','#276bb0'),('Piecewise monotone','#13936e')]:
 tr=min((x for x in tuning if x['family']==name),key=lambda x:x['val_WAPE']);h=tr['history'];axs[0].plot([x['epoch'] for x in h],[x['loss'] for x in h],marker='o',ms=3,color=col,label=name);axs[1].plot([x['epoch'] for x in h],[100*x['val_WAPE'] for x in h],marker='o',ms=3,color=col,label=name);ep=tr['best_iteration'];axs[1].scatter(ep,h[ep-1]['val_WAPE']*100,marker='*',s=90,color=col,edgecolor='black',lw=.4)
axs[0].set_title('(a) Training objective, seed zero');axs[0].set_ylabel('Mean Poisson loss');axs[1].set_title('(b) Validation selection');axs[1].set_ylabel('Validation WAPE (%)');axs[0].legend(frameon=False,fontsize=11.5)
for ax in axs:ax.set_xlabel('Epoch');ax.grid(alpha=.2)
fig.tight_layout();fig.savefig(O/'training.png',dpi=300);plt.close(fig)
fig,axs=plt.subplots(1,2,figsize=(8,2.9),gridspec_kw={'width_ratios':[1,1.25]})
rows=pd.read_csv(O/'results.csv');full=rows[(rows.model=='LightGBM Tweedie')&(rows.seed==0)].iloc[0];vals=[full.WAPE,*[R.loc[x,'WAPE'] for x in ['Tree no stock','Tree no discount','Tree availability check']]];axs[0].barh(['All inputs','No stock','No discount','Conservative'],np.array(vals)*100,color=['#276bb0','#8a9fae','#dc6632','#8756a0']);axs[0].invert_yaxis();axs[0].set_xlim(0,95);axs[0].set_xlabel('Test WAPE (%)');axs[0].set_title('(a) Retrained ablations')
for i,v in enumerate(vals):axs[0].text(v*100+1,i,f'{100*v:.2f}',va='center',fontsize=11.5)
# Importance taken from the actual selected seed-zero tree.
import sys
root=Path('outputs/COMP5064_Demo_and_Results').resolve();sys.path.insert(0,str(root/'vendor'));import lightgbm as lgb
m=lgb.Booster(model_file=str(O/'lightgbm_tweedie_0.txt'));im=pd.DataFrame({'feature':m.feature_name(),'gain':m.feature_importance('gain')});im['share']=100*im.gain/im.gain.sum();im=im.nlargest(7,'gain').iloc[::-1];names={'mean_sofar':'Mean past sales','lag1':'Last-week sales','inv_start':'Estimated stock','disc_change':'Discount change','t':'Product age','retail':'Store','deliv_prev':'Prior delivery','disc':'Discount','opening_stock':'Opening stock','cum_sales':'Cumulative sales','gtrend':'Search trends','price':'Price'};axs[1].barh([names.get(x,x) for x in im.feature],im.share,color='#13936e');axs[1].set_xlabel('Share of split gain (%)');axs[1].set_title('(b) Selected tree importance');fig.tight_layout(w_pad=1.8);fig.savefig(O/'ablations.png',dpi=300);plt.close(fig)
fig,axs=plt.subplots(1,2,figsize=(8,3.5));q=C.groupby(['group','model']).agg(WAPE=('WAPE','mean'),n=('n','first')).reset_index();order=q[q.model=='LightGBM Tweedie'].nlargest(10,'n').group.tolist()
for off,name,marker in [(-.12,'LightGBM Tweedie','o'),(.12,'MLP Poisson','s')]:
 z=q[q.model==name].set_index('group').loc[order];axs[0].plot(z.WAPE*100,np.arange(10)+off,linestyle='none',marker=marker,ms=4,color=colors[name],label='Tweedie' if off<0 else 'Poisson MLP')
axs[0].set_yticks(range(10),order);axs[0].invert_yaxis();axs[0].set_xlabel('Category WAPE (%)');axs[0].set_title('(a) Ten largest categories');axs[0].legend(frameon=False,fontsize=11.5,loc='lower center',bbox_to_anchor=(.5,-.32),ncol=2)
sw=pd.read_csv(O/'sweeps.csv');curve_summary=[]
for name,g in sw.groupby('model',sort=False):
 ss=[]
 for seed,z in g.groupby('seed'):
  a=z.pivot(index='category',columns='discount',values='mean');ss.append(a.div(a[0.],axis=0).mean().to_numpy())
 ss=np.array(ss);avg=ss.mean(0);sd=ss.std(0,ddof=1);label={'LightGBM Tweedie':'Tweedie','MLP Poisson':'Poisson MLP','Piecewise monotone':'Monotone piecewise','Piecewise free':'Free piecewise'}[name];axs[1].plot(np.arange(7)*10,avg,marker='o',ms=3,color=colors[name],label=label);axs[1].fill_between(np.arange(7)*10,avg-sd,avg+sd,color=colors[name],alpha=.12);curve_summary.append(dict(model=name,ratios=avg.tolist(),sd=sd.tolist()))
axs[1].set_xlabel('Scenario discount (%)');axs[1].set_ylabel('Prediction / prediction at 0%');axs[1].set_title('(b) Fixed-history scenarios');axs[1].legend(frameon=False,fontsize=12,loc='upper left');axs[1].set_xticks([0,20,40,60]);fig.tight_layout(w_pad=1.2);fig.subplots_adjust(bottom=.23);fig.savefig(O/'response.png',dpi=300);plt.close(fig);(O/'curves_summary.json').write_text(json.dumps(curve_summary,indent=2))
fig,axs=plt.subplots(1,2,figsize=(8,3));f=pd.DataFrame(json.loads((O/'rolling_folds.json').read_text()));models=['LightGBM Tweedie','MLP Poisson','Piecewise monotone','Paired free control'];short=['Tweedie','Poisson MLP','Monotone','Paired free'];origins=sorted(f.origin.unique())
for name,col in zip(models,['#dc6632','#276bb0','#13936e','#8756a0']):
 z=f[f.model==name].set_index('origin').loc[origins];axs[0].plot(range(3),z.WAPE*100,'o-',label=short[models.index(name)],color=col)
axs[0].set_xticks(range(3),['May 2018','Nov 2018','May 2019'],rotation=15);axs[0].set_ylabel('Period WAPE (%)');axs[0].set_title('(a) Earlier rolling evaluations');axs[0].legend(frameon=False,fontsize=11,loc='upper center',bbox_to_anchor=(.5,-.3),ncol=2)
b=pd.DataFrame(json.loads((O/'refit_replicates.json').read_text()));axs[1].hist(b.gap_pp,bins=8,color='#276bb0',edgecolor='white');axs[1].axvline(0,color='black',ls='--');axs[1].set_xlabel('Poisson minus tree WAPE (pp)');axs[1].set_ylabel('Training resamples');axs[1].set_title('(b) Refit and reselect sensitivity');fig.tight_layout();fig.subplots_adjust(bottom=.33);fig.savefig(O/'robustness.png',dpi=300);plt.close(fig)
D=pd.read_csv(O/'occurrence_metrics.csv');fig,axs=plt.subplots(1,2,figsize=(8,2.8))
for ax,name,title in zip(axs,['LightGBM Tweedie','MLP Poisson'],['Tweedie, validation threshold','Poisson MLP, validation threshold']):
 r=D[(D.model==name)&(D.seed==0)].iloc[0];cm=np.array([[r.TN,r.FP],[r.FN,r.TP]],int);share=cm/cm.sum(1,keepdims=True);ax.imshow(share,cmap='Blues',vmin=0,vmax=1,aspect='auto')
 for i in range(2):
  for j in range(2):ax.text(j,i,f'{cm[i,j]:,}\n{100*share[i,j]:.1f}%',ha='center',va='center',fontsize=12,color='white' if share[i,j]>.65 else 'black')
 ax.set_xticks([0,1],['No sale','Sale']);ax.set_yticks([0,1],['No sale','Sale']);ax.set_xlabel('Predicted');ax.set_ylabel('Actual');ax.set_title(title,fontsize=11.5)
fig.tight_layout();fig.savefig(O/'occurrence.png',dpi=300);plt.close(fig)
fig,axs=plt.subplots(1,2,figsize=(8,2.8));sup=pd.read_csv(O/'joint_support.csv');axs[0].plot(sup.discount*100,sup.outside_validation_distance_share*100,'o-',color='#dc6632');axs[0].set_ylim(0,100);axs[0].set_xlabel('Scenario discount (%)');axs[0].set_ylabel('Outside distance threshold (%)');axs[0].set_title('(a) Joint-input support check');life=pd.read_csv(O/'by_life_week.csv');z=life[life.model=='LightGBM Tweedie'].groupby('group').WAPE.mean();z.index=z.index.astype(int);z=z.sort_index();axs[1].plot(z.index,z*100,'o-',color='#276bb0');axs[1].set_xlabel('Life week');axs[1].set_ylabel('WAPE (%)');axs[1].set_title('(b) Selected tree by product age');fig.tight_layout();fig.savefig(O/'support_life.png',dpi=300);plt.close(fig)
print('Six updated multipanel figures saved')
