"""
combined_results_and_charts.py
==============================
Generates all figures from hardcoded final results (7 datasets).
Run this in Colab as ONE cell — no need to re-run experiments.

Outputs (auto-downloaded):
  fig1_heatmap_7ds.png
  fig2_line_7ds.png
  fig3_wiki_detail.png
  fig4_combined_7ds.png
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

METHODS  = ['BM25', 'DPR', 'Hybrid-RRF', 'Hybrid-α']
COLORS   = ['#2C5F8A', '#1E5C2E', '#8B6914', '#7B1D1D']
DATASETS = ['TriviaQA', 'HotpotQA', 'MS MARCO', 'Natural Q', 'SQuAD 2.0', 'FEVER', 'Wikipedia\n2023']
MARKERS  = ['o','s','^','D']
LINES    = ['-','--','-.', ':']

DATA = {
    'TriviaQA':        {'corpus':947,   'n':500, 'MRR':[0.8021,0.9195,0.8837,0.9262], 'CI_lo':[0.7727,0.8970,0.8582,0.9037], 'CI_hi':[0.8319,0.9399,0.9081,0.9449], 'p':[None,0.000,0.000,0.000]},
    'HotpotQA':        {'corpus':2986,  'n':500, 'MRR':[0.8778,0.8619,0.9079,0.8986], 'CI_lo':[0.8506,0.8358,0.8852,0.8740], 'CI_hi':[0.9036,0.8898,0.9312,0.9220], 'p':[None,0.278,0.001,0.082]},
    'MS MARCO':        {'corpus':13868, 'n':500, 'MRR':[0.3586,0.5886,0.5016,0.5787], 'CI_lo':[0.3257,0.5546,0.4676,0.5440], 'CI_hi':[0.3891,0.6225,0.5361,0.6120], 'p':[None,0.000,0.000,0.000]},
    'Natural Q':       {'corpus':596,   'n':500, 'MRR':[0.8481,0.9580,0.9205,0.9615], 'CI_lo':[0.8218,0.9403,0.8987,0.9475], 'CI_hi':[0.8769,0.9718,0.9417,0.9763], 'p':[None,0.000,0.000,0.000]},
    'SQuAD 2.0':       {'corpus':500,   'n':500, 'MRR':[0.7200,0.6849,0.7688,0.7675], 'CI_lo':[0.6854,0.6505,0.7337,0.7360], 'CI_hi':[0.7567,0.7181,0.8048,0.7999], 'p':[None,0.059,0.000,0.001]},
    'FEVER':           {'corpus':542,   'n':461, 'MRR':[0.7539,0.9520,0.8591,0.9561], 'CI_lo':[0.7171,0.9364,0.8335,0.9409], 'CI_hi':[0.7882,0.9675,0.8826,0.9695], 'p':[None,0.000,0.000,0.000]},
    'Wikipedia\n2023': {'corpus':13000, 'n':500, 'MRR':[0.9140,0.9207,0.9370,0.9232], 'CI_lo':[0.8927,0.8995,0.9185,0.9012], 'CI_hi':[0.9332,0.9440,0.9545,0.9443], 'p':[None,0.526,0.005,0.367]},
}

def sig(p):
    if p is None: return ''
    if p < 0.001: return '***'
    if p < 0.01:  return '**'
    if p < 0.05:  return '*'
    return 'n.s.'

def ds_label(ds): return ds.replace('\n',' ') + '\n(' + str(DATA[ds]['corpus']) + ')'

mrr_matrix = np.array([[DATA[ds]['MRR'][i] for ds in DATASETS] for i in range(4)])
x = np.arange(7)

# Fig 1 — Heatmap
fig, ax = plt.subplots(figsize=(15, 5)); fig.patch.set_facecolor('#FAFAFA')
im = ax.imshow(mrr_matrix, cmap='RdYlGn', aspect='auto', vmin=0.3, vmax=1.0)
plt.colorbar(im, ax=ax, label='MRR', fraction=0.02, pad=0.01)
ax.set_xticks(range(7)); ax.set_xticklabels([ds_label(ds) for ds in DATASETS], fontsize=9)
ax.set_yticks(range(4)); ax.set_yticklabels(METHODS, fontsize=11, fontweight='bold')
for i in range(4):
    for j, ds in enumerate(DATASETS):
        v = DATA[ds]['MRR'][i]; p = DATA[ds]['p'][i]
        tc = 'white' if v < 0.55 or v > 0.90 else 'black'
        ax.text(j, i, str(round(v,3)) + '\n' + sig(p), ha='center', va='center',
                fontsize=8, fontweight='bold', color=tc)
        if i == np.argmax(mrr_matrix[:,j]):
            ax.add_patch(plt.Rectangle((j-0.49,i-0.49),0.98,0.98,fill=False,edgecolor='#FFD700',linewidth=3))
ax.set_title('Figure 1: MRR Heatmap — All 7 Datasets x 4 Methods', fontsize=12, fontweight='bold', pad=12)
plt.tight_layout(); plt.savefig('fig1_heatmap_7ds.png', dpi=160, bbox_inches='tight', facecolor='#FAFAFA')
plt.show(); print('Fig1 saved')

# Fig 2 — Line trend
fig, ax = plt.subplots(figsize=(16, 6)); fig.patch.set_facecolor('#FAFAFA')
for i,(method,color,marker,ls) in enumerate(zip(METHODS,COLORS,MARKERS,LINES)):
    vals  = [DATA[ds]['MRR'][i]   for ds in DATASETS]
    ci_lo = [DATA[ds]['CI_lo'][i] for ds in DATASETS]
    ci_hi = [DATA[ds]['CI_hi'][i] for ds in DATASETS]
    ax.plot(x, vals, color=color, marker=marker, linewidth=2.5, markersize=9,
            linestyle=ls, label=method, markeredgecolor='white', markeredgewidth=1.5)
    ax.fill_between(x, ci_lo, ci_hi, color=color, alpha=0.10)
    for j,v in enumerate(vals):
        offset = 0.025 if i%2==0 else -0.045
        ax.text(j, v+offset, str(round(v,3)), ha='center', fontsize=8, color=color, fontweight='bold')
ax.set_xticks(x); ax.set_xticklabels([ds_label(ds) for ds in DATASETS], fontsize=9)
ax.set_ylabel('MRR', fontsize=12); ax.set_ylim(0.25, 1.10)
ax.set_title('Figure 2: MRR Trends — All 7 Datasets (shaded = 95% CI)', fontsize=13, fontweight='bold')
ax.legend(fontsize=11, ncol=4, loc='upper center')
ax.set_facecolor('#F8F9FA'); ax.grid(alpha=0.3, linestyle='--')
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
ax.axvspan(3.5, 6.5, alpha=0.05, color='#7B1D1D')
ax.text(5.0, 0.28, 'New datasets', ha='center', fontsize=9, color='#7B1D1D', style='italic', fontweight='bold')
plt.tight_layout(); plt.savefig('fig2_line_7ds.png', dpi=160, bbox_inches='tight', facecolor='#FAFAFA')
plt.show(); print('Fig2 saved')

# Fig 3 — Wikipedia detail
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('Figure 3: Wikipedia 2023 — Detailed Analysis', fontsize=13, fontweight='bold')
fig.patch.set_facecolor('#FAFAFA')
wiki_mrr=[0.9140,0.9207,0.9370,0.9232]; wiki_lo=[0.8927,0.8995,0.9185,0.9012]
wiki_hi=[0.9332,0.9440,0.9545,0.9443];  wiki_d=[0,0.0067,0.0229,0.0091]
ax=axes[0]; lo=[m-c for m,c in zip(wiki_mrr,wiki_lo)]; hi=[c-m for m,c in zip(wiki_mrr,wiki_hi)]
bars=ax.bar(METHODS,wiki_mrr,color=COLORS,edgecolor='white',width=0.6,yerr=[lo,hi],capsize=5,ecolor='#333',error_kw={'linewidth':1.5})
bars[2].set_edgecolor('#FFD700'); bars[2].set_linewidth(3)
ax.axhline(y=wiki_mrr[0],color='#2C5F8A',linestyle='--',alpha=0.4,linewidth=1.5)
ax.set_title('MRR with 95% CI',fontweight='bold',fontsize=11); ax.set_ylim(0.85,0.98)
ax.set_ylabel('MRR',fontsize=10); ax.tick_params(axis='x',rotation=15,labelsize=9)
ax.set_facecolor('#F8F9FA'); ax.grid(axis='y',alpha=0.3,linestyle='--')
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
for bar,v in zip(bars,wiki_mrr): ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+0.002,str(round(v,4)),ha='center',fontsize=9,fontweight='bold')
ax=axes[1]; pvals=[1.0,0.526,0.005,0.367]
pval_labels=['BM25\nbaseline','DPR\np=0.526\nn.s.','Hybrid-RRF\np=0.005\n**','Hybrid-a\np=0.367\nn.s.']
ax.bar(range(4),[-np.log10(max(p,0.001)) for p in pvals],color=['#AAAAAA','#E07070','#1E5C2E','#E07070'],edgecolor='white',width=0.6)
ax.axhline(y=-np.log10(0.05),color='red',linestyle='--',linewidth=1.5,label='p=0.05')
ax.set_xticks(range(4)); ax.set_xticklabels(pval_labels,fontsize=8)
ax.set_ylabel('-log10(p-value)',fontsize=10); ax.set_title('Significance vs BM25',fontweight='bold',fontsize=11)
ax.legend(fontsize=8); ax.set_facecolor('#F8F9FA'); ax.grid(axis='y',alpha=0.3,linestyle='--')
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
ax=axes[2]; delta_colors=['#AAAAAA','#E07070','#1E5C2E','#E07070']
bars=ax.bar(METHODS,wiki_d,color=delta_colors,edgecolor='white',width=0.6)
ax.axhline(y=0,color='black',linewidth=0.8)
ax.set_title('Delta MRR vs BM25',fontweight='bold',fontsize=11); ax.set_ylabel('Delta MRR',fontsize=10)
ax.tick_params(axis='x',rotation=15,labelsize=9); ax.set_facecolor('#F8F9FA'); ax.grid(axis='y',alpha=0.3,linestyle='--')
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
for bar,v in zip(bars,wiki_d): ax.text(bar.get_x()+bar.get_width()/2,v+0.001,str(round(v,4)),ha='center',fontsize=9,fontweight='bold')
plt.tight_layout(); plt.savefig('fig3_wiki_detail.png',dpi=160,bbox_inches='tight',facecolor='#FAFAFA')
plt.show(); print('Fig3 saved')

# Fig 4 — Combined master
fig = plt.figure(figsize=(24, 30)); fig.patch.set_facecolor('#FAFAFA')
fig.text(0.5,0.985,'Complete RAG Retrieval Comparison — All 7 Datasets',ha='center',fontsize=20,fontweight='bold',color='#1A3A5C')
fig.text(0.5,0.972,'BM25 vs DPR vs Hybrid-RRF vs Hybrid-α | 2,961 Total Queries | 95% Bootstrap CI',ha='center',fontsize=11,color='#555555',style='italic')
ax1=fig.add_axes([0.05,0.825,0.88,0.125])
im=ax1.imshow(mrr_matrix,cmap='RdYlGn',aspect='auto',vmin=0.3,vmax=1.0)
plt.colorbar(im,ax=ax1,label='MRR',fraction=0.012,pad=0.01)
ax1.set_xticks(range(7)); ax1.set_xticklabels([ds_label(ds) for ds in DATASETS],fontsize=9)
ax1.set_yticks(range(4)); ax1.set_yticklabels(METHODS,fontsize=10,fontweight='bold')
for i in range(4):
    for j,ds in enumerate(DATASETS):
        v=DATA[ds]['MRR'][i]; p=DATA[ds]['p'][i]
        tc='white' if v<0.55 or v>0.90 else 'black'
        ax1.text(j,i,str(round(v,3))+'\n'+sig(p),ha='center',va='center',fontsize=8,fontweight='bold',color=tc)
        if i==np.argmax(mrr_matrix[:,j]):
            ax1.add_patch(plt.Rectangle((j-0.49,i-0.49),0.98,0.98,fill=False,edgecolor='#FFD700',linewidth=3))
ax1.set_title('MRR Heatmap — Gold = best per dataset',fontweight='bold',fontsize=11)
bar_pos=[[0.03,0.565,0.20,0.21],[0.24,0.565,0.20,0.21],[0.45,0.565,0.20,0.21],[0.66,0.565,0.20,0.21],
         [0.085,0.315,0.20,0.21],[0.335,0.315,0.20,0.21],[0.585,0.315,0.20,0.21]]
for pos,ds in zip(bar_pos,DATASETS):
    ax=fig.add_axes(pos); d=DATA[ds]; mrr=d['MRR']
    lo=[m-c for m,c in zip(mrr,d['CI_lo'])]; hi=[c-m for m,c in zip(mrr,d['CI_hi'])]
    bars=ax.bar(METHODS,mrr,color=COLORS,edgecolor='white',width=0.6,yerr=[lo,hi],capsize=4,ecolor='#333',error_kw={'linewidth':1.5})
    best=mrr.index(max(mrr)); bars[best].set_edgecolor('#FFD700'); bars[best].set_linewidth(3)
    ax.axhline(y=mrr[0],color='#2C5F8A',linestyle='--',alpha=0.35,linewidth=1.2)
    for bar,v,p,h in zip(bars,mrr,d['p'],hi):
        s=sig(p); lbl=str(round(v,3))+('\n'+s if s else '')
        ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+h+0.012,lbl,ha='center',va='bottom',fontsize=7.5,fontweight='bold')
    ax.set_title(ds.replace('\n',' ')+'\n(n='+str(d['corpus'])+')',fontweight='bold',fontsize=9.5)
    ax.set_ylim(0,1.22); ax.set_ylabel('MRR',fontsize=8); ax.tick_params(axis='x',rotation=22,labelsize=8)
    ax.set_facecolor('#F8F9FA'); ax.grid(axis='y',alpha=0.3,linestyle='--')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
ax3=fig.add_axes([0.05,0.065,0.88,0.215])
for i,(method,color,marker,ls) in enumerate(zip(METHODS,COLORS,MARKERS,LINES)):
    vals=[DATA[ds]['MRR'][i] for ds in DATASETS]; ci_lo=[DATA[ds]['CI_lo'][i] for ds in DATASETS]; ci_hi=[DATA[ds]['CI_hi'][i] for ds in DATASETS]
    ax3.plot(x,vals,color=color,marker=marker,linewidth=2.5,markersize=10,linestyle=ls,label=method,markeredgecolor='white',markeredgewidth=2)
    ax3.fill_between(x,ci_lo,ci_hi,color=color,alpha=0.12)
    for j,v in enumerate(vals):
        offset=0.028 if i%2==0 else -0.048
        ax3.text(j,v+offset,str(round(v,3)),ha='center',fontsize=8.5,color=color,fontweight='bold')
ax3.set_xticks(x); ax3.set_xticklabels([ds_label(ds) for ds in DATASETS],fontsize=9)
ax3.set_ylabel('MRR',fontsize=12); ax3.set_ylim(0.22,1.12)
ax3.set_title('MRR Trends Across All 7 Datasets — Shaded = 95% CI',fontweight='bold',fontsize=12)
ax3.legend(fontsize=10,ncol=4,loc='upper center',bbox_to_anchor=(0.5,1.02),framealpha=0.9)
ax3.set_facecolor('#F8F9FA'); ax3.grid(alpha=0.3,linestyle='--')
ax3.spines['top'].set_visible(False); ax3.spines['right'].set_visible(False)
ax3.axvspan(3.5,6.5,alpha=0.05,color='#7B1D1D')
ax3.text(5.0,0.26,'New datasets',ha='center',fontsize=10,color='#7B1D1D',style='italic',fontweight='bold')
for j,ds in enumerate(DATASETS):
    best_i=np.argmax(mrr_matrix[:,j]); best_v=mrr_matrix[best_i,j]
    ax3.annotate('★',xy=(j,best_v),xytext=(j,best_v+0.06),ha='center',fontsize=13,color='#B8860B',fontweight='bold')
patches=[mpatches.Patch(color=c,label=m) for c,m in zip(COLORS,METHODS)]
fig.legend(handles=patches,loc='lower center',ncol=4,fontsize=12,bbox_to_anchor=(0.5,0.005),framealpha=0.95)
plt.savefig('fig4_combined_7ds.png',dpi=150,bbox_inches='tight',facecolor='#FAFAFA')
plt.show(); print('Fig4 saved')

# Download
try:
    from google.colab import files
    for fname in ['fig1_heatmap_7ds.png','fig2_line_7ds.png','fig3_wiki_detail.png','fig4_combined_7ds.png']:
        files.download(fname); print('Downloaded: '+fname)
except:
    print('Run in Colab to download. Files saved locally.')

print('All done!')
