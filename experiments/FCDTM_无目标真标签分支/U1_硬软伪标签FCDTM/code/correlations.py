"""Descriptive within-class correlations after prediction sealing; no fitting."""
import json,sys
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
BR=Path(__file__).resolve().parents[2];sys.path.insert(0,str(BR/'U0_标签隔离与无标签缓存/code'));import common as c
BASE=Path(__file__).resolve().parents[1]
def main():
    for p,h in json.loads((c.OUT/'seal.json').read_text()).items():assert c.sha(BR/p)==h,p
    data=c.read(c.U2/'artifacts/逐次评价.csv');rows=[];methods=['hard','soft','uniform_hard','confidence','margin','ridge6']
    for budget in [8,16,24,40]:
        for repeat in range(1 if budget==40 else 5):
            for cls in range(1,9):
                part=[r for r in data if int(r['budget'])==budget and int(r['repeat'])==repeat and int(r['cls'])==cls]
                for method in methods:
                    valid=[r for r in part if np.isfinite(float(r[method])) and np.isfinite(float(r['true_F1']))];x=np.array([float(r[method]) for r in valid]);y=np.array([float(r['true_F1']) for r in valid]);x=-x if method in ['hard','soft','uniform_hard'] else x
                    rho=float(spearmanr(x,y).statistic) if len(valid)>=3 and np.std(x)>1e-12 and np.std(y)>1e-12 else np.nan
                    rows.append(dict(budget=budget,repeat=repeat,cls=cls,method=method,task_points=len(valid),rho=rho))
    c.save(BASE/'artifacts/类别内相关.csv',rows)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    matrix=[]
    for method in methods:
        row=[]
        for cls in range(1,9):
            means=[]
            for budget in [8,16,24]:
                values=[r['rho'] for r in rows if r['method']==method and r['cls']==cls and r['budget']==budget and np.isfinite(r['rho'])]
                if values:means.append(float(np.mean(values)))
            row.append(float(np.mean(means)) if means else np.nan)
        matrix.append(row)
    fig,ax=plt.subplots(figsize=(10,4.5));cmap=plt.get_cmap('coolwarm').copy();cmap.set_bad('.85');im=ax.imshow(np.ma.masked_invalid(matrix),cmap=cmap,vmin=-1,vmax=1,aspect='auto');ax.set(xticks=np.arange(8),xticklabels=['Crop','Forest','Grass','Shrub','Wetland','Water','Built','Bare'],yticks=np.arange(len(methods)),yticklabels=methods)
    for i,row in enumerate(matrix):
        for j,v in enumerate(row):ax.text(j,i,f'{v:.2f}' if np.isfinite(v) else 'N/A',ha='center',va='center',fontsize=8)
    fig.colorbar(im,ax=ax,label='Within-class Spearman (descriptive)');fig.tight_layout()
    for ext in ['png','pdf']:fig.savefig(BASE/('artifacts/class_correlation.'+ext),dpi=180)
    plt.close(fig)
    report=BASE/'实验结果分析.md';marker='\n## 类别内相关性补充\n';text=report.read_text().split(marker)[0];report.write_text(text+marker+'\n每个预算/重复内，按类别在不同源—目标任务间计算Spearman，再对重复及8/16/24预算等权。分数方向均统一为高值表示高目标F1。少于3任务或常量时N/A，不填0；灰色不是失败分数。40图相关另存CSV。\n\n这不是同目标候选组的选择正确率，跨任务/源模型尺度仍可能影响相关；重复共享数据，不作显著性检验。不将混类高相关代替同类选模证据。\n\n![类别内描述性相关](artifacts/class_correlation.png)\n')
    print('Descriptive correlations archived; no predictions changed')
if __name__=='__main__':main()
