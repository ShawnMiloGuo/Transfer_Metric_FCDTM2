"""Post-evaluation typography only; frozen forecasts and metrics stay unchanged."""
from collections import defaultdict
import batch4 as b
import matplotlib
matplotlib.use('Agg')
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator
original_save = Figure.savefig
def readable_save(fig, *args, **kwargs):
    for ax in fig.axes:
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
        ax.tick_params(labelsize=9)
    return original_save(fig, *args, **kwargs)
Figure.savefig = readable_save
tasks = defaultdict(list)
for r in b.c.read(b.OUT/'散点数据.csv'):
    for name in b.c.RISK+b.c.SCORE+['source_F1','true_F1','true_AD']:
        r[name] = float(r[name])
    tasks[int(r['cls']),r['src'],r['tgt']].append(r)
summary = b.c.read(b.OUT/'相关汇总.csv')
for r in summary:
    r['defined_tasks'] = int(r['defined_tasks'])
    for name in ['Pearson','Spearman']:r[name] = float(r[name])
b.plot(tasks,summary)
print('Exported 54 plot pairs; forecasts and metrics unchanged')
