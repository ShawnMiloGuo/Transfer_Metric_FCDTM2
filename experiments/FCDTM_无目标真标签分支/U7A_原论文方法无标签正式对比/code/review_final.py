from pathlib import Path
import csv,json,math,hashlib
from collections import defaultdict
u=Path(__file__).resolve().parents[1]
def read(name):return list(csv.DictReader((u/'artifacts'/name).open()))
rows=read('逐项评价.csv');groups=defaultdict(list)
for r in rows:
 if r['kind']=='batch4':groups[int(r['cls']),r['src'],r['tgt'],int(r['seed'])].append(r)
def pearson(x,y):
 if len(x)<4 or max(x)==min(x) or max(y)==min(y):return float('nan')
 x=[(v-min(x))/(max(x)-min(x)) for v in x];y=[(v-min(y))/(max(y)-min(y)) for v in y];mx=sum(x)/len(x);my=sum(y)/len(y);a=[v-mx for v in x];b=[v-my for v in y];return math.fsum(p*q for p,q in zip(a,b))/math.sqrt(math.fsum(p*p for p in a)*math.fsum(q*q for q in b))
def ranks(v):
 order=sorted(range(len(v)),key=lambda i:v[i]);out=[0.]*len(v);i=0
 while i<len(v):
  j=i+1
  while j<len(v) and v[order[j]]==v[order[i]]:j+=1
  for k in order[i:j]:out[k]=(i+j-1)/2+1
  i=j
 return out
checked=0
for r in read('逐任务相关.csv'):
 key=int(r['cls']),r['src'],r['tgt'],int(r['seed']);method=r['method'];rr=[x for x in groups[key] if r['scope']=='own' or x['common']=='True'];direction=-1 if method in ['DS_pixel','GBC_spherical_PL','GBC_diagonal_PL'] else 1;pts=[(direction*float(x[method]),float(x['true_AD'])) for x in rr if math.isfinite(float(x[method])) and math.isfinite(float(x['true_AD']))];x=[v[0] for v in pts];y=[v[1] for v in pts];assert len(pts)==int(r['K'])
 pr=pearson(x,y) if pts else float('nan');sr=pearson(ranks(x),ranks(y)) if pts else float('nan')
 for actual,record in [(pr,float(r['Pearson'])),(sr,float(r['Spearman']))]:assert (math.isnan(actual) and math.isnan(record)) or abs(actual-record)<1e-9,(key,method,actual,record)
 checked+=1
assert checked==4000
plots=list((u/'artifacts/figures').glob('*.png'));assert len(plots)==54 and len(list((u/'artifacts/figures').glob('*.pdf')))==54
result=dict(status='passed',correlation_rows_independently_recomputed=checked,PNG_PDF_pairs=54,visual_QA='four D42 summaries and representative task inspected',prediction_method_changes=False)
(u/'artifacts/最终复核.json').write_text(json.dumps(result,indent=1));print(json.dumps(result))
