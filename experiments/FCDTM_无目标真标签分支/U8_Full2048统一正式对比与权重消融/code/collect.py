# -*- coding: utf-8 -*-
"""Score-only merge; complete all methods before permitting evaluation."""
from common import *
import math
score_guard();m=json.loads((OLD/'artifacts/manifest_full.json').read_text());seal=json.loads((OLD/'artifacts/seal_full.json').read_text())
for p,h in seal.items():assert sha(OLD/p)==h,p
keys=['cls','src','tgt','panel','kind','budget','seed','repeat'];uid=lambda r:tuple(str(r[k]) for k in keys)
old={uid(r):r for r in read(OLD/'artifacts/predictions_full.csv') if r['panel']=='full2048'};new={};ot={};audits=[]
for x in m['models']:
 stem=f'{x["cls"]}_{x["src"]}'
 for folder,target in [('workers',new),('ot_workers',ot)]:
  audit=json.loads((O/folder/f'{stem}_audit.json').read_text());assert audit['target_reference_reads']==0 and audit['guard_probe_blocked'] and not audit['unexpected_denials'];audits.append(audit)
  for name,h in audit['code_sha'].items():assert sha(B/'code'/name)==h,name
  for p,h in audit['inputs'].items():assert sha(p)==h,p
  for r in read(O/folder/f'{stem}.csv'):assert uid(r) not in target;target[uid(r)]=r
assert set(new)==set(old)==set(ot) and len(old)==3300
rows=[]
for k,r in old.items():
 a=new[k];b=ot[k];assert r['indices']==a['indices']==b['indices']
 for method in METHODS:
  if method in ['OTCE_PL_SrcCal','MMD_RBF_global']:continue
  aa,bb=float(a[method]),float(r[method]);assert aa==bb or math.isnan(aa) and math.isnan(bb),(k,method)
 row={**a,'OTCE_PL_SrcCal':b['OTCE_PL_SrcCal'],'OTCE_approximation':'Q1024_full_input_product_lift','OTCE_source_points':b['source_points'],'OTCE_target_points':b['target_points'],'OTCE_marginal_error':b['marginal_error']};rows.append(row)
save(O/'predictions_sealed.csv',rows)
dump(O/'manifest.json',dict(methods=METHODS,ablation=ABL,cohorts=m['cohorts'],tasks=m['tasks'],models=[{k:x[k] for k in ['cls','src','ck','sha']} for x in m['models']],panels=['full2048'],source_images=64,target_pool_images=40,pixels_per_image=2048,records=len(rows),old_scores_preserved=True,OTCE_approximate=True,target_reference_reads=0,removed_results=['NCE_Pred_PL','ATC_NE']))
dump(O/'score_audit.json',dict(target_reference_reads=0,records=len(rows),worker_audits=len(audits),all_inputs_and_code_verified=True,old_native_scores_exactly_preserved=True,OTCE_full_input_point_counts=True))
paths=[O/'predictions_sealed.csv',O/'manifest.json',O/'score_audit.json',B/'协议.md',B/'OTCE近似协议.md']+list((B/'code').glob('*.py'));dump(O/'score_seal.json',{str(p.relative_to(B)):sha(p) for p in paths});print('ALL_SCORES_SEALED',len(rows),flush=True)
