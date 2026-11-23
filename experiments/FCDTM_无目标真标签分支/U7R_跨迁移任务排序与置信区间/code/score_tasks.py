"""Score-only projection of frozen U7B 40-image outputs; never opens evaluation."""
import sys,os,json,math
from pathlib import Path
from common import *
def main():
 assert not (O/'score_seal.json').exists(),'Task score seal exists: never overwrite.'
 denied=[]
 def guard(e,a):
  if e=='open' and a and isinstance(a[0],(str,bytes,Path)):
   p=os.fsdecode(a[0])
   if '/annotations/' in p or any(t in p for t in ['逐项评价','评价真值','true_F1','evaluation_full.json']):denied.append(p);raise PermissionError('U7R score stage forbids target-reference/evaluation access')
 sys.addaudithook(guard)
 try:(OLD/'逐项评价_full.csv').open();raise AssertionError('Guard probe failed')
 except PermissionError:pass
 oldseal=json.loads((OLD/'seal_full.json').read_text());assert oldseal['artifacts/predictions_full.csv']==sha(OLD/'predictions_full.csv')
 m=json.loads((OLD/'manifest_full.json').read_text());source=read(OLD/'predictions_full.csv');assert len(source)==6600 and all('true_F1' not in r and 'true_AD' not in r for r in source)
 rows=[];seen=set()
 for r in source:
  if r['kind']!='pool' or int(r['budget'])!=40:continue
  assert int(r['seed'])==0 and sorted(json.loads(r['indices']))==list(range(40));key=(r['panel'],int(r['cls']),r['src'],r['tgt']);assert key not in seen;seen.add(key)
  v={k:r[k] for k in ['panel','cls','src','tgt','source_F1','indices']};v['scenario']=scenario(r['src'],r['tgt']);v.update({k:r[k] for k in METHODS});v['score_common']=all(math.isfinite(float(r[k])) for k in PANELS[r['panel']]);v['undefined_methods']=json.dumps([k for k in PANELS[r['panel']] if not math.isfinite(float(r[k]))]);rows.append(v)
 assert len(rows)==100
 save(O/'task_scores_sealed.csv',rows)
 (O/'score_audit.json').write_text(json.dumps(dict(target_annotation_reads=0,evaluation_cache_reads=0,guard_probe_blocked=True,denied=denied,input_score_sha256=sha(OLD/'predictions_full.csv'),source_score_seal_sha256=sha(OLD/'seal_full.json'),task_rows=len(rows),task_score_rule='existing whole 40-image native score; no outcome fitting'),indent=2))
 (O/'task_manifest.json').write_text(json.dumps(dict(cohorts=m['cohorts'],methods=PANELS,bootstrap_replicates=REPS,seed=SEED,primary_block='class',sensitivity_block='source_model',protocol_sha256=sha(B/'协议.md')),ensure_ascii=False,indent=2))
 files=[O/'task_scores_sealed.csv',O/'score_audit.json',O/'task_manifest.json',B/'协议.md']+[B/'code'/n for n in ['common.py','score_tasks.py','ranking_stats.py','evaluate_tasks.py']]
 (O/'score_seal.json').write_text(json.dumps({str(p.relative_to(B)):sha(p) for p in files},indent=2));print('SEALED',len(rows),'task scores before target evaluation',flush=True)
if __name__=='__main__':main()
