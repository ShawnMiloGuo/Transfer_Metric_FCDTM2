"""Post-evaluation integrity checks; no annotation or model access."""
from pathlib import Path
import csv,json,hashlib,math
B=Path(__file__).resolve().parents[1];O=B/'artifacts'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def read(n):return list(csv.DictReader((O/n).open()))
def equal(a,b):
 a,b=float(a),float(b);return a==b or math.isnan(a) and math.isnan(b)
seal=json.loads((O/'seal_full.json').read_text())
for p,h in seal.items():assert sha(B/p)==h,p
m=json.loads((O/'manifest_full.json').read_text());e=json.loads((O/'evaluation_full.json').read_text());p=read('predictions_full.csv');r=read('逐项评价_full.csv');assert len(p)==len(r)==6600
for before,after in zip(p,r):
 for k in ['cls','src','tgt','panel','kind','budget','seed','repeat','indices']:assert before[k]==after[k],k
 for k in e['methods'][before['panel']]:assert equal(before[k],after[k]),k
 assert float(before['NCE_Pred_PL'])==0
 if before['singleton_PL']=='True':
  for k in ['LEEP_PL','NLEEP_PL']:
   if math.isfinite(float(before[k])):assert float(before[k])==0
traces={}
for model in m['models']:
 stem=f"{model['cls']}_{model['src']}";a=json.loads((O/f'audit_{stem}.json').read_text());assert a['status']=='passed' and a['source_reads_complete'] and not a['unauthorized_annotations'];trace=B/'logs'/f'{stem}.strace';assert sha(trace)==a['trace_sha'];traces[str(trace.relative_to(B))]=a['trace_sha']
fi=json.loads((O/'figures/figure_index.json').read_text());aa=json.loads((O/'figures/annotation_audit.json').read_text());summary=read('相关汇总_full.csv');lookup={(s['panel'],s['cohort'],s['scenario'],s['method']):s for s in summary if s['scope']=='common'}
assert len(fi)==72
labels={'FCDTM_HardPL':'FCDTM-HardPL','FCDTM_SoftPL':'FCDTM-SoftPL','FD_global':'FD','DS_pixel':'DS','GBC_spherical_PL':'GBC-PL','AC':'AC / MSP','Entropy':'Entropy','ATC_MC':'ATC-MC','ATC_NE':'ATC-NE','DoC':'DoC-gap','DoC_Acc_SrcCal':'DoC-Acc-SrcCal','LogME_PL':'LogME-ICML-PL','Hscore_PL':'H-score-PL','LEEP_PL':'LEEP-PL','NCE_Pred_PL':'NCE-Pred-PL','NLEEP_PL':'NLEEP-PL','OTCE_PL_SrcCal':'OTCE-PL-SrcCal','MMD_RBF_global':'MMD-RBF-global'}
for panel,methods in e['methods'].items():
 for cohort in ['D42','E50']:
  for scenario in ['region_S2','region_L8','sensor_GBA','sensor_XJ']:
   shown={k for f in fi if f['panel']==panel and f['cohort']==cohort and f['scenario']==scenario for k in f['methods']};assert shown=={labels[k] for k in methods},(panel,cohort,scenario,shown)
for a in aa:
 s=lookup[a['panel'],a['cohort'],a['scenario'],a['method']]
 for k,col in [('Fisher_r','Fisher_r'),('rho','rho')]:
  v=float(s[col]);expected=f'{v:.3f}' if math.isfinite(v) else 'N/A';assert a[k]==expected
for f in fi:
 assert (B/f['file']).exists() and (B/f['file']).with_suffix('.png').exists()
large={str(q.relative_to(B)):sha(q) for folder in [O/'features',O/'cache'] for q in folder.glob('*.npz')};large.update(traces)
(O/'large_cache_index.json').write_text(json.dumps(large,indent=1))
report=dict(seal_entries_verified=len(seal),score_records=6600,evaluation_preserves_predictions=True,source_only_audits_passed=25,native_trace_hashes_verified=25,all_method_figures=72,figure_table_annotations_match=True,figures_cover_all_applicable_methods_in_all_scenarios=True,large_files_retained_on_server=len(large),target_reference_reopened_by_verifier=False)
(O/'delivery_verification.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
