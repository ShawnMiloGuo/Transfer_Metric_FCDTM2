# -*- coding: utf-8 -*-
from pathlib import Path
import csv,json,hashlib,sys,os
B=Path(__file__).resolve().parents[1];BR=B.parent;OLD=BR/'U7B_审稿新增度量与伪标签适配';O=B/'artifacts'
METHODS=['FCDTM_HardPL','FCDTM_SoftPL','FD_global','DS_pixel','GBC_spherical_PL','AC','Entropy','ATC_MC','DoC','DoC_Acc_SrcCal','LogME_PL','Hscore_PL','LEEP_PL','NLEEP_PL','OTCE_PL_SrcCal','MMD_RBF_global']
ABL=['FCDTM_HardSigned','FCDTM_HardAbs','FCDTM_SoftSigned','FCDTM_SoftAbs']
BAD={'FCDTM_HardPL','FCDTM_SoftPL','FD_global','Entropy','DoC','OTCE_PL_SrcCal','MMD_RBF_global',*ABL}
LABEL=dict(zip(METHODS,['FCDTM-HardPL','FCDTM-SoftPL','FD','DS','GBC-PL','AC / MSP','Entropy','ATC-MC','DoC-gap','DoC-Acc-SrcCal','LogME-ICML-PL','H-score-PL','LEEP-PL','NLEEP-PL','OTCE-PL-SrcCal','MMD-RBF-global']))
LABEL.update(dict(zip(ABL,['Hard: feature difference','Hard: difference + abs','Soft: feature difference','Soft: difference + abs'])))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return list(csv.DictReader(Path(p).open()))
def save(p,rows):
 Path(p).parent.mkdir(parents=True,exist_ok=True)
 with Path(p).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def dump(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def score_guard():
 denied=[]
 def hook(e,a):
  if e=='open' and a and isinstance(a[0],(str,bytes,os.PathLike)):
   p=os.fsdecode(a[0])
   if '/annotations/' in p or any(v in p for v in ['逐项评价','逐任务相关','评价真值','true_F1','evaluation_full.json']):denied.append(p);raise PermissionError('U8 score may not read target reference/evaluation')
 sys.addaudithook(hook)
 try:open('/not-a-real-path/annotations/U8_probe')
 except PermissionError:pass
 assert len(denied)==1
 return denied
