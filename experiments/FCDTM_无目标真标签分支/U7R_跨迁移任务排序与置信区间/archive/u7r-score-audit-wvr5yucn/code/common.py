from pathlib import Path
import csv,json,hashlib,math
B=Path(__file__).resolve().parents[1];O=B/'artifacts';OLD=B.parent/'U7B_审稿新增度量与伪标签适配'/'artifacts'
METHODS=['FCDTM_HardPL','FCDTM_SoftPL','FD_global','DS_pixel','GBC_spherical_PL','AC','Entropy','ATC_MC','ATC_NE','DoC','DoC_Acc_SrcCal','LogME_PL','Hscore_PL','LEEP_PL','NCE_Pred_PL','NLEEP_PL','OTCE_PL_SrcCal','MMD_RBF_global']
BAD={'FCDTM_HardPL','FCDTM_SoftPL','FD_global','Entropy','DoC','OTCE_PL_SrcCal','MMD_RBF_global'}
PANELS={'full2048':[k for k in METHODS if k not in ['OTCE_PL_SrcCal','MMD_RBF_global']],'shared256':METHODS}
LABEL=dict(zip(METHODS,['FCDTM-HardPL','FCDTM-SoftPL','FD','DS','GBC-PL','AC / MSP','Entropy','ATC-MC','ATC-NE','DoC-gap','DoC-Acc-SrcCal','LogME-ICML-PL','H-score-PL','LEEP-PL','NCE-Pred-PL','NLEEP-PL','OTCE-PL-SrcCal','MMD-RBF-global']))
SCENARIOS=['all','region_S2','region_L8','sensor_GBA','sensor_XJ'];COHORTS=['D42','P37','E50'];REPS=5000;SEED=20261123
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return list(csv.DictReader(Path(p).open()))
def save(p,rows):
 with Path(p).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def scenario(src,tgt):return ('region_S2' if 'sentinel2' in src else 'region_L8') if src.split('_')[0]!=tgt.split('_')[0] else ('sensor_GBA' if src.startswith('dwq') else 'sensor_XJ')
def fmt(x,n=3):return f'{float(x):.{n}f}' if math.isfinite(float(x)) else 'N/A'
