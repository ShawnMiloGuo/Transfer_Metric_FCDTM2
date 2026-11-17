import json
from pathlib import Path
BR=Path(__file__).resolve().parents[2];EXPS=BR.parent;DATA=Path('/home/Shanxin.Guo/ZhangtuosCode/1_dataset/dataset')
def main():
    old=EXPS/'E17_源域伪迁移校准与新网格验证/artifacts';used={d:set(v) for d,v in json.loads((old/'historical_names.json').read_text()).items()};e17=json.loads((old/'manifest.json').read_text());e18=json.loads((EXPS/'E18_E13融合模型权重分布差异/artifacts/manifest.json').read_text());report={}
    for d in used:
        used[d].update(r['name'] for role in ['cal','test'] for r in e17['targets'][d][role]);used[d].update(r['name'] for r in e18['pools'][d]);files=[p for split in ['train','val'] for p in sorted((DATA/d/'train_val/images'/split).glob('*.tif'))];remaining=[p.name for p in files if p.name not in used[d]];report[d]=dict(total=len(files),remaining_unused_names=len(remaining),can_form_40_image_pool=len(remaining)>=40,remaining_names=remaining)
    out=BR/'U4_确认数据可用性审计/artifacts';out.mkdir(exist_ok=True);(out/'库存审计.json').write_text(json.dumps(dict(target_labels_read=False,strict_geographic_independence_verified=False,domains=report),ensure_ascii=False,indent=1));print(json.dumps({d:{k:v for k,v in r.items() if k!='remaining_names'} for d,r in report.items()},indent=1))
if __name__=='__main__':main()
