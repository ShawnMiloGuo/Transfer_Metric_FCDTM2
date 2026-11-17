"""Role-aware filename audit. Never opens target annotations or NPZ truth arrays."""
import csv,json,re
from collections import defaultdict
from pathlib import Path
import numpy as np
BR=Path(__file__).resolve().parents[2];EXPS=BR.parent;DATA=Path('/home/Shanxin.Guo/ZhangtuosCode/1_dataset/dataset/xj_landsat8/train_val/images');RX=re.compile(r'xjl8_i\d+j\d+\.tif')
def main():
    roles={k:defaultdict(set) for k in ['source','target','unclassified']};cache_target=set();cache_source=set()
    def walk(x,path,file):
        if isinstance(x,dict):
            for k,v in x.items():walk(v,path+'/'+k,file)
        elif isinstance(x,list):
            for v in x:walk(v,path+'[]',file)
        elif isinstance(x,str):
            names=RX.findall(x)
            target=any(k in path for k in ['/calibration','/cal/','/cal[]','/test/','/test[]','/pools/','/targets/'])
            source=any(k in path for k in ['/source','/ref'])
            role='target' if target else 'source' if source else 'unclassified'
            for n in names:roles[role][n].add(file+':'+path)
    evidence=[]
    for exp in sorted(EXPS.glob('E*')):
        if not re.match(r'E\d+[AB]?_',exp.name):continue
        for p in sorted(exp.rglob('*.json')):
            if p.name not in ['manifest.json','protocol.json']:continue
            walk(json.loads(p.read_text()),'',str(p.relative_to(EXPS)))
        for p in sorted(exp.rglob('*.npz')):
            if '__' not in p.stem:continue
            left,right=p.stem.split('__',1)
            if not right.startswith('xj_landsat8'):continue
            with np.load(p,allow_pickle=False) as z:
                if 'names' not in z.files:continue
                names={n for v in z['names'].tolist() for n in RX.findall(str(v))}
            role='source' if left.endswith('xj_landsat8') else 'target'
            if role=='target':cache_target.update(names)
            else:cache_source.update(names)
            for n in names:roles[role][n].add(str(p.relative_to(EXPS))+':NPZ.names')
    files={p.name:p for split in ['train','val'] for p in sorted((DATA/split).iterdir()) if p.is_file() and p.suffix.lower() in ['.tif','.tiff']};alln=set(files)
    source=set(roles['source'])&alln;target=set(roles['target'])&alln
    old=EXPS/'E17_源域伪迁移校准与新网格验证/artifacts';strict=set(json.loads((old/'historical_names.json').read_text())['xj_landsat8']);e17=json.loads((old/'manifest.json').read_text());e18=json.loads((EXPS/'E18_E13融合模型权重分布差异/artifacts/manifest.json').read_text());strict.update(r['name'] for k in ['cal','test'] for r in e17['targets']['xj_landsat8'][k]);strict.update(r['name'] for r in e18['pools']['xj_landsat8']);strict &= alln
    report=dict(total=len(files),train=sum(p.parent.name=='train' for p in files.values()),val=sum(p.parent.name=='val' for p in files.values()),strict_list_used=len(strict),strict_list_unseen=len(alln-strict),source_reference_used=len(source),target_declared_or_processed=len(target),source_and_target=len(source&target),source_only=len(source-target),not_used_as_target=len(alln-target),target_cache_evidence=len(cache_target&alln),target_cache_unseen=len(alln-cache_target),declaration_without_cache_evidence=len(target-cache_target),old_strict_marked_but_no_source_or_target_role=sorted(strict-source-target),target_labels_read=False,strict_unseen_not_guaranteed_never_opened=True)
    out=BR/'U4_确认数据可用性审计/artifacts';out.mkdir(exist_ok=True)
    (out/'使用口径复核.json').write_text(json.dumps(report,ensure_ascii=False,indent=1))
    rows=[dict(name=n,split=files[n].parent.name,strict_used=n in strict,source_reference=n in source,target_history=n in target,target_cache_evidence=n in cache_target,can_be_target_candidate_under_role_rule=n not in target,source_evidence=json.dumps(sorted(roles['source'][n]),ensure_ascii=False),target_evidence=json.dumps(sorted(roles['target'][n]),ensure_ascii=False)) for n in sorted(alln)]
    with (out/'逐图角色与来源.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    (out/'未作目标候选清单.json').write_text(json.dumps([dict(name=n,split=files[n].parent.name) for n in sorted(alln-target)],indent=1))
    text=f'''# 新疆L8历史使用标记复核

原“只剩4张”的表述不完整：它表示不在指定历史source/target/protocol/cache清单里的图名，并不是可作为目标数据的总数，也不能保证从未被库存审计打开过。

本次不读取目标标注，逐图追踪清单角色及NPZ.names证据。总图{report['total']}，train {report['train']}、val {report['val']}；严格历史名单未见{report['strict_list_unseen']}；仅作为源参考而未作目标{report['source_only']}；未作为目标的候选{report['not_used_as_target']}。目标角色历史{report['target_declared_or_processed']}，有实际target缓存names证据{report['target_cache_evidence']}；仅声明但缺target缓存证据{report['declaration_without_cache_evidence']}。

因此，应保留原库存数字但纠正其含义。之前把源参考也全局排除，用这一条件断言无法做U4过于绝对。可以为不在新疆L8训练的候选模型，从“未作目标候选清单”重新预登记目标池；这些图也许被其他模型用作源参考，不能称全研究从未见过，更不能直接称独立地理区域。

U4下一步应按“具体候选模型、目标角色、训练足迹、已参与方法开发程度”定义可用性，检查图像重复与空间重叠后固定清单。若严格要求整个研究从未用过，则目前该口径仍只有{report['strict_list_unseen']}张；若只要求未参与目标评分，则候选明显更多。不得把两种确认强度混淆。原U0/U1/U2成绩与冻结清单不改变。

完整逐图证据见artifacts/逐图角色与来源.csv。新候选没有通过标签或精度筛选；本次仅审计，尚未运行新的U4性能实验。
'''
    (out.parent/'标记复核说明.md').write_text(text)
    for rel in ['首轮实验综合分析.md','README.md','TODO.md']:
        p=BR/rel;s=p.read_text();marker='\n## U4口径更正\n';s=s.split(marker)[0];p.write_text(s+marker+f"\n原4张仅指全局历史清单未见，不是目标可用图总数。本次角色审计：总{report['total']}，未作目标候选{report['not_used_as_target']}，源参考专用{report['source_only']}。此前据4张直接判断不能做U4过于绝对；先按候选训练角色/空间重叠预登记确认方案。详见[标记复核](U4_确认数据可用性审计/标记复核说明.md)。原实验结果不变，新U4尚未运行。\n")
    print(json.dumps(report,ensure_ascii=False,indent=1))
if __name__=='__main__':main()
