"""Read-only manuscript/task audit. No image annotations or model inference."""
import argparse,csv,hashlib,json,re
from pathlib import Path
from zipfile import ZipFile
import xml.etree.ElementTree as E

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--manuscript',required=True);ap.add_argument('--bib',required=True);ap.add_argument('--reviews',required=True);ap.add_argument('--branch',required=True);args=ap.parse_args()
    manuscript=Path(args.manuscript);bib=Path(args.bib);reviews=Path(args.reviews);br=Path(args.branch);out=br/'U7_原论文与审稿新增方法正式对比/artifacts';out.mkdir(parents=True,exist_ok=True)
    text=manuscript.read_text();classes={'Cropland':1,'Forest':2,'Grassland':3,'Shrub':4,'Wetland':5,'Water':6,'Built-up':7,'Bareland':8};regions={'GBA':'dwq','XJ':'xj'};sensors={'Sentinel-2':'sentinel2','Landsat-8':'landsat8'};records=[]
    for label in ['cross_region','cross_sensor','ablation_region','ablation_sensor']:
        block=next(x for x in re.findall(r'\\begin\{table\*?\}.*?\\end\{table\*?\}',text,re.S) if '\\label{tab:'+label+'}' in x)
        for line in block.splitlines():
            if not re.match(r'^(GBA|XJ|Sentinel-2|Landsat-8)\s*&',line):continue
            a,b,c,cl=[x.strip() for x in line.split('&')[:4]]
            if 'region' in label:src=regions[a]+'_'+sensors[c];tgt=regions[b]+'_'+sensors[c]
            else:src=regions[c]+'_'+sensors[a];tgt=regions[c]+'_'+sensors[b]
            records.append(dict(table=label,cls=classes[cl],class_name=cl,src=src,tgt=tgt))
    expected=[]
    for sensor,cl in [('sentinel2',[1,2,3,6,7,8]),('landsat8',[1,6,7,8])]:
        for s,t in [('dwq','xj'),('xj','dwq')]:
            for k in cl:expected.append((k,s+'_'+sensor,t+'_'+sensor))
    for region,cl in [('dwq',[1,2,6,7,8]),('xj',[1,3,5,6,7,8])]:
        for s,t in [('sentinel2','landsat8'),('landsat8','sentinel2')]:
            for k in cl:expected.append((k,region+'_'+s,region+'_'+t))
    existing=json.loads((br/'U0_标签隔离与无标签缓存/artifacts/manifest.json').read_text());available={(t['cls'],t['src'],t['tgt']) for t in existing['tasks']};published={(r['cls'],r['src'],r['tgt']) for r in records if r['table'] in ['cross_region','cross_sensor']}
    counts={k:sum(r['table']==k for r in records) for k in sorted({r['table'] for r in records})}
    inv=[]
    for k in sorted(set(expected)|published|available):inv.append(dict(cls=k[0],src=k[1],tgt=k[2],published_main=k in published,declared_III_C=k in expected,current_manifest=k in available))
    with (out/'论文与当前任务对照.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(inv[0]));w.writeheader();w.writerows(inv)
    with ZipFile(reviews) as z:
        root=E.fromstring(z.read('word/document.xml'));ns={'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main'};paragraphs=[]
        for i,p in enumerate(root.findall('.//w:body//w:p',ns)):
            t=''.join(x.text or '' for x in p.findall('.//w:t',ns))
            if t.strip():paragraphs.append(f'P{i:03d} '+t)
    (out/'审稿意见逐段提取.txt').write_text('\n'.join(paragraphs))
    snapshot=out/'服务器实现只读快照';report=dict(date='2026-11-17',inputs={str(p):sha(p) for p in [manuscript,bib,reviews]},table_task_counts=counts,published_main_tasks=len(published),declared_setup_tasks=len(expected),current_models=len(existing['models']),current_tasks=len(available),declared_missing_from_main=[dict(cls=k[0],src=k[1],tgt=k[2]) for k in sorted(set(expected)-published)],declared_unavailable_current=[dict(cls=k[0],src=k[1],tgt=k[2]) for k in sorted(set(expected)-available)],snapshot_sha={str(p.relative_to(snapshot)):sha(p) for p in snapshot.rglob('*.py')},target_labels_read=False,performance_experiment_run=False)
    (out/'核对证据.json').write_text(json.dumps(report,ensure_ascii=False,indent=2));print(json.dumps({k:v for k,v in report.items() if k not in ['snapshot_sha','inputs']},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
