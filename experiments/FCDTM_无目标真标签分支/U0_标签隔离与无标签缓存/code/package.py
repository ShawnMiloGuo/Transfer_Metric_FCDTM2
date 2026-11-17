"""Post-evaluation replay and archive; never changes the frozen predictor."""
import json,platform,sys
from collections import defaultdict
import numpy as np
import common as c
def key(r):return (int(r['cls']),r['src'],r['tgt'],int(r['budget']),int(r['repeat']))
def main():
    c.check()
    for p,h in json.loads((c.OUT/'seal.json').read_text()).items():assert c.sha(c.BR/p)==h,p
    risk=c.read(c.U2/'artifacts/predictions.csv');score={key(r):r for r in c.read(c.U1/'artifacts/predictions.csv')};assert len(risk)==len(score)==800
    m=json.loads((c.OUT/'manifest.json').read_text());by=defaultdict(list)
    for r in risk:by[key(r)[:3]].append(r)
    replay=0;maxdiff=0.
    for t in m['tasks']:
        cls,src,tgt=t['cls'],t['src'],t['tgt'];source=json.loads((c.OUT/f'source/{cls}_{src}.json').read_text());sf=source['source_F1'];draws={(b,rep):ix for b,rep,ix in c.draws(tgt)}
        with np.load(c.OUT/f'unlabeled_cache/{cls}_{src}__{tgt}.npz') as z:
            assert z['names'].tolist()==[r['name'] for r in m['pools'][tgt]]
            for r in by[(cls,src,tgt)]:
                ix=np.array(json.loads(r['indices']));assert np.array_equal(ix,draws[int(r['budget']),int(r['repeat'])]);assert r['indices']==score[key(r)]['indices']
                assert float(r['source_only'])==sf
                for mode in c.RISK:
                    value=sf if mode=='source_only' else c.f1(z['risk_'+mode][ix]);pred=float(r[mode]);assert np.isclose(value,pred,equal_nan=True,atol=1e-12);assert np.isclose(float(r[mode+'_AD']),sf-pred,equal_nan=True,atol=1e-12)
                    if np.isfinite(value):maxdiff=max(maxdiff,abs(value-pred))
                replay+=1
    sels=c.read(c.U2/'artifacts/选择汇总.csv');ad=c.read(c.U2/'artifacts/AD汇总.csv');overall={r['method']:r for r in ad if r['partition']=='overall'};sm={r['method']:r for r in sels if r['budget']=='0'};whole={r['method']:r for r in sels if r['budget']=='40'}
    coverage=[]
    for folder,methods in [(c.U1,['hard','soft','uniform_hard','confidence','neg_entropy','source_only','margin']),(c.U2,c.RISK+['confidence','neg_entropy'])]:
        choices=c.read(folder/'artifacts/模型选择.csv');keys={method:{(r['cls'],r['tgt'],r['budget'],r['repeat']) for r in choices if r['method']==method and np.isfinite(float(r['regret'])) and int(r['budget'])<40} for method in methods};common=set.intersection(*keys.values());rows=[]
        for method in methods:
            rs=[r for r in choices if r['method']==method and (r['cls'],r['tgt'],r['budget'],r['repeat']) in common];rows.append(dict(method=method,configs=len(rs),regret=float(np.mean([float(r['regret']) for r in rs])) if rs else np.nan,Top1=float(np.mean([float(r['Top1']) for r in rs])) if rs else np.nan))
        c.save(folder/'artifacts/主方法共同选择.csv',rows);coverage.append(dict(experiment=folder.name,common_configs=len(common),potential=300))
    check=json.loads((c.OUT/'最终核验.json').read_text());check.update(risk_forecast_replayed=replay,max_replay_difference=maxdiff,AD_identity_verified=True,shared_indices_verified=True,main_comparison_coverage=coverage)
    (c.OUT/'预测复算核验.json').write_text(json.dumps(check,ensure_ascii=False,indent=1))
    import scipy,torch,matplotlib
    from osgeo import gdal
    (c.OUT/'环境与版本.json').write_text(json.dumps(dict(client_archive_date='2026-11-17',python=sys.version,numpy=np.__version__,scipy=scipy.__version__,torch=torch.__version__,matplotlib=matplotlib.__version__,gdal=gdal.VersionInfo(),platform=platform.platform(),precompute_model_hashes_frozen=True,input_raster_content_hashes_precompute=False),ensure_ascii=False,indent=1))
    fmt=lambda x:f'{float(x):.6f}'
    intro='\n## 首轮结果的判断\n\n'
    extra1=intro+'硬伪标签原式变体选择遗憾0.049632、Top-1 57.67%；软版0.044501/61.33%。相同硬分组无权重为0.116714/31.00%，说明头权重方向在这一原式变体中含有排序信息。硬/软的遗憾低于原生置信度0.064275，但命中率低于置信度64.00%；不能只报有利指标。\n\n最强无标签选择基线是源6维错误代理ridge6（0.023341/74.00%），原式替代尚未超过它；margin-only风险也更强（0.034999/72.00%）。soft尚无同软分组的无权重消融，不能把全部soft收益归给权重。原生分数不做AD数值宣称。\n'
    extra2=intro+f"加权双侧风险AD MAE={fmt(overall['weighted']['MAE'])}，低于源先验{fmt(overall['source_only']['MAE'])}，但高于同容量无权重{fmt(overall['uniform']['MAE'])}和margin-only {fmt(overall['margin']['MAE'])}。加权选模遗憾{fmt(sm['weighted']['regret'])}，也未超过无权重{fmt(sm['uniform']['regret'])}、margin-only {fmt(sm['margin']['regret'])}或ridge6 {fmt(sm['ridge6']['regret'])}。本轮不支持新增权重距离的独立增益。\n\n"
    extra2+='跨传感器加权MAE0.097624，略低于margin-only的0.103390，但几乎与无权重0.097710相同；跨地域加权0.292302，高于margin-only的0.277418。不能据跨传感器局部改善就宣称头权重必需。40图完整池加权MAE0.184737，无权重0.182704，margin-only 0.182663；覆盖更充分仍未产生独立优势。\n\n'
    extra2+='8/16/24的真实评价对象是各自子池，40图是完整部署池；随着图像数增加指标变化也包含真实子池组成变化，不能把下降曲线解释成固定总体上已证明收敛。没有目标标签的source-risk不具E13目标残差校正，数值不得直接与少标签旧实验作同条件比较。\n'
    for folder,extra in [(c.U1,extra1),(c.U2,extra2)]:
        path=folder/'实验结果分析.md';text=path.read_text().split(intro)[0];path.write_text(text+extra+'\n主比较样本数、全部失败与40图结果见CSV。所有主比较均300/300，不存在靠删除失败候选获得本轮主要增益。主比较口径说明在评价前已写于本地，但因网络超时补传较晚；本轮全部方法交集同样完整，口径没有改变所报主数值。\n')
    audits=[json.loads(p.read_text()) for p in c.OUT.glob('audit_*.json')];assert len(audits)==25 and all(r['status']=='passed' and r['source_reads_complete'] and not r['unauthorized_annotations'] and not r['historical_data_paths'] for r in audits)
    source_models=[json.loads(p.read_text()) for p in (c.OUT/'source').glob('*.json')]
    source_diag=dict(models=len(source_models),missing_fit_class=sum(r['missing_source_class'] for r in source_models),models_relative_floor_triggered=sum(r['prototype_floors']>0 for r in source_models),total_relative_floors=sum(r['prototype_floors'] for r in source_models))
    (c.OUT/'源支持诊断.json').write_text(json.dumps(source_diag,ensure_ascii=False,indent=1))
    auditpath=c.BASE/'实验结果分析.md';auditpath.write_text(auditpath.read_text()+'\n最终确认：25进程均exit0；25系统调用审计通过且各自实际读到64个源标注白名单。目标annotation打开次数0，旧目标缓存打开次数0。800个风险预测和AD/抽样重建通过，最大差0；源1600+目标2000次神经网络图像推理。\n\n说明：审计记录实际打开，不提供OS层强沙箱；标签评价另进程在封存后进行。原始图像内容没有在运行前逐文件哈希，输入路径和模型哈希已固定，该限制明确记录。\n')
    root=['# 零目标真标签分支：首轮综合结果','','已完成U0/U1/U2及U3揭盲后首轮机制诊断，全部新文件位于本分支独立子目录。U4仅完成库存可用性审计，没有进行独立性能确认。', '', '## 关键对照（8/16/24无标签图像预算等权）','', '|方法|AD MAE|选择遗憾|Top-1|','|---|---:|---:|---:|']
    for method in ['source_only','confidence','hard','soft','uniform_hard','ridge6','weighted','uniform','margin']:
        root.append('|'+method+'|'+(fmt(overall[method]['MAE']) if method in overall else '无原生AD单位')+'|'+fmt(sm[method]['regret'])+'|'+fmt(sm[method]['Top1'])+'|')
    root+=['','原式硬伪标签加权方向有一定排序价值；新双侧风险降低了源先验误差，但未超过margin-only/同容量无权重，故未证实新权重模块的独立收益。ridge6是本轮最强平均选模基线，不能忽略。','', '## 标签隔离与局限','', '25worker底层访问审计通过；目标真实标签只在封存后的独立评价及U3诊断读取。此为开发池重推理，非独立区域验证。高置信错误诊断见U3；该诊断不能用来调阈值后仍声称前瞻有效。','', '## 后续约束','', '先保留本轮冻结结果，分析源条件概率为什么跨域失效、原式何处确有权重贡献；不立即增加模块或扫伪标签阈值。新疆L8历史未用图名仅4张，不能照搬40图四域独立确认。U3完整增强一致性/伪标签筛选未执行，U4性能确认待新数据或预登记限定范围协议。']
    (c.BR/'首轮实验综合分析.md').write_text('\n'.join(root)+'\n')
    todo=c.BR/'TODO.md';text=todo.read_text();text=text.replace('- [ ] U0：','- [x] U0：').replace('- [ ] U1：','- [x] U1：').replace('- [ ] U2：','- [x] U2：');text+='\n## 本轮执行记录（2026-11-17）\n\n- [x] U0：25个访问审计通过，800预测复算通过，结果与系统调用日志独立归档。\n- [x] U1/U2：800子池预测、320多候选配置（主预算300+40图20）、图表与结果分析完成。\n- [x] U3：仅完成封存后2000逐图伪标签/失配诊断；未完成增强一致性及筛选新实验。\n- [x] U4：完成纯图名库存审计，新疆L8仅剩4图，不满足四域40图确认协议。\n- [ ] U4性能确认：需补充独立数据或预登记其他范围，不将历史池重抽称为独立确认。\n';todo.write_text(text)
    readme=c.BR/'README.md';text=readme.read_text();text=text.replace('当前仅完成实验设计，尚未执行U0–U4或产生新的性能结果。','已完成首轮U0/U1/U2及U3封存后诊断，详见[首轮综合分析](首轮实验综合分析.md)。U4只完成库存审计，未进行独立性能确认。');readme.write_text(text)
    for folder in [c.BASE,c.U1,c.U2]:
        (folder/'TODO.md').write_text('# 本实验清单\n\n- [x] 协议、代码与输入清单冻结。\n- [x] 源/目标角色与无目标标签预测访问审计。\n- [x] 预测封存、独立评价与指标复算。\n- [x] 代码、CSV、图表、结果分析与版本归档。\n')
    files=sorted(p for p in c.BR.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name!='文件校验.json');(c.BR/'文件校验.json').write_text(json.dumps({str(p.relative_to(c.BR)):c.sha(p) for p in files},ensure_ascii=False,indent=1));print('PACKAGED',len(files),'files; all frozen predictions unchanged')
if __name__=='__main__':main()
