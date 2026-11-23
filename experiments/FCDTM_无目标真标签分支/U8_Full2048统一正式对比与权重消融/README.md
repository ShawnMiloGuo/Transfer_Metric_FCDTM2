# U8 Full2048统一正式对比与权重消融

先阅读实验结果分析.md。25模型、50任务、3300记录、16正式条目及4个消融分数；35对PNG/PDF图。所有预测入口均禁止目标真实标签。

代码：worker.py/kernel.py计算精确分块MMD与消融；ot_worker.py/ot_quantized.py计算全输入Q1024可行运输近似并重新源域校准；finalize.py先封存后评价；verify.py独立SciPy复核；outputs.py --paper /本地/fcdtm-label-free生成结果表图。

原缓存与完整kernel_sums/ot_cache保存在服务器本实验artifacts下，未上传大型缓存或依赖。旧U7B、U7R评分不覆盖。失败OTCE代码和日志在archive中。

论文及本地集成脚本保存在fcdtm-label-free；不向公开仓库上传完整论文。实验代码与结果发布独立codex/fcdtm-label-free分支。

正文全方法连续图：outputs.py后运行main_all_methods.py /本地/fcdtm-label-free；图6–9各两页8面板。
