from pathlib import Path
import time,subprocess,sys,os
b=Path.cwd()
for i in range(1440):
 if (b/"artifacts/evaluation_full.json").exists():break
 time.sleep(10)
else:raise RuntimeError("U7B evaluation not complete after four hours")
subprocess.check_call([sys.executable,"code/write_analysis.py"])
subprocess.check_call([sys.executable,"code/plot_full.py"])
print("POSTPROCESS COMPLETE",flush=True)
