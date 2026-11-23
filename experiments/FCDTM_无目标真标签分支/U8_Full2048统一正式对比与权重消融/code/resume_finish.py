from pathlib import Path
import time,subprocess,os,sys
b=Path(__file__).resolve().parents[1]
p=Path('/proc/28112/stat')
while p.exists():
 try:
  if p.read_text().split()[2]=='Z':break
 except FileNotFoundError:break
 time.sleep(2)
subprocess.check_call([sys.executable,str(b/'code/launch_ot.py'),'8'])
subprocess.check_call([sys.executable,str(b/'code/finalize.py')])
