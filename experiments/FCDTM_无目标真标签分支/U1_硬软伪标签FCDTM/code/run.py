"""Shared inference entry point; outputs for U1 and U2 stay in separate folders."""
import json,subprocess,sys
from pathlib import Path
SHARED=Path(__file__).resolve().parents[2]/'U0_标签隔离与无标签缓存'
if __name__=='__main__':
    if (SHARED/'artifacts/seal.json').exists():raise SystemExit('已有封存结果，禁止覆盖；见本实验artifacts和分析。')
    if not (SHARED/'artifacts/manifest.json').exists():subprocess.run([sys.executable,str(SHARED/'code/prepare.py')],check=True)
    subprocess.run([sys.executable,str(SHARED/'code/launch.py')],check=True)
