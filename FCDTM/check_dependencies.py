#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
依赖检查脚本

检查所有必要的Python包是否已安装。
"""

def check_dependencies():
    """检查所有依赖"""
    print("="*60)
    print("检查依赖包")
    print("="*60)
    
    dependencies = {
        'torch': 'PyTorch',
        'numpy': 'NumPy',
        'scipy': 'SciPy',
        'matplotlib': 'Matplotlib',
        'tqdm': 'tqdm',
        'torchmetrics': 'TorchMetrics',
        'cv2': 'OpenCV',
        'PIL': 'Pillow',
    }
    
    missing = []
    
    for module, name in dependencies.items():
        try:
            __import__(module)
            print(f"  ✓ {name} 已安装")
        except ImportError:
            print(f"  ✗ {name} 未安装")
            missing.append(name)
    
    if missing:
        print(f"\n缺失的依赖包: {', '.join(missing)}")
        print("\n请安装缺失的依赖:")
        print(f"  pip install {' '.join(m.lower() for m in missing)}")
        return False
    else:
        print("\n✓ 所有依赖包已安装!")
        return True


if __name__ == "__main__":
    import sys
    sys.exit(0 if check_dependencies() else 1)
