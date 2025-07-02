#!/usr/bin/env python
"""
EdgeRLHF环境设置脚本
自动检查和配置运行环境
"""

import os
import sys
import subprocess
import importlib
from pathlib import Path

def check_python_version():
    """检查Python版本"""
    print("🐍 检查Python版本...")
    version = sys.version_info
    print(f"   当前Python版本: {version.major}.{version.minor}.{version.micro}")
    
    if version.major != 3 or version.minor < 9:
        print("   ❌ 需要Python 3.9或更高版本")
        return False
    else:
        print("   ✅ Python版本符合要求")
        return True

def check_cuda():
    """检查CUDA环境"""
    print("\n🔧 检查CUDA环境...")
    try:
        import torch
        if torch.cuda.is_available():
            print(f"   ✅ CUDA可用: {torch.version.cuda}")
            print(f"   🎯 GPU设备: {torch.cuda.get_device_name(0)}")
            print(f"   💾 GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
            return True
        else:
            print("   ❌ CUDA不可用，将使用CPU")
            return False
    except ImportError:
        print("   ⚠️ PyTorch未安装，无法检查CUDA")
        return False

def check_required_packages():
    """检查必需的包"""
    print("\n📦 检查必需包...")
    
    required_packages = [
        "torch",
        "transformers", 
        "datasets",
        "accelerate",
        "peft",
        "trl",
        "numpy",
        "pandas",
        "matplotlib"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            importlib.import_module(package)
            print(f"   ✅ {package}")
        except ImportError:
            print(f"   ❌ {package}")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️ 缺少以下包: {', '.join(missing_packages)}")
        print("请运行: pip install -r requirements.txt")
        return False
    else:
        print("\n✅ 所有必需包已安装")
        return True

def setup_directories():
    """设置项目目录结构"""
    print("\n📁 设置项目目录...")
    
    directories = [
        "data",
        "models", 
        "models/sft",
        "models/rm",
        "models/rm/bf16",
        "models/rm/int8", 
        "models/rm/int4",
        "results",
        "logs",
        "scripts"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"   ✅ {directory}")
    
    print("✅ 目录结构设置完成")

def check_data_files():
    """检查数据文件"""
    print("\n📊 检查数据文件...")
    
    data_files = [
        "data/train_prefs.jsonl",
        "data/test_prefs.jsonl"
    ]
    
    all_present = True
    for file_path in data_files:
        if Path(file_path).exists():
            size = Path(file_path).stat().st_size / 1024**2  # MB
            print(f"   ✅ {file_path} ({size:.1f} MB)")
        else:
            print(f"   ❌ {file_path} 不存在")
            all_present = False
    
    if not all_present:
        print("   ⚠️ 请先运行01_Data_Preparation.ipynb生成数据文件")
    
    return all_present

def check_git_config():
    """检查Git配置"""
    print("\n🔄 检查Git配置...")
    
    try:
        # 检查是否在git仓库中
        result = subprocess.run(["git", "status"], 
                              capture_output=True, text=True, 
                              cwd=Path.cwd())
        if result.returncode == 0:
            print("   ✅ Git仓库已初始化")
            
            # 检查是否有未提交的更改
            if "nothing to commit" in result.stdout:
                print("   ✅ 工作目录干净")
            else:
                print("   ⚠️ 有未提交的更改")
                
            return True
        else:
            print("   ❌ 不在Git仓库中")
            return False
            
    except FileNotFoundError:
        print("   ❌ Git未安装")
        return False

def test_config_import():
    """测试配置文件导入"""
    print("\n⚙️ 测试配置文件...")
    
    try:
        # 添加项目根目录到路径
        sys.path.insert(0, str(Path.cwd()))
        
        from config import (
            MODEL_CONFIG, SFT_CONFIG, REWARD_MODEL_CONFIG, 
            PPO_CONFIG, EXPERIMENT_CONFIG, HARDWARE_CONFIG,
            print_config_summary
        )
        
        print("   ✅ 配置文件导入成功")
        print_config_summary()
        return True
        
    except ImportError as e:
        print(f"   ❌ 配置文件导入失败: {e}")
        return False

def create_quick_start_script():
    """创建快速启动脚本"""
    print("\n🚀 创建快速启动脚本...")
    
    script_content = '''#!/usr/bin/env python
"""快速启动EdgeRLHF训练"""

import subprocess
import sys
from pathlib import Path

def run_notebook(notebook_path):
    """运行Jupyter notebook"""
    cmd = [
        sys.executable, "-m", "jupyter", "nbconvert", 
        "--to", "notebook", "--execute", "--inplace", notebook_path
    ]
    
    print(f"🚀 运行 {notebook_path}...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print(f"✅ {notebook_path} 完成")
        return True
    else:
        print(f"❌ {notebook_path} 失败:")
        print(result.stderr)
        return False

def main():
    """主函数"""
    notebooks = [
        "00_Setup.ipynb",
        "01_Data_Preparation.ipynb", 
        "02_SFT_Finetuning.ipynb",
        "03_Reward_Modeling.ipynb",
        "04_PPO_Alignment.ipynb"
    ]
    
    print("🎬 开始EdgeRLHF完整训练流程...")
    
    for notebook in notebooks:
        if not Path(notebook).exists():
            print(f"❌ {notebook} 不存在")
            return False
            
        success = run_notebook(notebook)
        if not success:
            print(f"❌ 训练在 {notebook} 阶段失败")
            return False
    
    print("🎉 EdgeRLHF训练流程完成!")
    return True

if __name__ == "__main__":
    main()
'''
    
    script_path = Path("scripts/quick_start.py")
    script_path.write_text(script_content, encoding='utf-8')
    print(f"   ✅ 创建快速启动脚本: {script_path}")

def main():
    """主函数"""
    print("🔧 EdgeRLHF环境设置向导")
    print("=" * 50)
    
    # 检查各个组件
    checks = [
        check_python_version(),
        check_cuda(),
        check_required_packages(),
        check_data_files(),
        check_git_config(),
        test_config_import()
    ]
    
    # 设置目录结构
    setup_directories()
    create_quick_start_script()
    
    # 总结
    print("\n" + "=" * 50)
    print("📋 环境检查总结:")
    
    passed = sum(checks)
    total = len(checks)
    
    print(f"   通过: {passed}/{total} 项检查")
    
    if passed == total:
        print("🎉 环境设置完成! 可以开始使用EdgeRLHF了")
        print("\n🚀 快速开始:")
        print("   python scripts/quick_start.py  # 运行完整流程")
        print("   jupyter lab                    # 启动Jupyter")
        print("   python config.py               # 查看配置")
    else:
        print("⚠️ 环境设置未完成，请解决上述问题后重试")
    
    return passed == total

if __name__ == "__main__":
    main() 