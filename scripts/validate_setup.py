#!/usr/bin/env python
"""
EdgeRLHF项目验证脚本
验证环境配置、数据完整性和模型可用性
"""

import sys
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple, Any
import importlib

def check_environment() -> Dict[str, Any]:
    """检查基础环境"""
    print("🔍 检查基础环境...")
    
    results = {}
    
    # Python版本
    version = sys.version_info
    results['python_version'] = f"{version.major}.{version.minor}.{version.micro}"
    python_ok = version.major == 3 and version.minor >= 9
    print(f"   Python版本: {results['python_version']} {'✅' if python_ok else '❌'}")
    results['python_ok'] = python_ok
    
    # CUDA检查
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            results['cuda_version'] = torch.version.cuda
            results['gpu_name'] = torch.cuda.get_device_name(0)
            results['gpu_memory'] = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"   CUDA版本: {results['cuda_version']} ✅")
            print(f"   GPU设备: {results['gpu_name']} ✅")
            print(f"   GPU内存: {results['gpu_memory']:.1f} GB ✅")
        else:
            print("   CUDA: 不可用 ❌")
        results['cuda_available'] = cuda_available
    except ImportError:
        print("   PyTorch: 未安装 ❌")
        results['cuda_available'] = False
    
    return results

def check_packages() -> Dict[str, bool]:
    """检查必需包"""
    print("\n📦 检查必需包...")
    
    required_packages = {
        'torch': 'PyTorch深度学习框架',
        'transformers': 'HuggingFace Transformers',
        'datasets': 'HuggingFace数据集',
        'accelerate': '分布式训练加速',
        'peft': '参数高效微调',
        'trl': '强化学习训练',
        'bitsandbytes': '量化库',
        'numpy': '数值计算',
        'pandas': '数据处理',
        'matplotlib': '可视化',
        'seaborn': '统计可视化',
        'jupyter': 'Jupyter Notebook',
        'ipykernel': 'Jupyter内核',
        'tqdm': '进度条'
    }
    
    results = {}
    
    for package, description in required_packages.items():
        try:
            importlib.import_module(package)
            print(f"   ✅ {package:<12} - {description}")
            results[package] = True
        except ImportError:
            print(f"   ❌ {package:<12} - {description}")
            results[package] = False
    
    missing = [pkg for pkg, installed in results.items() if not installed]
    if missing:
        print(f"\n⚠️ 缺少包: {', '.join(missing)}")
        print("建议运行: pip install -r requirements.txt")
    
    return results

def check_project_structure() -> Dict[str, Any]:
    """检查项目结构"""
    print("\n📁 检查项目结构...")
    
    required_files = [
        "config.py",
        "requirements.txt", 
        "environment.yml",
        ".gitignore",
        "README.md",
        "EdgeRLHF_Research_Report.md"
    ]
    
    required_dirs = [
        "data",
        "models",
        "models/sft",
        "models/rm",
        "results",
        "logs",
        "scripts"
    ]
    
    notebooks = [
        "00_Setup.ipynb",
        "01_Data_Preparation.ipynb",
        "02_SFT_Finetuning.ipynb",
        "03_Reward_Modeling.ipynb",
        "04_PPO_Alignment.ipynb"
    ]
    
    results = {
        'files': {},
        'directories': {},
        'notebooks': {}
    }
    
    # 检查文件
    for file_path in required_files:
        exists = Path(file_path).exists()
        size = Path(file_path).stat().st_size / 1024 if exists else 0  # KB
        print(f"   {'✅' if exists else '❌'} {file_path:<30} {f'({size:.1f} KB)' if exists else '(不存在)'}")
        results['files'][file_path] = {'exists': exists, 'size_kb': size}
    
    # 检查目录
    for dir_path in required_dirs:
        exists = Path(dir_path).exists()
        print(f"   {'✅' if exists else '❌'} {dir_path}/")
        results['directories'][dir_path] = exists
    
    # 检查Notebook
    for notebook in notebooks:
        exists = Path(notebook).exists()
        size = Path(notebook).stat().st_size / 1024 if exists else 0  # KB
        print(f"   {'✅' if exists else '❌'} {notebook:<25} {f'({size:.1f} KB)' if exists else '(不存在)'}")
        results['notebooks'][notebook] = {'exists': exists, 'size_kb': size}
    
    return results

def check_data_integrity() -> Dict[str, Any]:
    """检查数据完整性"""
    print("\n📊 检查数据完整性...")
    
    data_files = [
        ("data/train_prefs.jsonl", "训练偏好数据"),
        ("data/test_prefs.jsonl", "测试偏好数据")
    ]
    
    results = {}
    
    for file_path, description in data_files:
        path = Path(file_path)
        file_result = {'description': description}
        
        if not path.exists():
            print(f"   ❌ {file_path} - 文件不存在")
            file_result['exists'] = False
            results[file_path] = file_result
            continue
        
        file_result['exists'] = True
        file_result['size_mb'] = path.stat().st_size / 1024**2
        
        try:
            # 读取并验证JSONL格式
            lines = []
            with open(path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        data = json.loads(line.strip())
                        lines.append(data)
                        if line_num > 5:  # 只检查前5行结构
                            break
                    except json.JSONDecodeError as e:
                        print(f"   ❌ {file_path} - 第{line_num}行JSON格式错误: {e}")
                        file_result['valid'] = False
                        break
                else:
                    file_result['valid'] = True
            
            if file_result['valid']:
                # 统计总行数
                with open(path, 'r', encoding='utf-8') as f:
                    file_result['total_lines'] = sum(1 for _ in f)
                
                # 检查数据结构
                if lines:
                    sample = lines[0]
                    required_keys = ['chosen', 'rejected']
                    has_all_keys = all(key in sample for key in required_keys)
                    file_result['structure_valid'] = has_all_keys
                    
                    if has_all_keys:
                        print(f"   ✅ {file_path} - {file_result['total_lines']} 条记录 ({file_result['size_mb']:.1f} MB)")
                    else:
                        print(f"   ⚠️ {file_path} - 数据结构不完整，缺少必需字段")
                else:
                    print(f"   ⚠️ {file_path} - 文件为空")
                    file_result['structure_valid'] = False
        
        except Exception as e:
            print(f"   ❌ {file_path} - 读取错误: {e}")
            file_result['valid'] = False
        
        results[file_path] = file_result
    
    return results

def check_model_files() -> Dict[str, Any]:
    """检查模型文件"""
    print("\n🤖 检查模型文件...")
    
    model_paths = [
        ("models/sft", "SFT微调模型"),
        ("models/rm/bf16", "BF16奖励模型"),
        ("models/rm/int8", "INT8奖励模型"),
        ("models/rm/int4", "INT4奖励模型")
    ]
    
    results = {}
    
    for model_path, description in model_paths:
        path = Path(model_path)
        model_result = {'description': description, 'path': model_path}
        
        if not path.exists():
            print(f"   ❌ {model_path} - 目录不存在")
            model_result['exists'] = False
            results[model_path] = model_result
            continue
        
        model_result['exists'] = True
        
        # 检查关键模型文件
        key_files = [
            "adapter_model.safetensors",
            "adapter_config.json",
            "training_args.bin"
        ]
        
        found_files = []
        total_size = 0
        
        for file_name in key_files:
            file_path = path / file_name
            if file_path.exists():
                size = file_path.stat().st_size / 1024**2  # MB
                found_files.append((file_name, size))
                total_size += size
        
        model_result['files'] = found_files
        model_result['total_size_mb'] = total_size
        
        if found_files:
            files_str = ", ".join([f"{name}({size:.1f}MB)" for name, size in found_files])
            print(f"   ✅ {model_path} - {files_str}")
            model_result['valid'] = True
        else:
            print(f"   ⚠️ {model_path} - 目录存在但缺少关键文件")
            model_result['valid'] = False
        
        results[model_path] = model_result
    
    return results

def check_config_validity() -> Dict[str, Any]:
    """检查配置文件有效性"""
    print("\n⚙️ 检查配置文件...")
    
    results = {'config_importable': False}
    
    try:
        # 添加项目根目录到路径
        sys.path.insert(0, str(Path.cwd()))
        
        from config import (
            MODEL_CONFIG, SFT_CONFIG, REWARD_MODEL_CONFIG,
            PPO_CONFIG, EXPERIMENT_CONFIG, HARDWARE_CONFIG
        )
        
        print("   ✅ 配置文件导入成功")
        results['config_importable'] = True
        
        # 检查配置完整性
        configs_to_check = {
            'MODEL_CONFIG': MODEL_CONFIG,
            'SFT_CONFIG': SFT_CONFIG, 
            'REWARD_MODEL_CONFIG': REWARD_MODEL_CONFIG,
            'PPO_CONFIG': PPO_CONFIG,
            'EXPERIMENT_CONFIG': EXPERIMENT_CONFIG,
            'HARDWARE_CONFIG': HARDWARE_CONFIG
        }
        
        for config_name, config in configs_to_check.items():
            if hasattr(config, '__dict__'):
                print(f"   ✅ {config_name} - 配置对象完整")
                results[config_name.lower()] = True
            else:
                print(f"   ❌ {config_name} - 配置对象无效")
                results[config_name.lower()] = False
        
        # 验证关键配置值
        max_memory_mb = int(HARDWARE_CONFIG.vram_gb * 1024 * HARDWARE_CONFIG.max_memory_allocation)
        if max_memory_mb <= 8192:
            print(f"   ✅ 内存配置适合8GB GPU: {max_memory_mb}MB")
        else:
            print(f"   ⚠️ 内存配置可能超出8GB GPU限制: {max_memory_mb}MB")
        
    except ImportError as e:
        print(f"   ❌ 配置文件导入失败: {e}")
        results['config_importable'] = False
    except Exception as e:
        print(f"   ❌ 配置验证失败: {e}")
        results['config_error'] = str(e)
    
    return results

def check_git_status() -> Dict[str, Any]:
    """检查Git状态"""
    print("\n🔄 检查Git状态...")
    
    results = {}
    
    try:
        # 检查是否为Git仓库
        result = subprocess.run(
            ["git", "rev-parse", "--is-inside-work-tree"],
            capture_output=True, text=True, cwd=Path.cwd()
        )
        
        if result.returncode == 0:
            print("   ✅ Git仓库已初始化")
            results['is_git_repo'] = True
            
            # 检查远程仓库
            result = subprocess.run(
                ["git", "remote", "-v"],
                capture_output=True, text=True, cwd=Path.cwd()
            )
            
            if result.stdout.strip():
                print(f"   ✅ 远程仓库已配置")
                results['has_remote'] = True
            else:
                print("   ⚠️ 未配置远程仓库")
                results['has_remote'] = False
            
            # 检查工作目录状态
            result = subprocess.run(
                ["git", "status", "--porcelain"],
                capture_output=True, text=True, cwd=Path.cwd()
            )
            
            if result.stdout.strip():
                print("   ⚠️ 有未提交的更改")
                results['clean_working_dir'] = False
            else:
                print("   ✅ 工作目录干净")
                results['clean_working_dir'] = True
                
        else:
            print("   ❌ 不是Git仓库")
            results['is_git_repo'] = False
            
    except FileNotFoundError:
        print("   ❌ Git未安装")
        results['git_installed'] = False
    except Exception as e:
        print(f"   ❌ Git检查失败: {e}")
        results['git_error'] = str(e)
    
    return results

def run_quick_tests() -> Dict[str, Any]:
    """运行快速功能测试"""
    print("\n🧪 运行快速功能测试...")
    
    results = {}
    
    # 测试PyTorch基本功能
    try:
        import torch
        
        # 创建小张量测试
        x = torch.randn(2, 3)
        y = torch.randn(3, 2)
        z = torch.mm(x, y)
        
        print("   ✅ PyTorch基本运算测试通过")
        results['pytorch_basic'] = True
        
        # 测试CUDA（如果可用）
        if torch.cuda.is_available():
            x_cuda = x.cuda()
            y_cuda = y.cuda()
            z_cuda = torch.mm(x_cuda, y_cuda)
            print("   ✅ CUDA运算测试通过")
            results['cuda_computation'] = True
        else:
            results['cuda_computation'] = False
            
    except Exception as e:
        print(f"   ❌ PyTorch测试失败: {e}")
        results['pytorch_basic'] = False
    
    # 测试Transformers加载
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
        tokens = tokenizer("Hello world", return_tensors="pt")
        print("   ✅ Transformers tokenizer测试通过")
        results['transformers_basic'] = True
    except Exception as e:
        print(f"   ❌ Transformers测试失败: {e}")
        results['transformers_basic'] = False
    
    return results

def generate_report(all_results: Dict[str, Any]) -> str:
    """生成验证报告"""
    report = ["# EdgeRLHF项目验证报告", ""]
    report.append(f"验证时间: {Path.cwd()}")
    report.append("")
    
    # 统计各部分状态
    sections = [
        ("环境检查", all_results.get('environment', {})),
        ("包检查", all_results.get('packages', {})),
        ("项目结构", all_results.get('structure', {})),
        ("数据完整性", all_results.get('data', {})),
        ("模型文件", all_results.get('models', {})),
        ("配置文件", all_results.get('config', {})),
        ("Git状态", all_results.get('git', {})),
        ("功能测试", all_results.get('tests', {}))
    ]
    
    # 计算总体得分
    total_score = 0
    max_score = 0
    
    for section_name, section_data in sections:
        if isinstance(section_data, dict):
            section_score = sum(1 for v in section_data.values() if v is True)
            section_max = len([v for v in section_data.values() if isinstance(v, bool)])
            total_score += section_score
            max_score += section_max
            
            if section_max > 0:
                percentage = (section_score / section_max) * 100
                status = "✅" if percentage >= 80 else "⚠️" if percentage >= 60 else "❌"
                report.append(f"- {status} {section_name}: {section_score}/{section_max} ({percentage:.1f}%)")
    
    if max_score > 0:
        overall_percentage = (total_score / max_score) * 100
        overall_status = "✅" if overall_percentage >= 80 else "⚠️" if overall_percentage >= 60 else "❌"
        report.append("")
        report.append(f"## 总体状态: {overall_status} {total_score}/{max_score} ({overall_percentage:.1f}%)")
    
    return "\n".join(report)

def main():
    """主函数"""
    print("🔍 EdgeRLHF项目验证工具")
    print("=" * 50)
    
    all_results = {}
    
    # 执行各项检查
    all_results['environment'] = check_environment()
    all_results['packages'] = check_packages()
    all_results['structure'] = check_project_structure()
    all_results['data'] = check_data_integrity()
    all_results['models'] = check_model_files()
    all_results['config'] = check_config_validity()
    all_results['git'] = check_git_status()
    all_results['tests'] = run_quick_tests()
    
    # 生成报告
    print("\n" + "=" * 50)
    report = generate_report(all_results)
    print(report)
    
    # 保存详细结果
    results_file = Path("logs/validation_results.json")
    results_file.parent.mkdir(exist_ok=True)
    
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"\n📄 详细结果已保存到: {results_file}")
    
    return all_results

if __name__ == "__main__":
    main() 