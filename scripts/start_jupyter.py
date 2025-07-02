#!/usr/bin/env python
"""
EdgeRLHF Jupyter Lab启动脚本
自动配置并启动Jupyter Lab环境
"""

import os
import sys
import subprocess
import webbrowser
from pathlib import Path
import time

def check_jupyter_installed():
    """检查Jupyter是否已安装"""
    try:
        result = subprocess.run([sys.executable, "-m", "jupyter", "--version"], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Jupyter已安装")
            return True
        else:
            print("❌ Jupyter未正确安装")
            return False
    except FileNotFoundError:
        print("❌ Jupyter未安装")
        return False

def setup_jupyter_config():
    """设置Jupyter配置"""
    print("⚙️ 配置Jupyter环境...")
    
    # 设置环境变量
    env_vars = {
        'JUPYTER_CONFIG_DIR': str(Path.cwd() / '.jupyter'),
        'PYTHONPATH': str(Path.cwd()),
    }
    
    for key, value in env_vars.items():
        os.environ[key] = value
        print(f"   设置环境变量: {key}={value}")
    
    # 创建Jupyter配置目录
    jupyter_config_dir = Path(os.environ['JUPYTER_CONFIG_DIR'])
    jupyter_config_dir.mkdir(exist_ok=True)
    
    # 创建Jupyter配置文件
    config_content = '''
# EdgeRLHF Jupyter配置

c = get_config()

# 基本配置
c.ServerApp.port = 8888
c.ServerApp.open_browser = True
c.ServerApp.root_dir = '.'
c.ServerApp.notebook_dir = '.'

# 允许远程连接（如果需要）
c.ServerApp.ip = '127.0.0.1'
c.ServerApp.allow_remote_access = False

# 禁用令牌（开发环境）
# c.ServerApp.token = ''
# c.ServerApp.password = ''

# 启用扩展
c.ServerApp.jpserver_extensions = {
    'jupyter_lsp': True,
    'jupyterlab': True
}

# 内存和性能设置
c.ServerApp.max_buffer_size = 268435456  # 256MB
c.KernelManager.autorestart = True

# 显示隐藏文件
c.ContentsManager.allow_hidden = True

print("🔧 Jupyter配置已加载")
'''
    
    config_file = jupyter_config_dir / 'jupyter_server_config.py'
    config_file.write_text(config_content, encoding='utf-8')
    print(f"   ✅ 配置文件已创建: {config_file}")

def install_jupyter_extensions():
    """安装Jupyter扩展"""
    print("🔌 检查Jupyter扩展...")
    
    extensions = [
        'jupyterlab-git',
        'jupyterlab-lsp',
        '@jupyter-widgets/jupyterlab-manager',
        'jupyterlab-plotly'
    ]
    
    # 检查已安装的扩展
    try:
        result = subprocess.run([sys.executable, "-m", "jupyter", "labextension", "list"], 
                              capture_output=True, text=True)
        installed_extensions = result.stdout
        
        for ext in extensions:
            if ext in installed_extensions:
                print(f"   ✅ {ext} 已安装")
            else:
                print(f"   ⚠️ {ext} 未安装，可选择安装")
                # 这里不自动安装，因为可能需要时间
                
    except Exception as e:
        print(f"   ⚠️ 扩展检查失败: {e}")

def create_notebook_templates():
    """创建Notebook模板"""
    print("📝 创建Notebook模板...")
    
    templates_dir = Path("templates")
    templates_dir.mkdir(exist_ok=True)
    
    # 创建基础实验模板
    experiment_template = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# EdgeRLHF实验 - [实验名称]\\n",
                    "\\n",
                    "## 实验目标\\n",
                    "- [ ] 目标1\\n",
                    "- [ ] 目标2\\n",
                    "\\n",
                    "## 实验参数\\n",
                    "```python\\n",
                    "EXPERIMENT_NAME = \\"your_experiment\\"\\n",
                    "MODEL_SIZE = \\"distilgpt2\\"\\n",
                    "PRECISION = \\"bf16\\"\\n",
                    "```\\n",
                    "\\n",
                    "## 实验记录\\n",
                    "日期: \\n",
                    "研究者: \\n"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "source": [
                    "# 导入基础配置\\n",
                    "import sys\\n",
                    "sys.path.append('.')\\n",
                    "\\n",
                    "from config import *\\n",
                    "import torch\\n",
                    "import numpy as np\\n",
                    "import matplotlib.pyplot as plt\\n",
                    "\\n",
                    "print(f\\"设备: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}\\")\\n",
                    "print(f\\"CUDA可用: {torch.cuda.is_available()}\\")\\n",
                    "print(f\\"内存配置: {int(HARDWARE_CONFIG.vram_gb * 1024 * HARDWARE_CONFIG.max_memory_allocation)}MB\\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 实验代码\\n",
                    "在下面的单元格中编写实验代码："
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "source": [
                    "# 实验代码\\n",
                    "pass"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 结果分析\\n",
                    "在下面记录实验结果和分析："
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "source": [
                    "# 结果可视化和分析\\n",
                    "pass"
                ]
            }
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.11.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    
    import json
    template_file = templates_dir / "experiment_template.ipynb"
    with open(template_file, 'w', encoding='utf-8') as f:
        json.dump(experiment_template, f, indent=2, ensure_ascii=False)
    
    print(f"   ✅ 实验模板已创建: {template_file}")

def start_jupyter_lab():
    """启动Jupyter Lab"""
    print("🚀 启动Jupyter Lab...")
    
    # 构建启动命令
    cmd = [
        sys.executable, "-m", "jupyter", "lab",
        "--port=8888",
        "--no-browser",  # 我们稍后手动打开
        "--allow-root",
        f"--notebook-dir={Path.cwd()}"
    ]
    
    print(f"   命令: {' '.join(cmd)}")
    
    try:
        # 启动Jupyter Lab
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        # 等待服务器启动并获取URL
        url = None
        for line in iter(process.stdout.readline, ''):
            print(f"   {line.rstrip()}")
            
            # 查找服务器URL
            if "http://127.0.0.1:8888" in line and "token=" in line:
                # 提取URL
                start = line.find("http://127.0.0.1:8888")
                url = line[start:].split()[0]
                break
            
            # 检查是否启动失败
            if "ERROR" in line or process.poll() is not None:
                print("❌ Jupyter Lab启动失败")
                return False
        
        if url:
            print(f"\\n✅ Jupyter Lab已启动!")
            print(f"🌐 访问地址: {url}")
            
            # 等待一下再打开浏览器
            print("⏳ 等待3秒后打开浏览器...")
            time.sleep(3)
            
            # 打开浏览器
            try:
                webbrowser.open(url)
                print("🌐 浏览器已打开")
            except Exception as e:
                print(f"⚠️ 无法自动打开浏览器: {e}")
                print(f"请手动访问: {url}")
            
            print("\\n💡 使用提示:")
            print("   - Ctrl+C 停止Jupyter Lab")
            print("   - 模板文件在 templates/ 目录下")
            print("   - 配置文件在 .jupyter/ 目录下")
            
            # 保持进程运行
            try:
                process.wait()
            except KeyboardInterrupt:
                print("\\n🛑 正在停止Jupyter Lab...")
                process.terminate()
                process.wait()
                print("✅ Jupyter Lab已停止")
            
            return True
        else:
            print("❌ 未能获取Jupyter Lab URL")
            return False
            
    except Exception as e:
        print(f"❌ 启动失败: {e}")
        return False

def show_usage_tips():
    """显示使用提示"""
    print("\\n📚 EdgeRLHF Jupyter使用指南:")
    print("=" * 50)
    print("🔧 配置文件:")
    print("   - config.py: 项目配置中心")
    print("   - .jupyter/: Jupyter配置目录")
    print("")
    print("📝 Notebook文件:")
    print("   - 00_Setup.ipynb: 环境设置")
    print("   - 01_Data_Preparation.ipynb: 数据准备")
    print("   - 02_SFT_Finetuning.ipynb: 监督微调")
    print("   - 03_Reward_Modeling.ipynb: 奖励建模")
    print("   - 04_PPO_Alignment.ipynb: PPO对齐")
    print("")
    print("🗂️ 目录结构:")
    print("   - data/: 数据文件")
    print("   - models/: 模型文件")
    print("   - results/: 实验结果")
    print("   - logs/: 日志文件")
    print("   - templates/: Notebook模板")
    print("")
    print("🛠️ 实用脚本:")
    print("   - python scripts/setup_environment.py")
    print("   - python scripts/validate_setup.py")
    print("   - python scripts/cleanup.py")
    print("")
    print("💡 最佳实践:")
    print("   - 使用模板创建新实验")
    print("   - 定期备份重要结果")
    print("   - 遵循命名约定")
    print("   - 添加详细注释")

def main():
    """主函数"""
    print("🔬 EdgeRLHF Jupyter Lab启动器")
    print("=" * 50)
    
    # 检查Jupyter安装
    if not check_jupyter_installed():
        print("❌ 请先安装Jupyter: pip install jupyter jupyterlab")
        return False
    
    # 设置Jupyter配置
    setup_jupyter_config()
    
    # 检查扩展
    install_jupyter_extensions()
    
    # 创建模板
    create_notebook_templates()
    
    # 显示使用提示
    show_usage_tips()
    
    # 询问是否启动
    print("\\n" + "=" * 50)
    response = input("🚀 是否现在启动Jupyter Lab? (y/N): ").strip().lower()
    
    if response in ['y', 'yes']:
        return start_jupyter_lab()
    else:
        print("👋 稍后可运行此脚本启动Jupyter Lab")
        return True

if __name__ == "__main__":
    main() 