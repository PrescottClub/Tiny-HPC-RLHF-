# EdgeRLHF项目管理Makefile
# 提供便捷的开发和管理命令

.PHONY: help setup install test clean validate jupyter notebook lab environment requirements

# 默认目标
help:
	@echo "🔬 EdgeRLHF项目管理工具"
	@echo "=================================="
	@echo "📦 环境管理:"
	@echo "  setup        - 设置开发环境"
	@echo "  install      - 安装依赖包"
	@echo "  environment  - 创建conda环境"
	@echo "  requirements - 更新requirements.txt"
	@echo ""
	@echo "🧪 开发工具:"
	@echo "  jupyter      - 启动Jupyter Lab"
	@echo "  notebook     - 启动Jupyter Notebook"
	@echo "  lab          - 启动Jupyter Lab（同jupyter）"
	@echo "  validate     - 验证项目设置"
	@echo "  test         - 运行测试"
	@echo ""
	@echo "🧹 维护工具:"
	@echo "  clean        - 清理临时文件"
	@echo "  clean-cache  - 清理缓存文件"
	@echo "  clean-models - 清理模型文件"
	@echo "  clean-logs   - 清理日志文件"
	@echo ""
	@echo "📊 数据管理:"
	@echo "  data-check   - 检查数据完整性"
	@echo "  data-stats   - 显示数据统计"
	@echo ""
	@echo "🔄 Git管理:"
	@echo "  git-status   - 显示Git状态"
	@echo "  git-clean    - 清理Git未跟踪文件"
	@echo "  push         - 提交并推送到GitHub"
	@echo ""
	@echo "📋 项目信息:"
	@echo "  status       - 显示项目状态"
	@echo "  info         - 显示系统信息"
	@echo "  structure    - 显示项目结构"

# 环境管理
setup: install validate
	@echo "✅ 开发环境设置完成"

install:
	@echo "📦 安装依赖包..."
	@pip install -r requirements.txt
	@echo "✅ 依赖包安装完成"

environment:
	@echo "🐍 创建conda环境..."
	@conda env create -f environment.yml
	@echo "✅ conda环境创建完成"
	@echo "💡 激活环境: conda activate edgerlhf"

requirements:
	@echo "📝 更新requirements.txt..."
	@pip freeze > requirements.txt
	@echo "✅ requirements.txt已更新"

# 开发工具
jupyter:
	@python scripts/start_jupyter.py

lab: jupyter

notebook:
	@echo "📓 启动Jupyter Notebook..."
	@jupyter notebook --port=8888 --allow-root

validate:
	@python scripts/validate_setup.py

test:
	@echo "🧪 运行项目测试..."
	@python scripts/setup_environment.py
	@python scripts/validate_setup.py

# 维护工具
clean:
	@echo "🧹 清理临时文件..."
	@python scripts/cleanup.py
	@echo "✅ 清理完成"

clean-cache:
	@echo "🗑️ 清理缓存文件..."
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name ".ipynb_checkpoints" -exec rm -rf {} + 2>/dev/null || true
	@find . -name "*.pyc" -delete 2>/dev/null || true
	@find . -name "*.pyo" -delete 2>/dev/null || true
	@echo "✅ 缓存文件清理完成"

clean-models:
	@echo "⚠️ 确认删除所有模型文件吗? [y/N] " && read ans && [ $${ans:-N} = y ]
	@rm -rf models/sft/* models/rm/* 2>/dev/null || true
	@echo "✅ 模型文件清理完成"

clean-logs:
	@echo "🗑️ 清理日志文件..."
	@rm -rf logs/*.log logs/*.json 2>/dev/null || true
	@echo "✅ 日志文件清理完成"

# 数据管理
data-check:
	@echo "📊 检查数据完整性..."
	@python -c "from scripts.validate_setup import check_data_integrity; check_data_integrity()"

data-stats:
	@echo "📈 数据统计信息:"
	@echo "训练数据:"
	@wc -l data/train_prefs.jsonl 2>/dev/null || echo "  文件不存在"
	@echo "测试数据:"
	@wc -l data/test_prefs.jsonl 2>/dev/null || echo "  文件不存在"

# Git管理
git-status:
	@echo "🔄 Git状态:"
	@git status --short

git-clean:
	@echo "⚠️ 确认清理Git未跟踪文件吗? [y/N] " && read ans && [ $${ans:-N} = y ]
	@git clean -fd

push:
	@echo "📤 提交并推送代码..."
	@git add .
	@git commit -m "Update: $(shell date +'%Y-%m-%d %H:%M:%S')"
	@git push
	@echo "✅ 代码已推送到GitHub"

# 项目信息
status:
	@echo "📋 EdgeRLHF项目状态"
	@echo "===================="
	@echo "📁 项目目录: $(PWD)"
	@echo "🐍 Python版本: $(shell python --version)"
	@echo "💾 可用内存: $(shell python -c 'import psutil; print(f"{psutil.virtual_memory().available/1024**3:.1f}GB")')"
	@echo "🎯 CUDA设备: $(shell python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU Only")')"
	@echo "📊 项目文件:"
	@find . -name "*.py" -o -name "*.ipynb" | wc -l | sed 's/^/  代码文件: /'
	@find data -name "*.jsonl" 2>/dev/null | wc -l | sed 's/^/  数据文件: /' || echo "  数据文件: 0"
	@find models -name "*.safetensors" -o -name "*.bin" 2>/dev/null | wc -l | sed 's/^/  模型文件: /' || echo "  模型文件: 0"

info:
	@echo "💻 系统信息"
	@echo "============"
	@python -c "import platform, psutil, torch; print(f'操作系统: {platform.system()} {platform.release()}'); print(f'CPU: {platform.processor()}'); print(f'内存: {psutil.virtual_memory().total/1024**3:.1f}GB'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else "无"}'); print(f'CUDA: {torch.version.cuda if torch.cuda.is_available() else "无"}')"

structure:
	@echo "📁 项目结构"
	@echo "============"
	@tree -I '__pycache__|.git|*.pyc|.ipynb_checkpoints' -L 3 . 2>/dev/null || \
	 find . -type d \( -name __pycache__ -o -name .git -o -name .ipynb_checkpoints \) -prune -o -type f -print | \
	 head -20 | sed 's|^\./||' | sort

# Windows兼容性
ifeq ($(OS),Windows_NT)
clean-cache:
	@echo "🗑️ 清理缓存文件..."
	@for /d /r . %%d in (__pycache__) do @if exist "%%d" rd /s /q "%%d"
	@for /d /r . %%d in (.pytest_cache) do @if exist "%%d" rd /s /q "%%d"
	@for /d /r . %%d in (.ipynb_checkpoints) do @if exist "%%d" rd /s /q "%%d"
	@del /s /q *.pyc 2>nul
	@del /s /q *.pyo 2>nul
	@echo "✅ 缓存文件清理完成"

structure:
	@echo "📁 项目结构"
	@echo "============"
	@dir /s /b | findstr /v __pycache__ | findstr /v .git | findstr /v .ipynb_checkpoints
endif 