#!/usr/bin/env python
"""
EdgeRLHF项目清理脚本
清理临时文件、缓存和不必要的文件
"""

import os
import shutil
import glob
from pathlib import Path
from typing import List

def get_file_size(path: Path) -> float:
    """获取文件大小（MB）"""
    if path.is_file():
        return path.stat().st_size / (1024 * 1024)
    elif path.is_dir():
        total = 0
        try:
            for file_path in path.rglob('*'):
                if file_path.is_file():
                    total += file_path.stat().st_size
        except (PermissionError, OSError):
            pass
        return total / (1024 * 1024)
    return 0

def cleanup_jupyter_checkpoints():
    """清理Jupyter检查点文件"""
    print("🧹 清理Jupyter检查点...")
    
    checkpoint_dirs = list(Path(".").rglob(".ipynb_checkpoints"))
    total_size = 0
    
    for checkpoint_dir in checkpoint_dirs:
        size = get_file_size(checkpoint_dir)
        total_size += size
        print(f"   删除: {checkpoint_dir} ({size:.2f} MB)")
        shutil.rmtree(checkpoint_dir, ignore_errors=True)
    
    print(f"   ✅ 清理Jupyter检查点完成，释放 {total_size:.2f} MB")
    return total_size

def cleanup_python_cache():
    """清理Python缓存文件"""
    print("\n🐍 清理Python缓存...")
    
    cache_patterns = [
        "**/__pycache__",
        "**/*.pyc",
        "**/*.pyo",
        "**/*.egg-info",
        "**/build",
        "**/dist"
    ]
    
    total_size = 0
    
    for pattern in cache_patterns:
        paths = list(Path(".").glob(pattern))
        for path in paths:
            size = get_file_size(path)
            total_size += size
            print(f"   删除: {path} ({size:.2f} MB)")
            
            if path.is_dir():
                shutil.rmtree(path, ignore_errors=True)
            else:
                path.unlink(missing_ok=True)
    
    print(f"   ✅ 清理Python缓存完成，释放 {total_size:.2f} MB")
    return total_size

def cleanup_logs():
    """清理日志文件（保留最近的）"""
    print("\n📋 清理旧日志文件...")
    
    logs_dir = Path("logs")
    if not logs_dir.exists():
        print("   ⏭️ 没有日志目录")
        return 0
    
    total_size = 0
    log_files = list(logs_dir.rglob("*.log"))
    
    # 按修改时间排序，保留最新的5个
    log_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    
    files_to_delete = log_files[5:]  # 删除超过5个的旧文件
    
    for log_file in files_to_delete:
        size = get_file_size(log_file)
        total_size += size
        print(f"   删除: {log_file} ({size:.2f} MB)")
        log_file.unlink(missing_ok=True)
    
    print(f"   ✅ 清理日志文件完成，释放 {total_size:.2f} MB")
    return total_size

def cleanup_temp_files():
    """清理临时文件"""
    print("\n🗂️ 清理临时文件...")
    
    temp_patterns = [
        "**/.DS_Store",
        "**/Thumbs.db",
        "**/*.tmp",
        "**/*.temp",
        "**/.cache",
        "**/tmp",
        "**/*~"
    ]
    
    total_size = 0
    
    for pattern in temp_patterns:
        paths = list(Path(".").glob(pattern))
        for path in paths:
            size = get_file_size(path)
            total_size += size
            print(f"   删除: {path} ({size:.2f} MB)")
            
            if path.is_dir():
                shutil.rmtree(path, ignore_errors=True)
            else:
                path.unlink(missing_ok=True)
    
    print(f"   ✅ 清理临时文件完成，释放 {total_size:.2f} MB")
    return total_size

def cleanup_empty_dirs():
    """清理空目录"""
    print("\n📁 清理空目录...")
    
    deleted_count = 0
    
    # 从深层目录开始清理
    for path in sorted(Path(".").rglob("*"), key=lambda x: len(x.parts), reverse=True):
        if path.is_dir() and not any(path.iterdir()):
            # 跳过重要目录
            if path.name in [".git", "data", "models", "results", "logs", "scripts"]:
                continue
            
            print(f"   删除空目录: {path}")
            path.rmdir()
            deleted_count += 1
    
    print(f"   ✅ 清理空目录完成，删除 {deleted_count} 个目录")
    return deleted_count

def cleanup_large_files(size_threshold_mb: float = 100):
    """列出大文件（不自动删除）"""
    print(f"\n📏 查找大于 {size_threshold_mb} MB的文件...")
    
    large_files = []
    
    try:
        for path in Path(".").rglob("*"):
            if path.is_file():
                size = get_file_size(path)
                if size > size_threshold_mb:
                    large_files.append((path, size))
    except (PermissionError, OSError):
        pass
    
    if large_files:
        large_files.sort(key=lambda x: x[1], reverse=True)
        print("   🔍 发现以下大文件:")
        for path, size in large_files:
            print(f"      {path} ({size:.2f} MB)")
        print("   ⚠️ 大文件需要手动检查和删除")
    else:
        print(f"   ✅ 没有发现大于 {size_threshold_mb} MB的文件")
    
    return large_files

def cleanup_model_checkpoints():
    """清理模型检查点（保留最终模型）"""
    print("\n🤖 清理模型检查点...")
    
    models_dir = Path("models")
    if not models_dir.exists():
        print("   ⏭️ 没有模型目录")
        return 0
    
    total_size = 0
    checkpoint_dirs = list(models_dir.rglob("checkpoint-*"))
    
    for checkpoint_dir in checkpoint_dirs:
        # 检查是否有同级的最终模型
        parent_dir = checkpoint_dir.parent
        has_final_model = any(
            (parent_dir / name).exists() 
            for name in ["adapter_model.safetensors", "pytorch_model.bin", "model.safetensors"]
        )
        
        if has_final_model:
            size = get_file_size(checkpoint_dir)
            total_size += size
            print(f"   删除检查点: {checkpoint_dir} ({size:.2f} MB)")
            shutil.rmtree(checkpoint_dir, ignore_errors=True)
        else:
            print(f"   保留检查点: {checkpoint_dir} (无最终模型)")
    
    print(f"   ✅ 清理模型检查点完成，释放 {total_size:.2f} MB")
    return total_size

def get_disk_usage():
    """获取项目目录磁盘使用情况"""
    print("\n💾 项目磁盘使用情况:")
    
    directories = ["data", "models", "results", "logs", ".git"]
    
    for dir_name in directories:
        dir_path = Path(dir_name)
        if dir_path.exists():
            size = get_file_size(dir_path)
            print(f"   {dir_name}: {size:.2f} MB")
        else:
            print(f"   {dir_name}: 不存在")

def interactive_cleanup():
    """交互式清理"""
    print("🔧 EdgeRLHF项目清理工具")
    print("=" * 50)
    
    options = [
        ("1", "清理Jupyter检查点", cleanup_jupyter_checkpoints),
        ("2", "清理Python缓存", cleanup_python_cache),
        ("3", "清理日志文件", cleanup_logs),
        ("4", "清理临时文件", cleanup_temp_files),
        ("5", "清理空目录", cleanup_empty_dirs),
        ("6", "清理模型检查点", cleanup_model_checkpoints),
        ("7", "查找大文件", lambda: cleanup_large_files()),
        ("8", "查看磁盘使用", get_disk_usage),
        ("a", "全部清理（除大文件）", None),
        ("q", "退出", None)
    ]
    
    while True:
        print("\n📋 清理选项:")
        for key, desc, _ in options:
            print(f"   {key}. {desc}")
        
        choice = input("\n请选择操作 (1-8, a, q): ").strip().lower()
        
        if choice == "q":
            break
        elif choice == "a":
            print("\n🚀 执行全面清理...")
            total_saved = 0
            cleanup_funcs = [func for _, _, func in options if func and func != get_disk_usage]
            
            for func in cleanup_funcs[:-1]:  # 跳过查找大文件
                try:
                    saved = func()
                    if isinstance(saved, (int, float)):
                        total_saved += saved
                except Exception as e:
                    print(f"   ❌ 清理失败: {e}")
            
            print(f"\n🎉 全面清理完成！总共释放 {total_saved:.2f} MB空间")
            get_disk_usage()
        else:
            # 查找对应的功能
            for key, _, func in options:
                if key == choice and func:
                    try:
                        result = func()
                        if isinstance(result, (int, float)):
                            print(f"释放空间: {result:.2f} MB")
                    except Exception as e:
                        print(f"❌ 操作失败: {e}")
                    break
            else:
                print("❌ 无效选择，请重试")

def main():
    """主函数"""
    print("🧹 EdgeRLHF项目清理脚本")
    print("自动清理临时文件和缓存")
    print("=" * 50)
    
    # 显示当前磁盘使用情况
    get_disk_usage()
    
    # 启动交互式清理
    interactive_cleanup()
    
    print("\n👋 清理完成，感谢使用！")

if __name__ == "__main__":
    main() 