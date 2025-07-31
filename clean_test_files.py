#!/usr/bin/env python3
"""
清理测试代码脚本

用于删除项目中的测试、调试、示例和临时文件，
以便整合成一个连贯的生产系统。
"""

import os
import shutil
import glob
import sys

# 要删除的文件模式
TEST_FILE_PATTERNS = [
    "test_*.py",       # 测试文件
    "debug_*.py",      # 调试文件
    "verify_*.py",     # 验证文件
    "*_test.py",       # 测试文件
    "*_mock.py",       # 模拟文件
    "getting_started_*.py",  # 入门示例
    "usage_tutorial.py",     # 使用教程
    "analyze_trades.py",     # 分析工具
    "benchmark_performance.py", # 性能测试
    "final_ml_verification.py"  # 最终验证
]

# 要删除的单独文件
SPECIFIC_FILES = [
    "ml_trading_system.py"  # 单文件版本的ML系统
]

# 要删除的目录
DIRECTORIES_TO_REMOVE = [
    "debug_models",
    "demo_models",
    "test_models"
]

def confirm_deletion(path):
    """确认是否删除文件或目录"""
    response = input(f"确认删除 '{path}'? (y/n): ")
    return response.lower() in ["y", "yes"]

def delete_files():
    """删除测试和调试文件"""
    deleted_files = []
    skipped_files = []
    
    # 删除匹配模式的文件
    for pattern in TEST_FILE_PATTERNS:
        for file_path in glob.glob(pattern):
            if os.path.isfile(file_path):
                if confirm_deletion(file_path):
                    try:
                        os.remove(file_path)
                        deleted_files.append(file_path)
                        print(f"已删除文件: {file_path}")
                    except Exception as e:
                        print(f"删除文件 {file_path} 失败: {e}")
                else:
                    skipped_files.append(file_path)
                    print(f"已跳过: {file_path}")
    
    # 删除特定文件
    for file_path in SPECIFIC_FILES:
        if os.path.isfile(file_path):
            if confirm_deletion(file_path):
                try:
                    os.remove(file_path)
                    deleted_files.append(file_path)
                    print(f"已删除文件: {file_path}")
                except Exception as e:
                    print(f"删除文件 {file_path} 失败: {e}")
            else:
                skipped_files.append(file_path)
                print(f"已跳过: {file_path}")
    
    return deleted_files, skipped_files

def delete_directories():
    """删除测试和示例目录"""
    deleted_dirs = []
    skipped_dirs = []
    
    for dir_name in DIRECTORIES_TO_REMOVE:
        if os.path.isdir(dir_name):
            if confirm_deletion(dir_name):
                try:
                    shutil.rmtree(dir_name)
                    deleted_dirs.append(dir_name)
                    print(f"已删除目录: {dir_name}")
                except Exception as e:
                    print(f"删除目录 {dir_name} 失败: {e}")
            else:
                skipped_dirs.append(dir_name)
                print(f"已跳过: {dir_name}")
    
    return deleted_dirs, skipped_dirs

def main():
    """主函数"""
    print("===== 测试代码清理工具 =====")
    print("此工具将删除项目中的测试、调试和示例文件，以整合成一个连贯的系统。")
    print("警告：删除操作不可恢复，请确保已备份重要文件！\n")
    
    response = input("是否继续? (y/n): ")
    if response.lower() not in ["y", "yes"]:
        print("操作已取消")
        return
    
    print("\n=== 开始清理 ===")
    
    # 删除文件
    print("\n删除测试和调试文件...")
    deleted_files, skipped_files = delete_files()
    
    # 删除目录
    print("\n删除测试和示例目录...")
    deleted_dirs, skipped_dirs = delete_directories()
    
    # 显示结果
    print("\n=== 清理完成 ===")
    print(f"已删除 {len(deleted_files)} 个文件")
    print(f"已删除 {len(deleted_dirs)} 个目录")
    print(f"已跳过 {len(skipped_files)} 个文件")
    print(f"已跳过 {len(skipped_dirs)} 个目录")
    
    print("\n系统清理完成，现在可以使用整合后的系统。")

if __name__ == "__main__":
    main() 