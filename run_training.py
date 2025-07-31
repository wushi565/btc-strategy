#!/usr/bin/env python3
"""
ML训练器快速启动脚本

使用方法:
    python run_training.py              # 启动交互式训练管理器
    python run_training.py --auto       # 自动恢复未完成训练
    python run_training.py --new        # 直接开始新训练
    python run_training.py --status     # 查看训练状态
    python run_training.py --cleanup    # 清理检查点
"""

import sys
import os
import argparse
from ml_training_manager import TrainingManager

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='ML训练管理器',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
    python run_training.py              # 启动交互式界面
    python run_training.py --auto       # 自动恢复训练
    python run_training.py --new        # 开始新训练
    python run_training.py --status     # 查看状态
    python run_training.py --cleanup    # 清理检查点
        """
    )
    
    parser.add_argument(
        '--config', '-c',
        default='config.yaml',
        help='配置文件路径 (默认: config.yaml)'
    )
    
    parser.add_argument(
        '--auto', '-a',
        action='store_true',
        help='自动恢复最新的未完成训练'
    )
    
    parser.add_argument(
        '--new', '-n',
        action='store_true',
        help='直接开始新训练'
    )
    
    parser.add_argument(
        '--status', '-s',
        action='store_true',
        help='查看训练状态'
    )
    
    parser.add_argument(
        '--cleanup',
        action='store_true',
        help='清理已完成的检查点'
    )
    
    parser.add_argument(
        '--list', '-l',
        action='store_true',
        help='列出所有训练记录'
    )
    
    parser.add_argument(
        '--resume',
        metavar='SESSION_ID',
        help='恢复指定的训练会话'
    )
    
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='静默模式，减少输出'
    )
    
    return parser.parse_args()

def main():
    """主函数"""
    args = parse_arguments()
    
    try:
        # 创建训练管理器
        if not args.quiet:
            print("🚀 初始化ML训练管理器...")
        
        manager = TrainingManager(args.config)
        
        # 根据参数执行相应操作
        if args.status:
            # 查看状态
            manager.view_training_status()
            
        elif args.list:
            # 列出训练记录
            manager.view_training_history()
            
        elif args.cleanup:
            # 清理检查点
            manager.cleanup_checkpoints()
            
        elif args.resume:
            # 恢复指定训练
            print(f"🔄 恢复训练会话: {args.resume}")
            success = manager.resume_specific_training(args.resume)
            if not success:
                sys.exit(1)
                
        elif args.auto:
            # 自动恢复训练
            print("🔍 检查未完成的训练...")
            resumed = manager.startup_check()
            
            if not resumed:
                print("📭 没有找到未完成的训练")
                if not args.quiet:
                    choice = input("是否开始新训练？ (y/n): ").strip().lower()
                    if choice == 'y':
                        manager.start_new_training()
                        
        elif args.new:
            # 开始新训练
            print("🚀 开始新训练...")
            manager.start_new_training()
            
        else:
            # 默认启动交互式界面
            manager.run()
            
    except KeyboardInterrupt:
        print("\n⚡ 操作被用户取消")
        sys.exit(1)
    except Exception as e:
        print(f"❌ 运行失败: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()