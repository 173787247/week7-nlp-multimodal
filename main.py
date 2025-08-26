#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第7周作业主运行脚本
整合所有四个任务：Word2Vec、BERT、OPT、CLIP
"""

import os
import sys
import time
from datetime import datetime

def run_task1_word2vec():
    """运行任务1：Word2Vec词嵌入模型"""
    print("\n" + "="*60)
    print(" 任务1：Word2Vec词嵌入模型")
    print("="*60)
    
    try:
        from src.word2vec_model import main as word2vec_main
        word2vec_main()
        return True
    except Exception as e:
        print(f" Word2Vec任务运行失败: {e}")
        return False

def run_task2_bert():
    """运行任务2：BERT句子编码器"""
    print("\n" + "="*60)
    print(" 任务2：BERT句子编码器")
    print("="*60)
    
    try:
        from src.bert_encoder import main as bert_main
        bert_main()
        return True
    except Exception as e:
        print(f" BERT任务运行失败: {e}")
        return False

def run_task3_opt():
    """运行任务3：OPT文本生成器"""
    print("\n" + "="*60)
    print(" 任务3：OPT文本生成器")
    print("="*60)
    
    try:
        from src.opt_generator import main as opt_main
        opt_main()
        return True
    except Exception as e:
        print(f" OPT任务运行失败: {e}")
        return False

def run_task4_clip():
    """运行任务4：CLIP多模态训练"""
    print("\n" + "="*60)
    print(" 任务4：CLIP多模态训练")
    print("="*60)
    
    try:
        from src.clip_multimodal import main as clip_main
        clip_main()
        return True
    except Exception as e:
        print(f" CLIP任务运行失败: {e}")
        return False

def check_dependencies():
    """检查依赖包"""
    print(" 检查依赖包...")
    
    required_packages = [
        "torch", "transformers", "gensim", "open_clip", 
        "sklearn", "matplotlib", "numpy", "jieba"
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"   {package}")
        except ImportError:
            print(f"   {package}")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n  缺少依赖包: {', '.join(missing_packages)}")
        print("请运行: pip install -r requirements.txt")
        return False
    
    print(" 所有依赖包已安装")
    return True

def create_directories():
    """创建必要的目录"""
    directories = ["models", "results", "data/text_data", "data/image_data", "data/multimodal_data"]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f" 创建目录: {directory}")

def main():
    """主函数"""
    print(" 第7周作业：NLP与多模态学习")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 检查依赖
    if not check_dependencies():
        print(" 依赖检查失败，请先安装所需包")
        return
    
    # 创建目录
    print("\n 创建项目目录...")
    create_directories()
    
    # 运行任务
    tasks = [
        ("Word2Vec词嵌入模型", run_task1_word2vec),
        ("BERT句子编码器", run_task2_bert),
        ("OPT文本生成器", run_task3_opt),
        ("CLIP多模态训练", run_task4_clip)
    ]
    
    results = []
    
    for task_name, task_func in tasks:
        print(f"\n{'='*20} 开始 {task_name} {'='*20}")
        start_time = time.time()
        
        success = task_func()
        end_time = time.time()
        
        duration = end_time - start_time
        status = " 成功" if success else " 失败"
        
        results.append((task_name, success, duration))
        
        print(f"{task_name}: {status} (耗时: {duration:.2f}秒)")
    
    # 总结结果
    print("\n" + "="*60)
    print(" 任务执行总结")
    print("="*60)
    
    successful_tasks = 0
    for task_name, success, duration in results:
        status = " 成功" if success else " 失败"
        print(f"{task_name:<20} {status:<10} {duration:>8.2f}秒")
        if success:
            successful_tasks += 1
    
    print(f"\n总任务数: {len(tasks)}")
    print(f"成功任务: {successful_tasks}")
    print(f"失败任务: {len(tasks) - successful_tasks}")
    print(f"成功率: {successful_tasks/len(tasks)*100:.1f}%")
    
    if successful_tasks == len(tasks):
        print("\n 恭喜！所有任务都成功完成！")
    else:
        print(f"\n  有 {len(tasks) - successful_tasks} 个任务失败，请检查错误信息")
    
    print(f"\n结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(" 生成的文件:")
    print("  - models/ (训练好的模型)")
    print("  - results/ (实验结果和可视化)")
    print("  - data/ (数据文件)")

if __name__ == "__main__":
    main()
