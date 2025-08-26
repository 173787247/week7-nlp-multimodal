#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
环境测试脚本
检查第7周作业所需的所有依赖包是否正确安装
"""

import sys
import importlib

def test_import(package_name, display_name=None):
    """测试包导入"""
    if display_name is None:
        display_name = package_name
    
    try:
        importlib.import_module(package_name)
        print(f" {display_name}")
        return True
    except ImportError:
        print(f" {display_name}")
        return False

def main():
    """主函数"""
    print(" 第7周作业环境检查")
    print("="*40)
    
    # 核心依赖包
    core_packages = [
        ("torch", "PyTorch"),
        ("torchvision", "TorchVision"),
        ("transformers", "Transformers"),
        ("gensim", "Gensim"),
        ("open_clip", "OpenCLIP"),
        ("sklearn", "Scikit-learn"),
        ("matplotlib", "Matplotlib"),
        ("numpy", "NumPy"),
        ("jieba", "Jieba"),
        ("PIL", "Pillow")
    ]
    
    # 测试导入
    success_count = 0
    total_count = len(core_packages)
    
    for package, display_name in core_packages:
        if test_import(package, display_name):
            success_count += 1
    
    print("\n" + "="*40)
    print(f" 检查结果: {success_count}/{total_count} 个包可用")
    
    if success_count == total_count:
        print(" 环境检查通过！可以开始第7周作业")
        print("\n 运行方式:")
        print("  python main.py                    # 运行所有任务")
        print("  python src/word2vec_model.py      # 运行Word2Vec任务")
        print("  python src/bert_encoder.py        # 运行BERT任务")
        print("  python src/opt_generator.py       # 运行OPT任务")
        print("  python src/clip_multimodal.py     # 运行CLIP任务")
    else:
        print(" 环境检查失败！请安装缺失的包")
        print("\n 安装命令:")
        print("  pip install -r requirements.txt")
    
    return success_count == total_count

if __name__ == "__main__":
    main()
