#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第7周作业 - 任务1：Word2Vec词嵌入模型
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from gensim.models import Word2Vec
from sklearn.manifold import TSNE
import jieba
from collections import Counter

def load_sample_texts():
    """加载示例文本数据"""
    sample_texts = [
        "人工智能是计算机科学的一个分支，致力于创建能够执行通常需要人类智能的任务的系统。",
        "机器学习是人工智能的一个子集，它使计算机能够在没有明确编程的情况下学习和改进。",
        "深度学习是机器学习的一个分支，使用多层神经网络来模拟人脑的学习过程。",
        "自然语言处理是人工智能的一个领域，专注于计算机理解和生成人类语言的能力。",
        "计算机视觉是人工智能的一个分支，使计算机能够从图像和视频中获取信息。",
        "神经网络是受生物神经网络启发的计算模型，用于模式识别和机器学习。",
        "大数据是指传统数据处理软件无法处理的庞大、复杂的数据集。",
        "云计算是一种通过互联网提供计算服务的模型，包括服务器、存储、数据库等。",
        "物联网是指通过互联网连接的物理设备网络，能够收集和交换数据。",
        "区块链是一种分布式账本技术，以安全、透明和不可变的方式记录交易。"
    ]
    
    # 分词处理
    processed_sentences = []
    for text in sample_texts:
        words = jieba.lcut(text)
        words = [word for word in words if len(word) > 1]
        if len(words) > 2:
            processed_sentences.append(words)
    
    print(f"加载了 {len(processed_sentences)} 个训练句子")
    return processed_sentences

def train_word2vec(sentences):
    """训练Word2Vec模型"""
    print("开始训练Word2Vec模型...")
    
    model = Word2Vec(
        sentences=sentences,
        vector_size=100,
        window=5,
        min_count=1,
        workers=4,
        sg=0,
        epochs=100
    )
    
    print("Word2Vec模型训练完成！")
    print(f"词汇表大小: {len(model.wv.key_to_index)}")
    
    return model

def find_similar_words(model, word, top_n=5):
    """查找相似词"""
    try:
        similar_words = model.wv.most_similar(word, topn=top_n)
        print(f"\n与 '{word}' 最相似的词:")
        for i, (similar_word, similarity) in enumerate(similar_words, 1):
            print(f"  {i}. {similar_word} (相似度: {similarity:.4f})")
        return similar_words
    except KeyError:
        print(f"词汇 '{word}' 不在词汇表中")
        return []

def visualize_word_vectors(model, words, save_path):
    """可视化词向量"""
    # 提取词向量
    word_vectors = []
    valid_words = []
    for word in words:
        if word in model.wv.key_to_index:
            word_vectors.append(model.wv[word])
            valid_words.append(word)
    
    if len(word_vectors) < 2:
        print("有效词汇数量不足")
        return
    
    # t-SNE降维
    tsne = TSNE(n_components=2, random_state=42)
    word_vectors_2d = tsne.fit_transform(word_vectors)
    
    # 绘制
    plt.figure(figsize=(10, 8))
    plt.scatter(word_vectors_2d[:, 0], word_vectors_2d[:, 1])
    
    for i, word in enumerate(valid_words):
        plt.annotate(word, (word_vectors_2d[i, 0], word_vectors_2d[i, 1]))
    
    plt.title("Word2Vec词向量可视化")
    plt.xlabel("t-SNE维度1")
    plt.ylabel("t-SNE维度2")
    plt.grid(True)
    
    # 保存
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"可视化图已保存到: {save_path}")
    
    plt.show()

def main():
    print("=== 第7周作业 - 任务1：Word2Vec词嵌入模型 ===")
    
    # 1. 加载数据
    print("\n1. 加载训练数据...")
    sentences = load_sample_texts()
    
    # 2. 训练模型
    print("\n2. 训练Word2Vec模型...")
    model = train_word2vec(sentences)
    
    # 3. 保存模型
    print("\n3. 保存模型...")
    os.makedirs("models", exist_ok=True)
    model.save("models/word2vec_model.model")
    
    # 4. 测试相似词
    print("\n4. 测试相似词查找...")
    test_words = ["人工智能", "机器学习", "深度学习"]
    for word in test_words:
        find_similar_words(model, word)
    
    # 5. 可视化
    print("\n5. 词向量可视化...")
    common_words = ["人工智能", "机器学习", "深度学习", "神经网络", "自然语言处理"]
    visualize_word_vectors(model, common_words, "results/word2vec_visualization.png")
    
    print("\n Word2Vec实验完成！")

if __name__ == "__main__":
    main()
