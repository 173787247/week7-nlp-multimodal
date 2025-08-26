#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第7周作业 - 任务2：BERT句子编码器
使用BERT预训练模型提取句子级别的向量表示
"""

import os
import torch
import numpy as np
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

class BERTEncoder:
    """BERT句子编码器"""
    
    def __init__(self, model_name="bert-base-chinese"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {self.device}")
        
        # 加载BERT模型和分词器
        print(f"正在加载BERT模型: {model_name}")
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        print(" BERT模型加载完成！")
    
    def encode_sentence(self, sentence):
        """编码单个句子"""
        # 分词
        inputs = self.tokenizer(sentence, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # 编码
        with torch.no_grad():
            outputs = self.model(**inputs)
            # 使用[CLS]标记的输出作为句子表示
            sentence_embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        
        return sentence_embedding.flatten()
    
    def encode_sentences(self, sentences):
        """编码多个句子"""
        embeddings = []
        for sentence in sentences:
            embedding = self.encode_sentence(sentence)
            embeddings.append(embedding)
        
        return np.array(embeddings)
    
    def calculate_similarity(self, sentence1, sentence2):
        """计算两个句子的相似度"""
        emb1 = self.encode_sentence(sentence1)
        emb2 = self.encode_sentence(sentence2)
        
        similarity = cosine_similarity(emb1.reshape(1, -1), emb2.reshape(1, -1))[0][0]
        return similarity
    
    def find_most_similar(self, query_sentence, candidate_sentences, top_k=3):
        """找到最相似的句子"""
        query_embedding = self.encode_sentence(query_sentence)
        candidate_embeddings = self.encode_sentences(candidate_sentences)
        
        similarities = cosine_similarity(query_embedding.reshape(1, -1), candidate_embeddings)[0]
        
        # 排序
        sorted_indices = np.argsort(similarities)[::-1]
        
        results = []
        for i in range(min(top_k, len(sorted_indices))):
            idx = sorted_indices[i]
            results.append((candidate_sentences[idx], similarities[idx]))
        
        return results
    
    def visualize_sentences(self, sentences, save_path=None):
        """可视化句子向量"""
        embeddings = self.encode_sentences(sentences)
        
        # t-SNE降维
        print("使用t-SNE降维...")
        tsne = TSNE(n_components=2, random_state=42)
        embeddings_2d = tsne.fit_transform(embeddings)
        
        # 绘制
        plt.figure(figsize=(12, 8))
        plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.7)
        
        # 添加标签
        for i, sentence in enumerate(sentences):
            plt.annotate(f"S{i+1}", (embeddings_2d[i, 0], embeddings_2d[i, 1]), 
                        fontsize=10, ha="center", va="center")
        
        plt.title("BERT句子向量可视化 (t-SNE降维)")
        plt.xlabel("t-SNE维度1")
        plt.ylabel("t-SNE维度2")
        plt.grid(True, alpha=0.3)
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"可视化图已保存到: {save_path}")
        
        plt.show()

def main():
    print("=== 第7周作业 - 任务2：BERT句子编码器 ===")
    
    # 创建BERT编码器
    encoder = BERTEncoder()
    
    # 测试句子
    test_sentences = [
        "人工智能是计算机科学的一个分支。",
        "机器学习使计算机能够自动学习和改进。",
        "深度学习使用神经网络进行模式识别。",
        "自然语言处理专注于计算机理解人类语言。",
        "计算机视觉使计算机能够理解图像和视频。",
        "今天天气很好，适合出去散步。",
        "我喜欢吃苹果和橙子。",
        "这本书讲述了人工智能的发展历史。",
        "神经网络模拟人脑的学习过程。",
        "大数据分析帮助企业做出更好的决策。"
    ]
    
    # 1. 句子编码测试
    print("\n1. 句子编码测试...")
    for i, sentence in enumerate(test_sentences[:3], 1):
        embedding = encoder.encode_sentence(sentence)
        print(f"句子{i}: {sentence}")
        print(f"向量维度: {embedding.shape}")
        print(f"向量范围: [{embedding.min():.4f}, {embedding.max():.4f}]")
        print()
    
    # 2. 句子相似度计算
    print("\n2. 句子相似度计算...")
    sentence_pairs = [
        ("人工智能是计算机科学的一个分支。", "机器学习使计算机能够自动学习和改进。"),
        ("深度学习使用神经网络进行模式识别。", "神经网络模拟人脑的学习过程。"),
        ("今天天气很好，适合出去散步。", "我喜欢吃苹果和橙子。")
    ]
    
    for sent1, sent2 in sentence_pairs:
        similarity = encoder.calculate_similarity(sent1, sent2)
        print(f"句子1: {sent1}")
        print(f"句子2: {sent2}")
        print(f"相似度: {similarity:.4f}")
        print()
    
    # 3. 最相似句子查找
    print("\n3. 最相似句子查找...")
    query = "人工智能和机器学习的关系"
    similar_sentences = encoder.find_most_similar(query, test_sentences, top_k=3)
    
    print(f"查询: {query}")
    print("最相似的句子:")
    for i, (sentence, similarity) in enumerate(similar_sentences, 1):
        print(f"  {i}. {sentence} (相似度: {similarity:.4f})")
    
    # 4. 句子向量可视化
    print("\n4. 句子向量可视化...")
    encoder.visualize_sentences(test_sentences, "results/bert_sentences_visualization.png")
    
    print("\n BERT句子编码器实验完成！")

if __name__ == "__main__":
    main()
