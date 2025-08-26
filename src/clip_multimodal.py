#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第7周作业 - 任务4：CLIP多模态训练
基于文本-图像对数据，微调增强CLIP模型的跨模态表示学习能力
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import open_clip
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
import json

class MultimodalDataset(Dataset):
    """多模态数据集"""
    
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.samples = []
        
        # 创建示例数据（文本-图像对）
        self.samples = [
            {"text": "一只可爱的小猫", "image": "cat.jpg", "category": "cat"},
            {"text": "一只忠诚的狗", "image": "dog.jpg", "category": "dog"},
            {"text": "一朵美丽的花", "image": "flower.jpg", "category": "flower"},
            {"text": "一辆红色的汽车", "image": "car.jpg", "category": "car"},
            {"text": "一座古老的建筑", "image": "building.jpg", "category": "building"},
            {"text": "一片绿色的森林", "image": "forest.jpg", "category": "forest"},
            {"text": "一杯香浓的咖啡", "image": "coffee.jpg", "category": "coffee"},
            {"text": "一本有趣的书", "image": "book.jpg", "category": "book"}
        ]
        
        print(f"创建了 {len(self.samples)} 个多模态样本")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        text = sample["text"]
        image_path = os.path.join(self.data_dir, sample["image"])
        category = sample["category"]
        
        # 处理图像
        try:
            image = Image.open(image_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
        except:
            # 如果图像不存在，创建占位图像
            image = torch.zeros(3, 224, 224)
        
        return text, image, category

class CLIPMultimodalTrainer:
    """CLIP多模态训练器"""
    
    def __init__(self, model_name="ViT-B-32", pretrained="openai"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {self.device}")
        
        # 加载CLIP模型
        print(f"正在加载CLIP模型: {model_name}")
        self.model, self.preprocess, _ = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, device=self.device
        )
        print(" CLIP模型加载完成！")
        
        # 设置训练模式
        self.model.train()
        
        # 损失函数和优化器
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(self.model.parameters(), lr=1e-5)
    
    def create_dummy_images(self, data_dir):
        """创建虚拟图像数据用于演示"""
        os.makedirs(data_dir, exist_ok=True)
        
        # 创建简单的彩色图像
        dummy_images = {
            "cat.jpg": [255, 100, 100],      # 红色
            "dog.jpg": [100, 255, 100],      # 绿色
            "flower.jpg": [100, 100, 255],   # 蓝色
            "car.jpg": [255, 255, 100],      # 黄色
            "building.jpg": [255, 100, 255], # 紫色
            "forest.jpg": [100, 255, 255],   # 青色
            "coffee.jpg": [150, 75, 0],      # 棕色
            "book.jpg": [128, 128, 128]      # 灰色
        }
        
        for filename, color in dummy_images.items():
            # 创建224x224的图像
            img_array = np.full((224, 224, 3), color, dtype=np.uint8)
            img = Image.fromarray(img_array)
            img.save(os.path.join(data_dir, filename))
        
        print(f" 创建了 {len(dummy_images)} 个虚拟图像文件")
    
    def train_epoch(self, dataloader):
        """训练一个epoch"""
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, (texts, images, categories) in enumerate(dataloader):
            images = images.to(self.device)
            
            # 编码图像和文本
            image_features = self.model.encode_image(images)
            text_features = self.model.encode_text(texts)
            
            # 归一化特征
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
            # 计算相似度矩阵
            logits = torch.matmul(image_features, text_features.T) * 100
            
            # 创建标签（对角线为匹配的文本-图像对）
            labels = torch.arange(len(texts)).to(self.device)
            
            # 计算损失
            loss = self.criterion(logits, labels)
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            if batch_idx % 10 == 0:
                print(f"Batch {batch_idx}, Loss: {loss.item():.4f}")
        
        return total_loss / num_batches
    
    def evaluate_model(self, dataloader):
        """评估模型性能"""
        self.model.eval()
        
        all_image_features = []
        all_text_features = []
        all_categories = []
        
        with torch.no_grad():
            for texts, images, categories in dataloader:
                images = images.to(self.device)
                
                # 编码特征
                image_features = self.model.encode_image(images)
                text_features = self.model.encode_text(texts)
                
                # 归一化
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                
                all_image_features.append(image_features.cpu())
                all_text_features.append(text_features.cpu())
                all_categories.extend(categories)
        
        # 合并所有特征
        image_features = torch.cat(all_image_features, dim=0)
        text_features = torch.cat(all_text_features, dim=0)
        
        # 计算相似度矩阵
        similarity_matrix = cosine_similarity(image_features, text_features)
        
        # 计算准确率（对角线匹配）
        correct = 0
        total = len(similarity_matrix)
        
        for i in range(total):
            if i < len(similarity_matrix[i]):
                predicted = np.argmax(similarity_matrix[i])
                if predicted == i:
                    correct += 1
        
        accuracy = correct / total
        
        print(f"模型评估结果:")
        print(f"  总样本数: {total}")
        print(f"  正确匹配: {correct}")
        print(f"  准确率: {accuracy:.4f}")
        
        return accuracy, similarity_matrix
    
    def zero_shot_classification(self, image_path, candidate_texts):
        """零样本分类"""
        self.model.eval()
        
        # 加载图像
        try:
            image = Image.open(image_path).convert("RGB")
            image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        except:
            print(f"无法加载图像: {image_path}")
            return None
        
        # 编码图像和候选文本
        with torch.no_grad():
            image_features = self.model.encode_image(image_tensor)
            text_features = self.model.encode_text(candidate_texts)
            
            # 归一化
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
            # 计算相似度
            similarities = torch.matmul(image_features, text_features.T)[0]
        
        # 排序结果
        results = []
        for i, (text, similarity) in enumerate(zip(candidate_texts, similarities)):
            results.append((text, similarity.item()))
        
        results.sort(key=lambda x: x[1], reverse=True)
        return results
    
    def save_model(self, filepath):
        """保存训练好的模型"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save(self.model.state_dict(), filepath)
        print(f" 模型已保存到: {filepath}")

def main():
    print("=== 第7周作业 - 任务4：CLIP多模态训练 ===")
    
    # 创建训练器
    trainer = CLIPMultimodalTrainer()
    
    # 创建虚拟数据
    data_dir = "data/multimodal_data"
    trainer.create_dummy_images(data_dir)
    
    # 创建数据集和数据加载器
    dataset = MultimodalDataset(data_dir, trainer.preprocess)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    # 1. 训练前评估
    print("\n1. 训练前模型评估...")
    initial_accuracy, initial_similarity = trainer.evaluate_model(dataloader)
    
    # 2. 微调训练
    print("\n2. 开始微调训练...")
    num_epochs = 5
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        avg_loss = trainer.train_epoch(dataloader)
        print(f"平均损失: {avg_loss:.4f}")
    
    # 3. 训练后评估
    print("\n3. 训练后模型评估...")
    final_accuracy, final_similarity = trainer.evaluate_model(dataloader)
    
    # 4. 零样本分类测试
    print("\n4. 零样本分类测试...")
    candidate_texts = [
        "一只猫", "一只狗", "一朵花", "一辆车",
        "一座建筑", "一片森林", "一杯咖啡", "一本书"
    ]
    
    # 测试几个图像
    test_images = ["cat.jpg", "dog.jpg", "flower.jpg"]
    for img_name in test_images:
        img_path = os.path.join(data_dir, img_name)
        results = trainer.zero_shot_classification(img_path, candidate_texts)
        
        if results:
            print(f"\n图像 {img_name} 的分类结果:")
            for i, (text, similarity) in enumerate(results[:3], 1):
                print(f"  {i}. {text} (相似度: {similarity:.4f})")
    
    # 5. 保存模型
    print("\n5. 保存训练好的模型...")
    trainer.save_model("models/clip_multimodal_finetuned.pth")
    
    # 6. 结果对比
    print("\n6. 训练效果对比:")
    print(f"  训练前准确率: {initial_accuracy:.4f}")
    print(f"  训练后准确率: {final_accuracy:.4f}")
    print(f"  提升: {final_accuracy - initial_accuracy:.4f}")
    
    print("\n CLIP多模态训练实验完成！")

if __name__ == "__main__":
    main()
