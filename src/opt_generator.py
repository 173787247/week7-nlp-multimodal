#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第7周作业 - 任务3：OPT文本生成器
利用OPT模型进行文本生成实验，掌握Next Token Prediction原理
"""

import os
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
import matplotlib.pyplot as plt

class OPTGenerator:
    """OPT文本生成器"""
    
    def __init__(self, model_name="facebook/opt-125m"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {self.device}")
        
        # 加载OPT模型和分词器
        print(f"正在加载OPT模型: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.to(self.device)
        print(" OPT模型加载完成！")
        
        # 设置pad_token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def generate_text(self, prompt, max_length=100, temperature=0.7, top_p=0.9, do_sample=True):
        """生成文本"""
        # 编码输入
        inputs = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        
        # 生成文本
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_length=max_length,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # 解码输出
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return generated_text
    
    def continue_story(self, story_start, max_new_tokens=50):
        """续写故事"""
        print(f"故事开头: {story_start}")
        
        # 生成续写
        full_story = self.generate_text(
            story_start, 
            max_length=len(self.tokenizer.encode(story_start)) + max_new_tokens,
            temperature=0.8
        )
        
        # 提取续写部分
        continuation = full_story[len(story_start):].strip()
        print(f"续写内容: {continuation}")
        
        return full_story, continuation
    
    def generate_with_different_temperatures(self, prompt, temperatures=[0.5, 0.7, 1.0, 1.2]):
        """使用不同温度生成文本"""
        print(f"原始提示: {prompt}")
        print("\n不同温度下的生成结果:")
        
        results = {}
        for temp in temperatures:
            generated = self.generate_text(prompt, temperature=temp, max_length=80)
            results[temp] = generated
            print(f"\n温度 {temp}:")
            print(f"  {generated}")
        
        return results
    
    def analyze_generation_quality(self, prompt, num_samples=5):
        """分析生成质量"""
        print(f"分析提示: {prompt}")
        print(f"生成 {num_samples} 个样本进行分析...")
        
        samples = []
        for i in range(num_samples):
            sample = self.generate_text(prompt, max_length=60, temperature=0.7)
            samples.append(sample)
            print(f"\n样本 {i+1}: {sample}")
        
        # 计算多样性（基于长度和内容）
        lengths = [len(sample) for sample in samples]
        avg_length = np.mean(lengths)
        length_std = np.std(lengths)
        
        print(f"\n生成质量分析:")
        print(f"  平均长度: {avg_length:.1f} 字符")
        print(f"  长度标准差: {length_std:.1f}")
        print(f"  样本数量: {len(samples)}")
        
        return samples, {"avg_length": avg_length, "length_std": length_std}
    
    def demonstrate_next_token_prediction(self, prompt, num_tokens=10):
        """演示Next Token Prediction原理"""
        print(f"演示Next Token Prediction:")
        print(f"提示: {prompt}")
        
        # 逐步生成，展示每一步的预测
        current_text = prompt
        for i in range(num_tokens):
            # 编码当前文本
            inputs = self.tokenizer.encode(current_text, return_tensors="pt").to(self.device)
            
            # 获取下一个token的概率分布
            with torch.no_grad():
                outputs = self.model(inputs)
                next_token_logits = outputs.logits[0, -1, :]
                next_token_probs = torch.softmax(next_token_logits, dim=-1)
            
            # 选择最可能的token
            next_token_id = torch.argmax(next_token_probs).item()
            next_token = self.tokenizer.decode([next_token_id])
            
            # 更新文本
            current_text += next_token
            
            print(f"步骤 {i+1}: 预测token '{next_token}' -> 当前文本: {current_text}")
        
        return current_text

def main():
    print("=== 第7周作业 - 任务3：OPT文本生成器 ===")
    
    # 创建OPT生成器
    generator = OPTGenerator()
    
    # 1. 基础文本生成
    print("\n1. 基础文本生成...")
    prompts = [
        "人工智能的未来是",
        "今天我想学习",
        "最好的编程语言是"
    ]
    
    for prompt in prompts:
        generated = generator.generate_text(prompt, max_length=50)
        print(f"提示: {prompt}")
        print(f"生成: {generated}")
        print()
    
    # 2. 故事续写
    print("\n2. 故事续写实验...")
    story_starts = [
        "从前有一个程序员，他每天都在写代码。有一天，",
        "在未来的世界里，机器人已经成为了人类的伙伴。",
        "小明是一个热爱学习的学生，他最喜欢研究"
    ]
    
    for story_start in story_starts:
        full_story, continuation = generator.continue_story(story_start)
        print("-" * 50)
    
    # 3. 不同温度生成
    print("\n3. 不同温度生成实验...")
    test_prompt = "机器学习的应用包括"
    generator.generate_with_different_temperatures(test_prompt)
    
    # 4. 生成质量分析
    print("\n4. 生成质量分析...")
    analysis_prompt = "深度学习在图像识别中的应用"
    samples, metrics = generator.analyze_generation_quality(analysis_prompt)
    
    # 5. Next Token Prediction演示
    print("\n5. Next Token Prediction原理演示...")
    demo_prompt = "人工智能"
    generator.demonstrate_next_token_prediction(demo_prompt, num_tokens=8)
    
    print("\n OPT文本生成器实验完成！")

if __name__ == "__main__":
    main()
