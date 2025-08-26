# 第7周作业：NLP与多模态学习

## 项目概述

本项目实现了第7周作业的四个核心任务：
1. **词表示学习** - Word2Vec词嵌入模型训练
2. **句子表示学习** - BERT预训练模型句子编码
3. **生成模型实验** - OPT模型文本生成
4. **多模态训练** - CLIP模型跨模态表示学习

## 项目结构

```
week7-nlp-multimodal/
 src/                    # 源代码
    word2vec_model.py  # Word2Vec词嵌入模型
    bert_encoder.py    # BERT句子编码器
    opt_generator.py   # OPT文本生成器
    clip_multimodal.py # CLIP多模态训练
 data/                   # 数据文件
 models/                 # 训练好的模型
 results/               # 实验结果
 requirements.txt       # 依赖包
 main.py               # 主运行脚本
 README.md             # 项目说明
```

## 快速开始

1. 安装依赖：`pip install -r requirements.txt`
2. 运行所有任务：`python main.py`
3. 运行单个任务：`python src/word2vec_model.py`

## 作者
AI-FullStack课程学员
