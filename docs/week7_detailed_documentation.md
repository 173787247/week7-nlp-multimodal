# 第7周作业：NLP与多模态学习详细文档

## 项目概述

本项目实现了第7周作业的四个核心任务，涵盖了自然语言处理和多模态学习的核心概念和实践应用。

## 任务详解

### 任务1：Word2Vec词嵌入模型

#### 目标
训练一个Word2Vec词嵌入模型，理解词向量的生成与语义捕捉。

#### 实现原理
- 使用CBOW（Continuous Bag of Words）和Skip-gram模型
- 通过神经网络学习词汇的分布式表示
- 捕捉词汇之间的语义相似性

#### 核心功能
- 模型训练：基于中文和英文文本数据
- 相似词查找：计算词汇间的语义相似度
- 可视化：使用t-SNE降维展示词向量分布

#### 使用方法
```python
from src.word2vec_model import train_word2vec, find_similar_words

# 训练模型
model = train_word2vec()

# 查找相似词
find_similar_words(model, "人工智能")
```

### 任务2：BERT句子编码器

#### 目标
使用BERT预训练模型，提取句子级别的向量表示，感受上下文语义的编码能力。

#### 实现原理
- 基于Transformer架构的预训练语言模型
- 双向编码器，能够理解词汇的上下文信息
- 使用掩码语言模型和下一句预测任务预训练

#### 核心功能
- 句子编码：将句子转换为固定维度的向量
- 相似度计算：计算句子间的语义相似度
- 可视化：展示句子向量在二维空间中的分布

#### 使用方法
```python
from src.bert_encoder import BERTEncoder

encoder = BERTEncoder()
embedding = encoder.encode_sentence("这是一个测试句子")
```

### 任务3：OPT文本生成器

#### 目标
利用OPT模型进行文本生成实验，掌握Next Token Prediction原理。

#### 实现原理
- 基于Transformer的解码器架构
- 自回归生成：根据前面的词汇预测下一个词汇
- 支持不同的采样策略（贪婪搜索、束搜索、温度采样）

#### 核心功能
- 文本生成：根据提示词生成连续文本
- 故事续写：基于开头续写完整故事
- 质量分析：评估生成文本的质量和多样性

#### 使用方法
```python
from src.opt_generator import OPTGenerator

generator = OPTGenerator()
text = generator.generate_text("人工智能的未来是", max_length=50)
```

### 任务4：CLIP多模态训练

#### 目标
尝试基于文本-图像对数据，微调增强CLIP模型的跨模态表示学习能力。

#### 实现原理
- 对比学习：学习文本和图像的对应关系
- 双塔架构：分别编码文本和图像特征
- 跨模态对齐：将不同模态的特征映射到同一空间

#### 核心功能
- 图像-文本匹配：判断图像和文本是否匹配
- 零样本分类：无需训练即可进行图像分类
- 跨模态检索：根据文本查找相关图像或反之

#### 使用方法
```python
from src.clip_multimodal import CLIPMultimodalTrainer

trainer = CLIPMultimodalTrainer()
# 训练和测试CLIP模型
```

## 技术架构

### 项目结构
```
week7-nlp-multimodal/
 src/                    # 核心源代码
    word2vec_model.py  # Word2Vec实现
    bert_encoder.py    # BERT编码器
    opt_generator.py   # OPT生成器
    clip_multimodal.py # CLIP多模态训练
 data/                   # 数据文件
    text_data/         # 文本数据
 models/                 # 训练好的模型
 results/                # 实验结果和可视化
 notebooks/              # Jupyter笔记本
 docs/                   # 详细文档
 requirements.txt        # 依赖包
 main.py                # 主运行脚本
 train_all_models.py    # 模型训练脚本
```

### 依赖技术栈
- **深度学习框架**: PyTorch
- **NLP库**: Transformers, Gensim, NLTK, Jieba
- **数据处理**: NumPy, Pandas
- **可视化**: Matplotlib, Seaborn
- **机器学习**: Scikit-learn

## 实验结果

### Word2Vec模型
- 成功训练词嵌入模型
- 能够捕捉词汇语义相似性
- 生成高质量的词向量可视化

### BERT编码器
- 成功加载预训练模型
- 生成上下文感知的句子向量
- 准确计算句子相似度

### OPT生成器
- 成功实现文本生成功能
- 支持多种生成策略
- 生成质量良好

### CLIP多模态
- 成功实现多模态训练框架
- 支持图像-文本匹配
- 具备零样本分类能力

## 使用说明

### 环境要求
- Python 3.8+
- CUDA支持（可选，用于GPU加速）
- 8GB+ 内存

### 安装步骤
1. 克隆项目
2. 安装依赖：`pip install -r requirements.txt`
3. 运行主程序：`python main.py`
4. 训练模型：`python train_all_models.py`

### 运行示例
```bash
# 运行所有实验
python main.py

# 训练所有模型
python train_all_models.py

# 运行Jupyter笔记本
jupyter notebook notebooks/week7_experiments.ipynb
```

## 注意事项

1. **模型下载**: BERT和OPT模型首次运行时会自动下载，需要网络连接
2. **内存要求**: 大模型需要足够的内存，建议8GB以上
3. **GPU加速**: 如有CUDA环境，可显著提升训练速度
4. **数据路径**: 确保数据文件路径正确

## 扩展建议

1. **数据增强**: 添加更多训练数据提升模型性能
2. **模型优化**: 尝试不同的超参数和架构
3. **应用场景**: 将模型应用到实际业务场景
4. **性能评估**: 添加更全面的评估指标

## 总结

本第7周作业成功实现了NLP与多模态学习的四个核心任务，涵盖了从词嵌入到句子编码，从文本生成到多模态学习的完整技术栈。通过实践，深入理解了这些技术的原理和应用，为后续的AI应用开发奠定了坚实基础。
