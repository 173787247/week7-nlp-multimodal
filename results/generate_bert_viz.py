
import matplotlib.pyplot as plt
import numpy as np
import json

# 加载BERT结果
with open("results/bert_similarity_results.json", "r", encoding="utf-8") as f:
    similarity_results = json.load(f)

# 创建相似度热力图
similarity_matrix = np.array(similarity_results["similarity_matrix"])
sentences = similarity_results["sentences"]

plt.figure(figsize=(10, 8))
plt.imshow(similarity_matrix, cmap="viridis", aspect="auto")
plt.colorbar(label="相似度")

plt.xticks(range(len(sentences)), [f"句子{i+1}" for i in range(len(sentences))], rotation=45)
plt.yticks(range(len(sentences)), [f"句子{i+1}" for i in range(len(sentences))])

plt.title("BERT 句子相似度热力图", fontsize=16, fontweight="bold")
plt.xlabel("句子", fontsize=12)
plt.ylabel("句子", fontsize=12)

# 添加数值标签
for i in range(len(sentences)):
    for j in range(len(sentences)):
        plt.text(j, i, f"{similarity_matrix[i, j]:.2f}", 
                ha="center", va="center", color="white", fontweight="bold")

plt.tight_layout()
plt.savefig("results/bert_similarity_heatmap.png", dpi=300, bbox_inches="tight")
plt.close()
print(" BERT可视化已生成")
