
import matplotlib.pyplot as plt
import numpy as np
import json

# 加载Word2Vec结果
with open("results/word2vec_similarity_results.json", "r", encoding="utf-8") as f:
    similarity_results = json.load(f)

# 创建可视化
plt.figure(figsize=(12, 8))
plt.title("Word2Vec 相似词关系图", fontsize=16, fontweight="bold")

y_pos = np.arange(len(similarity_results))
words = list(similarity_results.keys())
similarity_counts = [len(similarity_results[word]) for word in words]

plt.barh(y_pos, similarity_counts, color="skyblue", alpha=0.7)
plt.yticks(y_pos, words, fontsize=12)
plt.xlabel("相似词数量", fontsize=12)
plt.grid(axis="x", alpha=0.3)

for i, count in enumerate(similarity_counts):
    plt.text(count + 0.1, i, str(count), va="center", fontsize=11)

plt.tight_layout()
plt.savefig("results/word2vec_visualization.png", dpi=300, bbox_inches="tight")
plt.close()
print(" Word2Vec可视化已生成")
