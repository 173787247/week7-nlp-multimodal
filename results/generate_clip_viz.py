
import matplotlib.pyplot as plt
import json

# 加载CLIP训练结果
with open("results/clip_training_curves.json", "r", encoding="utf-8") as f:
    training_curves = json.load(f)

epochs = training_curves["epochs"]
loss_values = training_curves["loss"]
accuracy_values = training_curves["accuracy"]

# 创建训练曲线
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# 损失曲线
ax1.plot(epochs, loss_values, "b-o", linewidth=2, markersize=8)
ax1.set_title("CLIP 训练损失曲线", fontsize=14, fontweight="bold")
ax1.set_xlabel("训练轮次", fontsize=12)
ax1.set_ylabel("损失值", fontsize=12)
ax1.grid(True, alpha=0.3)
ax1.set_xticks(epochs)

# 准确率曲线
ax2.plot(epochs, accuracy_values, "r-o", linewidth=2, markersize=8)
ax2.set_title("CLIP 训练准确率曲线", fontsize=14, fontweight="bold")
ax2.set_xlabel("训练轮次", fontsize=12)
ax2.set_ylabel("准确率", fontsize=12)
ax2.grid(True, alpha=0.3)
ax2.set_xticks(epochs)

plt.tight_layout()
plt.savefig("results/clip_training_curves.png", dpi=300, bbox_inches="tight")
plt.close()
print(" CLIP训练曲线已生成")
