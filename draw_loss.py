import json
import matplotlib.pyplot as plt

# 读取文件内容
file_path = "output/pretrain_dir/2025-03-12_21-54-09/log.txt"  # 请替换为实际文件路径

epochs = []
losses = []

with open(file_path, "r") as f:
    for line in f:
        data = json.loads(line.strip())
        epochs.append(data["epoch"])
        losses.append(data["train_loss"])

sampled_epochs = epochs[:5]
samples_losses = losses[:5]

# for epochs > 20, we sample them in range of 10
for i in range(5, len(epochs), 10):
    sampled_epochs.append(epochs[i] + 1)
    samples_losses.append(losses[i])
sampled_epochs[-1] = epochs[-1] + 1
samples_losses[-1] = losses[-1]

highlight_epochs = [24, max(epochs)] if 24 in epochs else [max(epochs)]
highlight_losses = [losses[epochs.index(ep)] for ep in highlight_epochs]

# 绘制 loss 与 epoch 的曲线
plt.figure(figsize=(8, 5))
plt.plot(
    sampled_epochs, samples_losses, marker="o", markerfacecolor="white", linestyle="-", color="r", label="Train Loss"
)

# 计算文本框的位置，确保文本保持在图像范围内
for ep, loss in zip(highlight_epochs, highlight_losses):
    text_x = ep + 10 if ep + 10 <= max(epochs) else ep - 45  # 避免超出右边界
    text_y = loss + 0.1 if loss + 0.1 <= max(losses) else loss - 0.02  # 避免超出上边界
    plt.annotate(
        f"Epoch {ep+1}\nLoss={loss:.4f}",
        xy=(ep + 1, loss),
        xytext=(text_x, text_y),
        fontsize=18,
        fontname="Times New Roman",
        arrowprops=dict(facecolor="black", arrowstyle="->"),
    )
plt.xlabel("Epoch", fontname="Times New Roman", fontsize=18)
plt.ylabel("Train Loss", fontname="Times New Roman", fontsize=18)
plt.title("Training Loss vs Epoch", fontname="Times New Roman", fontsize=20)
plt.legend(prop={"family": "Times New Roman"})
plt.xlim(1, 200)
plt.ylim(0.2, 0.7)
plt.xticks(fontname="Times New Roman", fontsize=16)
plt.yticks(fontname="Times New Roman", fontsize=16)
plt.grid(False)  # 取消内部网格
# 调整图形边距，使其更紧凑
plt.tight_layout()
plt.savefig("./loss_curve.png", dpi=300, bbox_inches="tight")  # 以紧凑模式保存图片
plt.show()
# plt.show()
