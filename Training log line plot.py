import re
import matplotlib.pyplot as plt

# Path to log file
log_file = "training_log.txt"

# Prepare lists for each metric
epochs, D, G, G_KD, D_KD = [], [], [], [], []

# Regex pattern to parse the log lines
pattern = re.compile(
    r"Epoch (\d+) \| D: ([\d.]+) \| G: ([\d.]+) \| G_KD: ([\d.]+) \| D_KD: ([\d.]+)"
)

# Read and parse the file
with open(log_file, "r") as f:
    for line in f:
        match = pattern.search(line)
        if match:
            epochs.append(int(match.group(1)))
            D.append(float(match.group(2)))
            G.append(float(match.group(3)))
            G_KD.append(float(match.group(4)))
            D_KD.append(float(match.group(5)))

# --- Plotting ---

plt.figure(figsize=(14, 8))
plt.plot(epochs, D, label="Discriminator Loss (D)", linewidth=2)
plt.plot(epochs, G, label="Generator Loss (G)", linewidth=2)
plt.plot(epochs, G_KD, label="Generator Knowledge Distillation Loss", linewidth=2, linestyle="--")
plt.plot(epochs, D_KD, label="Discriminator Knowledge Distillation Loss", linewidth=2, linestyle="--")

plt.title("Training Progress of GAN + Knowledge Distillation", fontsize=22, weight='bold', pad=16)
plt.xlabel("Epoch", fontsize=18, weight='bold')
plt.ylabel("Loss", fontsize=18, weight='bold')
plt.grid(True, which='both', linestyle=':', alpha=0.5)
plt.legend(fontsize=16, frameon=True, loc="best")
plt.tight_layout()
plt.savefig("training_progress.png", dpi=1200)
plt.show()

print("✅ Training plot saved as training_progress.png (1200 dpi) and shown above.")
