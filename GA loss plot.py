import numpy as np
import matplotlib.pyplot as plt

# Load data
data = np.loadtxt("ga_loss_Fear.txt", delimiter=",")
iterations = data[:, 0]
loss = data[:, 1]

plt.figure(figsize=(8, 8))

# Line plot with circles at each point
plt.plot(
    iterations, loss,
    label="Genetic Algorithm Loss (Fear)",
    color="#0066cc", linewidth=2, marker="o", markersize=5, markerfacecolor="white", markeredgewidth=2
)

plt.title("Genetic Algorithm Loss Progression (Fear Class)", fontsize=16, weight='bold', pad=16)
plt.xlabel("Iteration (Epoch)", fontsize=18, weight='bold')
plt.ylabel("Loss", fontsize=18, weight='bold')
plt.grid(True, which='both', linestyle=':', alpha=0.5)
plt.legend(fontsize=16, frameon=True, loc="best")
plt.tight_layout()
plt.savefig("ga_loss_Fear.png", dpi=1200)
plt.show()

print("✅ Genetic Algorithm loss plot saved as ga_loss_neutral.png (1200 dpi) and shown above.")
