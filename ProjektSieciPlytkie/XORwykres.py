import matplotlib.pyplot as plt
import numpy as np

# dane XOR
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 1, 1, 0])

colors = np.array(['#4A90E2' if label == 0 else '#E94E4E' for label in y])

plt.figure(figsize=(5, 5))
plt.scatter(X[:, 0], X[:, 1], c=colors, s=200, edgecolor='black', linewidth=1.2)

for i, (x1, x2) in enumerate(X):
    plt.text(x1 + 0.03, x2 + 0.03, f"({x1},{x2})", fontsize=10)

plt.xlabel("$x_1$", fontsize=12)
plt.ylabel("$x_2$", fontsize=12)
plt.title("Rozkład punktów dla problemu XOR w przestrzeni")
plt.grid(True, linestyle='--', alpha=0.5)
plt.xlim(-0.2, 1.2)
plt.ylim(-0.2, 1.2)
plt.scatter([], [], c="#4A90E2", s=200, edgecolor='black', label='klasa 0')
plt.scatter([], [], c="#E94E4E", s=200, edgecolor='black', label='klasa 1')
plt.legend(loc="upper right")
plt.tight_layout()
plt.savefig("xor_points.png", dpi=200)
plt.show()
