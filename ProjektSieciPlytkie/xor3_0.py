import numpy as np
import matplotlib.pyplot as plt

# Dane
x = np.array([
    [0., 0.],
    [0., 1.],
    [1., 0.],
    [1., 1.]
], dtype=float)

y_true = np.array([
    [0.],
    [1.],
    [1.],
    [0.]
], dtype=float)

assert x.shape == (4, 2)
assert y_true.shape == (4, 1)
print(x.shape, y_true.shape, x.dtype)  # test


# Parametry
n_hidden  = 2
lr        = 0.5
n_epochs  = 10000
mse_target = 0.01

# Inicjalizacja wag
rng = np.random.default_rng()
W_h = rng.uniform(-0.5, 0.5, size=(n_hidden, x.shape[1]))
b_h = rng.uniform(-0.5, 0.5, size=(n_hidden, 1))
W_o = rng.uniform(-0.5, 0.5, size=(1, n_hidden))
b_o = rng.uniform(-0.5, 0.5, size=(1, 1))

print("\nW_h =", W_h)
print("b_h =", b_h)
print("W_o =", W_o)
print("b_o =", b_o)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def dsigmoid(a):
    return a * (1.0 - a)


# Historia
m = x.shape[0]
mse_hist, acc_hist = [], []
mse_per_sample_hist = []
W_h_hist, W_o_hist = [], []

for ep in range(1, n_epochs + 1):

    z_h  = (W_h @ x.T) + b_h
    h    = sigmoid(z_h)
    z_o  = (W_o @ h) + b_o
    y_hat = sigmoid(z_o)

    y_row = y_true.T
    mse   = np.mean((y_hat - y_row) ** 2)
    y_cls = (y_hat.T >= 0.5).astype(int)
    acc   = np.mean(y_cls == y_true)

    mse_hist.append(mse)
    acc_hist.append(acc)
    mse_per_sample_hist.append(((y_hat - y_row) ** 2).ravel())
    W_h_hist.append(W_h.copy())
    W_o_hist.append(W_o.copy())

    dZ_o = (y_hat - y_row) * dsigmoid(y_hat)
    dW_o = (dZ_o @ h.T) / m
    db_o = np.sum(dZ_o, axis=1, keepdims=True) / m

    dA_h = W_o.T @ dZ_o
    dZ_h = dA_h * dsigmoid(h)
    dW_h = (dZ_h @ x) / m
    db_h = np.sum(dZ_h, axis=1, keepdims=True) / m

    W_o -= lr * dW_o        # UPDATE
    b_o -= lr * db_o
    W_h -= lr * dW_h
    b_h -= lr * db_h

    if ep % 1000 == 0 or mse < mse_target:
        print(f"epoka {ep:5d}: MSE={mse:.6f}, acc={acc*100:.1f}%")
    if mse < mse_target:
        print(f"Spelniono prog blędu {mse_target} w epoce {ep}.")
        break

z_h  = (W_h @ x.T) + b_h
h    = sigmoid(z_h)
z_o  = (W_o @ h) + b_o
y_hat_final = sigmoid(z_o).T
y_cls_final = (y_hat_final >= 0.5).astype(int)

print("\nPrzewidywania po uczeniu:")
for i in range(x.shape[0]):
    print(f"{x[i]} -> y_hat={y_hat_final[i,0]:.4f}, y={y_cls_final[i,0]}")

# Wyniki
epochs_axis = np.arange(1, len(mse_hist) + 1)
mse_per_sample_hist = np.array(mse_per_sample_hist)
W_h_hist = np.array(W_h_hist)
W_o_hist = np.array(W_o_hist)


plt.figure()
plt.plot(epochs_axis, mse_hist)
plt.xlabel("Epoka"); plt.ylabel("MSE"); plt.title("MSE na zbiorze uczacym")
plt.grid(True); plt.tight_layout(); plt.show()

plt.figure()
plt.plot(epochs_axis, acc_hist)
plt.xlabel("Epoka"); plt.ylabel("Accuracy"); plt.title("Dokladnosc (prog 0.5)")
plt.ylim(-0.05, 1.05)
plt.grid(True); plt.tight_layout(); plt.show()


plt.figure()
for i in range(mse_per_sample_hist.shape[1]):
    plt.plot(epochs_axis, mse_per_sample_hist[:, i], label=f"Probka {i+1}")
plt.xlabel("Epoka"); plt.ylabel("MSE"); plt.title("MSE na przykladach uczacych")
plt.legend(); plt.grid(True); plt.tight_layout(); plt.show()

plt.figure()
plt.plot(epochs_axis, 1 - np.array(acc_hist))
plt.xlabel("Epoka"); plt.ylabel("Blad klasyfikacji (1-acc)")
plt.title("Blad klasyfikacji (próg 0.5)")
plt.grid(True); plt.tight_layout(); plt.show()


plt.figure()
for i in range(W_h_hist.shape[1]):
    for j in range(W_h_hist.shape[2]):
        plt.plot(epochs_axis, W_h_hist[:, i, j], label=f"W_h[{i},{j}]")
plt.xlabel("Epoka"); plt.ylabel("Wartość wagi")
plt.title("Trajektorie wag warstwy ukrytej W_h")
plt.legend(); plt.grid(True); plt.tight_layout(); plt.show()


plt.figure()
for j in range(W_o_hist.shape[2]):
    plt.plot(epochs_axis, W_o_hist[:, 0, j], label=f"W_o[0,{j}]")
plt.xlabel("Epoka"); plt.ylabel("Wartość wagi")
plt.title("Trajektorie wag warstwy wyjściowej W_o")
plt.legend(); plt.grid(True); plt.tight_layout(); plt.show()
