
import numpy as np
import matplotlib.pyplot as plt


use_momentum     = True
use_adaptive_lr  = True
batch_size       = 2

# Parametry
n_hidden   = 2
lr         = 0.5
lr_min     = 1e-3
lr_max     = 0.5
inc_factor = 1.02
dec_factor = 0.95

alpha      = 0.9
n_epochs   = 10000
mse_target = 0.01

# Dane
x = np.array([[0., 0.],
              [0., 1.],
              [1., 0.],
              [1., 1.]], dtype=float)

y_true = np.array([[0.],
                   [1.],
                   [1.],
                   [0.]], dtype=float)

assert x.shape == (4, 2)
assert y_true.shape == (4, 1)
print(x.shape, y_true.shape, x.dtype)  # test


rng = np.random.default_rng()  # ustaw np.random.default_rng(1) dla powtarzalności
W_h = rng.uniform(-0.5, 0.5, size=(n_hidden, x.shape[1]))
b_h = rng.uniform(-0.5, 0.5, size=(n_hidden, 1))
W_o = rng.uniform(-0.5, 0.5, size=(1, n_hidden))
b_o = rng.uniform(-0.5, 0.5, size=(1, 1))

print("\nW_h =", W_h)
print("b_h =", b_h)
print("W_o =", W_o)
print("b_o =", b_o)


vW_h = np.zeros_like(W_h); vb_h = np.zeros_like(b_h)
vW_o = np.zeros_like(W_o); vb_o = np.zeros_like(b_o)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def dsigmoid(a):
    return a * (1.0 - a)

def forward(W_h, b_h, W_o, b_o, X):
    z_h  = (W_h @ X.T) + b_h
    h    = sigmoid(z_h)
    z_o  = (W_o @ h) + b_o
    yhat = sigmoid(z_o)
    return h, yhat

def compute_mse_acc(W_h, b_h, W_o, b_o, X, Y):
    h, yhat = forward(W_h, b_h, W_o, b_o, X)
    y_row = Y.T
    mse   = np.mean((yhat - y_row) ** 2)
    y_cls = (yhat.T >= 0.5).astype(int)
    acc   = np.mean(y_cls == Y)
    return mse, acc, yhat.T


m = x.shape[0]
mse_hist, acc_hist = [], []
mse_per_sample_hist = []
W_h_hist, W_o_hist = [], []
lr_hist = []

prev_mse = np.inf


for ep in range(1, n_epochs + 1):

    idx = rng.permutation(m)
    Xs, Ys = x[idx], y_true[idx]


    for start in range(0, m, batch_size):
        stop = start + batch_size
        xb = Xs[start:stop]   # (B, 2)
        yb = Ys[start:stop]   # (B, 1)
        B  = len(xb)


        z_h  = (W_h @ xb.T) + b_h
        h    = sigmoid(z_h)
        z_o  = (W_o @ h) + b_o
        yhat = sigmoid(z_o)


        y_row = yb.T
        dZ_o = (yhat - y_row) * dsigmoid(yhat)   # (1,B)
        dW_o = (dZ_o @ h.T) / B                  # (1,hidden)
        db_o = np.sum(dZ_o, axis=1, keepdims=True) / B  # (1,1)

        dA_h = W_o.T @ dZ_o                      # (hidden,B)
        dZ_h = dA_h * dsigmoid(h)                # (hidden,B)
        dW_h = (dZ_h @ xb) / B                   # (hidden,2)
        db_h = np.sum(dZ_h, axis=1, keepdims=True) / B  # (hidden,1)


        if use_momentum:
            vW_o = alpha * vW_o + lr * dW_o
            vb_o = alpha * vb_o + lr * db_o
            vW_h = alpha * vW_h + lr * dW_h
            vb_h = alpha * vb_h + lr * db_h

            W_o -= vW_o; b_o -= vb_o
            W_h -= vW_h; b_h -= vb_h
        else:
            W_o -= lr * dW_o; b_o -= lr * db_o
            W_h -= lr * dW_h; b_h -= lr * db_h


    mse, acc, yhat_full = compute_mse_acc(W_h, b_h, W_o, b_o, x, y_true)
    mse_hist.append(mse)
    acc_hist.append(acc)
    mse_per_sample_hist.append(((yhat_full - y_true) ** 2).ravel())
    W_h_hist.append(W_h.copy())
    W_o_hist.append(W_o.copy())
    lr_hist.append(lr)


    if ep % 1000 == 0 or mse < mse_target:
        print(f"epoka {ep:5d}: MSE={mse:.6f}, acc={acc*100:.1f}%, lr={lr:.5f}")

    if use_adaptive_lr:
        if mse < prev_mse:
            lr = min(lr * inc_factor, lr_max)
        else:
            lr = max(lr * dec_factor, lr_min)
        prev_mse = mse


    if mse < mse_target:
        print(f"Spełniono próg błędu {mse_target} w epoce {ep}.")
        break

# Wyniki
mse_per_sample_hist = np.array(mse_per_sample_hist)
W_h_hist = np.array(W_h_hist)
W_o_hist = np.array(W_o_hist)
epochs_axis = np.arange(1, len(mse_hist) + 1)

_, y_hat_full = forward(W_h, b_h, W_o, b_o, x)
y_hat_final = y_hat_full.T
y_cls_final = (y_hat_final >= 0.5).astype(int)
print("\nPrzewidywania po uczeniu:")
for i in range(x.shape[0]):
    print(f"{x[i]} -> y_hat={y_hat_final[i,0]:.4f}, y={y_cls_final[i,0]}")

plt.figure()
plt.plot(epochs_axis, mse_hist)
plt.xlabel("Epoka"); plt.ylabel("MSE"); plt.title("MSE na zbiorze uczącym")
plt.grid(True); plt.tight_layout(); plt.show()

plt.figure()
plt.plot(epochs_axis, acc_hist)
plt.xlabel("Epoka"); plt.ylabel("Accuracy"); plt.title("Dokładność (próg 0.5)")
plt.ylim(-0.05, 1.05)
plt.grid(True); plt.tight_layout(); plt.show()

plt.figure()
for i in range(mse_per_sample_hist.shape[1]):
    plt.plot(epochs_axis, mse_per_sample_hist[:, i], label=f"Próbka {i+1}")
plt.xlabel("Epoka"); plt.ylabel("MSE")
plt.title("MSE na przykładach uczących")
plt.legend(); plt.grid(True); plt.tight_layout(); plt.show()

plt.figure()
plt.plot(epochs_axis, 1 - np.array(acc_hist))
plt.xlabel("Epoka"); plt.ylabel("Błąd klasyfikacji (1-acc)")
plt.title("Błąd klasyfikacji (próg 0.5)")
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

plt.figure()
plt.plot(epochs_axis, lr_hist)
plt.xlabel("Epoka"); plt.ylabel("learning rate")
plt.title("Ewolucja współczynnika uczenia (lr)")
plt.grid(True); plt.tight_layout(); plt.show()
