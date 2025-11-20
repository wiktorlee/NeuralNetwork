import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD


use_momentum     = True
use_adaptive_lr  = True
batch_size       = 2

# --- HIPERPARAMETRY ---
n_hidden   = 2
lr         = 0.5
lr_min     = 1e-3
lr_max     = 0.5
inc_factor = 1.02
dec_factor = 0.95

alpha      = 0.9
n_epochs   = 10000
mse_target = 0.01

# --- DANE XOR ---
x = np.array([[0., 0.],
              [0., 1.],
              [1., 0.],
              [1., 1.]], dtype=np.float32)

y_true = np.array([[0.],
                   [1.],
                   [1.],
                   [0.]], dtype=np.float32)

assert x.shape == (4, 2)
assert y_true.shape == (4, 1)
print(x.shape, y_true.shape, x.dtype)


# --- INICJALIZACJA WAG (identyczna jak w oryginale) ---
rng = np.random.default_rng()  # ustaw np.random.default_rng(1) dla powtarzalności

# Inicjalizuj wagi ręcznie (identycznie jak w oryginale)
W_h_init = rng.uniform(-0.5, 0.5, size=(n_hidden, x.shape[1])).astype(np.float32)
b_h_init = rng.uniform(-0.5, 0.5, size=(n_hidden, 1)).astype(np.float32)
W_o_init = rng.uniform(-0.5, 0.5, size=(1, n_hidden)).astype(np.float32)
b_o_init = rng.uniform(-0.5, 0.5, size=(1, 1)).astype(np.float32)

# --- BUDOWA MODELU ---
model = Sequential([
    Dense(n_hidden, activation='sigmoid', input_shape=(2,), name='hidden'),
    Dense(1, activation='sigmoid', name='output')
])

# Ustaw wagi ręcznie (identyczne jak w oryginale)
model.get_layer('hidden').set_weights([W_h_init, b_h_init.ravel()])
model.get_layer('output').set_weights([W_o_init, b_o_init.ravel()])

print("\nW_h =", W_h_init)
print("b_h =", b_h_init)
print("W_o =", W_o_init)
print("b_o =", b_o_init)


# --- OPTYMALIZATOR ---
# Keras SGD z momentum implementuje: v = momentum * v - lr * gradient, w = w + v
# Oryginał: v = alpha * v + lr * gradient, w = w - v
# To jest równoważne, więc używamy wbudowanego momentum
optimizer = SGD(learning_rate=lr, momentum=alpha if use_momentum else 0.0)
model.compile(optimizer=optimizer, loss='mse')


# --- TRENING (własna pętla jak w oryginale) ---
m = x.shape[0]
mse_hist, acc_hist = [], []
mse_per_sample_hist = []
W_h_hist, W_o_hist = [], []
lr_hist = []
prev_mse = np.inf

for ep in range(1, n_epochs + 1):
    # Shuffle danych
    idx = rng.permutation(m)
    Xs, Ys = x[idx], y_true[idx]
    
    # Mini-batch training
    for start in range(0, m, batch_size):
        stop = start + batch_size
        xb = Xs[start:stop]
        yb = Ys[start:stop]
        
        # Forward + backward pass (automatycznie przez Keras)
        with tf.GradientTape() as tape:
            y_pred = model(xb, training=True)
            loss = tf.reduce_mean(tf.square(y_pred - yb))
        
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    # Oblicz metryki na całym zbiorze
    y_hat_full = model(x, training=False).numpy()
    mse = np.mean((y_hat_full - y_true) ** 2)
    y_cls = (y_hat_full >= 0.5).astype(int)
    acc = np.mean(y_cls == y_true)
    
    # Zapisz historię
    mse_hist.append(mse)
    acc_hist.append(acc)
    mse_per_sample_hist.append(((y_hat_full - y_true) ** 2).ravel())
    W_h_hist.append(model.get_layer('hidden').get_weights()[0].copy())
    W_o_hist.append(model.get_layer('output').get_weights()[0].copy())
    lr_hist.append(float(optimizer.learning_rate.numpy()))
    
    # Wyświetl postęp
    if ep % 1000 == 0 or mse < mse_target:
        current_lr = float(optimizer.learning_rate.numpy())
        print(f"epoka {ep:5d}: MSE={mse:.6f}, acc={acc*100:.1f}%, lr={current_lr:.5f}")
    
    # Adaptacyjny learning rate
    if use_adaptive_lr:
        if mse < prev_mse:
            lr = min(lr * inc_factor, lr_max)
        else:
            lr = max(lr * dec_factor, lr_min)
        optimizer.learning_rate.assign(lr)
        prev_mse = mse
    
    # Wczesne zatrzymanie
    if mse < mse_target:
        print(f"Spełniono próg błędu {mse_target} w epoce {ep}.")
        break


# --- PODSUMOWANIE I WYNIKI ---
mse_per_sample_hist = np.array(mse_per_sample_hist)
W_h_hist = np.array(W_h_hist)
W_o_hist = np.array(W_o_hist)
epochs_axis = np.arange(1, len(mse_hist) + 1)

y_hat_final = model(x, training=False).numpy()
y_cls_final = (y_hat_final >= 0.5).astype(int)
print("\nPrzewidywania po uczeniu:")
for i in range(x.shape[0]):
    print(f"{x[i]} -> y_hat={y_hat_final[i,0]:.4f}, y={y_cls_final[i,0]}")

# --- WYKRESY ---
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
