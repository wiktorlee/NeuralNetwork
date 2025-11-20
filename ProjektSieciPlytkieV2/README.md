# Projekt Sieci Płytkie V2 - Implementacja z Keras

Implementacja sieci neuronowej do problemu XOR używająca biblioteki Keras/TensorFlow, oparta na oryginalnej implementacji ręcznej z `xor4_0.py`.

## Różnice w stosunku do oryginalnej implementacji

### Uproszczenia dzięki bibliotece:
1. **Forward pass i backpropagation** - automatycznie obsługiwane przez Keras
2. **Momentum** - wbudowane w optymalizator SGD (parametr `momentum`)
3. **Mini-batch** - automatycznie obsługiwane przez `batch_size` w `fit()`
4. **Shuffle danych** - automatycznie przez `shuffle=True` w `fit()`

### Zachowane elementy:
- Identyczna inicjalizacja wag (uniform(-0.5, 0.5))
- Identyczna logika adaptacyjnego learning rate
- Identyczne wykresy i metryki
- Zbieranie wag w każdej epoce
- MSE per sample
- Accuracy z progiem 0.5
- Wczesne zatrzymanie przy osiągnięciu `mse_target`

## Wymagania

Zainstaluj zależności:
```bash
pip install -r requirements.txt
```

## Uruchomienie

```bash
python xor4_0_keras.py
```

## Struktura kodu

- **UniformInitializer** - własny inicjalizator wag (identyczny jak w oryginale)
- **MetricsCallback** - zbieranie wag, MSE, accuracy w każdej epoce
- **AdaptiveLRCallback** - adaptacyjny learning rate (identyczna logika jak w oryginale)
- **EarlyStopCallback** - wczesne zatrzymanie przy osiągnięciu progu błędu

## Wykresy

Program generuje te same wykresy co oryginalna implementacja:
1. MSE na zbiorze uczącym
2. Dokładność (próg 0.5)
3. MSE na przykładach uczących (per sample)
4. Błąd klasyfikacji (próg 0.5)
5. Trajektorie wag warstwy ukrytej W_h
6. Trajektorie wag warstwy wyjściowej W_o
7. Ewolucja współczynnika uczenia (lr)

