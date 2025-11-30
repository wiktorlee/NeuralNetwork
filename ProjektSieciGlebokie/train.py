import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

from load_data import load_flags_dataset
from model import build_model


def load_training_data(max_samples_per_class=None):
    """
    Wczytuje dane treningowe używając load_data.py.
    
    Args:
        max_samples_per_class: Maksymalna liczba próbek na klasę (None = wszystkie)
                               UWAGA: None wczytuje ~195k obrazów (~25GB RAM)!
    
    Returns:
        Tuple: (X_train, X_val, y_train, y_val, class_names)
    """
    print("=" * 70)
    print("WCZYTYWANIE DANYCH")
    print("=" * 70)
    
    if max_samples_per_class is None:
        print("⚠️  UWAGA: Wczytywanie wszystkich danych (~195k obrazów, ~25GB RAM)")
        print("   Jeśli masz problemy z pamięcią, użyj max_samples_per_class=100-200")
    
    try:
        X_train, X_val, X_test, y_train, y_val, y_test, class_names = load_flags_dataset(
            test_size=0.2,
            val_size=0.1,
            target_size=(128, 128),
            max_samples_per_class=max_samples_per_class
        )
    except MemoryError as e:
        print(f"\n✗ Błąd pamięci podczas wczytywania danych: {e}")
        print("   Spróbuj zmniejszyć max_samples_per_class (np. 30-50)")
        raise
    except Exception as e:
        print(f"\n✗ Błąd podczas wczytywania danych: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    # Walidacja danych
    if len(X_train) == 0 or len(X_val) == 0:
        raise ValueError("Błąd: Puste zbiory treningowe lub walidacyjne!")
    
    if X_train.shape[1:] != (128, 128, 3):
        raise ValueError(f"Błąd: Nieprawidłowy kształt danych: {X_train.shape[1:]}")
    
    print(f"\n✓ Dane wczytane pomyślnie")
    print(f"  Train: {X_train.shape[0]} obrazów")
    print(f"  Val:   {X_val.shape[0]} obrazów")
    print(f"  Liczba klas: {len(class_names)}")
    
    return X_train, X_val, y_train, y_val, class_names


def build_and_compile_model(num_classes=195, learning_rate=1e-3):
    """
    Buduje i kompiluje model używając model.py.
    
    Args:
        num_classes: Liczba klas (domyślnie 195)
        learning_rate: Learning rate dla optymalizatora Adam
    
    Returns:
        Skompilowany model Keras
    """
    print("\n" + "=" * 70)
    print("BUDOWA MODELU")
    print("=" * 70)
    
    model = build_model(
        input_shape=(128, 128, 3),
        num_classes=num_classes,
        base_filters=32
    )
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    print("\n✓ Model zbudowany i skompilowany")
    model.summary()
    
    return model


def create_callbacks(models_dir="models", patience=5):
    """
    Tworzy callbacks dla treningu: ModelCheckpoint i EarlyStopping.
    
    Args:
        models_dir: Katalog do zapisu modeli
        patience: Liczba epok bez poprawy przed zatrzymaniem (EarlyStopping)
    
    Returns:
        Lista callbacków
    """
    # Tworzenie katalogu jeśli nie istnieje
    Path(models_dir).mkdir(parents=True, exist_ok=True)
    
    # ModelCheckpoint - zapis najlepszego modelu
    checkpoint_path = os.path.join(models_dir, "best_model.h5")
    checkpoint_callback = ModelCheckpoint(
        filepath=checkpoint_path,
        monitor="val_accuracy",
        save_best_only=True,
        save_weights_only=False,
        mode="max",
        verbose=1
    )
    
    # EarlyStopping - zatrzymanie przy braku poprawy
    early_stopping_callback = EarlyStopping(
        monitor="val_accuracy",
        patience=patience,
        restore_best_weights=True,
        mode="max",
        verbose=1
    )
    
    callbacks = [checkpoint_callback, early_stopping_callback]
    
    print(f"\n✓ Callbacks utworzone:")
    print(f"  - ModelCheckpoint: {checkpoint_path}")
    print(f"  - EarlyStopping: patience={patience}")
    
    return callbacks


def train_model(model, X_train, y_train, X_val, y_val, 
                batch_size=32, epochs=30, callbacks=None):
    """
    Trenuje model.
    
    Args:
        model: Model Keras do treningu
        X_train, y_train: Dane treningowe
        X_val, y_val: Dane walidacyjne
        batch_size: Rozmiar batcha
        epochs: Maksymalna liczba epok
        callbacks: Lista callbacków
    
    Returns:
        Historia treningu (History object)
    """
    print("\n" + "=" * 70)
    print("ROZPOCZĘCIE TRENINGU")
    print("=" * 70)
    print(f"  Batch size: {batch_size}")
    print(f"  Max epochs: {epochs}")
    print(f"  Learning rate: {model.optimizer.learning_rate.numpy():.6f}")
    print()
    
    try:
        history = model.fit(
            X_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=1
        )
        
        print("\n✓ Trening zakończony pomyślnie")
        
        # Wyświetlenie najlepszej accuracy
        best_val_acc = max(history.history['val_accuracy'])
        best_epoch = history.history['val_accuracy'].index(best_val_acc) + 1
        print(f"  Najlepsza val_accuracy: {best_val_acc:.4f} (epoka {best_epoch})")
        
        return history
        
    except Exception as e:
        print(f"\n✗ Błąd podczas treningu: {e}")
        raise


def visualize_history(history, plots_dir="plots"):
    """
    Wizualizuje historię treningu (accuracy i loss).
    
    Args:
        history: Historia treningu z model.fit()
        plots_dir: Katalog do zapisu wykresów
    """
    # Tworzenie katalogu jeśli nie istnieje
    Path(plots_dir).mkdir(parents=True, exist_ok=True)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Wykres accuracy
    ax1.plot(history.history['accuracy'], label='Train Accuracy', linewidth=2)
    ax1.plot(history.history['val_accuracy'], label='Val Accuracy', linewidth=2)
    ax1.set_title('Model Accuracy', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Wykres loss
    ax2.plot(history.history['loss'], label='Train Loss', linewidth=2)
    ax2.plot(history.history['val_loss'], label='Val Loss', linewidth=2)
    ax2.set_title('Model Loss', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Loss', fontsize=12)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Zapis wykresu
    plot_path = os.path.join(plots_dir, "training_history.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Wykres zapisany: {plot_path}")
    
    plt.close()


def main(
    batch_size=32,
    epochs=5,
    learning_rate=1e-3,
    patience=3,
    max_samples_per_class=10
):
    """
    Główna funkcja orkiestrująca proces treningu.
    
    Args:
        batch_size: Rozmiar batcha
        epochs: Maksymalna liczba epok
        learning_rate: Learning rate
        patience: Patience dla EarlyStopping
        max_samples_per_class: Maksymalna liczba próbek na klasę (None = wszystkie)
    """
    try:
        # 1. Wczytanie danych
        X_train, X_val, y_train, y_val, class_names = load_training_data(
            max_samples_per_class=max_samples_per_class
        )
        
        # 2. Budowa i kompilacja modelu
        model = build_and_compile_model(
            num_classes=len(class_names),
            learning_rate=learning_rate
        )
        
        # 3. Tworzenie callbacków
        callbacks = create_callbacks(patience=patience)
        
        # 4. Trening
        history = train_model(
            model, X_train, y_train, X_val, y_val,
            batch_size=batch_size,
            epochs=epochs,
            callbacks=callbacks
        )
        
        # 5. Wizualizacja
        visualize_history(history)
        
        print("\n" + "=" * 70)
        print("TRENING ZAKOŃCZONY POMYŚLNIE")
        print("=" * 70)
        print(f"  Model zapisany: models/best_model.h5")
        print(f"  Wykres zapisany: plots/training_history.png")
        print("=" * 70)
        
    except Exception as e:
        print("\n" + "=" * 70)
        print("BŁĄD PODCZAS TRENINGU")
        print("=" * 70)
        print(f"  {e}")
        print("=" * 70)
        raise


if __name__ == "__main__":
    # Domyślne parametry treningu
    # UWAGA: max_samples_per_class=None wczytuje wszystkie dane (~195k obrazów, ~25GB RAM)
    # Dla większości komputerów lepiej użyć ograniczenia, np. 50-100 próbek na klasę
    # W Colab czasem trzeba zmniejszyć do 50, żeby uniknąć problemów z pamięcią
    main(
        batch_size=32,
        epochs=30,
        learning_rate=1e-3,
        patience=5,
        max_samples_per_class=50  # 50 próbek na klasę = ~9.75k obrazów (bezpieczne dla Colab)
        # max_samples_per_class=100  # Możesz zwiększyć jeśli masz więcej RAM
        # max_samples_per_class=None  # Odkomentuj dla pełnego datasetu (wymaga dużo RAM)
    )

