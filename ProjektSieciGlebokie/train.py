import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, 
    LambdaCallback, Callback
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator

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


class TrainingMetricsCallback(Callback):
    """
    Custom callback do zbierania metryk podczas treningu:
    - Learning rate w każdej epoce
    - Loss per class
    - Wagi wybranych warstw
    - Gradient norms
    """
    def __init__(self, X_train, y_train, class_names):
        super().__init__()
        self.X_train = X_train
        self.y_train = y_train
        self.class_names = class_names
        self.learning_rates = []
        self.loss_per_class = {i: [] for i in range(len(class_names))}
        self.weight_trajectories = {}  # Będzie wypełnione w on_train_begin
        self.gradient_norms = []
        
    def on_train_begin(self, logs=None):
        """Inicjalizacja - wybierz warstwy do śledzenia wag"""
        # Wybierz ostatnią warstwę Dense przed softmax (przedostatnia warstwa)
        dense_layers = [i for i, layer in enumerate(self.model.layers) 
                       if 'dense' in layer.name.lower() and 'softmax' not in layer.name.lower()]
        
        if dense_layers:
            # Weź ostatnią warstwę Dense
            self.tracked_layer_idx = dense_layers[-1]
            self.tracked_layer = self.model.layers[self.tracked_layer_idx]
            # Śledź kilka wag (np. pierwsze 4 wagi z pierwszej kolumny)
            self.weight_indices = [(0, 0), (0, 1), (1, 0), (1, 1)]
            for idx in self.weight_indices:
                self.weight_trajectories[f'W[{idx[0]},{idx[1]}]'] = []
        else:
            self.tracked_layer_idx = None
            self.tracked_layer = None
    
    def on_epoch_end(self, epoch, logs=None):
        """Zbieranie metryk na końcu każdej epoki"""
        # 1. Learning rate
        lr = float(self.model.optimizer.learning_rate.numpy())
        self.learning_rates.append(lr)
        
        # 2. Loss per class (użyj małej próbki dla wydajności)
        sample_size = min(1000, len(self.X_train))
        sample_indices = np.random.choice(len(self.X_train), sample_size, replace=False)
        X_sample = self.X_train[sample_indices]
        y_sample = self.y_train[sample_indices]
        
        # Oblicz predykcje i loss per class
        y_pred_proba = self.model.predict(X_sample, verbose=0, batch_size=32)
        
        for class_idx in range(len(self.class_names)):
            class_mask = y_sample == class_idx
            if np.sum(class_mask) > 0:
                # Oblicz loss dla tej klasy
                class_y_true = y_sample[class_mask]
                class_y_pred = y_pred_proba[class_mask]
                class_losses = keras.losses.sparse_categorical_crossentropy(
                    class_y_true, class_y_pred
                )
                avg_loss = float(np.mean(class_losses.numpy()))
                self.loss_per_class[class_idx].append(avg_loss)
            else:
                self.loss_per_class[class_idx].append(np.nan)
        
        # 3. Wagi wybranej warstwy
        if self.tracked_layer is not None:
            try:
                weights = self.tracked_layer.get_weights()[0]  # Wagi (bez bias)
                for idx in self.weight_indices:
                    if len(weights.shape) >= 2 and idx[0] < weights.shape[0] and idx[1] < weights.shape[1]:
                        weight_val = weights[idx[0], idx[1]]
                        self.weight_trajectories[f'W[{idx[0]},{idx[1]}]'].append(float(weight_val))
            except Exception as e:
                # Jeśli nie można odczytać wag, pomiń
                pass
        
        # 4. Gradient norms (oblicz na małej próbce)
        try:
            X_batch = X_sample[:32]  # Mały batch
            y_batch = y_sample[:32]
            
            with tf.GradientTape() as tape:
                y_pred_batch = self.model(X_batch, training=True)
                loss = keras.losses.sparse_categorical_crossentropy(y_batch, y_pred_batch)
                loss = tf.reduce_mean(loss)
            
            # Oblicz gradienty
            gradients = tape.gradient(loss, self.model.trainable_variables)
            # Oblicz normę gradientów
            total_norm = 0
            for grad in gradients:
                if grad is not None:
                    grad_norm = tf.norm(grad).numpy()
                    total_norm += grad_norm ** 2
            total_norm = np.sqrt(total_norm)
            self.gradient_norms.append(float(total_norm))
        except Exception as e:
            # Jeśli obliczenie gradientów się nie powiedzie, użyj NaN
            self.gradient_norms.append(np.nan)


def create_callbacks(models_dir="models", patience=5, X_train=None, y_train=None, class_names=None):
    """
    Tworzy callbacks dla treningu: ModelCheckpoint, EarlyStopping, ReduceLROnPlateau,
    oraz custom callback do zbierania metryk.
    
    Args:
        models_dir: Katalog do zapisu modeli
        patience: Liczba epok bez poprawy przed zatrzymaniem (EarlyStopping)
        X_train, y_train: Dane treningowe (dla custom callback)
        class_names: Nazwy klas (dla custom callback)
    
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
    
    # ReduceLROnPlateau - zmniejszanie learning rate
    lr_scheduler = ReduceLROnPlateau(
        monitor='val_accuracy',
        factor=0.5,           # Zmniejsz LR o połowę
        patience=3,           # Czekaj 3 epoki bez poprawy
        min_lr=1e-6,         # Minimalny LR
        verbose=1
    )
    
    callbacks = [checkpoint_callback, early_stopping_callback, lr_scheduler]
    
    # Custom callback do zbierania metryk (jeśli dane są dostępne)
    if X_train is not None and y_train is not None and class_names is not None:
        metrics_callback = TrainingMetricsCallback(X_train, y_train, class_names)
        callbacks.append(metrics_callback)
    else:
        metrics_callback = None
    
    print(f"\n✓ Callbacks utworzone:")
    print(f"  - ModelCheckpoint: {checkpoint_path}")
    print(f"  - EarlyStopping: patience={patience}")
    print(f"  - ReduceLROnPlateau: factor=0.5, patience=3, min_lr=1e-6")
    if metrics_callback:
        print(f"  - TrainingMetricsCallback: zbieranie metryk analitycznych")
    
    return callbacks, metrics_callback


def create_data_generators(X_train, y_train, X_val, y_val, batch_size=32, use_augmentation=True):
    """
    Tworzy generator danych z augmentacją dla treningu.
    Walidacja używa bezpośrednio tablic (bez generatora) dla stabilności.
    
    Args:
        X_train, y_train: Dane treningowe
        X_val, y_val: Dane walidacyjne
        batch_size: Rozmiar batcha
        use_augmentation: Czy używać augmentacji danych (domyślnie True)
    
    Returns:
        Tuple: (train_generator, validation_data, steps_per_epoch)
    """
    if use_augmentation:
        # Augmentacja dla danych treningowych
        train_datagen = ImageDataGenerator(
            rotation_range=10,        # Obrót ±10 stopni
            width_shift_range=0.1,     # Przesunięcie w poziomie ±10%
            height_shift_range=0.1,   # Przesunięcie w pionie ±10%
            brightness_range=[0.8, 1.2],  # Zmiana jasności ±20%
            zoom_range=0.1,           # Zoom ±10%
            fill_mode='nearest',      # Wypełnianie pikseli przy transformacjach
            rescale=1.0               # Dane już są znormalizowane (0-1)
        )
        print("✓ Augmentacja danych włączona:")
        print("  - Obrót: ±10°")
        print("  - Przesunięcie: ±10%")
        print("  - Jasność: ±20%")
        print("  - Zoom: ±10%")
    else:
        train_datagen = ImageDataGenerator(rescale=1.0)
        print("✓ Augmentacja danych wyłączona")
    
    # Generator tylko dla treningu (z augmentacją)
    train_generator = train_datagen.flow(
        X_train, y_train,
        batch_size=batch_size,
        shuffle=True
    )
    
    # Walidacja używa bezpośrednio tablic (bez generatora) - bardziej stabilne
    validation_data = (X_val, y_val)
    
    # Oblicz steps_per_epoch - zaokrąglij w górę, żeby pokryć wszystkie dane
    steps_per_epoch = (len(X_train) + batch_size - 1) // batch_size
    
    print(f"  Steps per epoch: {steps_per_epoch}")
    print(f"  Validation samples: {len(X_val)}")
    
    return train_generator, validation_data, steps_per_epoch


def train_model(model, X_train, y_train, X_val, y_val, 
                batch_size=32, epochs=30, callbacks=None, use_augmentation=True, metrics_callback=None):
    """
    Trenuje model z augmentacją danych.
    
    Args:
        model: Model Keras do treningu
        X_train, y_train: Dane treningowe
        X_val, y_val: Dane walidacyjne
        batch_size: Rozmiar batcha
        epochs: Maksymalna liczba epok
        callbacks: Lista callbacków
        use_augmentation: Czy używać augmentacji danych (domyślnie True)
    
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
    
    # Tworzenie generatora danych (tylko dla treningu)
    train_generator, validation_data, steps_per_epoch = create_data_generators(
        X_train, y_train, X_val, y_val, batch_size, use_augmentation
    )
    
    try:
        history = model.fit(
            train_generator,
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            validation_data=validation_data,  # Bezpośrednio tablice, nie generator
            callbacks=callbacks,
            verbose=1
        )
        
        print("\n✓ Trening zakończony pomyślnie")
        
        # Wyświetlenie najlepszej accuracy
        best_val_acc = max(history.history['val_accuracy'])
        best_epoch = history.history['val_accuracy'].index(best_val_acc) + 1
        print(f"  Najlepsza val_accuracy: {best_val_acc:.4f} (epoka {best_epoch})")
        
        # Dodaj metryki z custom callback do historii
        if metrics_callback is not None:
            history.metrics_callback = metrics_callback
        
        return history
        
    except Exception as e:
        print(f"\n✗ Błąd podczas treningu: {e}")
        raise


def visualize_history(history, plots_dir="plots", class_names=None):
    """
    Wizualizuje historię treningu (accuracy, loss) oraz dodatkowe wykresy analityczne.
    
    Args:
        history: Historia treningu z model.fit()
        plots_dir: Katalog do zapisu wykresów
        class_names: Nazwy klas (dla wykresu loss per class)
    """
    # Tworzenie katalogu jeśli nie istnieje
    Path(plots_dir).mkdir(parents=True, exist_ok=True)
    
    # 1. Podstawowe wykresy (accuracy i loss)
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
    plot_path = os.path.join(plots_dir, "training_history.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"✓ Wykres zapisany: {plot_path}")
    plt.close()
    
    # 2. Błąd klasyfikacji (1 - Accuracy)
    fig, ax = plt.subplots(figsize=(10, 6))
    train_error = [1 - acc for acc in history.history['accuracy']]
    val_error = [1 - acc for acc in history.history['val_accuracy']]
    ax.plot(train_error, label='Train Error (1-acc)', linewidth=2, color='blue')
    ax.plot(val_error, label='Val Error (1-acc)', linewidth=2, color='orange')
    ax.set_title('Błąd klasyfikacji (1 - Accuracy)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Epoka', fontsize=12)
    ax.set_ylabel('Błąd klasyfikacji (1-acc)', fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plot_path = os.path.join(plots_dir, "classification_error.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"✓ Wykres zapisany: {plot_path}")
    plt.close()
    
    # Sprawdź czy mamy metryki z custom callback
    metrics_callback = getattr(history, 'metrics_callback', None)
    
    if metrics_callback is not None:
        # 3. Ewolucja Learning Rate
        if metrics_callback.learning_rates:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(metrics_callback.learning_rates, linewidth=2, color='blue')
            ax.set_title('Ewolucja współczynnika uczenia (lr)', fontsize=14, fontweight='bold')
            ax.set_xlabel('Epoka', fontsize=12)
            ax.set_ylabel('learning rate', fontsize=12)
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plot_path = os.path.join(plots_dir, "learning_rate_evolution.png")
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            print(f"✓ Wykres zapisany: {plot_path}")
            plt.close()
        
        # 4. Loss per Class (wybierz kilka klas do pokazania)
        if metrics_callback.loss_per_class and class_names:
            # Wybierz 4 klasy do pokazania (np. pierwsze 4)
            num_classes_to_show = min(4, len(class_names))
            fig, ax = plt.subplots(figsize=(12, 6))
            
            for i in range(num_classes_to_show):
                if metrics_callback.loss_per_class[i]:
                    losses = metrics_callback.loss_per_class[i]
                    # Usuń NaN
                    losses_clean = [l for l in losses if not np.isnan(l)]
                    if losses_clean:
                        epochs = range(len(losses_clean))
                        ax.plot(epochs, losses_clean, label=f'{class_names[i]}', linewidth=2)
            
            ax.set_title('MSE na przykładach uczących', fontsize=14, fontweight='bold')
            ax.set_xlabel('Epoka', fontsize=12)
            ax.set_ylabel('MSE', fontsize=12)
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plot_path = os.path.join(plots_dir, "loss_per_class.png")
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            print(f"✓ Wykres zapisany: {plot_path}")
            plt.close()
        
        # 5. Trajektorie wag
        if metrics_callback.weight_trajectories:
            fig, ax = plt.subplots(figsize=(12, 6))
            for weight_name, trajectory in metrics_callback.weight_trajectories.items():
                if trajectory:
                    ax.plot(trajectory, label=weight_name, linewidth=2)
            ax.set_title('Trajektorie wag warstwy wyjściowej W_o', fontsize=14, fontweight='bold')
            ax.set_xlabel('Epoka', fontsize=12)
            ax.set_ylabel('Wartość wagi', fontsize=12)
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plot_path = os.path.join(plots_dir, "weight_trajectories.png")
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            print(f"✓ Wykres zapisany: {plot_path}")
            plt.close()
        
        # 6. Gradient Norms
        if metrics_callback.gradient_norms:
            # Usuń NaN z gradient norms
            gradient_norms_clean = [g for g in metrics_callback.gradient_norms if not np.isnan(g)]
            if gradient_norms_clean:
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(gradient_norms_clean, linewidth=2, color='purple')
                ax.set_title('Normy gradientów', fontsize=14, fontweight='bold')
                ax.set_xlabel('Epoka', fontsize=12)
                ax.set_ylabel('Norma gradientu', fontsize=12)
                ax.grid(True, alpha=0.3)
                plt.tight_layout()
                plot_path = os.path.join(plots_dir, "gradient_norms.png")
                plt.savefig(plot_path, dpi=150, bbox_inches='tight')
                print(f"✓ Wykres zapisany: {plot_path}")
                plt.close()
    else:
        print("⚠️  Brak metryk z custom callback - niektóre wykresy nie zostały wygenerowane")


def main(
    batch_size=32,
    epochs=30,
    learning_rate=1e-3,
    patience=5,
    max_samples_per_class=100,
    use_augmentation=True
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
        
        # 3. Tworzenie callbacków (z danymi dla custom callback)
        callbacks, metrics_callback = create_callbacks(
            patience=patience,
            X_train=X_train,
            y_train=y_train,
            class_names=class_names
        )
        
        # 4. Trening
        history = train_model(
            model, X_train, y_train, X_val, y_val,
            batch_size=batch_size,
            epochs=epochs,
            callbacks=callbacks,
            use_augmentation=use_augmentation,
            metrics_callback=metrics_callback
        )
        
        # 5. Wizualizacja
        visualize_history(history, class_names=class_names)
        
        print("\n" + "=" * 70)
        print("TRENING ZAKOŃCZONY POMYŚLNIE")
        print("=" * 70)
        print(f"  Model zapisany: models/best_model.h5")
        print(f"  Wygenerowane wykresy:")
        print(f"    - plots/training_history.png (accuracy i loss)")
        print(f"    - plots/classification_error.png (błąd klasyfikacji)")
        print(f"    - plots/learning_rate_evolution.png (ewolucja LR)")
        print(f"    - plots/loss_per_class.png (loss per class)")
        print(f"    - plots/weight_trajectories.png (trajektorie wag)")
        print(f"    - plots/gradient_norms.png (normy gradientów)")
        print("=" * 70)
        
    except Exception as e:
        print("\n" + "=" * 70)
        print("BŁĄD PODCZAS TRENINGU")
        print("=" * 70)
        print(f"  {e}")
        print("=" * 70)
        raise


if __name__ == "__main__":
    # ETAP 5: Optymalizacja
    # - Więcej danych: 100 próbek na klasę (zamiast 50)
    # - Augmentacja danych: włączona (obrót, przesunięcie, jasność, zoom)
    # 
    # UWAGA: max_samples_per_class=None wczytuje wszystkie dane (~195k obrazów, ~25GB RAM)
    # Dla większości komputerów lepiej użyć ograniczenia, np. 100-200 próbek na klasę
    # W Colab czasem trzeba zmniejszyć do 50, żeby uniknąć problemów z pamięcią
    main(
        batch_size=32,
        epochs=30,
        learning_rate=1e-3,
        patience=5,
        max_samples_per_class=75,  # ETAP 5A: 75 próbek na klasę (kompromis między 50 a 100)
        use_augmentation=False       # ETAP 5B: Augmentacja danych WYŁĄCZONA (test)
    )

