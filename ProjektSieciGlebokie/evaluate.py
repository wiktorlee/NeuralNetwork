import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import (
    confusion_matrix, 
    classification_report, 
    precision_recall_fscore_support
)
import seaborn as sns
from tensorflow import keras

from load_data import load_flags_dataset


def load_test_data(max_samples_per_class=None):
    """
    Wczytuje dane testowe.
    
    Args:
        max_samples_per_class: Maksymalna liczba próbek na klasę (None = wszystkie)
                              MUSI być takie samo jak w train.py!
    
    Returns:
        Tuple: (X_test, y_test, class_names)
    """
    print("=" * 70)
    print("WCZYTYWANIE DANYCH TESTOWYCH")
    print("=" * 70)
    
    X_train, X_val, X_test, y_train, y_val, y_test, class_names = load_flags_dataset(
        test_size=0.2,
        val_size=0.1,
        target_size=(128, 128),
        max_samples_per_class=max_samples_per_class
    )
    
    print(f"\n✓ Dane testowe wczytane")
    print(f"  Test: {X_test.shape[0]} obrazów")
    print(f"  Liczba klas: {len(class_names)}")
    
    return X_test, y_test, class_names


def load_trained_model(model_path="models/best_model.h5"):
    """
    Wczytuje wytrenowany model.
    
    Args:
        model_path: Ścieżka do zapisanego modelu
    
    Returns:
        Wczytany model Keras
    """
    print("\n" + "=" * 70)
    print("WCZYTYWANIE MODELU")
    print("=" * 70)
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model nie znaleziony: {model_path}\n"
            f"Upewnij się, że najpierw wytrenowałeś model (uruchom train.py)"
        )
    
    model = keras.models.load_model(model_path)
    print(f"✓ Model wczytany z: {model_path}")
    print(f"  Liczba parametrów: {model.count_params():,}")
    
    return model


def evaluate_model(model, X_test, y_test, class_names):
    """
    Ewaluuje model na zbiorze testowym.
    
    Args:
        model: Model Keras
        X_test: Dane testowe
        y_test: Etykiety testowe
        class_names: Lista nazw klas
    
    Returns:
        Dictionary z wynikami
    """
    print("\n" + "=" * 70)
    print("EWALUACJA MODELU")
    print("=" * 70)
    
    # Predykcje
    print("\nGenerowanie predykcji...")
    print("(To może chwilę potrwać, ale nie wymaga GPU)")
    y_pred_proba = model.predict(X_test, verbose=1, batch_size=32)
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    # Top-3 accuracy
    print("\nObliczanie Top-3 accuracy...")
    top3_correct = 0
    for i in range(len(y_test)):
        top3_preds = np.argsort(y_pred_proba[i])[-3:][::-1]
        if y_test[i] in top3_preds:
            top3_correct += 1
    top3_accuracy = top3_correct / len(y_test)
    
    # Podstawowe metryki
    print("\nObliczanie metryk...")
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0, batch_size=32)
    
    # Precision, Recall, F1 per class
    precision, recall, f1, support = precision_recall_fscore_support(
        y_test, y_pred, average=None, zero_division=0
    )
    
    # Macro averages
    macro_precision = np.mean(precision)
    macro_recall = np.mean(recall)
    macro_f1 = np.mean(f1)
    
    # Weighted averages
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average='weighted', zero_division=0
    )
    
    results = {
        'test_loss': test_loss,
        'test_accuracy': test_accuracy,
        'top3_accuracy': top3_accuracy,
        'y_test': y_test,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'support': support,
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'macro_f1': macro_f1,
        'weighted_precision': weighted_precision,
        'weighted_recall': weighted_recall,
        'weighted_f1': weighted_f1,
    }
    
    # Wyświetlenie wyników
    print(f"\n{'='*70}")
    print("WYNIKI EWALUACJI")
    print(f"{'='*70}")
    print(f"Test Loss:        {test_loss:.4f}")
    print(f"Test Accuracy:    {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    print(f"Top-3 Accuracy:   {top3_accuracy:.4f} ({top3_accuracy*100:.2f}%)")
    print(f"\nMetryki ogólne (macro average):")
    print(f"  Precision:      {macro_precision:.4f}")
    print(f"  Recall:         {macro_recall:.4f}")
    print(f"  F1-score:       {macro_f1:.4f}")
    print(f"\nMetryki ogólne (weighted average):")
    print(f"  Precision:      {weighted_precision:.4f}")
    print(f"  Recall:         {weighted_recall:.4f}")
    print(f"  F1-score:       {weighted_f1:.4f}")
    
    return results


def create_confusion_matrix(y_test, y_pred, class_names, plots_dir="plots", max_classes=50):
    """
    Tworzy i zapisuje confusion matrix.
    
    Args:
        y_test: Prawdziwe etykiety
        y_pred: Predykcje
        class_names: Lista nazw klas
        plots_dir: Katalog do zapisu
        max_classes: Maksymalna liczba klas do wyświetlenia (dla czytelności)
    """
    print("\n" + "=" * 70)
    print("GENEROWANIE CONFUSION MATRIX")
    print("=" * 70)
    
    Path(plots_dir).mkdir(parents=True, exist_ok=True)
    
    # Pełna confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Jeśli za dużo klas, pokaż tylko top N najczęstszych
    if len(class_names) > max_classes:
        print(f"⚠️  Zbyt wiele klas ({len(class_names)}). Tworzenie confusion matrix dla top {max_classes} klas...")
        
        # Znajdź klasy z największą liczbą próbek
        class_counts = np.bincount(y_test)
        top_classes_idx = np.argsort(class_counts)[-max_classes:][::-1]
        
        # Filtruj dane
        mask = np.isin(y_test, top_classes_idx)
        y_test_filtered = y_test[mask]
        y_pred_filtered = y_pred[mask]
        
        # Mapowanie do nowych indeksów
        idx_map = {old_idx: new_idx for new_idx, old_idx in enumerate(top_classes_idx)}
        y_test_mapped = np.array([idx_map[idx] for idx in y_test_filtered])
        y_pred_mapped = np.array([idx_map.get(idx, -1) for idx in y_pred_filtered])
        y_pred_mapped = np.where(y_pred_mapped == -1, 0, y_pred_mapped)  # Fallback
        
        cm_filtered = confusion_matrix(y_test_mapped, y_pred_mapped, labels=range(len(top_classes_idx)))
        class_names_filtered = [class_names[i] for i in top_classes_idx]
        
        # Wizualizacja
        plt.figure(figsize=(20, 16))
        sns.heatmap(cm_filtered, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names_filtered, 
                   yticklabels=class_names_filtered,
                   cbar_kws={'label': 'Liczba próbek'})
        plt.title(f'Confusion Matrix - Top {max_classes} klas (najczęściej występujących)', 
                 fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Predykcja', fontsize=12)
        plt.ylabel('Prawdziwa klasa', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        plot_path = os.path.join(plots_dir, "confusion_matrix_top_classes.png")
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"✓ Confusion matrix zapisana: {plot_path}")
        plt.close()
    else:
        # Pełna confusion matrix
        plt.figure(figsize=(30, 25))
        sns.heatmap(cm, annot=False, fmt='d', cmap='Blues', 
                   xticklabels=class_names, 
                   yticklabels=class_names,
                   cbar_kws={'label': 'Liczba próbek'})
        plt.title('Confusion Matrix - Wszystkie klasy', fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Predykcja', fontsize=12)
        plt.ylabel('Prawdziwa klasa', fontsize=12)
        plt.xticks(rotation=90, ha='right', fontsize=6)
        plt.yticks(rotation=0, fontsize=6)
        plt.tight_layout()
        
        plot_path = os.path.join(plots_dir, "confusion_matrix_full.png")
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"✓ Confusion matrix zapisana: {plot_path}")
        plt.close()
    
    # Zapisanie confusion matrix do pliku tekstowego
    cm_path = os.path.join(plots_dir, "confusion_matrix.txt")
    np.savetxt(cm_path, cm, fmt='%d')
    print(f"✓ Confusion matrix (raw) zapisana: {cm_path}")


def analyze_errors(y_test, y_pred, y_pred_proba, class_names, X_test, plots_dir="plots", top_n=10):
    """
    Analizuje błędy klasyfikacji.
    
    Args:
        y_test: Prawdziwe etykiety
        y_pred: Predykcje
        y_pred_proba: Prawdopodobieństwa predykcji
        class_names: Lista nazw klas
        X_test: Obrazy testowe
        plots_dir: Katalog do zapisu
        top_n: Liczba najtrudniejszych klas do pokazania
    """
    print("\n" + "=" * 70)
    print("ANALIZA BŁĘDÓW")
    print("=" * 70)
    
    Path(plots_dir).mkdir(parents=True, exist_ok=True)
    
    # Znajdź błędne klasyfikacje
    errors = y_test != y_pred
    error_indices = np.where(errors)[0]
    
    print(f"\nLiczba błędnych klasyfikacji: {len(error_indices)} / {len(y_test)} ({len(error_indices)/len(y_test)*100:.2f}%)")
    
    # Które klasy są najtrudniejsze (najwięcej błędów)
    error_per_class = {}
    for idx in error_indices:
        true_class = y_test[idx]
        if true_class not in error_per_class:
            error_per_class[true_class] = 0
        error_per_class[true_class] += 1
    
    # Sortuj po liczbie błędów
    sorted_errors = sorted(error_per_class.items(), key=lambda x: x[1], reverse=True)
    
    print(f"\nTop {top_n} najtrudniejszych klas (najwięcej błędów):")
    print("-" * 70)
    print(f"{'Klasa':<30} {'Błędy':<10} {'% błędów':<15}")
    print("-" * 70)
    
    for class_idx, error_count in sorted_errors[:top_n]:
        total_samples = np.sum(y_test == class_idx)
        error_rate = (error_count / total_samples * 100) if total_samples > 0 else 0
        print(f"{class_names[class_idx]:<30} {error_count:<10} {error_rate:.2f}%")
    
    # Zapis do pliku
    errors_path = os.path.join(plots_dir, "error_analysis.txt")
    with open(errors_path, 'w', encoding='utf-8') as f:
        f.write("ANALIZA BŁĘDÓW KLASYFIKACJI\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Liczba błędnych klasyfikacji: {len(error_indices)} / {len(y_test)} ({len(error_indices)/len(y_test)*100:.2f}%)\n\n")
        f.write(f"Top {top_n} najtrudniejszych klas:\n")
        f.write("-" * 70 + "\n")
        f.write(f"{'Klasa':<30} {'Błędy':<10} {'% błędów':<15}\n")
        f.write("-" * 70 + "\n")
        for class_idx, error_count in sorted_errors[:top_n]:
            total_samples = np.sum(y_test == class_idx)
            error_rate = (error_count / total_samples * 100) if total_samples > 0 else 0
            f.write(f"{class_names[class_idx]:<30} {error_count:<10} {error_rate:.2f}%\n")
    
    print(f"\n✓ Analiza błędów zapisana: {errors_path}")
    
    # Przykłady błędnych klasyfikacji (wizualizacja)
    if len(error_indices) > 0:
        n_examples = min(12, len(error_indices))
        example_indices = np.random.choice(error_indices, n_examples, replace=False)
        
        fig, axes = plt.subplots(3, 4, figsize=(16, 12))
        axes = axes.flatten()
        
        for i, idx in enumerate(example_indices):
            ax = axes[i]
            ax.imshow(X_test[idx])
            ax.axis('off')
            
            true_class = class_names[y_test[idx]]
            pred_class = class_names[y_pred[idx]]
            confidence = y_pred_proba[idx][y_pred[idx]] * 100
            
            ax.set_title(f"Prawdziwa: {true_class}\nPredykcja: {pred_class}\nPewność: {confidence:.1f}%", 
                        fontsize=9, pad=5)
        
        plt.suptitle('Przykłady błędnych klasyfikacji', fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()
        
        plot_path = os.path.join(plots_dir, "error_examples.png")
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"✓ Przykłady błędów zapisane: {plot_path}")
        plt.close()


def generate_classification_report(y_test, y_pred, class_names, plots_dir="plots"):
    """
    Generuje szczegółowy raport klasyfikacji.
    
    Args:
        y_test: Prawdziwe etykiety
        y_pred: Predykcje
        class_names: Lista nazw klas
        plots_dir: Katalog do zapisu
    """
    print("\n" + "=" * 70)
    print("GENEROWANIE RAPORTU KLASYFIKACJI")
    print("=" * 70)
    
    Path(plots_dir).mkdir(parents=True, exist_ok=True)
    
    # Raport per class
    report = classification_report(
        y_test, y_pred, 
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )
    
    # Zapis do pliku
    report_path = os.path.join(plots_dir, "classification_report.txt")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("RAPORT KLASYFIKACJI - SZCZEGÓŁOWY\n")
        f.write("=" * 70 + "\n\n")
        
        # Per class metrics
        f.write("METRYKI PER KLASA:\n")
        f.write("-" * 70 + "\n")
        f.write(f"{'Klasa':<30} {'Precision':<12} {'Recall':<12} {'F1-score':<12} {'Support':<10}\n")
        f.write("-" * 70 + "\n")
        
        for class_name in class_names:
            if class_name in report:
                metrics = report[class_name]
                f.write(f"{class_name:<30} {metrics['precision']:<12.4f} {metrics['recall']:<12.4f} "
                       f"{metrics['f1-score']:<12.4f} {int(metrics['support']):<10}\n")
        
        # Ogólne metryki
        f.write("\n" + "-" * 70 + "\n")
        f.write("METRYKI OGÓLNE:\n")
        f.write("-" * 70 + "\n")
        f.write(f"Accuracy:           {report['accuracy']:.4f}\n")
        f.write(f"Macro avg Precision: {report['macro avg']['precision']:.4f}\n")
        f.write(f"Macro avg Recall:    {report['macro avg']['recall']:.4f}\n")
        f.write(f"Macro avg F1-score:  {report['macro avg']['f1-score']:.4f}\n")
        f.write(f"Weighted avg Precision: {report['weighted avg']['precision']:.4f}\n")
        f.write(f"Weighted avg Recall:    {report['weighted avg']['recall']:.4f}\n")
        f.write(f"Weighted avg F1-score:  {report['weighted avg']['f1-score']:.4f}\n")
    
    print(f"✓ Raport klasyfikacji zapisany: {report_path}")
    
    # Wyświetlenie krótkiego podsumowania
    print("\nKrótkie podsumowanie (per class):")
    print(f"  Accuracy: {report['accuracy']:.4f}")
    print(f"  Macro avg F1: {report['macro avg']['f1-score']:.4f}")
    print(f"  Weighted avg F1: {report['weighted avg']['f1-score']:.4f}")


def main(model_path="models/best_model.h5", max_samples_per_class=50):
    """
    Główna funkcja ewaluacji.
    
    Args:
        model_path: Ścieżka do modelu
        max_samples_per_class: Maksymalna liczba próbek na klasę
                              MUSI być takie samo jak w train.py!
    """
    try:
        print("=" * 70)
        print("ETAP 4: EWALUACJA MODELU")
        print("=" * 70)
        print(f"\nUWAGA: Używam max_samples_per_class={max_samples_per_class}")
        print("Upewnij się, że to jest takie samo jak w train.py!\n")
        
        # 1. Wczytanie danych testowych
        X_test, y_test, class_names = load_test_data(max_samples_per_class=max_samples_per_class)
        
        # 2. Wczytanie modelu
        model = load_trained_model(model_path)
        
        # 3. Ewaluacja
        results = evaluate_model(model, X_test, y_test, class_names)
        
        # 4. Confusion matrix
        create_confusion_matrix(
            results['y_test'], 
            results['y_pred'], 
            class_names,
            max_classes=50  # Dla czytelności
        )
        
        # 5. Analiza błędów
        analyze_errors(
            results['y_test'],
            results['y_pred'],
            results['y_pred_proba'],
            class_names,
            X_test,
            top_n=20
        )
        
        # 6. Raport klasyfikacji
        generate_classification_report(
            results['y_test'],
            results['y_pred'],
            class_names
        )
        
        print("\n" + "=" * 70)
        print("EWALUACJA ZAKOŃCZONA POMYŚLNIE")
        print("=" * 70)
        print("Wygenerowane pliki:")
        print("  - plots/confusion_matrix_*.png")
        print("  - plots/confusion_matrix.txt")
        print("  - plots/error_analysis.txt")
        print("  - plots/error_examples.png")
        print("  - plots/classification_report.txt")
        print("=" * 70)
        
    except Exception as e:
        print("\n" + "=" * 70)
        print("BŁĄD PODCZAS EWALUACJI")
        print("=" * 70)
        print(f"  {e}")
        import traceback
        traceback.print_exc()
        print("=" * 70)
        raise


if __name__ == "__main__":
    # UWAGA: max_samples_per_class MUSI być takie samo jak w train.py!
    # W train.py jest ustawione na 50, więc tutaj też 50
    main(
        model_path="models/best_model.h5",
        max_samples_per_class=50  # MUSI być takie samo jak w train.py!
    )

