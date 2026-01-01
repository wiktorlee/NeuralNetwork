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


def plot_top_n_accuracy(y_test, y_pred_proba, class_names, plots_dir="plots", max_n=5):
    """
    Tworzy wykres Top-N accuracy.
    
    Args:
        y_test: Prawdziwe etykiety
        y_pred_proba: Prawdopodobieństwa predykcji
        class_names: Lista nazw klas
        plots_dir: Katalog do zapisu
        max_n: Maksymalna wartość N do sprawdzenia
    """
    print("\n" + "=" * 70)
    print("GENEROWANIE WYKRESU TOP-N ACCURACY")
    print("=" * 70)
    
    Path(plots_dir).mkdir(parents=True, exist_ok=True)
    
    # Oblicz Top-N accuracy dla N od 1 do max_n
    top_n_accuracies = []
    n_values = list(range(1, max_n + 1))
    
    for n in n_values:
        correct = 0
        for i in range(len(y_test)):
            top_n_preds = np.argsort(y_pred_proba[i])[-n:][::-1]
            if y_test[i] in top_n_preds:
                correct += 1
        accuracy = correct / len(y_test)
        top_n_accuracies.append(accuracy)
        print(f"  Top-{n} Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Wizualizacja
    plt.figure(figsize=(10, 6))
    plt.plot(n_values, top_n_accuracies, marker='o', linewidth=2, markersize=8, color='#2E86AB')
    plt.xlabel('N (Top-N)', fontsize=12, fontweight='bold')
    plt.ylabel('Accuracy', fontsize=12, fontweight='bold')
    plt.title('Top-N Accuracy', fontsize=14, fontweight='bold', pad=15)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.ylim([min(top_n_accuracies) - 0.05, 1.0])
    
    # Dodaj wartości na wykresie
    for n, acc in zip(n_values, top_n_accuracies):
        plt.text(n, acc + 0.02, f'{acc*100:.2f}%', ha='center', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    
    plot_path = os.path.join(plots_dir, "top_n_accuracy.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"✓ Top-N accuracy zapisana: {plot_path}")
    plt.close()


def plot_confidence_distribution(y_test, y_pred, y_pred_proba, plots_dir="plots"):
    """
    Analizuje rozkład pewności modelu dla poprawnych vs błędnych predykcji.
    
    Args:
        y_test: Prawdziwe etykiety
        y_pred: Predykcje
        y_pred_proba: Prawdopodobieństwa predykcji
        plots_dir: Katalog do zapisu
    """
    print("\n" + "=" * 70)
    print("ANALIZA ROZKŁADU PEWNOŚCI MODELU")
    print("=" * 70)
    
    Path(plots_dir).mkdir(parents=True, exist_ok=True)
    
    # Oblicz pewność dla każdej predykcji (maksymalne prawdopodobieństwo)
    confidences = np.max(y_pred_proba, axis=1)
    
    # Podziel na poprawne i błędne
    correct_mask = y_test == y_pred
    correct_confidences = confidences[correct_mask]
    incorrect_confidences = confidences[~correct_mask]
    
    print(f"  Średnia pewność dla poprawnych: {np.mean(correct_confidences):.4f}")
    print(f"  Średnia pewność dla błędnych: {np.mean(incorrect_confidences):.4f}")
    
    # Wizualizacja
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram
    ax1 = axes[0]
    ax1.hist(correct_confidences, bins=30, alpha=0.7, label='Poprawne', color='green', edgecolor='black')
    ax1.hist(incorrect_confidences, bins=30, alpha=0.7, label='Błędne', color='red', edgecolor='black')
    ax1.set_xlabel('Pewność modelu (maksymalne prawdopodobieństwo)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Liczba próbek', fontsize=11, fontweight='bold')
    ax1.set_title('Rozkład pewności modelu', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Box plot
    ax2 = axes[1]
    data_to_plot = [correct_confidences, incorrect_confidences]
    bp = ax2.boxplot(data_to_plot, labels=['Poprawne', 'Błędne'], patch_artist=True)
    bp['boxes'][0].set_facecolor('lightgreen')
    bp['boxes'][1].set_facecolor('lightcoral')
    ax2.set_ylabel('Pewność modelu', fontsize=11, fontweight='bold')
    ax2.set_title('Porównanie pewności: poprawne vs błędne', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    plot_path = os.path.join(plots_dir, "confidence_distribution.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"✓ Rozkład pewności zapisany: {plot_path}")
    plt.close()


def plot_precision_recall_per_class(y_test, y_pred, class_names, plots_dir="plots", top_n=30):
    """
    Tworzy wykres Precision i Recall per class - pokazuje które klasy są łatwe/trudne.
    
    Args:
        y_test: Prawdziwe etykiety
        y_pred: Predykcje
        class_names: Lista nazw klas
        plots_dir: Katalog do zapisu
        top_n: Liczba klas do pokazania (najtrudniejsze lub wszystkie jeśli mniej)
    """
    print("\n" + "=" * 70)
    print("ANALIZA PRECISION/RECALL PER CLASS")
    print("=" * 70)
    
    Path(plots_dir).mkdir(parents=True, exist_ok=True)
    
    # Oblicz precision i recall per class
    from sklearn.metrics import precision_recall_fscore_support
    precision, recall, f1, support = precision_recall_fscore_support(
        y_test, y_pred, average=None, zero_division=0
    )
    
    # Oblicz średnią F1-score jako miarę trudności klasy
    # Sortuj po F1-score (najtrudniejsze na górze)
    class_scores = list(zip(range(len(class_names)), precision, recall, f1, support))
    class_scores.sort(key=lambda x: x[3])  # Sortuj po F1-score
    
    # Weź top N najtrudniejszych klas
    if len(class_scores) > top_n:
        class_scores = class_scores[:top_n]
        title_suffix = f"Top {top_n} najtrudniejszych klas"
    else:
        title_suffix = "Wszystkie klasy"
    
    # Przygotuj dane do wykresu
    indices, precisions, recalls, f1_scores, supports = zip(*class_scores)
    class_labels = [class_names[i] for i in indices]
    
    # Wizualizacja
    x = np.arange(len(class_labels))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(max(14, len(class_labels) * 0.5), 8))
    
    bars1 = ax.bar(x - width/2, precisions, width, label='Precision', color='#3498db', alpha=0.8)
    bars2 = ax.bar(x + width/2, recalls, width, label='Recall', color='#e74c3c', alpha=0.8)
    
    # Dodaj linię dla F1-score
    ax2 = ax.twinx()
    line = ax2.plot(x, f1_scores, 'o-', color='#2ecc71', linewidth=2, markersize=6, label='F1-score')
    ax2.set_ylabel('F1-score', fontsize=11, fontweight='bold', color='#2ecc71')
    ax2.tick_params(axis='y', labelcolor='#2ecc71')
    ax2.set_ylim([0, 1.05])
    
    ax.set_xlabel('Klasa', fontsize=11, fontweight='bold')
    ax.set_ylabel('Precision / Recall', fontsize=11, fontweight='bold')
    ax.set_title(f'Precision i Recall per class - {title_suffix}', fontsize=13, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(class_labels, rotation=45, ha='right', fontsize=8)
    ax.set_ylim([0, 1.05])
    ax.legend(loc='upper left')
    ax2.legend(loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    plot_path = os.path.join(plots_dir, "precision_recall_per_class.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"✓ Precision/Recall per class zapisany: {plot_path}")
    print(f"  Pokazano {len(class_labels)} klas (najtrudniejsze)")
    plt.close()


def plot_error_confusion_matrix(y_test, y_pred, class_names, plots_dir="plots", min_errors=1):
    """
    Tworzy confusion matrix tylko dla klas z błędami - pokazuje które pary klas są mylone.
    
    Args:
        y_test: Prawdziwe etykiety
        y_pred: Predykcje
        class_names: Lista nazw klas
        plots_dir: Katalog do zapisu
        min_errors: Minimalna liczba błędów, aby klasa została uwzględniona
    """
    print("\n" + "=" * 70)
    print("ANALIZA PAR KLAS - KTÓRE SĄ MYLONE")
    print("=" * 70)
    
    Path(plots_dir).mkdir(parents=True, exist_ok=True)
    
    # Znajdź klasy z błędami
    errors = y_test != y_pred
    error_indices = np.where(errors)[0]
    
    if len(error_indices) == 0:
        print("  Brak błędów - nie można utworzyć macierzy błędów")
        return
    
    # Zlicz błędy per para klas (true_class -> pred_class)
    error_pairs = {}
    for idx in error_indices:
        true_class = y_test[idx]
        pred_class = y_pred[idx]
        pair = (true_class, pred_class)
        error_pairs[pair] = error_pairs.get(pair, 0) + 1
    
    # Filtruj pary z min_errors lub więcej
    significant_pairs = {pair: count for pair, count in error_pairs.items() if count >= min_errors}
    
    if len(significant_pairs) == 0:
        print(f"  Brak par klas z co najmniej {min_errors} błędami")
        return
    
    # Przygotuj dane do wykresu - top 20 par z największą liczbą błędów
    sorted_pairs = sorted(significant_pairs.items(), key=lambda x: x[1], reverse=True)[:20]
    
    true_classes = [class_names[pair[0]] for pair, _ in sorted_pairs]
    pred_classes = [class_names[pair[1]] for pair, _ in sorted_pairs]
    counts = [count for _, count in sorted_pairs]
    
    # Wizualizacja - wykres słupkowy
    fig, ax = plt.subplots(figsize=(14, max(8, len(sorted_pairs) * 0.4)))
    
    y_pos = np.arange(len(sorted_pairs))
    bars = ax.barh(y_pos, counts, color='coral', alpha=0.8, edgecolor='darkred')
    
    # Dodaj etykiety
    labels = [f"{true} → {pred}" for true, pred in zip(true_classes, pred_classes)]
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel('Liczba błędnych klasyfikacji', fontsize=11, fontweight='bold')
    ax.set_title(f'Top {len(sorted_pairs)} par klas najczęściej mylonych', fontsize=13, fontweight='bold', pad=15)
    ax.grid(True, alpha=0.3, axis='x')
    
    # Dodaj wartości na słupkach
    for i, (bar, count) in enumerate(zip(bars, counts)):
        ax.text(count + 0.5, i, str(count), va='center', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    
    plot_path = os.path.join(plots_dir, "error_confusion_matrix.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"✓ Analiza par klas zapisana: {plot_path}")
    print(f"  Pokazano {len(sorted_pairs)} par klas z błędami")
    plt.close()


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
        
        # 7. Top-N Accuracy
        plot_top_n_accuracy(
            results['y_test'],
            results['y_pred_proba'],
            class_names
        )
        
        # 8. Rozkład pewności modelu
        plot_confidence_distribution(
            results['y_test'],
            results['y_pred'],
            results['y_pred_proba']
        )
        
        # 9. Precision/Recall per class
        plot_precision_recall_per_class(
            results['y_test'],
            results['y_pred'],
            class_names,
            top_n=30
        )
        
        # 10. Analiza par klas (które są mylone)
        plot_error_confusion_matrix(
            results['y_test'],
            results['y_pred'],
            class_names,
            min_errors=1
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
        print("  - plots/top_n_accuracy.png")
        print("  - plots/confidence_distribution.png (NOWY - analityczny)")
        print("  - plots/precision_recall_per_class.png (NOWY - analityczny)")
        print("  - plots/error_confusion_matrix.png (ZMODYFIKOWANY - analityczny)")
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
    # W train.py jest ustawione na 75 (ETAP 5A), więc tutaj też 75
    main(
        model_path="models/best_model.h5",
        max_samples_per_class=75  # MUSI być takie samo jak w train.py! (ETAP 5A)
    )

