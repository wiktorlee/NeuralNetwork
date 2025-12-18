# Etapy projektu - Klasyfikacja flag państw

## Wymagania systemowe

### Wersje oprogramowania:
- **Python:** 3.11 lub 3.12 (zalecane 3.11 dla lepszej kompatybilności z TensorFlow)
- **TensorFlow:** >= 2.10.0
- **NumPy:** >= 1.22.0
- **Pillow:** >= 9.0.0
- **scikit-learn:** >= 1.0.0
- **matplotlib:** >= 3.5.0
- **seaborn:** >= 0.12.0
- **kagglehub:** >= 0.3.0
- **pandas:** >= 1.4.0 (opcjonalne)

**Uwaga:** Python 3.14 nie jest jeszcze wspierany przez TensorFlow. Użyj Python 3.11 lub 3.12.

### Wymagania sprzętowe:
- **RAM:** Minimum 4GB (zalecane 8GB+)
- **Dysk:** ~1GB wolnego miejsca (dane + model)
- **GPU:** Opcjonalne (przyspiesza trening, ale nie jest wymagane)

---

## Pierwsze uruchomienie – krótkie instrukcje

### Lokalnie (na własnym komputerze):
1. Utwórz i aktywuj środowisko: 
   - Windows: `py -3.11 -m venv .venv` oraz `.\.venv\Scripts\Activate.ps1`
   - Linux/Mac: `python3.11 -m venv .venv` oraz `source .venv/bin/activate`
2. Zainstaluj zależności: `pip install -r requirements.txt`.
3. Uruchom `test_etap1.py`, aby pobrać dane i potwierdzić, że pipeline działa (to normalne, że pobieranie zajmuje ~500 MB i chwilę trwa).
4. Uruchom `model.py` (lub `test_model.py`), żeby sprawdzić, czy model buduje się poprawnie.

### W Google Colab (szybszy trening dzięki GPU):
1. Otwórz [Google Colab](https://colab.research.google.com/) i utwórz nowy notebook.
2. Włącz GPU: **Runtime → Change runtime type → Hardware accelerator: GPU → Save**.
3. W pierwszej komórce zainstaluj zależności:
   ```python
   !pip install kagglehub tensorflow matplotlib scikit-learn pillow numpy
   ```
4. Prześlij pliki projektu: kliknij ikonę folderu (📁 Files) po lewej → **Upload to session storage** → wybierz `train.py`, `model.py`, `load_data.py`.
5. Uruchom trening w nowej komórce:
   ```python
   !python /content/train.py
   ```
6. Po zakończeniu treningu pobierz wyniki: **Files → models/best_model.h5** (prawym → Download) oraz **plots/training_history.png**.
   
**Uwaga:** Trening w Colab na GPU trwa ~5-10 minut (vs ~75 minut na CPU lokalnie). Dane i wyniki są przechowywane tylko podczas sesji Colab.

## Cel projektu
Zbudowanie systemu klasyfikacji obrazów flag państw świata używając sieci neuronowych głębokich. Zbiór danych zawiera 195 krajów, po około 1001 obrazów na kraj.

## MVP - Minimalna wersja działająca
System zdolny do klasyfikacji flag z dokładnością powyżej 50% na zbiorze testowym, z pełnym pipeline od pobrania danych do ewaluacji modelu.

---

## ETAP 1: Pobieranie i przygotowanie danych [ZREALIZOWANY]

### Zadania wykonane:
- Implementacja automatycznego pobierania danych z Kaggle (kagglehub)
- Wczytywanie obrazów z folderów zorganizowanych według krajów
- Preprocessing:
  - Konwersja do RGB
  - Zmiana rozmiaru do 128x128 pikseli
  - Normalizacja wartości pikseli do zakresu [0, 1]
- Podział danych na zbiory: train (70%), validation (10%), test (20%)

### Pliki:
- `download_data.py` - pobieranie danych z Kaggle
- `load_data.py` - wczytywanie i preprocessing danych
- `requirements.txt` - lista zależności Python

### Status: Zakończony

---

## ETAP 2: Projektowanie i implementacja modelu [ZREALIZOWANY]

### Zadania:
- Zaprojektowanie architektury CNN odpowiedniej dla 195 klas
- Implementacja modelu w TensorFlow/Keras
- Wybór warstw:
  - Warstwy konwolucyjne (Conv2D)
  - Warstwy pooling (MaxPooling2D)
  - Warstwy dropout dla regularyzacji
  - Warstwy gęste (Dense)
  - Warstwa wyjściowa z softmax
- Kompilacja modelu z odpowiednim optimizer i loss function
- Test czy model się kompiluje i ma poprawny kształt wyjściowy

### Pliki:
- `model.py` - definicja architektury CNN + sekcja testowa w `__main__`
- `test_model.py` - sanity check kształtu i softmaxu

### Kryteria sukcesu:
- Model kompiluje się bez błędów
- Kształt wyjściowy: (batch_size, 195)
- Model gotowy do treningu

---

## ETAP 3: Trening modelu [ZREALIZOWANY]

### Zadania wykonane:
- ✅ Implementacja skryptu treningowego (`train.py`)
- ✅ Konfiguracja hiperparametrów:
  - Learning rate: `1e-3` (Adam optimizer)
  - Batch size: `32`
  - Maksymalna liczba epok: `30`
  - EarlyStopping patience: `5`
  - Liczba próbek na klasę: `50` (dla Colab, można zmienić w `train.py`)
- ✅ Implementacja callbacks:
  - **ModelCheckpoint** - zapisywanie najlepszego modelu (`models/best_model.h5`) na podstawie `val_accuracy`
  - **EarlyStopping** - zatrzymanie przy braku poprawy przez 5 epok, przywrócenie najlepszych wag
- ✅ Wizualizacja procesu uczenia:
  - Wykres accuracy (train vs validation) - `plots/training_history.png`
  - Wykres loss (train vs validation) - `plots/training_history.png`
- ✅ Zapis wytrenowanego modelu: `models/best_model.h5`

### Pliki:
- `train.py` - skrypt treningowy z funkcjami modułowymi
- `models/best_model.h5` - wytrenowany model (najlepsza wersja)
- `plots/training_history.png` - wykresy historii treningu

### Wyniki treningu:
- **Val accuracy:** 99.57% (epoka 11 - najlepsza)
- **Train accuracy:** 98.83% (epoka 11)
- **Liczba epok:** 16 (zatrzymane przez EarlyStopping)
- **Czas treningu:** ~5-10 minut na GPU (Colab), ~75 minut na CPU (lokalnie)
- **Zbieżność:** Szybka zbieżność od epoki 4, brak overfittingu

### Parametry użyte w treningu:
```python
batch_size=32
epochs=30
learning_rate=1e-3
patience=5
max_samples_per_class=50  # ~5,850 obrazów (30 na klasę × 195 klas)
```

### Status: Zakończony

---

## ETAP 4: Ewaluacja modelu [ZREALIZOWANY]

### Zadania wykonane:
- ✅ Ewaluacja na zbiorze testowym:
  - Obliczenie accuracy i loss
  - Confusion matrix
- ✅ Analiza błędów:
  - Które flagi są najtrudniejsze do rozpoznania
  - Przykłady błędnych klasyfikacji
  - Wizualizacja confusion matrix
- ✅ Obliczenie metryk szczegółowych:
  - Precision, Recall, F1-score per class
  - Top-3 accuracy

### Pliki:
- `evaluate.py` - skrypt ewaluacji
- `plots/confusion_matrix_top_classes.png` - wizualizacja confusion matrix (top 50 klas)
- `plots/confusion_matrix.txt` - surowe dane confusion matrix
- `plots/error_analysis.txt` - analiza najtrudniejszych klas
- `plots/error_examples.png` - przykłady błędnych klasyfikacji
- `plots/classification_report.txt` - szczegółowy raport z metrykami per class

### Wyniki ewaluacji:
- **Test Accuracy:** 93.85%
- **Test Loss:** 1.47
- **Top-3 Accuracy:** 94.36%
- **Liczba błędów:** 120 / 1950 (6.15%)
- **Metryki ogólne (macro average):**
  - Precision: 93.59%
  - Recall: 93.85%
  - F1-score: 93.68%
- **Metryki ogólne (weighted average):**
  - Precision: 93.59%
  - Recall: 93.85%
  - F1-score: 93.68%

### Obserwacje:
- Model osiąga 93.85% accuracy na zbiorze testowym
- Dla top 50 klas (najczęściej występujących) accuracy wynosi 100%
- Główne problemy: podobne flagi są mylone (np. Chad-Romania, Dominican Republic-DRC)
- Wiele klas z 100% błędów wynika z małej liczby próbek w test set (10 próbek na klasę)
- Model jest bardzo pewny swoich predykcji, nawet przy błędach (pewność 87-100%)

### Status: Zakończony

---

## ETAP 5: Optymalizacja (opcjonalnie)

### Zadania:
- Eksperymenty z hiperparametrami
- Augmentacja danych (obroty, przesunięcia, zmiana jasności/kontrastu)
- Transfer learning (użycie pre-trenowanych modeli jak ResNet, VGG)
- Porównanie różnych architektur

### Pliki do stworzenia:
- `augment_data.py` - augmentacja danych (opcjonalnie)
- `train_advanced.py` - zaawansowany trening (opcjonalnie)

### Uwaga:
Ten etap jest opcjonalny i zależy od wyników z etapu 4. Jeśli podstawowy model osiąga zadowalające wyniki, można go pominąć.

---

## ETAP 6: Sprawozdanie

### Zawartość sprawozdania:
- Opis problemu i zbioru danych
- Opis przygotowania danych (ETAP 1)
- Opis architektury modelu (ETAP 2)
- Obserwacje procesu uczenia:
  - Wykresy loss i accuracy
  - Analiza zbieżności
  - Problemy napotkane i rozwiązania
- Wyniki i wnioski:
  - Dokładność modelu na zbiorze testowym
  - Analiza błędów
  - Wnioski końcowe

### Pliki:
- Dokumentacja/sprawozdanie w formacie Markdown lub PDF

---

## Podział pracy między członków zespołu

### Proponowany podział:
- **Osoba 1**: ETAP 2 (Model) + ETAP 3 (Trening)
- **Osoba 2**: ETAP 4 (Ewaluacja) + ETAP 6 (Sprawozdanie - część wyników)
- **Osoba 3**: ETAP 5 (Optymalizacja) + ETAP 6 (Sprawozdanie - część analizy)

### Uwaga:
ETAP 1 jest już zrealizowany i może być używany przez wszystkich. Każdy powinien mieć dostęp do danych i móc uruchomić `load_data.py`.

---

## Najważniejsze etapy dla MVP

1. **ETAP 2** - Bez modelu nie ma co trenować
2. **ETAP 3** - Trening jest kluczowy dla działania systemu
3. **ETAP 4** - Ewaluacja pokazuje czy projekt działa

ETAP 5 i 6 są ważne dla jakości projektu, ale MVP można zrealizować bez optymalizacji.

---

## Aktualny status

- ETAP 1: Zakończony ✅
- ETAP 2: Zakończony ✅
- ETAP 3: Zakończony ✅
- ETAP 4: Zakończony ✅
- ETAP 5: W kolejce (opcjonalny)
- ETAP 6: W kolejce (sprawozdanie)
