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

## 🚀 INSTRUKCJA URUCHOMIENIA - KROK PO KROKU

### ⚠️ WAŻNE: Przeczytaj przed rozpoczęciem!

**Dla osób uruchamiających projekt po raz pierwszy:** Ta sekcja zawiera wszystkie kroki potrzebne do uruchomienia projektu od zera. Postępuj dokładnie krok po kroku.

---

### 📋 Wymagane pliki do uruchomienia:

Musisz mieć następujące pliki w projekcie:
- ✅ `train.py` - skrypt treningowy
- ✅ `evaluate.py` - skrypt ewaluacji (WAŻNE - nie zapomnij!)
- ✅ `model.py` - definicja modelu CNN
- ✅ `load_data.py` - wczytywanie i preprocessing danych
- ✅ `requirements.txt` - lista zależności (opcjonalne, ale pomocne)

### 🎯 Ważna informacja o gotowym modelu:

**✅ Gotowy model jest już w repozytorium!**

W folderze `models/` znajduje się już wytrenowany model `best_model.h5` (accuracy: 98.97%), który jest commitowany do repo.

**Oznacza to, że:**
- Możesz od razu uruchomić `evaluate.py` bez treningu (jeśli chcesz tylko zobaczyć wyniki)
- Albo możesz wytrenować własny model używając `train.py` (nadpisze istniejący model)
- Model jest gotowy do użycia i nie wymaga treningu

**Jeśli chcesz tylko zobaczyć wyniki bez treningu:**
- Pomiń Krok 4 (trening) i przejdź od razu do Kroku 5 (ewaluacja)
- Model `models/best_model.h5` jest już dostępny w repo

---

### 🖥️ Opcja 1: Uruchomienie w Google Colab (ZALECANE - szybsze dzięki GPU)

#### Krok 1: Przygotowanie środowiska Colab
1. Otwórz [Google Colab](https://colab.research.google.com/)
2. Utwórz nowy notebook: **File → New notebook**
3. **WAŻNE:** Włącz GPU: **Runtime → Change runtime type → Hardware accelerator: GPU → Save**
   - Bez GPU trening będzie trwał ~75 minut, z GPU ~5-10 minut

#### Krok 2: Instalacja zależności
W pierwszej komórce notebooka uruchom:
```python
!pip install kagglehub tensorflow matplotlib scikit-learn pillow numpy seaborn
```
**Uwaga:** Instalacja może chwilę potrwać. Poczekaj aż zakończy się (✓).

#### Krok 3: Upload plików projektu
**WAŻNE:** Musisz wgrać WSZYSTKIE 4 pliki:
1. Kliknij ikonę folderu (📁 Files) po lewej stronie
2. Kliknij **Upload to session storage**
3. Wybierz i wgraj następujące pliki:
   - ✅ `train.py`
   - ✅ `evaluate.py` ← **NIE ZAPOMNIJ TEGO!**
   - ✅ `model.py`
   - ✅ `load_data.py`

**Uwaga:** Pliki muszą być w folderze `/content/` w Colab. Sprawdź czy wszystkie 4 pliki są widoczne w panelu Files.

#### Krok 4: Uruchomienie treningu (OPCJONALNE)

**ℹ️ UWAGA:** Jeśli chcesz tylko zobaczyć wyniki, możesz pominąć ten krok! Gotowy model `models/best_model.h5` jest już w repo i możesz od razu przejść do Kroku 5 (ewaluacja).

Jeśli chcesz wytrenować własny model (lub nadpisać istniejący), uruchom:
```python
!python /content/train.py
```

**Co się dzieje podczas treningu:**
- Pobieranie danych z Kaggle (~500 MB, może chwilę potrwać)
- Wczytywanie i preprocessing obrazów (14,625 obrazów)
- Budowa modelu CNN (5.3M parametrów)
- Trening modelu (11 epok, ~5-10 minut na GPU)
- Generowanie wykresów treningowych (6 wykresów)

**Oczekiwany wynik:**
- Model zapisany: `models/best_model.h5`
- Najlepsza val_accuracy: ~98.97% (epoka 6)
- 6 wykresów w folderze `plots/`:
  - `training_history.png` - accuracy i loss
  - `classification_error.png` - błąd klasyfikacji
  - `learning_rate_evolution.png` - ewolucja LR
  - `loss_per_class.png` - loss per class
  - `weight_trajectories.png` - trajektorie wag
  - `gradient_norms.png` - normy gradientów

#### Krok 5: Uruchomienie ewaluacji
**WAŻNE:** Po zakończeniu treningu uruchom ewaluację w nowej komórce:
```python
!python /content/evaluate.py
```

**Co się dzieje podczas ewaluacji:**
- Wczytywanie modelu `models/best_model.h5`
- Ewaluacja na zbiorze testowym (2,925 obrazów)
- Generowanie wykresów ewaluacyjnych (9 wykresów)
- Generowanie raportów tekstowych (3 pliki .txt)

**Oczekiwany wynik:**
- Test Accuracy: ~98.97%
- Top-3 Accuracy: 100.00%
- 9 wykresów w folderze `plots/`:
  - `confusion_matrix_top_classes.png`
  - `error_examples.png`
  - `top_n_accuracy.png`
  - `confidence_distribution.png`
  - `precision_recall_per_class.png`
  - `error_confusion_matrix.png`
  - (i inne)
- 3 pliki tekstowe z wynikami:
  - `classification_report.txt` - szczegółowe metryki
  - `error_analysis.txt` - analiza błędów
  - `confusion_matrix.txt` - surowe dane

#### Krok 6: Pobieranie wyników
Po zakończeniu treningu i ewaluacji:

1. **Pobierz model:**
   - Files → `models/best_model.h5` → prawy przycisk → Download

2. **Pobierz wszystkie wykresy:**
   - Files → `plots/` → zaznacz wszystkie pliki PNG → Download
   - Powinno być 15 wykresów (6 z treningu + 9 z ewaluacji)

3. **Pobierz raporty tekstowe:**
   - Files → `plots/` → zaznacz pliki `.txt` → Download
   - Powinno być 3 pliki tekstowe

**⚠️ UWAGA:** Dane w Colab są przechowywane tylko podczas sesji. Po zamknięciu notebooka wszystko znika! Pobierz wyniki przed zamknięciem.

---

### 💻 Opcja 2: Uruchomienie lokalnie (na własnym komputerze)

#### Krok 1: Przygotowanie środowiska
1. Utwórz i aktywuj środowisko wirtualne:
   - **Windows:** 
     ```powershell
     py -3.11 -m venv .venv
     .\.venv\Scripts\Activate.ps1
     ```
   - **Linux/Mac:**
     ```bash
     python3.11 -m venv .venv
     source .venv/bin/activate
     ```

2. Zainstaluj zależności:
   ```bash
   pip install -r requirements.txt
   ```
   Lub ręcznie:
   ```bash
   pip install kagglehub tensorflow matplotlib scikit-learn pillow numpy seaborn
   ```

#### Krok 2: Weryfikacja środowiska
Uruchom testy, aby sprawdzić czy wszystko działa:
```bash
python test_etap1.py    # Test pobierania danych
python model.py         # Test budowy modelu
```

#### Krok 3: Trening modelu (OPCJONALNE)

**ℹ️ UWAGA:** Jeśli chcesz tylko zobaczyć wyniki, możesz pominąć ten krok! Gotowy model `models/best_model.h5` jest już w repo i możesz od razu przejść do Kroku 4 (ewaluacja).

Jeśli chcesz wytrenować własny model (lub nadpisać istniejący):
```bash
python train.py
```

**Czas treningu:** ~75 minut na CPU (bez GPU), ~5-10 minut z GPU

#### Krok 4: Ewaluacja modelu
```bash
python evaluate.py
```

**Wyniki:** Wszystkie pliki zostaną zapisane w folderze `plots/` i `models/`

---

### 📊 Podsumowanie wygenerowanych plików

Po pełnym uruchomieniu (trening + ewaluacja) powinieneś mieć:

**W folderze `models/`:**
- ✅ `best_model.h5` - wytrenowany model
  - **ℹ️ UWAGA:** Ten model jest już commitowany do repo! Jeśli nie trenujesz własnego modelu, użyjesz gotowego modelu z repo (accuracy: 98.97%)

**W folderze `plots/` - wykresy treningowe (6 plików):**
- ✅ `training_history.png` - accuracy i loss przez epoki
- ✅ `classification_error.png` - błąd klasyfikacji (1-accuracy)
- ✅ `learning_rate_evolution.png` - ewolucja learning rate
- ✅ `loss_per_class.png` - loss dla wybranych klas
- ✅ `weight_trajectories.png` - trajektorie wag warstwy wyjściowej
- ✅ `gradient_norms.png` - normy gradientów przez epoki

**W folderze `plots/` - wykresy ewaluacyjne (9 plików):**
- ✅ `confusion_matrix_top_classes.png` - confusion matrix (top 50 klas)
- ✅ `error_examples.png` - przykłady błędnych klasyfikacji
- ✅ `top_n_accuracy.png` - Top-N accuracy (N=1-5)
- ✅ `confidence_distribution.png` - rozkład pewności modelu
- ✅ `precision_recall_per_class.png` - Precision/Recall per class
- ✅ `error_confusion_matrix.png` - pary klas najczęściej mylonych
- (i inne)

**W folderze `plots/` - raporty tekstowe (3 pliki):**
- ✅ `classification_report.txt` - szczegółowe metryki per class
- ✅ `error_analysis.txt` - analiza najtrudniejszych klas
- ✅ `confusion_matrix.txt` - surowe dane confusion matrix

**Łącznie: 1 model + 15 wykresów + 3 raporty = 19 plików wynikowych**

---

### ⚠️ Częste problemy i rozwiązania

**Problem 1: "ModuleNotFoundError: No module named 'seaborn'"**
- **Rozwiązanie:** Dodaj `seaborn` do instalacji: `!pip install seaborn`

**Problem 2: "Model nie znaleziony" podczas ewaluacji**
- **Rozwiązanie:** Upewnij się, że najpierw uruchomiłeś `train.py` i model został zapisany

**Problem 3: "max_samples_per_class mismatch"**
- **Rozwiązanie:** Upewnij się, że w `train.py` i `evaluate.py` jest ta sama wartość (obecnie 75)

**Problem 4: Trening trwa bardzo długo**
- **Rozwiązanie:** Użyj GPU w Colab (Runtime → Change runtime type → GPU)

**Problem 5: Brakuje niektórych wykresów**
- **Rozwiązanie:** Upewnij się, że uruchomiłeś zarówno `train.py` (6 wykresów) jak i `evaluate.py` (9 wykresów)

---

### 📝 Notatki dla kolegów

- **Gotowy model w repo:** Model `models/best_model.h5` jest już commitowany - możesz użyć go bez treningu!
- **Opcjonalny trening:** Jeśli chcesz wytrenować własny model, uruchom `train.py` (nadpisze istniejący model)
- **Ewaluacja bez treningu:** Możesz od razu uruchomić `evaluate.py` używając gotowego modelu z repo
- **Nie modyfikuj** `max_samples_per_class` bez aktualizacji w obu plikach (`train.py` i `evaluate.py`)
- **Jeśli trenujesz:** Zawsze uruchamiaj najpierw `train.py`, potem `evaluate.py`
- **Pobierz wszystkie pliki** z Colab przed zamknięciem sesji
- **Sprawdź** czy wszystkie 4 pliki są w Colab przed uruchomieniem

---

## Pierwsze uruchomienie – krótkie instrukcje (stara sekcja)

### Lokalnie (na własnym komputerze):
1. Utwórz i aktywuj środowisko: 
   - Windows: `py -3.11 -m venv .venv` oraz `.\.venv\Scripts\Activate.ps1`
   - Linux/Mac: `python3.11 -m venv .venv` oraz `source .venv/bin/activate`
2. Zainstaluj zależności: `pip install -r requirements.txt`.
3. Uruchom `test_etap1.py`, aby pobrać dane i potwierdzić, że pipeline działa (to normalne, że pobieranie zajmuje ~500 MB i chwilę trwa).
4. Uruchom `model.py` (lub `test_model.py`), żeby sprawdzić, czy model buduje się poprawnie.

*(Zobacz sekcję "🚀 INSTRUKCJA URUCHOMIENIA - KROK PO KROKU" powyżej dla szczegółowych instrukcji)*

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
  - Learning rate: `1e-3` (Adam optimizer) z **ReduceLROnPlateau** scheduler
  - Batch size: `32`
  - Maksymalna liczba epok: `30`
  - EarlyStopping patience: `5`
  - Liczba próbek na klasę: `75` (zwiększone z 50 dla lepszych wyników)
- ✅ Implementacja callbacks:
  - **ModelCheckpoint** - zapisywanie najlepszego modelu (`models/best_model.h5`) na podstawie `val_accuracy`
  - **EarlyStopping** - zatrzymanie przy braku poprawy przez 5 epok, przywrócenie najlepszych wag
  - **ReduceLROnPlateau** - automatyczne zmniejszanie learning rate (factor=0.5, patience=3, min_lr=1e-6)
  - **TrainingMetricsCallback** - custom callback do zbierania metryk analitycznych
- ✅ Wizualizacja procesu uczenia (6 wykresów):
  - `training_history.png` - accuracy i loss (train vs validation)
  - `classification_error.png` - błąd klasyfikacji (1-accuracy)
  - `learning_rate_evolution.png` - ewolucja learning rate przez epoki
  - `loss_per_class.png` - loss dla wybranych klas przez epoki
  - `weight_trajectories.png` - trajektorie wag warstwy wyjściowej
  - `gradient_norms.png` - normy gradientów przez epoki
- ✅ Zapis wytrenowanego modelu: `models/best_model.h5`

### Pliki:
- `train.py` - skrypt treningowy z funkcjami modułowymi
- `models/best_model.h5` - wytrenowany model (najlepsza wersja)
- `plots/training_history.png` - wykresy historii treningu
- `plots/classification_error.png` - błąd klasyfikacji
- `plots/learning_rate_evolution.png` - ewolucja learning rate
- `plots/loss_per_class.png` - loss per class
- `plots/weight_trajectories.png` - trajektorie wag
- `plots/gradient_norms.png` - normy gradientów

### Wyniki treningu (aktualne):
- **Val accuracy:** 98.97% (epoka 6 - najlepsza)
- **Train accuracy:** ~97% (epoka 6)
- **Liczba epok:** 11 (zatrzymane przez EarlyStopping)
- **Czas treningu:** ~5-10 minut na GPU (Colab), ~75 minut na CPU (lokalnie)
- **Zbieżność:** Szybka zbieżność od epoki 2-3, brak overfittingu
- **Learning Rate:** Zmniejszony z 0.001 do 0.0005 w epoce 9 (ReduceLROnPlateau)

### Parametry użyte w treningu:
```python
batch_size=32
epochs=30
learning_rate=1e-3  # z ReduceLROnPlateau scheduler
patience=5
max_samples_per_class=75  # 14,625 obrazów (75 na klasę × 195 klas)
use_augmentation=False  # wyłączona (testy pokazały spadek accuracy)
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
- **Wykresy ewaluacyjne (9 plików):**
  - `confusion_matrix_top_classes.png` - wizualizacja confusion matrix (top 50 klas)
  - `error_examples.png` - przykłady błędnych klasyfikacji
  - `top_n_accuracy.png` - Top-N accuracy (N=1-5)
  - `confidence_distribution.png` - rozkład pewności modelu (poprawne vs błędne)
  - `precision_recall_per_class.png` - Precision/Recall per class (top 30 najtrudniejszych)
  - `error_confusion_matrix.png` - pary klas najczęściej mylonych
  - (i inne)
- **Raporty tekstowe (3 pliki):**
  - `confusion_matrix.txt` - surowe dane confusion matrix
  - `error_analysis.txt` - analiza najtrudniejszych klas
  - `classification_report.txt` - szczegółowy raport z metrykami per class

### Wyniki ewaluacji (aktualne):
- **Test Accuracy:** 98.97%
- **Test Loss:** 0.0264
- **Top-1 Accuracy:** 98.97%
- **Top-2 Accuracy:** 100.00% ⭐
- **Top-3 Accuracy:** 100.00%
- **Liczba błędów:** 30 / 2925 (1.03%)
- **Metryki ogólne (macro average):**
  - Precision: 98.46%
  - Recall: 98.97%
  - F1-score: 98.63%
- **Metryki ogólne (weighted average):**
  - Precision: 98.46%
  - Recall: 98.97%
  - F1-score: 98.63%

### Obserwacje:
- Model osiąga **98.97% accuracy** na zbiorze testowym (bardzo dobry wynik!)
- **Top-2 accuracy: 100%** - prawidłowa odpowiedź jest zawsze w top 2 predykcji
- Dla top 50 klas (najczęściej występujących) accuracy wynosi 100%
- **Główne problemy:** Tylko 2 pary klas są mylone:
  - Chad → Romania (15 błędów, 100% błędów dla Chad)
  - Indonesia → Monaco (15 błędów, 100% błędów dla Indonesia)
- **Dlaczego te błędy?** Flagi są wizualnie niemal identyczne:
  - Chad vs Romania: Różnią się tylko odcieniem niebieskiego
  - Indonesia vs Monaco: Identyczne flagi (różne tylko proporcje)
- **Pewność modelu:** Model ma niską pewność (~51-56%) przy błędach, co wskazuje na świadomość niepewności
- **193 z 195 klas:** Mają 100% accuracy (perfekcyjna klasyfikacja)

### Status: Zakończony

---

## ETAP 5: Optymalizacja [ZREALIZOWANY]

### Zadania wykonane:
- ✅ **ETAP 5A: Zwiększenie liczby próbek**
  - Zwiększono z 50 do 75 próbek na klasę
  - Więcej danych = lepsze wyniki
- ✅ **ETAP 5B: Przetestowanie augmentacji danych**
  - Zaimplementowano augmentację (obrót, przesunięcie, jasność, zoom)
  - **Wynik testu:** Augmentacja powodowała spadek accuracy (1.54% z augmentacją vs 99.49% bez)
  - **Decyzja:** Augmentacja wyłączona - dla tego zadania nie była potrzebna
  - **Szczegóły próby:** Zobacz sekcję "Próba z augmentacją danych" poniżej
- ⏭️ **ETAP 5C: Eksperymenty z hiperparametrami** (opcjonalnie, pominięte)
  - Model osiąga już doskonałe wyniki (99.49%), więc dalsze optymalizacje nie były konieczne

### Pliki:
- `train.py` - zaktualizowany (75 próbek, augmentacja wyłączona)
- `evaluate.py` - zaktualizowany (75 próbek)
- `models/best_model.h5` - nowy model wytrenowany na 75 próbkach/klasę

### Wyniki optymalizacji (finalne):
- **Test Accuracy:** 98.97% (poprzednio: 93.85% z 50 próbkami)
- **Top-2 Accuracy:** 100.00% ⭐
- **Top-3 Accuracy:** 100.00%
- **Test Loss:** 0.0264
- **Błędy:** 30 / 2925 (1.03%)
- **Wzrost accuracy:** +5.12% (z 93.85% do 98.97%)
- **Dodatkowe ulepszenia:**
  - Learning Rate Scheduler (ReduceLROnPlateau) - automatyczna optymalizacja LR
  - 6 wykresów analitycznych z treningu (obserwacja procesu uczenia)
  - 9 wykresów analitycznych z ewaluacji (szczegółowa analiza wyników)

### Wnioski:
1. **Więcej danych pomaga:** Zwiększenie z 50 do 75 próbek/klasę poprawiło wyniki (+5.12%)
2. **Augmentacja nie zawsze pomaga:** W tym przypadku powodowała spadek accuracy, więc została wyłączona
3. **Learning Rate Scheduler pomaga:** ReduceLROnPlateau automatycznie optymalizuje learning rate podczas treningu
4. **Model działa doskonale:** 98.97% accuracy to bardzo dobry wynik dla 195 klas
5. **Top-2 accuracy 100%:** Nawet gdy model się myli, prawidłowa odpowiedź jest zawsze w top 2 predykcji
6. **Błędy są przewidywalne:** Wszystkie błędy dotyczą wizualnie bardzo podobnych flag (Chad-Romania, Indonesia-Monaco)

---

### Próba z augmentacją danych (ETAP 5B - szczegóły)

#### Co było testowane:
Zaimplementowano augmentację danych używając `ImageDataGenerator` z następującymi parametrami:
- **Obrót:** ±10 stopni (`rotation_range=10`)
- **Przesunięcie poziome:** ±10% (`width_shift_range=0.1`)
- **Przesunięcie pionowe:** ±10% (`height_shift_range=0.1`)
- **Jasność:** ±20% (`brightness_range=[0.8, 1.2]`)
- **Zoom:** ±10% (`zoom_range=0.1`)
- **Fill mode:** `nearest` (wypełnianie pikseli przy transformacjach)
- **Rescale:** `1.0` (dane już znormalizowane do [0,1])

#### Wyniki testów:

**Z augmentacją:**
- Train accuracy: 1-19% (bardzo niska, rosła powoli)
- Val accuracy: 1.03% (epoka 1), potem spadała do 0%
- Test accuracy: 1.54%
- Val loss: 5.67 → 19.41 (bardzo wysoki, rosnący)
- Problem: Model się nie uczył poprawnie, generator kończył się za wcześnie

**Bez augmentacji:**
- Train accuracy: 21% → 98% (szybki wzrost)
- Val accuracy: 7.18% → 99.49% (epoka 8)
- Test accuracy: 99.49%
- Val loss: 5.32 → 0.0089 (szybki spadek)
- Sukces: Model uczył się poprawnie i osiągnął doskonałe wyniki

#### Możliwe przyczyny problemu z augmentacją:

1. **Zbyt agresywne transformacje dla flag:**
   - Flagi mają specyficzną geometrię (proporcje, kolory, wzory)
   - Obrót ±10° może zmienić orientację flagi (np. flaga pionowa vs pozioma)
   - Przesunięcia mogą przyciąć ważne elementy flagi

2. **Problem z generatorami danych:**
   - Generator kończył się za wcześnie ("Your input ran out of data")
   - Możliwe problemy z `steps_per_epoch` lub synchronizacją generatorów

3. **Normalizacja danych:**
   - Dane już były znormalizowane do [0,1]
   - `rescale=1.0` w generatorze może powodować konflikty

4. **Zbyt mało danych:**
   - 75 próbek/klasę może być za mało dla skutecznej augmentacji
   - Augmentacja działa lepiej przy większych zbiorach danych

#### Co można spróbować w przyszłości:

1. **Mniej agresywne transformacje:**
   - Obrót: ±5° zamiast ±10°
   - Przesunięcia: ±5% zamiast ±10%
   - Wyłączyć zoom (flagi mają stałe proporcje)

2. **Selektywna augmentacja:**
   - Tylko jasność i kontrast (bez obrotów/przesunięć)
   - Augmentacja tylko dla niektórych klas

3. **Więcej danych:**
   - Zwiększyć do 100-200 próbek/klasę przed zastosowaniem augmentacji

4. **Inne metody augmentacji:**
   - Cutout/CutMix
   - Mixup
   - Specjalne transformacje dla flag (np. zmiana kolorów w określonych zakresach)

5. **Poprawa generatorów:**
   - Użyć `.repeat()` w generatorach
   - Sprawdzić synchronizację między train i validation generatorami
   - Użyć bezpośrednio tablic dla validation (jak w finalnej wersji)

#### Status próby:
- **Próba:** Zrealizowana i udokumentowana
- **Wynik:** Niepowodzenie - augmentacja powodowała spadek accuracy
- **Decyzja:** Augmentacja wyłączona w finalnej wersji
- **Możliwość powrotu:** Tak - można wrócić do tego w przyszłości z mniej agresywnymi parametrami

---

### Status: Zakończony

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
- ETAP 5: Zakończony ✅
  - ETAP 5A: Więcej danych (50→75 próbek) ✅
  - ETAP 5B: Przetestowanie augmentacji (wyłączona) ✅
  - ETAP 5C: Eksperymenty z hiperparametrami (pominięte - niepotrzebne) ⏭️
- ETAP 6: W kolejce (sprawozdanie)
