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

### Wyniki optymalizacji:
- **Test Accuracy:** 99.49% (poprzednio: 93.85% z 50 próbkami)
- **Top-3 Accuracy:** 100.00%
- **Test Loss:** 0.0089
- **Błędy:** 15 / 2925 (0.51%)
- **Wzrost accuracy:** +5.64% (z 93.85% do 99.49%)

### Wnioski:
1. **Więcej danych pomaga:** Zwiększenie z 50 do 75 próbek/klasę poprawiło wyniki
2. **Augmentacja nie zawsze pomaga:** W tym przypadku powodowała spadek accuracy, więc została wyłączona
3. **Model działa doskonale:** 99.49% accuracy to bardzo dobry wynik dla 195 klas

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
