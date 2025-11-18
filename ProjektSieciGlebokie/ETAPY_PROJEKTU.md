# Etapy projektu - Klasyfikacja flag państw

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

## ETAP 2: Projektowanie i implementacja modelu [NASTĘPNY - WAŻNY]

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

### Pliki do stworzenia:
- `model.py` - definicja architektury CNN

### Kryteria sukcesu:
- Model kompiluje się bez błędów
- Kształt wyjściowy: (batch_size, 195)
- Model gotowy do treningu

---

## ETAP 3: Trening modelu [WAŻNY]

### Zadania:
- Implementacja skryptu treningowego
- Konfiguracja hiperparametrów:
  - Learning rate
  - Batch size (32-64)
  - Liczba epok (początkowo 20-30)
- Implementacja callbacks:
  - ModelCheckpoint - zapisywanie najlepszego modelu
  - EarlyStopping - zatrzymanie przy braku poprawy
- Wizualizacja procesu uczenia:
  - Wykres accuracy (train vs validation)
  - Wykres loss (train vs validation)
- Zapis wytrenowanego modelu

### Pliki do stworzenia:
- `train.py` - skrypt treningowy

### Kryteria sukcesu:
- Model trenuje się bez błędów
- Wykresy pokazują zbieżność
- Zapisany model `best_model.h5`
- Wykresy zapisane do pliku

---

## ETAP 4: Ewaluacja modelu [WAŻNY]

### Zadania:
- Ewaluacja na zbiorze testowym:
  - Obliczenie accuracy i loss
  - Confusion matrix
- Analiza błędów:
  - Które flagi są najtrudniejsze do rozpoznania
  - Przykłady błędnych klasyfikacji
  - Wizualizacja confusion matrix
- Obliczenie metryk szczegółowych:
  - Precision, Recall, F1-score per class
  - Top-3 accuracy (opcjonalnie)

### Pliki do stworzenia:
- `evaluate.py` - skrypt ewaluacji

### Kryteria sukcesu:
- Raport z dokładnością na testach
- Confusion matrix wygenerowana
- Zidentyfikowane problematyczne klasy

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

- ETAP 1: Zakończony
- ETAP 2: Do realizacji
- ETAP 3-6: W kolejce
