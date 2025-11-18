import sys
from pathlib import Path

def test_imports():
    print("1. Sprawdzanie importów...")
    try:
        import numpy as np
        import kagglehub
        from PIL import Image
        from sklearn.model_selection import train_test_split
        print("   OK - wszystkie biblioteki dostępne")
        return True
    except ImportError as e:
        print(f"   BŁĄD - brakuje biblioteki: {e}")
        return False

def test_download_data():
    print("\n2. Sprawdzanie pobierania danych...")
    try:
        import kagglehub
        from pathlib import Path
        
        data_path = Path(kagglehub.dataset_download("shuvokumarbasak4004/world-flags-dataset-195"))
        data_path = data_path / "data"
        
        if data_path.exists():
            folders = [d for d in data_path.iterdir() if d.is_dir()]
            print(f"   OK - dane dostępne: {len(folders)} krajów")
            print(f"   Ścieżka: {data_path}")
            return True, data_path
        else:
            print("   BŁĄD - folder data nie istnieje")
            return False, None
    except Exception as e:
        print(f"   BŁĄD - problem z pobieraniem danych: {e}")
        return False, None

def test_load_data(data_path):
    print("\n3. Test wczytywania danych (load_data.py)...")
    try:
        from load_data import load_flags_dataset
        
        print("   Wczytywanie próbki danych (10 obrazów na kraj)...")
        X_train, X_val, X_test, y_train, y_val, y_test, class_names = load_flags_dataset(
            test_size=0.2,
            val_size=0.1,
            target_size=(128, 128),
            max_samples_per_class=10
        )
        
        print(f"\n   Weryfikacja danych:")
        print(f"   - Liczba klas: {len(class_names)}")
        print(f"   - Kształt X_train: {X_train.shape}")
        print(f"   - Kształt y_train: {y_train.shape}")
        print(f"   - Kształt X_val: {X_val.shape}")
        print(f"   - Kształt X_test: {X_test.shape}")
        print(f"   - Zakres wartości X_train: [{X_train.min():.3f}, {X_train.max():.3f}]")
        print(f"   - Zakres wartości y_train: [{y_train.min()}, {y_train.max()}]")
        
        if X_train.min() >= 0 and X_train.max() <= 1:
            print("   OK - wartości znormalizowane do [0, 1]")
        else:
            print("   UWAGA - wartości nie są w zakresie [0, 1]")
        
        if y_train.min() >= 0 and y_train.max() < len(class_names):
            print("   OK - etykiety są poprawne")
        else:
            print("   BŁĄD - etykiety poza zakresem")
            return False
        
        total = len(X_train) + len(X_val) + len(X_test)
        train_pct = len(X_train) / total * 100
        val_pct = len(X_val) / total * 100
        test_pct = len(X_test) / total * 100
        
        print(f"\n   Podział danych:")
        print(f"   - Train: {len(X_train)} ({train_pct:.1f}%)")
        print(f"   - Val:   {len(X_val)} ({val_pct:.1f}%)")
        print(f"   - Test:  {len(X_test)} ({test_pct:.1f}%)")
        
        print("\n   OK - load_data.py działa poprawnie")
        return True
        
    except Exception as e:
        print(f"   BŁĄD - problem z wczytywaniem danych: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_file_structure():
    print("\n4. Sprawdzanie struktury plików...")
    required_files = [
        'download_data.py',
        'load_data.py',
        'requirements.txt'
    ]
    
    missing = []
    for file in required_files:
        if Path(file).exists():
            print(f"   OK - {file}")
        else:
            print(f"   BŁĄD - brakuje {file}")
            missing.append(file)
    
    return len(missing) == 0

def main():
    print("="*70)
    print("WERYFIKACJA ETAPU 1 - Pobieranie i przygotowanie danych")
    print("="*70)
    
    results = []
    
    results.append(("Importy bibliotek", test_imports()))
    
    data_ok, data_path = test_download_data()
    results.append(("Pobieranie danych", data_ok))
    
    if data_ok:
        results.append(("Wczytywanie danych", test_load_data(data_path)))
    else:
        print("\n3. Pomijanie testu wczytywania - dane nie są dostępne")
        results.append(("Wczytywanie danych", False))
    
    results.append(("Struktura plików", test_file_structure()))
    
    print("\n" + "="*70)
    print("PODSUMOWANIE WERYFIKACJI")
    print("="*70)
    
    all_passed = True
    for test_name, passed in results:
        status = "OK" if passed else "BŁĄD"
        print(f"{test_name:30s}: {status}")
        if not passed:
            all_passed = False
    
    print("="*70)
    if all_passed:
        print("ETAP 1: WERYFIKACJA ZAKOŃCZONA POMYŚLNIE")
        print("Wszystkie testy przeszły. Dane są gotowe do użycia.")
    else:
        print("ETAP 1: WERYFIKACJA NIEUDANA")
        print("Niektóre testy nie przeszły. Sprawdź błędy powyżej.")
    print("="*70)
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
