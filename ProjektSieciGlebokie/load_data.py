import numpy as np
from pathlib import Path
from PIL import Image
from sklearn.model_selection import train_test_split
import kagglehub

def load_flags_dataset(test_size=0.2, val_size=0.1, target_size=(128, 128), max_samples_per_class=None):
    print("Wczytywanie danych...")
    
    data_path = Path(kagglehub.dataset_download("shuvokumarbasak4004/world-flags-dataset-195"))
    data_path = data_path / "data"
    
    print(f"Ścieżka do danych: {data_path}")
    
    country_folders = sorted([d for d in data_path.iterdir() if d.is_dir()])
    class_names = [folder.name for folder in country_folders]
    
    print(f"Znaleziono {len(class_names)} krajów")
    
    images = []
    labels = []
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
    
    for idx, country_folder in enumerate(country_folders):
        country_name = country_folder.name
        image_files = [f for f in country_folder.iterdir() 
                      if f.suffix.lower() in image_extensions]
        
        if max_samples_per_class:
            image_files = image_files[:max_samples_per_class]
        
        print(f"Wczytywanie {len(image_files)} obrazów z {country_name}...", end='\r')
        
        for img_path in image_files:
            try:
                img = Image.open(img_path)
                
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                img = img.resize(target_size, Image.Resampling.LANCZOS)
                
                img_array = np.array(img, dtype=np.float32) / 255.0
                
                images.append(img_array)
                labels.append(idx)
                
            except Exception as e:
                print(f"\nBłąd przy wczytywaniu {img_path}: {e}")
                continue
        
        if (idx + 1) % 20 == 0:
            print(f"\nPrzetworzono {idx + 1}/{len(class_names)} krajów...")
    
    print(f"\n[OK] Wczytano {len(images)} obrazow")
    
    X = np.array(images)
    y = np.array(labels)
    
    print(f"Kształt danych: {X.shape}")
    print(f"Kształt etykiet: {y.shape}")
    
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        min_samples_per_class = min([(y_train == i).sum() for i in range(len(class_names))])
        use_stratify_val = min_samples_per_class >= 2
        
        if use_stratify_val:
            X_train, X_val, y_train, y_val = train_test_split(
                X_train, y_train, test_size=val_size, random_state=42, stratify=y_train
            )
        else:
            X_train, X_val, y_train, y_val = train_test_split(
                X_train, y_train, test_size=val_size, random_state=42
            )
    except ValueError:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=val_size, random_state=42
        )
    
    print(f"\nPodział danych:")
    print(f"  Train: {X_train.shape[0]} obrazów")
    print(f"  Val:   {X_val.shape[0]} obrazów")
    print(f"  Test:  {X_test.shape[0]} obrazów")
    
    return X_train, X_val, X_test, y_train, y_val, y_test, class_names


if __name__ == "__main__":
    print("Test wczytania danych (max 10 próbek na klasę)...\n")
    X_train, X_val, X_test, y_train, y_val, y_test, class_names = load_flags_dataset(
        max_samples_per_class=10
    )
    
    print(f"\n✓ Test zakończony pomyślnie!")
    print(f"Liczba klas: {len(class_names)}")
    print(f"Przykładowe klasy: {class_names[:5]}")
