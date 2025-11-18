import kagglehub
from pathlib import Path

print("Pobieranie zbioru danych World Flags Dataset...")
print("To może potrwać kilka minut, zbiór jest duży...\n")

path = kagglehub.dataset_download("shuvokumarbasak4004/world-flags-dataset-195")

print(f"\n[OK] Pobrano dane!")
print(f"Sciezka do danych: {path}")

data_path = Path(path)
folders = [d for d in data_path.iterdir() if d.is_dir()]
print(f"\nZnaleziono {len(folders)} folderów z krajami")
