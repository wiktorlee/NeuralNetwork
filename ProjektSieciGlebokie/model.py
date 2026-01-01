from tensorflow import keras
from tensorflow.keras import layers

def build_model(input_shape=(128, 128, 3), num_classes=195, base_filters=32):
    """
    Buduje model CNN do klasyfikacji flag państw.
    
    Args:
        input_shape: Kształt obrazu wejściowego (height, width, channels)
        num_classes: Liczba klas do klasyfikacji (195 krajów)
        base_filters: Liczba filtrów w pierwszej warstwie konwolucyjnej
    
    Returns:
        Model Keras gotowy do kompilacji
    """
    model = keras.Sequential([
        # Blok konwolucyjny 1
        layers.Conv2D(base_filters, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D(2, 2),
        layers.Dropout(0.25),
        
        # Blok konwolucyjny 2
        layers.Conv2D(base_filters * 2, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        layers.Dropout(0.25),
        
        # Blok konwolucyjny 3
        layers.Conv2D(base_filters * 4, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        layers.Dropout(0.25),
        
        # Blok konwolucyjny 4
        layers.Conv2D(base_filters * 8, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        layers.Dropout(0.25),
        
        # Spłaszczenie i warstwy gęste
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        
        # Warstwa wyjściowa
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model


if __name__ == "__main__":
    # Test czy model się buduje poprawnie
    model = build_model()
    model.summary()



