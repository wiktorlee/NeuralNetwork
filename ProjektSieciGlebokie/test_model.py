import numpy as np

from model import build_model


def sanity_check_model():
    model = build_model()
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    assert model.output_shape == (None, 195), (
        "Oczekiwany kształt wyjścia (None, 195), "
        f"otrzymano {model.output_shape}"
    )

    dummy_input = np.zeros((2, 128, 128, 3), dtype=np.float32)
    preds = model.predict(dummy_input, verbose=0)

    assert preds.shape == (2, 195), (
        "Niepoprawny kształt predykcji: "
        f"spodziewano (2, 195), otrzymano {preds.shape}"
    )

    row_sums = np.sum(preds, axis=1)
    assert np.allclose(row_sums, 1.0, atol=1e-5), (
        "Wyniki softmax nie sumują się do 1 "
        f"(suma wierszy: {row_sums})"
    )


if __name__ == "__main__":
    sanity_check_model()
    print("✓ Model przeszedł sanity check (kształt i softmax).")

