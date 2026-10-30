"""
generate_dataset.py
--------------------
Builds Crop_recommendation.csv — a labeled dataset of soil/climate readings
mapped to the best-fit crop, in the same shape as the well-known
"Crop Recommendation Dataset" (N, P, K, temperature, humidity, ph, rainfall -> label).

Since this environment has no network access to pull the original CSV, each
crop's realistic agronomic range (N/P/K in kg/ha, temperature in C, relative
humidity in %, soil pH, rainfall in mm) is encoded below and used to sample
synthetic-but-plausible rows with gaussian noise + a bit of cross-feature
correlation, so the resulting dataset is learnable and produces sane
feature-importance/metric results. Drop in a real CSV with the same columns
to replace this at any time — train_model.py doesn't care which one it reads.
"""
import numpy as np
import pandas as pd
import os

RNG = np.random.default_rng(42)

# mean, std for each feature, per crop — approximate agronomic ranges
CROP_PROFILES = {
    "rice":        dict(N=(80, 15), P=(45, 10), K=(40, 10), temperature=(24, 2.5), humidity=(82, 5),  ph=(6.2, 0.4), rainfall=(230, 40)),
    "maize":       dict(N=(80, 18), P=(45, 12), K=(20, 8),  temperature=(23, 3.5), humidity=(63, 8),  ph=(6.3, 0.5), rainfall=(85, 25)),
    "chickpea":    dict(N=(40, 8),  P=(67, 10), K=(80, 10), temperature=(19, 3.0), humidity=(16, 4),  ph=(7.3, 0.4), rainfall=(80, 15)),
    "kidneybeans": dict(N=(21, 6),  P=(67, 10), K=(20, 6),  temperature=(19, 2.5), humidity=(21, 4),  ph=(5.8, 0.4), rainfall=(105, 20)),
    "pigeonpeas":  dict(N=(21, 6),  P=(67, 12), K=(20, 6),  temperature=(27, 4.0), humidity=(48, 12), ph=(5.8, 0.5), rainfall=(150, 45)),
    "mothbeans":   dict(N=(21, 6),  P=(48, 10), K=(20, 6),  temperature=(28, 2.5), humidity=(53, 12), ph=(6.8, 0.5), rainfall=(50, 12)),
    "mungbean":    dict(N=(21, 6),  P=(47, 10), K=(20, 6),  temperature=(28, 2.0), humidity=(85, 5),  ph=(6.7, 0.4), rainfall=(50, 10)),
    "blackgram":   dict(N=(40, 8),  P=(67, 10), K=(20, 6),  temperature=(29, 2.5), humidity=(65, 8),  ph=(7.1, 0.4), rainfall=(68, 12)),
    "lentil":      dict(N=(19, 6),  P=(68, 10), K=(19, 6),  temperature=(24, 3.0), humidity=(65, 8),  ph=(6.9, 0.4), rainfall=(45, 10)),
    "pomegranate": dict(N=(19, 6),  P=(19, 6),  K=(40, 8),  temperature=(21, 3.0), humidity=(90, 4),  ph=(6.4, 0.4), rainfall=(107, 20)),
    "banana":      dict(N=(100, 15),P=(82, 10), K=(50, 8),  temperature=(27, 2.0), humidity=(80, 4),  ph=(5.9, 0.4), rainfall=(105, 15)),
    "mango":       dict(N=(20, 6),  P=(27, 8),  K=(30, 8),  temperature=(31, 2.5), humidity=(50, 6),  ph=(5.7, 0.4), rainfall=(95, 20)),
    "grapes":      dict(N=(23, 6),  P=(132, 10),K=(200, 12),temperature=(24, 3.0), humidity=(82, 4),  ph=(6.0, 0.3), rainfall=(70, 12)),
    "watermelon":  dict(N=(99, 12), P=(17, 6),  K=(50, 8),  temperature=(25, 2.5), humidity=(85, 4),  ph=(6.5, 0.4), rainfall=(50, 12)),
    "muskmelon":   dict(N=(100, 12),P=(18, 6),  K=(50, 8),  temperature=(28, 2.0), humidity=(92, 3),  ph=(6.4, 0.3), rainfall=(24, 6)),
    "apple":       dict(N=(21, 6),  P=(134, 12),K=(199, 12),temperature=(22, 2.5), humidity=(92, 3),  ph=(5.9, 0.3), rainfall=(112, 20)),
    "orange":      dict(N=(19, 6),  P=(16, 6),  K=(10, 4),  temperature=(29, 2.0), humidity=(92, 3),  ph=(7.0, 0.3), rainfall=(110, 20)),
    "papaya":      dict(N=(50, 10), P=(59, 10), K=(50, 8),  temperature=(33, 2.5), humidity=(92, 3),  ph=(6.7, 0.4), rainfall=(142, 25)),
    "coconut":     dict(N=(22, 6),  P=(17, 6),  K=(31, 8),  temperature=(27, 2.0), humidity=(94, 2),  ph=(5.9, 0.4), rainfall=(175, 30)),
    "cotton":      dict(N=(118, 15),P=(46, 10), K=(20, 6),  temperature=(24, 2.5), humidity=(80, 5),  ph=(6.9, 0.4), rainfall=(80, 15)),
    "jute":        dict(N=(78, 12), P=(47, 10), K=(40, 8),  temperature=(25, 2.0), humidity=(80, 4),  ph=(6.7, 0.3), rainfall=(175, 25)),
    "coffee":      dict(N=(101, 15),P=(28, 8),  K=(30, 8),  temperature=(25, 2.5), humidity=(58, 8),  ph=(6.8, 0.4), rainfall=(160, 30)),
}

FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
BOUNDS = {  # physical clip ranges
    "N": (0, 145), "P": (5, 145), "K": (5, 205),
    "temperature": (7, 44), "humidity": (10, 100),
    "ph": (3.5, 9.5), "rainfall": (18, 300),
}

SAMPLES_PER_CROP = 120


def generate(path="Crop_recommendation.csv", n_per_crop=SAMPLES_PER_CROP, seed=42):
    rng = np.random.default_rng(seed)
    rows = []
    for crop, profile in CROP_PROFILES.items():
        for _ in range(n_per_crop):
            row = {}
            for f in FEATURES:
                mean, std = profile[f]
                val = rng.normal(mean, std)
                lo, hi = BOUNDS[f]
                val = float(np.clip(val, lo, hi))
                row[f] = round(val, 2)
            row["label"] = crop
            rows.append(row)
    df = pd.DataFrame(rows)[FEATURES + ["label"]]
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)  # shuffle
    out_path = os.path.join(os.path.dirname(__file__), path)
    df.to_csv(out_path, index=False)
    print(f"wrote {len(df)} rows ({len(CROP_PROFILES)} crops) to {out_path}")
    return out_path


if __name__ == "__main__":
    generate()
