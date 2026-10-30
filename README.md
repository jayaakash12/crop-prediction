# Field Ledger — Crop Recommendation (dashboard + trained model + API)

A working, network-connected project: `dashboard.html` (the UI you uploaded)
talks over HTTP to a Flask backend that serves a real, trained scikit-learn
model. Nothing in the dashboard changed — it already expected exactly this
API shape at `http://localhost:8000`.

```
crop-recommendation/
├── dashboard.html          the frontend (open in a browser)
├── app.py                  Flask API — connects the dashboard to the model
├── train_model.py          trains & compares 6 classifiers, saves the best
├── requirements.txt
├── data/
│   └── Crop_recommendation.csv  real 2,200-row crop recommendation dataset
│   └── generate_dataset.py      optional synthetic-data generator (not used by training)
└── models/                 model.joblib, metrics.json, etc. (created by training)
```

## 1. Install

```bash
cd crop-recommendation
pip install -r requirements.txt
```

## 2. Train the model (one-time, ~5 seconds)

```bash
python train_model.py
```

The project now uses the real `data/Crop_recommendation.csv` dataset
(2,200 labeled records across 22 crops and the 7 input features N, P, K,
temperature, humidity, ph, rainfall). The trainer validates the required
columns, trains RandomForest, GradientBoosting, DecisionTree, KNN,
LogisticRegression and SVM, evaluates each with a held-out split and 5-fold
cross-validation, and saves the best-performing model (by macro F1) into
`models/`.

The training script will **not** silently generate synthetic data if the CSV
is missing. This prevents accidentally deploying a model trained on the
illustrative synthetic dataset.

## 3. Run the backend

```bash
python app.py
```

Serves on `http://localhost:8000`. You'll see:
```
Field Ledger backend starting on http://localhost:8000  (model_loaded=True)
```

## 4. Open the dashboard

Open `dashboard.html` directly in a browser (double-click it, or `python -m
http.server` and visit it). The sidebar already points at
`http://localhost:8000` — click **Reconnect** if the status dot shows
OFFLINE before the backend finishes starting.

All seven tabs are live:
- **Predict** — single best-fit crop + confidence + runner-ups
- **Rank Top‑N** — full ranked list for N candidates
- **Crop Directory** — the 22 crops the model knows
- **Prediction Log** — every prediction, persisted to SQLite (`predictions.db`)
- **Model Bench** — accuracy/precision/recall/F1 for all 6 candidate models
- **Feature Weight** — which soil/climate features drive the model most
- **Active Model** — metadata on the deployed model

## Notes

- History logging uses SQLite and is on by default. Disable with
  `ENABLE_DB=false python app.py`.
- CORS is enabled for all origins so the dashboard can be opened as a local
  file (`file://`) or served separately from the API.
- To retrain after editing `data/generate_dataset.py`'s agronomic ranges,
  just delete `data/Crop_recommendation.csv` and rerun `train_model.py`.


## 5. Current trained model

The included model artifacts were retrained from the real `data/Crop_recommendation.csv` dataset. The dataset contains 2,200 records across 22 crop classes. The selected deployed model is RandomForest.

Training results from the held-out test split:
- Accuracy: 0.9932
- Weighted precision: 0.9935
- Weighted recall: 0.9932
- Weighted F1: 0.9932
- Macro F1: 0.9932
- 5-fold CV macro F1: 0.9964

These metrics describe performance on this dataset and should not be interpreted as a guarantee of real-world agronomic accuracy.
