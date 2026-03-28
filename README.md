# OncoSense v3 — Quantum ML Platform for Early Disease Detection

**Team MindMatrix | HackHustle 2.0 | Healthcare Domain**
**SDG 3: Good Health & Well-being | SDG 10: Reduced Inequalities | SDG 9: Innovation**

## What's New in v3

- **One-click image model training** — no Kaggle, no terminal, no dataset downloads
- **Synthetic histopathology generator** — creates training images inside the app
- **CPU-only PyTorch** — deploys on Railway without hitting size limits
- **Dual diagnosis modes** — Tabular (30 biopsy features) + Image (histopathology slides)

## Quick Start

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Architecture

### Hybrid CNN → Quantum Pipeline (Novel)
```
Histopathology Image → ResNet18 (512 features) → PCA (4 features)
→ ZZFeatureMap (4 qubits) → Fidelity Quantum Kernel → SVM → Diagnosis
```

### Tabular Pipeline
```
30 Biopsy Features → MinMaxScaler → PCA (4 features) → Quantum Kernel → SVM
```

## Project Structure

```
oncosense_v3/
├── app.py                              # Streamlit app (5 pages)
├── requirements.txt                    # Dependencies (CPU-only PyTorch)
├── railway.json                        # Railway deployment config
├── .python-version                     # Python 3.12
├── models/                             # Trained model artifacts (auto-generated)
└── utils/
    ├── preprocessing.py                # Tabular data pipeline
    ├── quantum_engine.py               # Quantum Kernel SVM
    ├── classical_engine.py             # Classical SVM baselines
    ├── image_feature_extractor.py      # ResNet18 CNN feature extractor
    ├── hybrid_quantum_pipeline.py      # Hybrid CNN→Quantum pipeline
    ├── synthetic_generator.py          # Synthetic histopathology image generator
    └── report_generator.py             # PDF diagnosis reports
```

## Deploy to Railway

1. Push to GitHub
2. railway.com → New Project → Deploy from GitHub
3. Generate public domain in Settings → Networking

*OncoSense — Where Quantum Meets Care*
