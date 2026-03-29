# 🔭 Classification of Gammas and Hadrons using the MAGIC Gamma Telescope

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0%2B-orange?logo=scikit-learn)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)

---

## 📌 Project Overview

This project applies **Logistic Regression** to classify high-energy particle events recorded by the **MAGIC (Major Atmospheric Gamma Imaging Cherenkov) Telescope** into two categories:

- **Gamma (signal)** — high-energy gamma rays from astrophysical sources
- **Hadron (background)** — cosmic ray background noise

The MAGIC Telescope, located on the Canary Island of La Palma, Spain, detects Cherenkov radiation emitted when high-energy particles enter Earth's atmosphere. Accurately distinguishing gamma rays from hadron background is crucial for gamma-ray astronomy.

---

## 📂 Dataset

- **Source:** [UCI Machine Learning Repository – MAGIC Gamma Telescope](https://archive.ics.uci.edu/ml/datasets/MAGIC+Gamma+Telescope)
- **Dataset ID:** 159
- **Instances:** 19,020
- **Features:** 10 continuous features
- **Target:** Binary — `g` (gamma) or `h` (hadron)

### Features

| Feature | Description |
|---|---|
| `fLength` | Major axis of ellipse (mm) |
| `fWidth` | Minor axis of ellipse (mm) |
| `fSize` | 10-log of sum of content of all pixels (photons) |
| `fConc` | Ratio of sum of two highest pixels over fSize |
| `fConc1` | Ratio of highest pixel over fSize |
| `fAsym` | Distance from highest pixel to center (mm) |
| `fM3Long` | 3rd root of third moment along major axis (mm) |
| `fM3Trans` | 3rd root of third moment along minor axis (mm) |
| `fAlpha` | Angle of major axis with vector to origin (deg) |
| `fDist` | Distance from origin to center of ellipse (mm) |

### Class Distribution

| Class | Label | Count |
|---|---|---|
| Gamma (signal) | g | 12,332 |
| Hadron (background) | h | 6,688 |

---

## 🛠️ Tech Stack

- **Language:** Python 3.8+
- **Libraries:**
  - `scikit-learn` — Model training & evaluation
  - `pandas` — Data manipulation
  - `numpy` — Numerical operations
  - `matplotlib` & `seaborn` — Visualization
  - `ucimlrepo` — Dataset loading

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/arun25bai10385-sys/Classification-of-gammas-and-hadrons-using-the-MAGIC-Gamma-Telescope-.git
cd Classification-of-gammas-and-hadrons-using-the-MAGIC-Gamma-Telescope-
```

### 2. Install Dependencies

```bash
pip install pandas numpy scikit-learn matplotlib seaborn ucimlrepo
```

### 3. Run the Script

```bash
python logistic_regression.py
```

---

## 🧠 Model Pipeline

```
Load Dataset → Encode Target → Train/Test Split (80/20)
      → Feature Scaling (StandardScaler)
          → Logistic Regression
              → Evaluation (Accuracy, ROC-AUC, Confusion Matrix)
```

### Steps:

1. **Load Data** — Fetched directly from UCI ML Repository using `ucimlrepo`
2. **Encode Target** — `g` (gamma) → `1`, `h` (hadron) → `0`
3. **Split Data** — 80% training, 20% testing with stratification
4. **Scale Features** — StandardScaler for normalization
5. **Train Model** — Logistic Regression with `max_iter=1000`
6. **Evaluate** — Accuracy, ROC-AUC, Classification Report, Confusion Matrix

---

## 📊 Results

| Metric | Score |
|---|---|
| Accuracy | ~80% |
| ROC-AUC | ~0.87 |

### Outputs:
- ✅ Confusion Matrix
- ✅ ROC Curve
- ✅ Feature Coefficient Plot

---

## 📁 Project Structure

```
📦 MAGIC-Gamma-Telescope-Classification
 ┣ 📜 logistic_regression.py   # Main model script
 ┣ 📜 README.md                # Project documentation
 ┗ 📜 requirements.txt         #dependencies to run the model
```

---

## 📋 Requirements

Create a `requirements.txt` with:

```
pandas
numpy
scikit-learn
matplotlib
seaborn
ucimlrepo
```

Install with:
```bash
pip install -r requirements.txt
```

---

## 👥 Team Members

| Name | GitHub |
|---|---|
| Arun Rajeev Nethran| [@arun25bai10385-sys](https://github.com/arun25bai10385-sys) |
---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgements

- [UCI Machine Learning Repository](https://archive.ics.uci.edu/) for the dataset
- [MAGIC Telescope Collaboration](https://magic.mpp.mpg.de/) for the original data collection
