# Alzheimer's Disease Prediction from MRI Data (OASIS)

Machine learning pipeline for automated Alzheimer’s disease classification using clinical and MRI-derived biomarkers.

Developed for the **Bioinformatics course (2025–2026)**.

---

## Project Overview

The goal of this project is to classify patients into **Demented** and **Non-demented** groups using clinical indicators and MRI-derived measurements.

The repository demonstrates a complete biomedical data science workflow:

* automated dataset acquisition
* data preprocessing and cleaning
* model training and comparison
* feature importance analysis
* visualization of biomarker contributions

This project is intended for **educational and research purposes only**.

---

## Dataset

**OASIS MRI Dataset (Open Access Series of Imaging Studies)**
Source: Kaggle — `jboysen/mri-and-alzheimers`

Key features used in the model:

* **MMSE** — Mini-Mental State Examination score
* **eTIV** — Estimated Total Intracranial Volume
* **nWBV** — Normalized Whole Brain Volume
* **ASF** — Atlas Scaling Factor
* **Diagnosis** — Target variable derived from the CDR scale

---

## Tech Stack

**Language**

* Python 3.9+

**Libraries**

* pandas
* numpy
* scikit-learn
* matplotlib
* seaborn
* opendatasets

---

## Project Structure

```
.
├── dataset.py            # Download dataset from Kaggle
├── preprocess.py         # Data cleaning and normalization
├── model_comparison.py   # Model training and evaluation
├── analysis.py           # Feature importance analysis
└── README.md
```

---

## Workflow

Run scripts in the following order:

```bash
python dataset.py
python preprocess.py
python model_comparison.py
python analysis.py
```

---

## Models Compared

The following models were evaluated:

* Logistic Regression
* Random Forest
* Support Vector Machine (SVM)
* Gradient Boosting

---

## Evaluation Metrics

Model performance is evaluated using:

* **AUC-ROC (primary metric)**
* Accuracy
* F1-Score

Special attention is given to reducing **false negatives**, which is critical in medical diagnostics.

---

## Academic Information

Course: **Bioinformatics (2025–2026)**
Defense date: **February 16, 2026**

Requirements satisfied:

* Real clinical dataset (OASIS)
* Reproducible ML pipeline
* Feature importance interpretation

**AI Disclosure:**
AI tools were used to assist in structuring the data processing pipeline and improving code organization.

---

## License

This project is intended for academic use only.

---

© 2026 | Bioinformatics Project Team
