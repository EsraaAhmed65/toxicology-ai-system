# 🧪 Toxicology AI Detection System

An AI-based decision support system for early detection and classification of toxic exposure patterns in livestock.

The system classifies cases into four categories:

- Normal
- Ricin-like
- Polonium-like
- Borderline

It uses clinical symptoms, exposure indicators, and severity scores to support toxicological screening and highlight uncertain or mixed patterns.

---

## 📌 Project Overview

This project was developed as an intelligent toxicology screening system that analyzes livestock clinical and exposure-related features to predict toxin-like patterns.

Rather than identifying the exact toxin chemically, the system works as a **pattern-based diagnostic support tool**. It helps detect cases that resemble:

- **Ricin-like toxicity**: acute gastrointestinal and feed-related toxic pattern
- **Polonium-like toxicity**: radiation-like systemic damage pattern
- **Borderline**: mixed or uncertain toxicological profile
- **Normal**: non-severe or non-toxic pattern

The model is designed to assist in **early screening**, especially in situations where direct laboratory confirmation may not be immediately available.

---

## 🚀 Main Features

- Multi-class toxicology case classification
- Confidence estimation using a calibrated model
- Borderline and uncertainty-aware prediction logic
- Review flag for mixed or ambiguous cases
- Explainable AI output with reasoning and risk alerts
- Interactive Streamlit web application
- Example cases for quick testing

---

## 🧠 Model Information

The core model is a **Calibrated Random Forest Classifier** trained on a synthetic toxicology dataset inspired by scientific literature and toxicological patterns.

### Model capabilities:
- Pattern recognition using clinical and exposure features
- Confidence scoring
- Second possible class identification
- Top-2 margin analysis
- Review recommendation for uncertain cases

### Output classes:
- **Normal**
- **Ricin-like**
- **Polonium-like**
- **Borderline**

---

## 📊 Input Features

The model uses the following features:

### Clinical and exposure features:
- Fever
- Vomiting
- Diarrhea
- Anorexia
- Dehydration
- Sudden onset
- Progressive deterioration
- Feed exposure
- Radiation pattern
- Herd cluster

### Severity scores:
- Weakness score
- Cell damage score
- GI damage score
- Multi-organ damage
- Marrow suppression
- Renal injury score
- Hepatic injury score

### Timing:
- Time to onset (hours)

---

## ⚠️ Important Note

This system does **not** perform laboratory or molecular toxin identification.

Instead, it predicts **toxin-like clinical patterns** based on symptom combinations and damage indicators.

That means:
- **Ricin-like** = resembles a ricin toxic pattern
- **Polonium-like** = resembles a radiation-like toxic pattern

So the system should be used as an **early screening and decision support tool**, not as a definitive laboratory diagnosis.

---

## 🧪 Explainability and Alerts

The application provides:
- Why this prediction?
- What may reduce certainty?
- Risk alerts for mixed or conflicting patterns
- Review recommendation for borderline or uncertain cases

This makes the system more practical for real-world use and improves transparency of the AI decision.

---

## 💻 Technologies Used

- Python
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- Joblib
- Matplotlib
- Seaborn

---

## 📁 Project Structure

```bash
.
├── app.py
├── model.ipynb
├── calibrated_random_forest_model.pkl
├── feature_columns.pkl
├── toxicology_dataset_strong_v5.csv
├── requirements.txt
└── README.md
