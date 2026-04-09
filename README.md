# 🧪 AI-Powered Toxicity Pattern Detection System

A Machine Learning-based decision support system for early detection and classification of toxic exposure patterns using clinical symptoms, exposure indicators, and severity signals.

The system classifies cases into four categories:

- Normal  
- Ricin-like  
- Polonium-like  
- Borderline  

The system is designed to assist in identifying complex and uncertain toxic patterns, especially in early-stage screening scenarios.

---

## 📌 Project Overview

This project was developed as an intelligent toxicology screening system that analyzes clinical symptoms and exposure-related features.

Rather than identifying the exact toxin chemically, the system works as a **pattern-based diagnostic support tool**. It helps identify patterns that resemble:

- **Ricin-like toxicity**: acute gastrointestinal and ingestion-related toxic pattern  
- **Polonium-like toxicity**: radiation-like systemic damage pattern  
- **Borderline**: mixed or uncertain toxicological profile  
- **Normal**: non-severe or non-toxic pattern  

The model is designed to assist in **early screening**, especially in situations where direct laboratory confirmation may not be immediately available.

**Note:** The system is domain-independent and can be applied to human, veterinary, and environmental toxicology scenarios.

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

## 📊 Model Performance

The model was evaluated using a held-out test set and cross-validation.

### Test Results:
- Accuracy: 91.8%  
- Macro Avg F1-score: 0.91  
- Weighted Avg F1-score: 0.92  

### Class-wise Performance:

| Class          | Precision | Recall | F1-score |
|---------------|----------|--------|---------|
| Borderline     | 0.79     | 0.78   | 0.78    |
| Normal         | 0.94     | 0.94   | 0.94    |
| Polonium-like  | 0.91     | 0.91   | 0.91    |
| Ricin-like     | 0.98     | 1.00   | 0.99    |

### Cross Validation:
- Mean Accuracy: 0.93  

### Notes:
- Strong performance on clearly distinguishable patterns (Ricin-like, Normal)  
- Borderline cases remain the most challenging due to overlapping patterns  
- Calibration improved confidence reliability without sacrificing accuracy  

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
- Radiation exposure pattern  
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
- **Ricin-like** = resembles a ricin-like toxic pattern  
- **Polonium-like** = resembles a radiation-like toxic pattern  

The system should be used as an **early screening and decision support tool**, not as a definitive clinical or laboratory diagnosis.

---

## 🧪 Explainability and Alerts

The application provides:
- Prediction reasoning  
- Confidence analysis  
- Risk alerts for mixed or conflicting patterns  
- Review recommendation for borderline or uncertain cases  

This enhances transparency and supports real-world decision-making.

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
