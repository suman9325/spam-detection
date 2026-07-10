# SMS Spam Detection Pipeline

An end-to-end Machine Learning pipeline implemented in Python to accurately classify SMS messages into **Ham** (legitimate) or **Spam**. 

This project explores text-preprocessing, vectorization (Bag-of-Words), and compares the performance of classical probabilistic and linear classifiers.

---

## 📌 Project Overview
Text-based spam classification is a vital task in Natural Language Processing (NLP). This repository contains a structured, reproducible implementation that trains, evaluates, and compares two primary algorithms:
1. **Multinomial Naive Bayes (MNB)** — A classic benchmark for text categorization tasks based on word frequencies.
2. **Logistic Regression** — A robust linear classification model used to establish reliable prediction probabilities.

---

## 🛠️ Tech Stack & Dependencies
The pipeline relies on standard Python data science and machine learning libraries:
* **Core:** Python 3.x
* **Data Manipulation:** `pandas`, `numpy`
* **Machine Learning & NLP:** `scikit-learn`

To install the necessary packages directly from your terminal, run:
```bash```
pip install pandas numpy scikit-learn

*(If running inside a Jupyter Notebook cell, use `%pip install pandas numpy scikit-learn` instead.)*

---

## 🗂️ Dataset Details
The model is trained on the public **SMS Spam Collection Dataset**, which contains over 5,500 real SMS messages.

* **Target Feature:** `label` (`ham` or `spam`)
* **Input Feature:** `message` (raw textual data)

### Preprocessing Pipeline:
* **Deduplication:** Dropped identical message rows to eliminate validation and training bias.
* **Label Encoding:** Mapped original categorical tags to numerical structures: `ham` $\rightarrow$ `0`, `spam` $\rightarrow$ `1`.
* **Data Splitting:** Maintained an **80/20** split for training and testing data respectively using a locked `random_state=42` to ensure reproducibility.
* **Vectorization:** Applied `CountVectorizer()` to extract a structural token vocabulary matrix based on frequency counts.

---

## 📊 Model Performance Results
Both classifiers achieve exceptional results, but exhibit slight differences in precision and recall characteristics on the holdout test set:

| Model Architecture | Test Accuracy | Macro Precision | Macro Recall | F1-Score (Spam) |
| :--- | :---: | :---: | :---: | :---: |
| **Multinomial Naive Bayes** | **98.45%** | 0.98 | 0.95 | 0.94 |
| **Logistic Regression** | **98.16%** | 0.98 | 0.93 | 0.93 |

### Confusion Matrix Insights:
* **Naive Bayes** demonstrates higher sensitivity (Recall) to catching malicious spam vectors.
* **Logistic Regression** offers slightly lower false-positive rates, ensuring important clean messages aren't misclassified into the spam filter.

---

## 🚀 Usage Guide

### 1. Training and Evaluation
Open and run all cells in the Jupyter Notebook `SpamDetection.ipynb` or run the standalone script execution:
```bash```
jupyter notebook SpamDetection.ipynb

2. Custom Testing (Inference)
You can seamlessly pass fresh string arrays to test the model's live logic block inside the notebook:

# Custom Inference Sandbox
sample_email = ["Meeting tomorrow at 11 am."]
vectorized_input = vectorizer.transform(sample_email)

prediction = model.predict(vectorized_input)
probabilities = model.predict_proba(vectorized_input)

print(f"Classification Label: {'Spam' if prediction[0] == 1 else 'Ham'}")
print(f"Spam Probability: {probabilities[0][1]:.4f}")
