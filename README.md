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
```bash
pip install pandas numpy scikit-learn
