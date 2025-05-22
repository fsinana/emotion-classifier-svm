# Emotion Classifier using SVM

## Overview
This project implements an emotion classification model that predicts the emotion of a given text using a Support Vector Machine (SVM) classifier with TF-IDF vectorization. The model is trained on a dataset labeled with emotions such as joy, sadness, anger, fear, love, and surprise.

## Features
- Text preprocessing: cleaning, removing punctuation, stopwords, URLs, and digits.
- TF-IDF vectorization of text data.
- Emotion classification using LinearSVC (Support Vector Machine).
- Evaluation with accuracy score, classification report, and confusion matrix visualization.

## Usage
Ensure your dataset CSV file (e.g., `emotions.csv`) is in the project directory or update the file path in `main.py`.

Run the main script:
```bash
python main.py
Results
Example accuracy: ~79.7%
