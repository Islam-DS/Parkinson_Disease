# Parkinson Disease Detection

This project uses classical machine learning to detect Parkinson’s disease from voice measurements using the UCI Parkinsons dataset (195 samples, 22 features + status).

## Dataset

- File: `parkinsons.csv`
- Target column: `status` (1 = Parkinson’s, 0 = healthy)
- Features: biomedical voice measures such as fundamental frequency, jitter, shimmer, and nonlinear measures.

## Methods

1. Load and explore the dataset in `PARKINSON_DISEASE.ipynb`.
2. Drop the non-numeric `name` column and split into features `X` and target `y`.
3. Perform a stratified train–test split (80% train, 20% test) with `random_state=42`.
4. Standardize features using `StandardScaler`.
5. Train a linear Support Vector Machine (SVM) classifier.
6. Evaluate with accuracy, precision, recall, F1-score, and a confusion matrix.

## Results

On the held-out test set, the linear SVM achieves:

- Accuracy: **0.949**
- Precision: **0.935**
- Recall: **1.000**
- F1-score: **0.967**

The confusion matrix shows 0 Parkinson cases missed and 2 healthy cases predicted as Parkinson.

## How to run
git clone https://github.com/Islam-DS/Parkinson_Disease.git
cd Parkinson_Disease
pip install -r requirements.txt
jupyter notebook
