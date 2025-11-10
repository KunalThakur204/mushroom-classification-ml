🌱 Mushroom Classification using Machine Learning 🍄

A supervised machine learning project designed to classify mushrooms as Edible or Poisonous using categorical biological features from the UCI Mushroom Dataset.

📌 Project Summary

This project applies multiple machine learning algorithms to predict mushroom toxicity based on 22 categorical attributes such as odor, cap color, gill-size, and spore-print color.

The best-performing models achieved an accuracy of 92%, making the system reliable for biological risk assessment and real-world mushroom identification support.

🚀 Features

✅ Uses 6 supervised ML algorithms

✅ Achieved 92% accuracy (Random Forest, SVM, KNN)

✅ Complete EDA with visualizations

✅ Full preprocessing for categorical data

✅ Evaluation using Accuracy, Confusion Matrix, ROC–AUC

✅ Feature importance analysis

🧰 Tech Stack

Python 3.10

Pandas

NumPy

Scikit-learn

Matplotlib

Seaborn

Jupyter / Google Colab

📂 Dataset

Source: UCI Machine Learning Repository

Instances: 8,124

Features: 22 categorical attributes

Target:

e → Edible

p → Poisonous

🔧 Project Workflow
1️⃣ Data Loading & Inspection

Load dataset into pandas DataFrame

Check structure, shape, and basic info

Verify missing values (none found)

2️⃣ Exploratory Data Analysis (EDA)

Distribution of edible vs poisonous mushrooms

Bar plots, count plots, heatmaps

Odor found to be the strongest indicator of toxicity

Study relationships among features

3️⃣ Data Preprocessing

Apply Label Encoding to all categorical columns

Split dataset:

80% training

20% testing

4️⃣ Model Training

Six supervised models were trained:

🌳 Random Forest

🧮 SVM (RBF Kernel)

🌿 Decision Tree

📈 Logistic Regression

📊 Naive Bayes

👥 KNN

5️⃣ Model Evaluation

Metrics used:

✅ Accuracy

✅ Confusion Matrix

✅ ROC–AUC Score

Best Models:

Random Forest – 92%

SVM – 92%

KNN – 92%

6️⃣ Feature Importance

From Random Forest:

⭐ Odor

⭐ Gill-size

⭐ Spore-print color

⭐ Cap-surface

These features heavily influence the classification decision.

📊 Results Summary
Model	Accuracy
Random Forest	⭐ 92%
SVM	⭐ 92%
KNN	⭐ 92%
Decision Tree	90%
Logistic Regression	90%
Naive Bayes	90%
🌟 Future Improvements

Implement Image-based detection (CNNs)

Develop a mobile app for real-time prediction

Use SHAP / LIME for deeper interpretability

Include seasonal & geographic features
