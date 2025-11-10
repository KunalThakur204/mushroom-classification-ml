🍄✨ Mushroom Classification Using Machine Learning
An Animated-Style ML Prototype that Predicts What’s Poisonous… Before Nature Does.
<p align="center"> <img src="https://img.shields.io/badge/Machine%20Learning-Classification-green?style=for-the-badge" /> <img src="https://img.shields.io/badge/Model-RandomForest%20%7C%20SVM-blue?style=for-the-badge" /> <img src="https://img.shields.io/badge/Dataset-UCI%20Mushroom-orange?style=for-the-badge" /> <img src="https://img.shields.io/badge/Status-Working%20Prototype-purple?style=for-the-badge" /> </p> <p align="center"> <img src="https://media.tenor.com/yYV3Qe9IuWwAAAAi/mushroom-dance.gif" width="200"> </p>

This project transforms a biological safety problem into a clean, modern AI system —
a system that quietly scans mushrooms and whispers:
“Yeh khane layak hai ya khatra?” 🍄⚠️

Designed with smooth ML flow, animated thinking, and safety as its backbone.

🌱 1. About the Project

Based on UCI’s 8,124-sample Mushroom Dataset, this ML model classifies mushrooms as:

✅ Edible (E)
❌ Poisonous (P)

Traditional identification depends on experts and countless risks.
Yeh model un sabko replace nahi karta, par unka kaam aasaan zaroor banata hai.

🎯 2. Objectives

Analyze mushroom dataset & discover patterns

Encode 22 categorical features

Train multiple models (RF, SVM, DT, LR, NB, KNN)

Compare accuracy, ROC-AUC and confusion matrix

Identify the most dangerous attributes like odor, gill-size, spore print color

Build a fully interpretable ML pipeline

🛠️ 3. Requirements
Software

Windows 10/11 or Ubuntu

Python 3.10+

Jupyter / VS Code / Google Colab

Libraries

Pandas

NumPy

Scikit-learn

Matplotlib

Seaborn

Dataset

UCI Mushroom Dataset
22 categorical attributes + 1 target class (edible/poisonous)

🧪 4. Implementation (Animated ML Workflow)
[Data Loading] ---> [Label Encoding] ---> [Train-Test Split] 
         ↓                   ↓                    ↓
   DataFrame Peek      Categorical Fix     80% Train | 20% Test
         ↓                   ↓                    ↓
 ---------------------------------------------------------------
                [Model Training: RF, SVM, DT, KNN]
 ---------------------------------------------------------------
         ↓
     [Accuracy + Confusion Matrix + ROC-AUC]
         ↓
  [Feature Importance Visualization]

✅ Code Snippets (Short & Clean)
Step 1 — Load Dataset
data = pd.read_csv("mushrooms.csv")

Step 2 — Encode Features
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
for col in data.columns:
    data[col] = le.fit_transform(data[col])

Step 3 — Prepare Data
X = data.drop('class', axis=1)
y = data['class']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

Step 4 — Train Random Forest
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

Step 5 — Evaluate
y_pred = model.predict(X_test)
accuracy_score(y_test, y_pred)
roc_auc_score(y_test, y_pred)

Step 6 — Feature Importance
sns.barplot(x=model.feature_importances_, y=X.columns)

📊 5. Output Summary
✅ Accuracy: 92%
✅ Best Models: Random Forest & SVM
✅ Top Features:

Odor

Gill-Size

Spore-Print-Color

Cap-Surface

✅ Confusion Matrix:

Minimal confusion between edible and poisonous groups.

✅ ROC-AUC:

High — strong classification capability.

<p align="center"> <img src="https://media.tenor.com/ZT6f-E8-l2UAAAAC/mushroom.gif" width="200"> </p>
🌟 6. Learning Outcomes

Hands-on experience with ML classification algorithms

Encoding categorical biological attributes

Understanding confusion matrix & ROC-AUC

Discovering which mushroom traits affect toxicity

Complete end-to-end ML workflow understanding

Realizing ML’s potential in biological & ecological safety

🔮 7. Future Enhancements

Deep learning classification (CNN on mushroom images)

Mobile app for real-time identification

Explainable AI (SHAP Values)

Advanced feature selection

Deployment via Flask / FastAPI

🙏 Credits

Developed with care, curiosity, and responsibility —
because biology me galti ki gunjāish nahi hoti.
