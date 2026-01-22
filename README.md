# Machine Learning Foundations Portfolio

A collection of technical implementations demonstrating core Machine Learning competencies, progressing from classical Supervised/Unsupervised learning to NLP and Deep Learning.

## 🛠 Tech Stack
* **Language:** Python 3.x
* **Libraries:** TensorFlow (Keras), Scikit-Learn, Pandas, NumPy, Matplotlib, Seaborn, XGBoost, NLTK/Regex.
* **Environment:** Google Colab / Jupyter Notebooks.

---

## 📂 Project Index

### 1. Titanic Survival Prediction (Binary Classification)
* **Goal:** Predict passenger survival probability based on demographics and ticket class.
* **Tech:** Logistic Regression, Data Imputation, One-Hot Encoding.
* **Key Insight:** Feature engineering on `Title` and `FamilySize` significantly improved model accuracy over raw data.

### 2. Ames Housing Price Prediction (Regression)
* **Goal:** Predict final residential home sale prices based on 79 explanatory variables.
* **Tech:** Linear Regression, Correlation Matrix, Multicollinearity Analysis.
* **Key Insight:** `OverallQual` and `GrLivArea` were the strongest predictors. Handled heteroscedasticity in high-value properties.

### 3. Customer Segmentation (Clustering)
* **Goal:** Segment mall customers into distinct target groups to inform marketing strategy.
* **Tech:** K-Means Clustering, Elbow Method, PCA (for visualization).
* **Key Insight:** Identified 5 distinct customer profiles, including "High Income/Low Spenders" (Savers) and "High Income/High Spenders" (VIPs).

### 4. Telco Customer Churn (Ensemble Methods)
* **Goal:** Predict customer churn to improve retention strategies.
* **Tech:** Random Forest, XGBoost, SMOTE (Oversampling), GridSearchCV (Hyperparameter Tuning).
* **Key Insight:** Optimized Recall by 12% using SMOTE. Identified `Tenure` and `Electronic Check` payments as primary churn drivers.

### 5. Movie Review Sentiment Analysis (NLP)
* **Goal:** Classify movie reviews as Positive or Negative from raw text.
* **Tech:** TF-IDF Vectorization, Regex (Text Cleaning), Linear SVC.
* **Key Insight:** Achieved **87.8% Accuracy**. Demonstrated the critical importance of proper tokenization (fixing "mashed" text) for vectorizer performance.

### 6. Handwritten Digit Recognition (Computer Vision)
* **Goal:** Correctly identify handwritten digits (0-9) from the MNIST dataset.
* **Tech:** Deep Learning, TensorFlow/Keras, Sequential Neural Network.
* **Key Insight:** Built a fully connected network (Flatten -> Dense 128 -> Softmax) achieving **98.6% Test Accuracy**.

---

## 🚀 How to Run
These projects are designed to run in a Jupyter environment.
1.  Clone the repository.
2.  Install dependencies: `pip install pandas numpy seaborn scikit-learn matplotlib tensorflow xgboost`
3.  Open any `.ipynb` file in Jupyter Lab or Google Colab.
