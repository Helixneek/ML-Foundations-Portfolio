# Machine Learning Foundations Portfolio

A collection of technical implementations demonstrating basic Machine Learning competencies in **Supervised (Classification & Regression)** and **Unsupervised Learning**.

These projects focus on the end-to-end data science lifecycle: Exploratory Data Analysis (EDA), rigorous Data Preprocessing, Feature Engineering, Model Training, and Evaluation.

## 🛠 Tech Stack
* **Language:** Python 3.x
* **Environment:** Google Colab / Jupyter Notebooks
* **Libraries:** Pandas, NumPy, Scikit-Learn, Matplotlib, Seaborn

---

## 📂 Project 1: Titanic Survival Prediction (Binary Classification)
**Objective:** Predict passenger survival probability based on demographics and ticket class features.

* **Key Methodologies:**
    * **Data Imputation:** Handled missing `Age` values using median imputation and `Embarked` values using mode, based on distribution analysis.
    * **Feature Engineering:** Created binary and one-hot encoded variables for categorical features (`Sex`, `Embarked`).
    * **Model:** Logistic Regression.
* **Evaluation:**
    * Achieved **~80% Accuracy** on unseen test data.
    * Analyzed performance using a **Confusion Matrix** to visualize the trade-off between False Positives and False Negatives.

## 📂 Project 2: Ames Housing Price Prediction (Regression)
**Objective:** Predict final residential home sale prices based on 79 explanatory variables.

* **Key Methodologies:**
    * **Feature Selection:** Utilized a **Correlation Matrix** to identify the top 7 highly correlated features (e.g., `OverallQual`, `GrLivArea`) to prevent overfitting.
    * **Multicollinearity Handling:** Analyzed and pruned redundant features (e.g., `GarageCars` vs. `GarageArea`).
    * **Model:** Linear Regression (OLS).
* **Evaluation:**
    * **R² Score:** ~0.80 (Explaining 80% of price variance).
    * **RMSE:** ~$39,600.
    * **Visualization:** Plotted "Actual vs. Predicted" values to detect heteroscedasticity in high-value properties.

## 📂 Project 3: Customer Segmentation (Unsupervised Clustering)
**Objective:** Segment mall customers into distinct target groups based on income and spending habits to inform marketing strategy.

* **Key Methodologies:**
    * **Algorithm:** K-Means Clustering.
    * **Optimal K Selection:** Implemented the **Elbow Method** to scientifically determine the optimal number of clusters (K=5) by minimizing WCSS (Within-Cluster Sum of Squares).
* **Results:**
    * Identified 5 distinct customer profiles, including "High Income/Low Spenders" (Savers) and "High Income/High Spenders" (VIPs).
    * Visualized clusters in 2D space with clearly demarcated centroids.

---

## 🚀 How to Run
These projects are designed to run in a Jupyter environment.
1.  Clone the repository.
2.  Install dependencies: `pip install pandas numpy seaborn scikit-learn matplotlib`
3.  Open the `.ipynb` files in Jupyter Lab or Google Colab.
