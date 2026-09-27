# Viva & Evaluation Defense Guide: Scholarship Eligibility Prediction

**Course:** AD4V71 / AD5302 – Data Science  
**Project Team:** Kirutick Siddhesh V (`210425243120`), Krishna H (`210425243124`)  
**Project Artifact:** Academic Machine Learning Classification Benchmark

---

### Q1: Why did you choose classification?
**Answer:** We chose classification because our problem requires predicting a discrete categorical label—whether an applicant is predicted as **Eligible** or **Not Eligible**—rather than a continuous numerical quantity. In machine learning, when the output belongs to a predefined set of categories, it is formulated as a supervised classification task.

---

### Q2: What is your target variable?
**Answer:** Our target variable is **`Eligibility`**, which is a binary categorical variable taking one of two discrete values: `'Eligible'` or `'Not Eligible'`. In our 1,000-record dataset, 638 records (63.8%) are labeled Eligible and 362 records (36.2%) are labeled Not Eligible.

---

### Q3: Why did you use four algorithms?
**Answer:** We used four distinct algorithms to compare different mathematical approaches to classification: a linear model (Logistic Regression), a single rule tree (Decision Tree), an ensemble bagging method (Random Forest), and a probabilistic model (Naïve Bayes). Comparing diverse model families ensures our final model selection is scientifically justified rather than chosen arbitrarily.

---

### Q4: Why Logistic Regression?
**Answer:** Logistic Regression serves as our linear baseline. It calculates the log-odds of eligibility as a linear combination of input features mapped through a sigmoid curve. If scholarship criteria were purely linear, Logistic Regression would suffice, which scored 86.0% accuracy in our tests.

---

### Q5: Why Decision Tree?
**Answer:** A Decision Tree splits data using recursive IF-THEN rules based on Gini impurity reduction. It closely mimics human administrative rule books (such as checking income cutoffs first, then community categories), achieving 90.5% accuracy in our experiments.

---

### Q6: Why Random Forest?
**Answer:** Random Forest is an ensemble method that combines 100 decorrelated decision trees using bootstrap aggregating (bagging) and random feature sub-sampling. It reduces the variance and overfitting tendencies of a single decision tree, achieving our highest performance at 93.0% accuracy and 0.944 F1-score.

---

### Q7: Why Naïve Bayes?
**Answer:** Naïve Bayes serves as a fast probabilistic benchmark based on Bayes' Theorem. It assumes all input features are conditionally independent given the class label. Because features like family income, community, and college type have correlations in practice, it achieved a lower accuracy of 84.0%.

---

### Q8: Why did Random Forest perform better than the other models?
**Answer:** Scholarship eligibility is governed by compound, non-linear step-functions (for instance, an income ceiling that applies differently depending on affirmative community categories). Linear models cannot capture these interactions without manual feature engineering, while a single tree easily overfits. Random Forest excels because it aggregates multiple randomized trees, smoothing decision boundaries and reducing variance.

---

### Q9: What is accuracy?
**Answer:** Accuracy is the ratio of correctly predicted student records (both True Positives and True Negatives) to the total number of evaluated records:
$$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$
In our test set of 200 records, Random Forest correctly classified 186 students, yielding 93.0% accuracy.

---

### Q10: What is precision?
**Answer:** Precision measures the exactness of positive predictions—out of all students the model labeled as Eligible, what fraction was genuinely labeled Eligible:
$$\text{Precision} = \frac{TP}{TP + FP}$$
Random Forest achieved 0.967 (96.7%) precision, meaning only 4 ineligible candidates were mistakenly predicted as eligible.

---

### Q11: What is recall?
**Answer:** Recall (or sensitivity) measures completeness—out of all students who actually qualified as Eligible, what fraction did the model successfully identify:
$$\text{Recall} = \frac{TP}{TP + FN}$$
Random Forest achieved 0.922 (92.2%) recall on the test set, successfully capturing 118 out of 128 eligible applicants.

---

### Q12: What is F1-score?
**Answer:** F1-score is the harmonic mean of precision and recall:
$$\text{F1-score} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$
It provides a balanced single metric between 0.0 and 1.0, ensuring that a model does not achieve high precision simply by being overly conservative or high recall by blindly approving everyone.

---

### Q13: Why is F1-score useful in this project?
**Answer:** Our dataset has a moderate class imbalance (63.8% Eligible vs 36.2% Not Eligible). A trivial dummy classifier that simply predicts "Eligible" for every student would achieve 63.8% accuracy while being completely useless. F1-score penalizes false predictions on both sides, making it a far more reliable metric than raw accuracy.

---

### Q14: What is overfitting?
**Answer:** Overfitting occurs when a machine learning model memorizes training noise and idiosyncrasies rather than learning generalizable underlying patterns. An overfitted model performs exceptionally well on training data but fails to predict accurately on unseen test data.

---

### Q15: How does Random Forest reduce overfitting?
**Answer:** Random Forest mitigates overfitting through two randomizing mechanisms: **bagging** (each tree trains on a different bootstrap sample of the training data) and **feature sub-sampling** (each split considers only a random subset of features). Averaging predictions across 100 decorrelated trees cancels out individual tree errors and reduces model variance.

---

### Q16: What is preprocessing?
**Answer:** Preprocessing is the stage where raw, unstructured, or mixed-type data is cleaned, validated, transformed, and formatted into numerical representations suitable for machine learning algorithms.

---

### Q17: Why encode categorical variables?
**Answer:** Most machine learning algorithms rely on mathematical calculations (matrix multiplications, distance metrics, dot products) and cannot compute directly on text strings like `'SC'`, `'Male'`, or `'Engineering'`. We use One-Hot Encoding with `handle_unknown='ignore'` to convert categorical features into binary indicator columns without imposing an artificial numerical ordering.

---

### Q18: Why scale numerical variables?
**Answer:** Numerical features like `FamilyIncome` (ranging in tens of thousands to lakhs) and `12thMarks` (ranging from 48 to 99) operate on drastically different numerical scales. If unscaled, models like Logistic Regression would assign disproportionate weight to income purely due to its larger magnitude. We apply `StandardScaler` to transform both features to mean $\mu=0$ and variance $\sigma=1$.

---

### Q19: What is data leakage?
**Answer:** Data leakage occurs when information from outside the training partition (such as test set statistics or future target information) inadvertently influences feature preprocessing or model training. This leads to unrealistically optimistic validation results that fail in real-world deployment.

---

### Q20: How did you prevent data leakage in your project?
**Answer:** We prevented leakage by encapsulating all scaling (`StandardScaler`), imputation (`SimpleImputer`), and categorical encoding (`OneHotEncoder`) within a scikit-learn `ColumnTransformer` inside a model `Pipeline`. The entire pipeline is fitted **strictly on the 80% training set** (`X_train`), and only `transform` is called on the 20% test set (`X_test`).

---

### Q21: What is Exploratory Data Analysis (EDA)?
**Answer:** EDA is the process of examining, summarizing, and visualizing a dataset prior to model training to uncover underlying distributions, detect outliers or missing data, and discover correlation patterns between features and the target.

---

### Q22: What does $r = -0.57$ mean for FamilyIncome?
**Answer:** A Pearson correlation coefficient of $r \approx -0.57$ indicates a moderate-to-strong **negative linear association** between family income and eligibility. As annual household income increases, the likelihood of being labeled Eligible decreases, which aligns with means-testing criteria.

---

### Q23: What does $r = +0.22$ mean for 12thMarks?
**Answer:** A Pearson correlation coefficient of $r \approx +0.22$ indicates a **weak-to-moderate positive linear association**. Higher 12th board marks are positively associated with eligibility (as seen in merit and merit-cum-means schemes), but marks alone are not the sole deciding factor.

---

### Q24: Does correlation mean causation?
**Answer:** **No, correlation does not imply causation.** A correlation of $r \approx -0.57$ shows that income and eligibility move in opposite directions within this dataset, but having low income does not automatically cause approval without satisfying other criteria (such as minimum marks and valid categories). Similarly, high marks do not cause an award if family income exceeds thresholds.

---

### Q25: What does feature importance mean in Random Forest?
**Answer:** In Random Forest, feature importance measures the average reduction in Gini Impurity brought about by splits on that feature across all 100 decision trees. It indicates **how heavily the model relied on each feature to make predictions**, NOT that the feature has a causal real-world impact. In our model, `FamilyIncome` contributed 46.9% and `12thMarks` contributed 21.3% of split importance.

---

### Q26: Why was StudentID removed?
**Answer:** `StudentID` is an arbitrary administrative identifier (`STU1001`, `STU1002`, etc.) with 1,000 unique values. It contains zero generalizable predictive signal. Retaining it would allow decision trees to memorize individual applicant rows, causing severe overfitting and false accuracy.

---

### Q27: Why was stratified splitting used?
**Answer:** Because our dataset is moderately imbalanced (63.8% Eligible vs 36.2% Not Eligible), a purely random split could draw unrepresentative class proportions in the test set. Stratified splitting enforces that both the 800-sample training set and 200-sample test set preserve the exact 63.8% / 36.2% class distribution.

---

### Q28: Why is the dataset split 80/20?
**Answer:** An 80/20 train/test split is a standard machine learning practice for 1,000-sample datasets. It allocates 800 records to provide sufficient sample diversity for tree learning while reserving 200 records to form a statistically reliable test set for evaluation.

---

### Q29: What are the limitations of this project?
**Answer:** 
1. **Synthetic Nature:** The dataset is a 1,000-record synthetic reference dataset created for academic experimentation; it does not contain real government student records.
2. **Tabular Scope:** The system evaluates tabular fields and cannot authenticate physical certificates or detect forged income certificates.
3. **No Causality:** The model identifies statistical patterns for decision-support; it does not establish legal eligibility.

---

### Q30: Is this an official scholarship eligibility system?
**Answer:** **No, absolutely not.** This is strictly an academic machine-learning classification and screening project. It demonstrates how classification algorithms benchmark and predict patterns on structured student data. It does not disburse funds, certify legal rights, or replace statutory government review bodies.
