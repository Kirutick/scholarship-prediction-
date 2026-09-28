# Presentation Plan & Slide-by-Slide Content (10 Slides)

**Project Title:** Scholarship Eligibility Prediction  
**Course Code:** AD4V71 / AD5302 – Data Science  
**Presenting Team:**  
- **Team Member 1:** Kirutick Siddhesh V (Regd No: `210425243120`, Section C)  
- **Team Member 2:** Krishna H (Regd No: `210425243124`, Section C)  
**PPT File Path:** `outputs/presentation/Scholarship_Eligibility_Prediction.pptx`  
**Speaking Script:** `outputs/presentation_speaking_script.md`  

---

## Slide 1 — Title & Team Details
- **Title:** Scholarship Eligibility Prediction
- **Course:** AD4V71 – Data Science
- **Presenting By:**
  - Team Member 1 Name: Kirutick Siddhesh V | Regd No: 210425243120 | Section: C
  - Team Member 2 Name: Krishna H | Regd No: 210425243124 | Section: C
- **Visuals:** Clean academic layout with institutional identifiers.

---

## Slide 2 — Problem Identification & Title Justification
- **Relevance of the Problem:** Scholarship screening in higher education traditionally requires checking multiple academic, financial, and demographic criteria across student applications. An automated prediction system can assist institutional review committees in screening applications efficiently based on objective criteria.
- **Problem Statement:** Determining scholarship eligibility involves evaluating multiple criteria. This project develops a machine-learning classification pipeline that predicts whether a student application is predicted as Eligible or Not Eligible based on available student attributes.
- **Title Justification and Scope:** The title "Scholarship Eligibility Prediction" accurately represents the machine-learning classification task using academic and financial attributes. The project scope includes data preprocessing, exploratory data analysis, feature engineering, classification model benchmarking, evaluation, and decision-support prediction, while explicitly excluding official scholarship approval, fund allocation, and document verification.

---

## Slide 3 — Basic Concepts
- **Machine Learning:** Enables computer systems to learn patterns from historical student data to make predictions without being explicitly hardcoded.
- **Classification:** A supervised learning task that categorizes student applications into discrete classes (Eligible or Not Eligible).
- **Dataset:** A structured collection of academic, financial, and demographic student attributes used for model training and evaluation.
- **Data Preprocessing:** Cleaning, handling categorical encodings, and scaling numerical features into standardized formats for algorithms.
- **Feature Selection / Engineering:** Identifying and transforming key predictor attributes (e.g., family income, board marks) that contribute to predictions.
- **Model Training:** Fitting classification algorithms on training data using supervised learning to minimize prediction error.
- **Prediction System:** Evaluating new student attribute inputs through the trained pipeline to output predicted eligibility with confidence scores.

---

## Slide 4 — Literature Survey & Research Gap
- **Related Systems Survey:**
  1. Scholarship Recommender: Suggests scholarships | Limitation: No eligibility prediction
  2. Financial Aid Predictor: Uses income data | Limitation: Limited academic factors considered
  3. AI Screening System: Faster evaluation | Limitation: Requires very large enterprise datasets
  4. Student Support System: Assists decisions | Limitation: Less personalized to individual criteria
  5. ML Eligibility Model: Good initial accuracy | Limitation: Restricted feature scope
- **Research Gap Addressed:**
  - *Multidimensional Feature Integration:* Combines academic, financial, and demographic attributes in a unified pipeline rather than isolated evaluations.
  - *Multi-Algorithm Benchmarking:* Systematically compares four distinct classification paradigms (linear, tree, ensemble, probabilistic).
  - *Zero-Leakage Pipeline:* Implements strict ColumnTransformer preprocessing fitted exclusively on training data.
- **Reference Context Note:**
  - Government scholarship guidelines and AISHE statistics provide conceptual domain context; the dataset is a synthetic reference benchmark created for academic experimentation.

---

## Slide 5 — Objectives
1. Develop a machine learning classification model for scholarship eligibility prediction.
2. Analyze academic, financial, and demographic student attributes affecting eligibility.
3. Perform structured data preprocessing and exploratory data analysis (EDA).
4. Train and benchmark multiple classification algorithms (Logistic Regression, Decision Tree, Random Forest, Naïve Bayes).
5. Evaluate model performance using standard evaluation metrics (Accuracy, Precision, Recall, F1-Score) to predict students labeled as eligible in the dataset.
6. Provide a decision-support prediction interface with probability estimation.

---

## Slide 6 — Project Planning & Time Schedule
- **Task Allocation:**
  - Kirutick Siddhesh V: Literature Survey, Data Collection, Documentation
  - Krishna H: Model Development, Testing, Presentation
- **Milestones:**
  - M1: Problem Definition | M2: Literature Survey | M3: Dataset Prepared | M4: Model Development | M5: Testing & Evaluation | M6: Final Submission
- **Timeline (W1-W8):**
  - W1-W2 $\rightarrow$ Literature Survey
  - W3-W4 $\rightarrow$ Data Collection & Preprocessing
  - W5-W6 $\rightarrow$ Model Development
  - W7 $\rightarrow$ Testing & Evaluation
  - W8 $\rightarrow$ Documentation & Presentation Rehearsal

---

## Slide 7 — Exploratory Data Analysis & Actual Findings
- **Dataset:** 1,000 synthetic student records (Community, FamilyIncome, 12thMarks, FirstGraduate, District, CollegeType, Course).
- **Target Distribution:** 63.8% Eligible (638) vs 36.2% Not Eligible (362) — moderately imbalanced classification target.
- **FamilyIncome Correlation:** $r \approx -0.57$ (Calculated $r = -0.5739$) — shows the strongest negative linear association with eligibility (means-testing criteria).
- **12thMarks Correlation:** $r \approx +0.22$ (Calculated $r = +0.2226$) — shows a moderate positive linear association with eligibility (academic merit).
- **Causality Note:** Correlation indicates linear statistical association within the dataset, not causation (income and marks do not independently cause eligibility).
- **Embedded Visuals:**
  - Left: `outputs/plots/community_distribution.png`
  - Right: `outputs/plots/family_income_distribution.png`

---

## Slide 8 — Model Comparison & Actual Results
- **Experimental Setup:** Four classification algorithms trained on 800 samples and evaluated on 200 held-out test samples (stratified 80/20 split).
- **Comparison Table (Actual Measured Scores):**
  | Model | Accuracy | Precision | Recall | F1-Score |
  | :--- | :---: | :---: | :---: | :---: |
  | Logistic Regression | 86.0% | 0.8906 | 0.8906 | 0.8906 |
  | Decision Tree | 90.5% | 0.9504 | 0.8984 | 0.9237 |
  | **Random Forest (Best)** | **93.0%** | **0.9672** | **0.9219** | **0.9440** |
  | Naïve Bayes | 84.0% | 0.8810 | 0.8672 | 0.8740 |
- **Random Forest Result:** Random Forest achieved **93.0% accuracy** and an **0.9440 F1-score** on the held-out test set, performing best in the experiment.
- **Feature Importance:** FamilyIncome (46.94%) and 12thMarks (21.28%) are the top predictors in the model.
- **Embedded Visual:** `outputs/plots/random_forest_feature_importance.png`

---

## Slide 9 — References
- **Academic Machine Learning Foundations:**
  - Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python. Journal of Machine Learning Research, 12, 2825-2830.
  - Breiman, L. (2001). Random Forests. Machine Learning, 45(1), 5-32.
  - Quinlan, J. R. (1986). Induction of Decision Trees. Machine Learning, 1(1), 81-106.
  - McKinney, W. (2010). Data Structures for Statistical Computing in Python (pandas). Proceedings of the 9th Python in Science Conference.
- **Conceptual Domain Context References:**
  - Government of Tamil Nadu: Post-Matric Scholarship Scheme Guidelines (Adi Dravidar & Tribal Welfare / Backward Classes Welfare Department) – Used as conceptual reference context for eligibility criteria.
  - Ministry of Education, Government of India: All India Survey on Higher Education (AISHE) Reports – Used as reference context for higher education enrolment demographics.
- **Clarification Note:** References provide algorithmic background and conceptual domain context; the project dataset is a synthetic reference dataset generated for academic machine-learning experimentation.

---

## Slide 10 — Thank You & Conclusion
- **Title:** Thank You!
- **Subtitle:** Academic Machine Learning Classification Demonstration
- **Course & Team:** Course: AD4V71 – Data Science | Team: Kirutick Siddhesh V (210425243120) & Krishna H (210425243124)
- **Closing Note:** Preliminary decision-support tool; official scholarship approvals remain under human authority.
