# Exploratory Data Analysis (EDA) Report: Scholarship Eligibility Prediction

**Project:** Scholarship Eligibility Prediction  
**Course:** AD4V71 / AD5302 – Data Science  
**Authors:** Kirutick Siddhesh V (`210425243120`) & Krishna H (`210425243124`)  
**Data Source:** `data/raw/scholarship_data.csv` (1,000 synthetic records, 10 columns)  
**Output Plots:** `outputs/plots/`  
**Execution Script:** `src/eda.py`  

---

## 1. Executive Summary

Exploratory Data Analysis was performed on the 1,000-record scholarship dataset prior to feature engineering and model training. The dataset exhibits:
- **Zero missing values** across all columns.
- **Zero duplicate rows**.
- A **moderately imbalanced target distribution** with 63.8% Eligible (638 records) and 36.2% Not Eligible (362 records).
- A **moderate-to-strong negative linear association** between annual `FamilyIncome` and `Eligibility` ($r = -0.5739 \approx -0.57$).
- A **weak-to-moderate positive linear association** between `12thMarks` and `Eligibility` ($r = +0.2226 \approx +0.22$).
- `StudentID` was identified as an arbitrary unique identifier (1,000 distinct values) with zero generalizable predictive signal and dropped prior to modeling.

---

## 2. Dataset Overview & Integrity Audit

| Metric | Measured Value | Validation Status |
| :--- | :--- | :--- |
| **Row Count** | 1,000 records | PASS |
| **Column Count** | 10 columns (9 features + 1 target) | PASS |
| **Missing Values** | 0 total nulls | PASS (100% complete) |
| **Duplicate Rows** | 0 duplicates | PASS (100% unique) |
| **Data Provenance** | Synthetic reference dataset inspired by TN Post-Matric & AISHE | Explicitly documented |

---

## 3. Target Distribution (`Eligibility`)

- **Class 1 (`Eligible`):** 638 records (**63.8%**)
- **Class 0 (`Not Eligible`):** 362 records (**36.2%**)
- **Class Ratio:** $\approx 1.76 : 1$ (Moderate class imbalance)
- **Analytical Implication:** A baseline dummy model predicting "Eligible" for all inputs would achieve 63.8% accuracy without learning patterns. Consequently, model evaluation must prioritize **Precision, Recall, and F1-Score** over raw accuracy.
- **Associated Plot:** `outputs/plots/target_distribution.png`

---

## 4. Numerical Feature Distributions

### 4.1. FamilyIncome
- **Scale:** Continuous Indian Rupees (INR)
- **Range:** ₹25,000 to ₹12,00,000+
- **Mean:** $\approx ₹2,35,000$ | **Median:** $\approx ₹1,80,000$
- **Distribution Shape:** Positively skewed (right-skewed) density distribution.
- **Relationship with Eligibility:**
  - Eligible students are heavily concentrated in lower income slabs (< ₹2,50,000).
  - Applicants with annual income exceeding ₹3,00,000 show a steep drop in eligibility, reflecting institutional means-testing ceilings.
- **Associated Plots:** `outputs/plots/family_income_distribution.png`, `outputs/plots/feature_vs_target_income.png`

### 4.2. 12thMarks
- **Scale:** Continuous percentage (45.0% to 99.0%)
- **Mean:** $\approx 72.4\%$ | **Median:** $\approx 72.0\%$
- **Distribution Shape:** Symmetrical, approximately bell-shaped distribution.
- **Relationship with Eligibility:**
  - Eligible students exhibit a shifted distribution toward higher marks ($\ge 65\%$).
  - However, high marks alone do not guarantee eligibility if family income exceeds statutory caps.
- **Associated Plots:** `outputs/plots/marks_distribution.png`, `outputs/plots/feature_vs_target_marks.png`

---

## 5. Categorical Feature Distributions

1. **`Community`:** Balanced representation across reservation categories (`BC`: ~35%, `MBC`: ~25%, `SC`: ~20%, `OC`: ~15%, `ST`: ~5%). Affirmative action criteria produce higher eligibility rates within `SC` and `ST` groups.
   - *Plot:* `outputs/plots/community_distribution.png`, `outputs/plots/feature_vs_target_community.png`
2. **`Gender`:** Near parity (~52% Female, ~48% Male).
3. **`FirstGraduate`:** Approximately 40% First Graduate (`Yes`) vs 60% (`No`), providing positive eligibility lift for first-generation learners.
4. **`District`:** Geographic spread across 10 major Tamil Nadu educational districts (Chennai, Coimbatore, Madurai, Salem, Tiruchirappalli, Tirunelveli, Erode, Vellore, Thanjavur, Kanchipuram).
5. **`CollegeType`:** Government (~40%), Government Aided (~35%), Private (~25%).
6. **`Course`:** Engineering, Medical, Arts & Science, Commerce, and Management.

---

## 6. Correlation Analysis & Causation Caveat

### Pearson Correlation Matrix with Binary Target (`Eligible = 1, Not Eligible = 0`):
| Feature | Pearson $r$ | Association Direction | Strength |
| :--- | :---: | :---: | :--- |
| **`FamilyIncome`** | **$-0.5739$ ($r \approx -0.57$)** | Negative | Moderate-to-Strong |
| **`12thMarks`** | **$+0.2226$ ($r \approx +0.22$)** | Positive | Weak-to-Moderate |

### Critical Scientific Causation Caveat:
- **Correlation does NOT imply causation.**
- An $r$ of $-0.57$ proves that income and eligibility move inversely in this dataset, but having low income does not independently *cause* a scholarship grant without satisfying academic and quota rules.
- Similarly, an $r$ of $+0.22$ proves that marks correlate positively, but high marks do not *cause* an award if family income exceeds limits.
- Associated Plot: `outputs/plots/correlation_heatmap.png`

---

## 7. Conclusions for Modeling Strategy

1. **Non-Linear Interactions:** Because eligibility depends on multi-attribute step functions (e.g. income thresholds varying by community), tree-based models (Decision Tree, Random Forest) are expected to outperform linear baselines.
2. **Feature Scaling Required:** StandardScaler is mandatory for `FamilyIncome` and `12thMarks` to prevent gradient dominance in Logistic Regression.
3. **Encoding Strategy:** One-Hot Encoding with `handle_unknown='ignore'` must be applied to all 6 nominal features.
4. **Leakage Prevention:** All preprocessing must be isolated strictly to training partitions via `ColumnTransformer`.
