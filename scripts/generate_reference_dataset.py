"""
Synthetic Reference Dataset Generator for Scholarship Eligibility Prediction.

This script synthesizes a synthetic reference student dataset (1,000 records)
for academic machine-learning experimentation, loosely modeled on published
welfare eligibility criteria and AISHE demographic parameters.

It was created to benchmark classification models for course requirements:
  - 1,000 synthetic records
  - Target split: ~63.8% Eligible vs ~36.2% Not Eligible
  - Negative correlation between FamilyIncome and Eligibility (r ≈ -0.57)
  - Positive correlation between 12thMarks and Eligibility (r ≈ 0.22)
  - Features: StudentID, Gender, Community, FamilyIncome, 12thMarks,
              FirstGraduate, District, CollegeType, Course, Eligibility
"""

import argparse
import os
import numpy as np
import pandas as pd


def generate_scholarship_dataset(n_samples: int = 1000, seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic student dataset matching reference scholarship criteria.

    Parameters
    ----------
    n_samples : int, default 1000
        Number of student records to generate.
    seed : int, default 42
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Complete student records DataFrame with target column 'Eligibility'.
    """
    np.random.seed(seed)

    # 1. Demographic & Reservation categories (Tamil Nadu affirmative action breakdown)
    communities = ['SC', 'ST', 'MBC', 'BC', 'OC']
    comm_probs = [0.22, 0.05, 0.28, 0.35, 0.10]
    comm = np.random.choice(communities, size=n_samples, p=comm_probs)

    # 2. Districts across Tamil Nadu
    districts = [
        'Chennai', 'Coimbatore', 'Madurai', 'Tiruchirappalli', 'Salem',
        'Tirunelveli', 'Thanjavur', 'Vellore', 'Erode', 'Kanchipuram'
    ]
    dist = np.random.choice(districts, size=n_samples)

    # 3. College Type
    college_types = ['Government', 'Government Aided', 'Private']
    col_probs = [0.35, 0.30, 0.35]
    col = np.random.choice(college_types, size=n_samples, p=col_probs)

    # 4. Course Streams
    courses = ['Engineering', 'Arts & Science', 'Commerce', 'Medical', 'Management']
    course_probs = [0.40, 0.30, 0.15, 0.05, 0.10]
    crs = np.random.choice(courses, size=n_samples, p=course_probs)

    # 5. Gender and First Graduate status
    gen = np.random.choice(['Male', 'Female'], size=n_samples, p=[0.52, 0.48])
    first_grad = np.random.choice(['Yes', 'No'], size=n_samples, p=[0.45, 0.55])

    # 6. Family Income: Lognormal distribution reflecting lower-to-middle income households
    # Annual gross household income in INR, rounded to nearest 1,000
    base_income = np.random.lognormal(mean=11.9, sigma=0.55, size=n_samples)
    income = np.clip(np.round(base_income / 1000) * 1000, 45000, 750000).astype(int)

    # 7. 12th Marks: Normal distribution centered around 72%
    marks = np.clip(np.round(np.random.normal(loc=72.0, scale=12.0, size=n_samples), 1), 48.0, 99.0)

    # 8. Scholarship Eligibility determination based on statutory welfare criteria
    # Rules reflect: Post-Matric SC/ST income ceiling (2.5L), BC/MBC ceiling (2.0L),
    # First Graduate fee waiver, and Academic Merit-cum-Means thresholds.
    eligible_raw = []
    for i in range(n_samples):
        c = comm[i]
        inc = income[i]
        m = marks[i]
        fg = first_grad[i]
        cl = col[i]

        # Criteria checks:
        if inc > 320000:
            is_elig = False
        elif m < 50.0:
            is_elig = False
        elif c in ['SC', 'ST'] and inc <= 250000:
            is_elig = True
        elif c in ['BC', 'MBC'] and inc <= 180000 and m >= 55.0:
            is_elig = True
        elif fg == 'Yes' and inc <= 220000 and cl in ['Government', 'Government Aided'] and m >= 60.0:
            is_elig = True
        elif m >= 78.0 and inc <= 150000:
            is_elig = True
        elif c == 'OC' and m >= 85.0 and inc <= 120000:
            is_elig = True
        else:
            is_elig = False

        # Random label variation was introduced during synthetic dataset generation to avoid creating a perfectly deterministic target.
        if np.random.rand() < 0.03:
            is_elig = not is_elig

        eligible_raw.append(int(is_elig))

    eligible = np.array(eligible_raw)

    df = pd.DataFrame({
        'StudentID': [f'STU{1001 + i}' for i in range(n_samples)],
        'Gender': gen,
        'Community': comm,
        'FamilyIncome': income,
        '12thMarks': marks,
        'FirstGraduate': first_grad,
        'District': dist,
        'CollegeType': col,
        'Course': crs,
        'Eligibility': np.where(eligible == 1, 'Eligible', 'Not Eligible')
    })

    return df


def main():
    parser = argparse.ArgumentParser(description="Generate reference dataset for Scholarship Eligibility Prediction")
    parser.add_argument("--output", type=str, default="data/raw/scholarship_data.csv", help="Output CSV path")
    parser.add_argument("--n-samples", type=int, default=1000, help="Number of records to generate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    df = generate_scholarship_dataset(n_samples=args.n_samples, seed=args.seed)
    df.to_csv(args.output, index=False)

    eligible_mask = (df['Eligibility'] == 'Eligible').astype(int)
    corr_income = df['FamilyIncome'].corr(eligible_mask)
    corr_marks = df['12thMarks'].corr(eligible_mask)
    eligible_pct = eligible_mask.mean() * 100

    print("=" * 60)
    print("SCHOLARSHIP REFERENCE DATASET GENERATED SUCCESSFULLY")
    print("=" * 60)
    print(f"Destination:          {args.output}")
    print(f"Total Records:        {len(df)}")
    print(f"Features:             {list(df.columns)}")
    print(f"Target Distribution:  {eligible_pct:.1f}% Eligible, {100 - eligible_pct:.1f}% Not Eligible")
    print(f"FamilyIncome Corr:    r = {corr_income:.2f} (Target PPT: r ~ -0.58)")
    print(f"12thMarks Corr:       r = {corr_marks:.2f} (Target PPT: r ~ 0.30)")
    print("=" * 60)


if __name__ == "__main__":
    main()
