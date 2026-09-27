"""
Exploratory Data Analysis (EDA) Module for Scholarship Eligibility Prediction.

This module generates comprehensive descriptive statistics and publication-grade plots:
1. Target Distribution (Eligible vs Not Eligible)
2. Community Distribution with Eligibility Breakdown
3. Family Income Distribution (KDE + Box Plot)
4. 12th Marks Distribution (KDE + Box Plot)
5. Pearson Correlation Matrix (Heatmap & Feature-to-Target ranking)
6. Feature vs Target Bivariate Visualizations
"""

import json
import os
from typing import Dict, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


PLOTS_DIR = os.path.join("outputs", "plots")
METRICS_DIR = os.path.join("outputs", "metrics")
DEFAULT_DATA_PATH = os.path.join("data", "raw", "scholarship_data.csv")


def set_plot_style():
    """Configure modern styling for all generated plots."""
    sns.set_theme(style="whitegrid", palette="muted")
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "axes.titlesize": 14,
        "axes.titleweight": "bold",
        "axes.labelsize": 12,
        "axes.labelweight": "semibold",
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.autolayout": True
    })


def plot_target_distribution(df: pd.DataFrame, output_dir: str = PLOTS_DIR) -> str:
    """Generate target variable distribution (Bar & Pie)."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    target_counts = df['Eligibility'].value_counts()
    colors = ["#2b8a3e", "#c92a2a"]

    # Bar chart
    sns.barplot(x=target_counts.index, y=target_counts.values, hue=target_counts.index, ax=axes[0], palette=colors, legend=False)
    axes[0].set_title("Scholarship Eligibility Count")
    axes[0].set_xlabel("Eligibility Status")
    axes[0].set_ylabel("Number of Students")
    for i, count in enumerate(target_counts.values):
        axes[0].text(i, count + 10, f"{count} ({count/len(df)*100:.1f}%)", ha="center", fontweight="bold")

    # Donut chart
    axes[1].pie(
        target_counts.values,
        labels=target_counts.index,
        autopct="%1.1f%%",
        startangle=140,
        colors=colors,
        wedgeprops=dict(width=0.4, edgecolor='white', linewidth=2)
    )
    axes[1].set_title("Target Proportion (Moderate Imbalance)")

    path = os.path.join(output_dir, "target_distribution.png")
    plt.savefig(path, dpi=300)
    plt.close()
    return path


def plot_community_distribution(df: pd.DataFrame, output_dir: str = PLOTS_DIR) -> str:
    """Plot distribution of applicants across Community categories with eligibility breakdown."""
    plt.figure(figsize=(10, 6))
    order = ['SC', 'ST', 'MBC', 'BC', 'OC']
    palette = {"Eligible": "#2b8a3e", "Not Eligible": "#e03131"}

    ax = sns.countplot(
        data=df,
        x='Community',
        hue='Eligibility',
        order=[c for c in order if c in df['Community'].unique()],
        palette=palette
    )
    plt.title("Community Distribution by Scholarship Eligibility", pad=15)
    plt.xlabel("Community Category")
    plt.ylabel("Number of Students")
    plt.legend(title="Status", frameon=True)

    path = os.path.join(output_dir, "community_distribution.png")
    plt.savefig(path, dpi=300)
    plt.close()
    return path


def plot_family_income_distribution(df: pd.DataFrame, output_dir: str = PLOTS_DIR) -> str:
    """Plot Family Income histogram, KDE, and boxplot."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    palette = {"Eligible": "#2b8a3e", "Not Eligible": "#e03131"}

    # KDE & Histogram
    sns.histplot(
        data=df,
        x='FamilyIncome',
        hue='Eligibility',
        kde=True,
        ax=axes[0],
        palette=palette,
        bins=30
    )
    axes[0].set_title("Family Income Distribution (KDE & Histogram)")
    axes[0].set_xlabel("Annual Family Income (INR)")
    axes[0].set_ylabel("Student Count")

    # Boxplot
    sns.boxplot(
        data=df,
        x='Eligibility',
        y='FamilyIncome',
        hue='Eligibility',
        ax=axes[1],
        palette=palette,
        legend=False
    )
    axes[1].set_title("Family Income by Eligibility (Box Plot)")
    axes[1].set_xlabel("Eligibility Status")
    axes[1].set_ylabel("Annual Family Income (INR)")

    path = os.path.join(output_dir, "family_income_distribution.png")
    plt.savefig(path, dpi=300)
    plt.close()
    return path


def plot_marks_distribution(df: pd.DataFrame, output_dir: str = PLOTS_DIR) -> str:
    """Plot 12th Marks histogram and box plot."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    palette = {"Eligible": "#2b8a3e", "Not Eligible": "#e03131"}

    # KDE & Histogram
    sns.histplot(
        data=df,
        x='12thMarks',
        hue='Eligibility',
        kde=True,
        ax=axes[0],
        palette=palette,
        bins=25
    )
    axes[0].set_title("12th Marks Distribution (KDE & Histogram)")
    axes[0].set_xlabel("12th Board Marks (%)")
    axes[0].set_ylabel("Student Count")

    # Boxplot
    sns.boxplot(
        data=df,
        x='Eligibility',
        y='12thMarks',
        hue='Eligibility',
        ax=axes[1],
        palette=palette,
        legend=False
    )
    axes[1].set_title("12th Marks by Eligibility (Box Plot)")
    axes[1].set_xlabel("Eligibility Status")
    axes[1].set_ylabel("12th Board Marks (%)")

    path = os.path.join(output_dir, "marks_distribution.png")
    plt.savefig(path, dpi=300)
    plt.close()
    return path


def plot_correlation_analysis(df: pd.DataFrame, output_dir: str = PLOTS_DIR) -> Dict[str, float]:
    """
    Compute correlation between numerical features and binary target.
    Target encoding: Eligible = 1, Not Eligible = 0.
    """
    df_corr = df.copy()
    df_corr['Target_Encoded'] = (df_corr['Eligibility'] == 'Eligible').astype(int)

    num_cols = ['FamilyIncome', '12thMarks', 'Target_Encoded']
    corr_matrix = df_corr[num_cols].corr()

    # Heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        square=True,
        cbar_kws={"label": "Pearson Correlation (r)"}
    )
    plt.title("Correlation Matrix: Academic & Financial Predictors vs Eligibility", pad=15)
    path = os.path.join(output_dir, "correlation_heatmap.png")
    plt.savefig(path, dpi=300)
    plt.close()

    r_income = float(corr_matrix.loc['FamilyIncome', 'Target_Encoded'])
    r_marks = float(corr_matrix.loc['12thMarks', 'Target_Encoded'])

    return {
        "FamilyIncome_r": r_income,
        "12thMarks_r": r_marks
    }


def plot_feature_vs_target_analysis(df: pd.DataFrame, output_dir: str = PLOTS_DIR):
    """Generate focused bivariate relationship plots."""
    palette = {"Eligible": "#2b8a3e", "Not Eligible": "#e03131"}

    # 1. Income vs Target
    plt.figure(figsize=(7, 5))
    sns.violinplot(data=df, x='Eligibility', y='FamilyIncome', hue='Eligibility', palette=palette, inner='quartile', legend=False)
    plt.title("Family Income Distribution across Eligibility Groups")
    plt.xlabel("Eligibility Status")
    plt.ylabel("Annual Family Income (INR)")
    plt.savefig(os.path.join(output_dir, "feature_vs_target_income.png"), dpi=300)
    plt.close()

    # 2. 12th Marks vs Target
    plt.figure(figsize=(7, 5))
    sns.violinplot(data=df, x='Eligibility', y='12thMarks', hue='Eligibility', palette=palette, inner='quartile', legend=False)
    plt.title("12th Marks Distribution across Eligibility Groups")
    plt.xlabel("Eligibility Status")
    plt.ylabel("12th Marks (%)")
    plt.savefig(os.path.join(output_dir, "feature_vs_target_marks.png"), dpi=300)
    plt.close()

    # 3. Community proportions vs Target
    plt.figure(figsize=(9, 5))
    comm_pct = pd.crosstab(df['Community'], df['Eligibility'], normalize='index') * 100
    comm_pct[['Eligible', 'Not Eligible']].plot(
        kind='bar',
        stacked=True,
        color=['#2b8a3e', '#e03131'],
        figsize=(9, 5)
    )
    plt.title("Eligibility Percentage within Each Community")
    plt.xlabel("Community")
    plt.ylabel("Percentage (%)")
    plt.legend(title="Eligibility")
    plt.xticks(rotation=0)
    plt.savefig(os.path.join(output_dir, "feature_vs_target_community.png"), dpi=300)
    plt.close()


def run_full_eda(
    data_path: str = DEFAULT_DATA_PATH,
    plots_dir: str = PLOTS_DIR,
    metrics_dir: str = METRICS_DIR
) -> Dict[str, Any]:
    """Execute complete EDA pipeline and save figures and metrics."""
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)
    set_plot_style()

    df = pd.read_csv(data_path)
    print("=" * 60)
    print("RUNNING EXPLORATORY DATA ANALYSIS (EDA)")
    print("=" * 60)
    print(f"Analyzing {len(df)} records from {data_path}...")

    # Plot generations
    plot_target_distribution(df, plots_dir)
    plot_community_distribution(df, plots_dir)
    plot_family_income_distribution(df, plots_dir)
    plot_marks_distribution(df, plots_dir)
    plot_feature_vs_target_analysis(df, plots_dir)
    corr_results = plot_correlation_analysis(df, plots_dir)

    # Reference values from presentation
    ref_income_r = -0.58
    ref_marks_r = 0.30

    calc_income_r = corr_results['FamilyIncome_r']
    calc_marks_r = corr_results['12thMarks_r']

    print(f"\n[CORRELATION ANALYSIS]")
    print(f"  Calculated FamilyIncome correlation (r): {calc_income_r:.2f}")
    print(f"  Reference PPT FamilyIncome (r):          {ref_income_r:.2f}")
    print(f"  Difference:                             {calc_income_r - ref_income_r:+.2f}")

    print(f"\n  Calculated 12thMarks correlation (r):    {calc_marks_r:.2f}")
    print(f"  Reference PPT 12thMarks (r):             {ref_marks_r:.2f}")
    print(f"  Difference:                             {calc_marks_r - ref_marks_r:+.2f}")

    print("\n[SCIENTIFIC CORRELATION EXPLANATION]")
    print("  1. FamilyIncome negative correlation (r < 0):")
    print("     Indicates that higher household earnings correspond to a lower likelihood")
    print("     of meeting need-based scholarship criteria (affirmative means-testing).")
    print("  2. 12thMarks positive correlation (r > 0):")
    print("     Indicates that higher academic performance correlates with a greater")
    print("     probability of qualification under merit and merit-cum-means schemes.")
    print("  3. Causation Caveat:")
    print("     Neither correlation implies direct singular causation. Scholarship awards")
    print("     depend on multi-variable compound policy rules (income thresholds, community,")
    print("     and graduation status combined).")

    eda_summary = {
        "total_records": len(df),
        "target_distribution": df['Eligibility'].value_counts().to_dict(),
        "target_percentages": (df['Eligibility'].value_counts(normalize=True) * 100).round(2).to_dict(),
        "calculated_correlations": {
            "FamilyIncome_r": round(calc_income_r, 4),
            "12thMarks_r": round(calc_marks_r, 4)
        },
        "reference_correlations": {
            "FamilyIncome_r": ref_income_r,
            "12thMarks_r": ref_marks_r
        },
        "discrepancies": {
            "FamilyIncome_diff": round(calc_income_r - ref_income_r, 4),
            "12thMarks_diff": round(calc_marks_r - ref_marks_r, 4)
        }
    }

    summary_file = os.path.join(metrics_dir, "eda_summary.json")
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(eda_summary, f, indent=4)

    print(f"\n[OUTPUT] Saved EDA metrics to: {summary_file}")
    print(f"[OUTPUT] Saved all 8 visual plots to: {plots_dir}")
    print("=" * 60)

    return eda_summary


if __name__ == "__main__":
    run_full_eda()
