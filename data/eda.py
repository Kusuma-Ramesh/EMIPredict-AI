from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "emi_prediction_cleaned.csv"
)

FIGURES_DIR = (
    PROJECT_ROOT
    / "reports"
    / "figures"
)

FIGURES_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# LOAD CLEANED DATA
# ============================================================

df = pd.read_csv(DATA_PATH)

print("=" * 70)
print("EMIPredict AI - EXPLORATORY DATA ANALYSIS")
print("=" * 70)

print(f"\nDataset shape: {df.shape}")

print("\nDataset loaded successfully.")

print("\nColumns:")
print(df.columns.tolist())

print("\nData types:")
print(df.dtypes)

print("\n" + "=" * 70)


# ============================================================
# TARGET DISTRIBUTION ANALYSIS
# ============================================================

print("\nCLASSIFICATION TARGET")
print("-" * 70)

classification_counts = df["emi_eligibility"].value_counts()

print(classification_counts)

print("\nClassification percentages:")

print(
    (classification_counts / len(df) * 100)
    .round(2)
)


print("\nREGRESSION TARGET")
print("-" * 70)

print(
    df["max_monthly_emi"].describe()
)


# ============================================================
# NUMERICAL FEATURE SUMMARY
# ============================================================

print("\nNUMERICAL FEATURE SUMMARY")
print("-" * 70)

numeric_columns = (
    df.select_dtypes(include="number")
    .columns
)

print(
    f"\nNumber of numerical columns: "
    f"{len(numeric_columns)}"
)

print("\nStatistical summary:")

print(
    df[numeric_columns]
    .describe()
    .T
    .round(2)
)


# ============================================================
# CATEGORICAL FEATURE SUMMARY
# ============================================================

print("\nCATEGORICAL FEATURE SUMMARY")
print("-" * 70)

categorical_columns = (
    df.select_dtypes(include="object")
    .columns
)

print(
    f"\nNumber of categorical columns: "
    f"{len(categorical_columns)}"
)

for column in categorical_columns:

    print(f"\n{column}")
    print("-" * 50)

    print(
        df[column]
        .value_counts()
    )


# ============================================================
# VISUAL EDA 1
# EMI ELIGIBILITY DISTRIBUTION
# ============================================================

plt.figure(figsize=(10, 6))

sns.countplot(
    data=df,
    x="emi_eligibility"
)

plt.title(
    "EMI Eligibility Distribution"
)

plt.xlabel(
    "EMI Eligibility"
)

plt.ylabel(
    "Number of Customers"
)

plt.tight_layout()

plt.savefig(
    FIGURES_DIR
    / "emi_eligibility_distribution.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


print("\nSaved:")
print(
    FIGURES_DIR
    / "emi_eligibility_distribution.png"
)


# ============================================================
# VISUAL EDA 2
# EMI ELIGIBILITY BY SCENARIO
# ============================================================

scenario_eligibility = pd.crosstab(
    df["emi_scenario"],
    df["emi_eligibility"]
)

print("\n" + "=" * 70)
print("EMI ELIGIBILITY BY SCENARIO")
print("=" * 70)

print(
    scenario_eligibility
)


scenario_eligibility.plot(
    kind="bar",
    figsize=(12, 6)
)

plt.title(
    "EMI Eligibility by EMI Scenario"
)

plt.xlabel(
    "EMI Scenario"
)

plt.ylabel(
    "Number of Customers"
)

plt.xticks(
    rotation=25
)

plt.legend(
    title="EMI Eligibility"
)

plt.tight_layout()

plt.savefig(
    FIGURES_DIR
    / "emi_eligibility_by_scenario.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


print("\nSaved:")
print(
    FIGURES_DIR
    / "emi_eligibility_by_scenario.png"
)


# ============================================================
# EDA COMPLETION
# ============================================================

print("\n" + "=" * 70)
print("EDA COMPLETED SUCCESSFULLY")
print("=" * 70)

print("\nGenerated figures:")

print(
    "1.",
    FIGURES_DIR
    / "emi_eligibility_distribution.png"
)

print(
    "2.",
    FIGURES_DIR
    / "emi_eligibility_by_scenario.png"
)

print("\n" + "=" * 70)