# ============================================================
# Script: generate_dataset.py
# Purpose: Generate synthetic dataset for Freelance Job Success Prediction
# ============================================================

import pandas as pd
import numpy as np
import os

np.random.seed(42)
print("🔄 Generating synthetic dataset...")

# ─── Parameters ───────────────────────────────────────────
n_records = 5000

# ─── Generate Raw Features ────────────────────────────────
project_budget          = np.random.randint(500, 50000, n_records).astype(float)
client_rating           = np.round(np.random.uniform(1.0, 5.0, n_records), 1)
num_existing_proposals  = np.random.randint(0, 100, n_records).astype(float)
freelancer_experience   = np.random.randint(0, 20, n_records)
proposal_quality_score  = np.random.randint(1, 10, n_records).astype(float)
project_duration        = np.random.randint(1, 90, n_records).astype(float)
freelancer_rating       = np.round(np.random.uniform(1.0, 5.0, n_records), 1)
previous_jobs_completed = np.random.randint(0, 500, n_records)

project_category   = np.random.choice(
    ["Web Development", "Mobile App", "Data Science", "Design", "Writing", "Marketing"],
    n_records
)
client_location    = np.random.choice(
    ["USA", "UK", "Canada", "Australia", "India", "Germany", "France", "UAE", "Singapore", "Brazil"],
    n_records
)
freelancer_country = np.random.choice(
    ["USA", "UK", "Canada", "Australia", "India", "Pakistan", "Philippines", "Ukraine", "Nigeria", "Bangladesh"],
    n_records
)

# ─── Generate Target Variable (Success) ───────────────────
#
# KEY FIX: Use a weighted score with small noise, then apply
# a percentile threshold — this preserves the feature→label
# relationship and gives models a learnable signal.
#
# Weights (must sum to 1.0):
#   proposal_quality_score  : 0.40  (most important)
#   freelancer_experience   : 0.30  (second)
#   freelancer_rating       : 0.20  (third)
#   client_rating           : 0.10  (fourth)
#
score = (
    0.40 * (proposal_quality_score / 9)
    + 0.30 * (freelancer_experience / 19)
    + 0.20 * (freelancer_rating / 5)
    + 0.10 * (client_rating / 5)
)

# Small realistic noise (σ=0.05)
noise = np.random.normal(0, 0.05, n_records)
final_score = np.clip(score + noise, 0, 1)

# Top 40% by score → success (preserves ~40% success rate with clean signal)
threshold_val = np.percentile(final_score, 60)
success = (final_score >= threshold_val).astype(int)

# ─── Build DataFrame ──────────────────────────────────────
df = pd.DataFrame({
    "project_budget":          project_budget,
    "client_rating":           client_rating,
    "num_existing_proposals":  num_existing_proposals,
    "freelancer_experience":   freelancer_experience,
    "proposal_quality_score":  proposal_quality_score,
    "project_category":        project_category,
    "client_location":         client_location,
    "project_duration":        project_duration,
    "freelancer_country":      freelancer_country,
    "freelancer_rating":       freelancer_rating,
    "previous_jobs_completed": previous_jobs_completed,
    "success":                 success,
})

# ─── Add Realistic Missing Values ─────────────────────────
print("⚠️  Introducing missing values...")
for col, rate in [("client_rating", 0.05), ("proposal_quality_score", 0.03),
                  ("freelancer_rating", 0.02), ("project_duration", 0.01)]:
    mask = np.random.random(len(df)) < rate
    df.loc[mask, col] = np.nan

# ─── Add Realistic Duplicates ─────────────────────────────
print("🔄 Adding duplicate records...")
dup_count = int(n_records * 0.02)
df = pd.concat([df, df.sample(n=dup_count, replace=True)], ignore_index=True)

# ─── Add Realistic Outliers ───────────────────────────────
print("📊 Introducing outliers...")
df.loc[np.random.choice(df.index, 30, replace=False), "project_budget"] = \
    np.random.randint(100000, 500000, 30).astype(float)
df.loc[np.random.choice(df.index, 20, replace=False), "num_existing_proposals"] = \
    np.random.randint(150, 300, 20).astype(float)

# ─── Feature Engineering ──────────────────────────────────
df["budget_per_proposal"]            = df["project_budget"] / (df["num_existing_proposals"] + 1)
df["client_rating_category"]         = pd.cut(df["client_rating"],
                                               bins=[0, 2, 3.5, 5],
                                               labels=["Low", "Medium", "High"])
df["experience_level"]               = pd.cut(df["freelancer_experience"],
                                               bins=[-1, 2, 5, 10, 20],
                                               labels=["Entry", "Junior", "Senior", "Expert"])

# ─── Save ─────────────────────────────────────────────────
os.makedirs("data", exist_ok=True)
df.to_csv("data/raw_dataset.csv", index=False)

# ─── Summary ──────────────────────────────────────────────
print("\n" + "=" * 60)
print("✅ DATASET GENERATED SUCCESSFULLY!")
print("=" * 60)
print(f"\n📊 Dataset Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
print(f"\n🎯 Target Distribution:")
print(f"   Success (1): {df['success'].sum():,} ({df['success'].mean()*100:.1f}%)")
print(f"   Failure (0): {(len(df)-df['success'].sum()):,} ({(1-df['success'].mean())*100:.1f}%)")
print(f"\n📁 Saved → data/raw_dataset.csv")
print("=" * 60)
