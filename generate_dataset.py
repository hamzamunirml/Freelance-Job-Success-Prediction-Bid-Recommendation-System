# ============================================================
# Script: generate_dataset.py
# Purpose: Generate synthetic dataset for Freelance Job Success Prediction
# ============================================================

import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

# Set random seed for reproducibility
np.random.seed(42)

print("🔄 Generating synthetic dataset...")

# ------------------ Dataset Parameters ------------------
n_records = 5000  # Total number of records

# ------------------ Generate Features ------------------

# 1. Project Budget (500 to 50,000 USD)
project_budget = np.random.randint(500, 50000, n_records)

# 2. Client Rating (1.0 to 5.0 stars)
client_rating = np.round(np.random.uniform(1.0, 5.0, n_records), 1)

# 3. Number of Existing Proposals (0 to 100)
num_existing_proposals = np.random.randint(0, 100, n_records)

# 4. Freelancer Experience (0 to 20 years)
freelancer_experience = np.random.randint(0, 20, n_records)

# 5. Proposal Quality Score (1 to 10)
proposal_quality_score = np.random.randint(1, 10, n_records)

# 6. Project Category (6 categories)
project_category = np.random.choice(
    ["Web Development", "Mobile App", "Data Science", "Design", "Writing", "Marketing"],
    n_records,
)

# 7. Client Location (added for more realism)
client_location = np.random.choice(
    [
        "USA",
        "UK",
        "Canada",
        "Australia",
        "India",
        "Germany",
        "France",
        "UAE",
        "Singapore",
        "Brazil",
    ],
    n_records,
)

# 8. Project Duration (in days)
project_duration = np.random.randint(1, 90, n_records)

# 9. Freelancer Country (added for more realism)
freelancer_country = np.random.choice(
    [
        "USA",
        "UK",
        "Canada",
        "Australia",
        "India",
        "Pakistan",
        "Philippines",
        "Ukraine",
        "Nigeria",
        "Bangladesh",
    ],
    n_records,
)

# 10. Freelancer Rating (1.0 to 5.0)
freelancer_rating = np.round(np.random.uniform(1.0, 5.0, n_records), 1)

# 11. Number of Previous Jobs Completed
previous_jobs_completed = np.random.randint(0, 500, n_records)

# ------------------ Generate Target Variable (Success) ------------------
# Success is influenced by multiple factors with different weights

# Calculate base probability using multiple features
success_probability = (
    0.20 * (client_rating / 5)  # Higher client rating = better chance
    + 0.15 * (freelancer_experience / 20)  # More experience = better chance
    + 0.15 * (proposal_quality_score / 10)  # Better proposal = better chance
    + 0.10 * (freelancer_rating / 5)  # Higher freelancer rating = better chance
    + 0.10 * (previous_jobs_completed / 500)  # More jobs = more reliable
    + 0.10 * (1 / (num_existing_proposals + 1))  # Less competition = better chance
    + 0.10 * (project_budget / 50000)  # Higher budget = more serious clients
    + 0.10 * np.random.normal(0.5, 0.15)  # Random noise
)

# Clip probability between 0 and 1
success_probability = np.clip(success_probability, 0.05, 0.95)

# Generate binary target using probability
success = np.random.binomial(1, success_probability)

# Ensure some imbalance (approx 35-40% success rate)
# If success rate is too high, adjust
current_success_rate = success.mean()
if current_success_rate > 0.45:
    # Reduce success by flipping some 1s to 0s
    success_indices = np.where(success == 1)[0]
    flip_count = int(len(success_indices) * 0.2)
    flip_indices = np.random.choice(success_indices, flip_count, replace=False)
    success[flip_indices] = 0

# ------------------ Create DataFrame ------------------
df = pd.DataFrame(
    {
        # Features
        "project_budget": project_budget,
        "client_rating": client_rating,
        "num_existing_proposals": num_existing_proposals,
        "freelancer_experience": freelancer_experience,
        "proposal_quality_score": proposal_quality_score,
        "project_category": project_category,
        "client_location": client_location,
        "project_duration": project_duration,
        "freelancer_country": freelancer_country,
        "freelancer_rating": freelancer_rating,
        "previous_jobs_completed": previous_jobs_completed,
        # Target
        "success": success,
    }
)

# ------------------ Add Some Missing Values (Realistic) ------------------
print("⚠️ Introducing missing values...")

# Missing in client_rating (5%)
missing_mask1 = np.random.random(n_records) < 0.05
df.loc[missing_mask1, "client_rating"] = np.nan

# Missing in proposal_quality_score (3%)
missing_mask2 = np.random.random(n_records) < 0.03
df.loc[missing_mask2, "proposal_quality_score"] = np.nan

# Missing in freelancer_rating (2%)
missing_mask3 = np.random.random(n_records) < 0.02
df.loc[missing_mask3, "freelancer_rating"] = np.nan

# Missing in project_duration (1%)
missing_mask4 = np.random.random(n_records) < 0.01
df.loc[missing_mask4, "project_duration"] = np.nan

# ------------------ Add Some Duplicates (Realistic) ------------------
print("🔄 Adding some duplicate records...")
dup_count = int(n_records * 0.02)  # 2% duplicates
dup_samples = df.sample(n=dup_count, replace=True)
df = pd.concat([df, dup_samples], ignore_index=True)

# ------------------ Add Some Outliers (Realistic) ------------------
print("📊 Introducing outliers...")

# Outliers in project_budget (very high values)
outlier_indices = np.random.choice(df.index, size=30, replace=False)
df.loc[outlier_indices, "project_budget"] = np.random.randint(100000, 500000, 30)

# Outliers in num_existing_proposals (very high values)
outlier_indices2 = np.random.choice(df.index, size=20, replace=False)
df.loc[outlier_indices2, "num_existing_proposals"] = np.random.randint(150, 300, 20)

# ------------------ Feature Engineering (for demonstration) ------------------
# Add some derived features
df["budget_per_proposal"] = df["project_budget"] / (df["num_existing_proposals"] + 1)
df["client_rating_category"] = pd.cut(
    df["client_rating"], bins=[0, 2, 3.5, 5], labels=["Low", "Medium", "High"]
)
df["experience_level"] = pd.cut(
    df["freelancer_experience"],
    bins=[-1, 2, 5, 10, 20],
    labels=["Entry", "Junior", "Senior", "Expert"],
)

# ------------------ Save Dataset ------------------
# Create data directory if it doesn't exist
os.makedirs("data", exist_ok=True)

# Save to CSV
df.to_csv("data/raw_dataset.csv", index=False)

# ------------------ Dataset Summary ------------------
print("\n" + "=" * 60)
print("✅ DATASET GENERATED SUCCESSFULLY!")
print("=" * 60)
print(f"\n📊 Dataset Shape: {df.shape}")
print(f"   - Records: {df.shape[0]:,}")
print(f"   - Features: {df.shape[1]}")

print(f"\n🎯 Target Variable Distribution:")
print(f"   - Success (1): {df['success'].sum():,} ({df['success'].mean()*100:.1f}%)")
print(
    f"   - Failure (0): {(len(df) - df['success'].sum()):,} {(1-df['success'].mean())*100:.1f}%)"
)

print(f"\n📋 Missing Values Summary:")
missing_counts = df.isnull().sum()
missing_cols = missing_counts[missing_counts > 0]
if len(missing_cols) > 0:
    for col, count in missing_cols.items():
        print(f"   - {col}: {count} ({count/len(df)*100:.1f}%)")
else:
    print("   ✅ No missing values!")

print(f"\n📊 Categorical Features:")
categorical_cols = df.select_dtypes(include=["object", "category"]).columns
for col in categorical_cols:
    print(f"   - {col}: {df[col].nunique()} unique values")

print(f"\n📊 Numerical Features Statistics:")
numerical_cols = df.select_dtypes(include=[np.number]).columns
for col in numerical_cols[:5]:  # Show first 5
    print(f"   - {col}:")
    print(f"       Mean: {df[col].mean():.2f}")
    print(f"       Std: {df[col].std():.2f}")
    print(f"       Min: {df[col].min():.2f}")
    print(f"       Max: {df[col].max():.2f}")

print(f"\n📁 File Saved: data/raw_dataset.csv")
print("\n🔍 Sample Data (First 5 rows):")
print(df.head())

print("\n💡 Dataset Info:")
print(df.info())

print("\n" + "=" * 60)
print("✅ Dataset generation complete! Ready for Phase 1.")
print("=" * 60)
