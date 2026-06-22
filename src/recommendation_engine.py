# ============================================================
# Phase 4: Intelligent Bid Recommendation Engine (FIXED)
# File: src/recommendation_engine.py
# ============================================================

import pandas as pd
import numpy as np
import joblib
import os
import json
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")


class BidRecommendationEngine:
    """
    Intelligent Bid Recommendation Engine
    Converts prediction probabilities into actionable business recommendations
    """

    def __init__(
        self,
        model_path="models/best_model.pkl",
        scaler_path="models/scaler.pkl",
        threshold_path="models/best_threshold.pkl",
    ):
        """
        Initialize the recommendation engine with trained model
        """
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)

        # Load threshold
        if os.path.exists(threshold_path):
            self.threshold = joblib.load(threshold_path)
        else:
            self.threshold = 0.40

        # IMPORTANT: Get feature names from scaler
        if hasattr(self.scaler, "feature_names_in_"):
            self.feature_columns = list(self.scaler.feature_names_in_)
        else:
            # Fallback: use training features
            self.feature_columns = [
                "project_budget",
                "client_rating",
                "num_existing_proposals",
                "freelancer_experience",
                "proposal_quality_score",
                "project_duration",
                "freelancer_rating",
                "previous_jobs_completed",
                "budget_per_proposal",
                "project_category_encoded",
                "client_location_encoded",
                "freelancer_country_encoded",
                "experience_quality_interaction",
                "combined_experience_rating",
                # Additional features from training
                "proposal_quality_score_x_freelancer_rating",
                "proposal_quality_score_x_freelancer_experience",
                "client_rating_x_freelancer_rating",
                "client_rating_category_encoded",
                "experience_level_encoded",
            ]

        print(f"✅ Bid Recommendation Engine initialized successfully!")
        print(f"📊 Model: {type(self.model).__name__}")
        print(f"📊 Threshold: {self.threshold:.2f}")
        print(f"📊 Features: {len(self.feature_columns)} features")

    def prepare_features(self, input_data):
        """
        Prepare features for prediction with all required columns
        """
        # Convert to DataFrame if dict
        if isinstance(input_data, dict):
            input_data = pd.DataFrame([input_data])

        # Create a DataFrame with all required features
        X = pd.DataFrame(index=input_data.index)

        # Add each required feature
        for col in self.feature_columns:
            if col in input_data.columns:
                X[col] = input_data[col]
            else:
                # Feature not provided, use default value
                if col in [
                    "project_budget",
                    "num_existing_proposals",
                    "freelancer_experience",
                    "proposal_quality_score",
                    "project_duration",
                    "previous_jobs_completed",
                ]:
                    X[col] = 0
                elif col in ["client_rating", "freelancer_rating"]:
                    X[col] = 3.0  # Average rating
                elif col in [
                    "budget_per_proposal",
                    "experience_quality_interaction",
                    "combined_experience_rating",
                ]:
                    X[col] = 0
                elif col in [
                    "project_category_encoded",
                    "client_location_encoded",
                    "freelancer_country_encoded",
                    "client_rating_category_encoded",
                    "experience_level_encoded",
                ]:
                    X[col] = 0
                elif "x_" in col:  # Interaction features
                    # Try to compute from existing columns
                    parts = col.split("_x_")
                    if len(parts) == 2:
                        col1, col2 = parts
                        if col1 in input_data.columns and col2 in input_data.columns:
                            X[col] = input_data[col1] * input_data[col2]
                        else:
                            X[col] = 0
                    else:
                        X[col] = 0
                else:
                    X[col] = 0

        # Ensure correct column order
        X = X[self.feature_columns]

        # Handle missing values
        X = X.fillna(0)

        # Scale features
        X_scaled = self.scaler.transform(X)

        return X_scaled

    def predict_success_probability(self, input_data):
        """
        Predict success probability for a project
        """
        X_scaled = self.prepare_features(input_data)
        probability = self.model.predict_proba(X_scaled)[:, 1][0]
        return probability

    def get_recommendation(self, probability):
        """
        Generate recommendation based on probability
        """
        if probability >= 0.70:
            return {
                "status": "HIGH_PROBABILITY",
                "level": "High",
                "action": "Apply Immediately",
                "color": "#2ecc71",
                "emoji": "✅",
                "recommendations": [
                    "Submit a personalized proposal immediately",
                    "Prioritize this opportunity",
                    "Invest extra time in proposal quality",
                    "Highlight your relevant experience",
                    "Follow up within 24 hours",
                ],
                "strategy": "High Priority Bid",
                "expected_success_rate": f"{probability*100:.1f}%",
            }
        elif probability >= 0.40:
            return {
                "status": "MEDIUM_PROBABILITY",
                "level": "Medium",
                "action": "Apply with Caution",
                "color": "#f39c12",
                "emoji": "⚠️",
                "recommendations": [
                    "Improve proposal quality",
                    "Refine pricing strategy",
                    "Customize the cover letter",
                    "Add relevant portfolio samples",
                    "Consider offering a competitive rate",
                ],
                "strategy": "Optimized Bid",
                "expected_success_rate": f"{probability*100:.1f}%",
            }
        else:
            return {
                "status": "LOW_PROBABILITY",
                "level": "Low",
                "action": "Avoid Applying",
                "color": "#e74c3c",
                "emoji": "❌",
                "recommendations": [
                    "Avoid applying to save resources",
                    "Focus on higher-probability opportunities",
                    "Consider improving your profile",
                    "Look for less competitive projects",
                    "Wait for better opportunities",
                ],
                "strategy": "Resource Preservation",
                "expected_success_rate": f"{probability*100:.1f}%",
            }

    def get_full_recommendation(self, input_data):
        """
        Get complete recommendation with all details
        """
        probability = self.predict_success_probability(input_data)
        recommendation = self.get_recommendation(probability)

        recommendation["probability"] = probability
        recommendation["threshold"] = self.threshold
        recommendation["timestamp"] = datetime.now().isoformat()

        if probability >= self.threshold:
            recommendation["decision"] = "PROCEED"
            recommendation["decision_reason"] = (
                f"Probability ({probability*100:.1f}%) exceeds threshold ({self.threshold*100:.1f}%)"
            )
        else:
            recommendation["decision"] = "SKIP"
            recommendation["decision_reason"] = (
                f"Probability ({probability*100:.1f}%) below threshold ({self.threshold*100:.1f}%)"
            )

        return recommendation

    def batch_recommendations(self, data):
        """
        Generate recommendations for multiple projects
        """
        X_scaled = self.prepare_features(data)
        probabilities = self.model.predict_proba(X_scaled)[:, 1]

        results = []
        for i, prob in enumerate(probabilities):
            rec = self.get_recommendation(prob)
            results.append(
                {
                    "project_id": i,
                    "probability": round(prob, 4),
                    "status": rec["status"],
                    "level": rec["level"],
                    "action": rec["action"],
                    "expected_success_rate": rec["expected_success_rate"],
                }
            )

        return pd.DataFrame(results)

    def generate_report(
        self, input_data, output_file="data/recommendation_report.json"
    ):
        """
        Generate detailed recommendation report
        """
        recommendation = self.get_full_recommendation(input_data)

        report = {
            "timestamp": datetime.now().isoformat(),
            "recommendation": recommendation,
            "input_data": (
                input_data
                if isinstance(input_data, dict)
                else input_data.to_dict("records")[0]
            ),
            "model_info": {
                "model_type": type(self.model).__name__,
                "threshold": self.threshold,
                "features_used": self.feature_columns,
            },
        }

        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, "w") as f:
            json.dump(report, f, indent=4, default=str)

        print(f"✅ Recommendation report saved to {output_file}")
        return report

    def get_decision_matrix(self):
        """
        Get the decision matrix for reference
        """
        return {
            "High (≥70%)": {
                "action": "Apply Immediately",
                "strategy": "High Priority Bid",
                "time_investment": "High (2-3 hours)",
                "win_probability": "70-100%",
            },
            "Medium (40-69%)": {
                "action": "Apply with Caution",
                "strategy": "Optimized Bid",
                "time_investment": "Medium (1-2 hours)",
                "win_probability": "40-70%",
            },
            "Low (<40%)": {
                "action": "Avoid Applying",
                "strategy": "Resource Preservation",
                "time_investment": "Minimal (0-1 hours)",
                "win_probability": "0-40%",
            },
        }


# ============================================================
# Helper Functions
# ============================================================


def get_feature_names_from_model(
    model_path="models/best_model.pkl", scaler_path="models/scaler.pkl"
):
    """
    Get the feature names used by the model
    """
    scaler = joblib.load(scaler_path)
    if hasattr(scaler, "feature_names_in_"):
        return list(scaler.feature_names_in_)
    else:
        return None


# ============================================================
# Example Usage
# ============================================================

if __name__ == "__main__":
    # Initialize engine
    engine = BidRecommendationEngine()

    # Get feature names for reference
    print("\n📋 Required Features:")
    for i, feat in enumerate(engine.feature_columns, 1):
        print(f"   {i}. {feat}")

    # Example 1: High probability project
    high_prob_project = {
        "project_budget": 5000,
        "client_rating": 4.8,
        "num_existing_proposals": 5,
        "freelancer_experience": 10,
        "proposal_quality_score": 9,
        "project_duration": 30,
        "freelancer_rating": 4.7,
        "previous_jobs_completed": 200,
        "budget_per_proposal": 833.33,
        "project_category_encoded": 0,
        "client_location_encoded": 3,
        "freelancer_country_encoded": 3,
        "experience_quality_interaction": 90,
        "combined_experience_rating": 80.5,
        # Interaction features (will be auto-calculated if missing)
        "proposal_quality_score_x_freelancer_rating": 42.3,
        "proposal_quality_score_x_freelancer_experience": 90,
        "client_rating_x_freelancer_rating": 22.56,
        "client_rating_category_encoded": 0,
        "experience_level_encoded": 1,
    }

    print("\n" + "=" * 60)
    print("EXAMPLE 1: HIGH PROBABILITY PROJECT")
    print("=" * 60)

    result = engine.get_full_recommendation(high_prob_project)
    print(f"\n📊 Success Probability: {result['probability']*100:.1f}%")
    print(f"📊 Status: {result['status']}")
    print(f"📊 Action: {result['action']}")
    print(f"📊 Decision: {result['decision']}")
    print(f"\n📋 Recommendations:")
    for rec in result["recommendations"]:
        print(f"   • {rec}")

    # Example 2: Low probability project
    low_prob_project = {
        "project_budget": 50000,
        "client_rating": 1.5,
        "num_existing_proposals": 95,
        "freelancer_experience": 2,
        "proposal_quality_score": 3,
        "project_duration": 45,
        "freelancer_rating": 2.1,
        "previous_jobs_completed": 10,
        "budget_per_proposal": 520.83,
        "project_category_encoded": 5,
        "client_location_encoded": 2,
        "freelancer_country_encoded": 4,
        "experience_quality_interaction": 6,
        "combined_experience_rating": 15.5,
        "proposal_quality_score_x_freelancer_rating": 6.3,
        "proposal_quality_score_x_freelancer_experience": 6,
        "client_rating_x_freelancer_rating": 3.15,
        "client_rating_category_encoded": 1,
        "experience_level_encoded": 2,
    }

    print("\n" + "=" * 60)
    print("EXAMPLE 2: LOW PROBABILITY PROJECT")
    print("=" * 60)

    result2 = engine.get_full_recommendation(low_prob_project)
    print(f"\n📊 Success Probability: {result2['probability']*100:.1f}%")
    print(f"📊 Status: {result2['status']}")
    print(f"📊 Action: {result2['action']}")
    print(f"\n📋 Recommendations:")
    for rec in result2["recommendations"]:
        print(f"   • {rec}")

    # Example 3: Batch predictions
    print("\n" + "=" * 60)
    print("EXAMPLE 3: BATCH PREDICTIONS")
    print("=" * 60)

    sample_projects = pd.DataFrame(
        [
            {
                "project_budget": 8000,
                "client_rating": 4.5,
                "num_existing_proposals": 12,
                "freelancer_experience": 15,
                "proposal_quality_score": 8,
                "project_duration": 25,
                "freelancer_rating": 4.6,
                "previous_jobs_completed": 350,
                "budget_per_proposal": 615.38,
                "project_category_encoded": 0,
                "client_location_encoded": 8,
                "freelancer_country_encoded": 5,
                "experience_quality_interaction": 120,
                "combined_experience_rating": 115.3,
                "proposal_quality_score_x_freelancer_rating": 36.8,
                "proposal_quality_score_x_freelancer_experience": 120,
                "client_rating_x_freelancer_rating": 20.7,
                "client_rating_category_encoded": 0,
                "experience_level_encoded": 1,
            },
            {
                "project_budget": 15000,
                "client_rating": 2.5,
                "num_existing_proposals": 75,
                "freelancer_experience": 3,
                "proposal_quality_score": 4,
                "project_duration": 45,
                "freelancer_rating": 2.8,
                "previous_jobs_completed": 25,
                "budget_per_proposal": 197.37,
                "project_category_encoded": 5,
                "client_location_encoded": 2,
                "freelancer_country_encoded": 4,
                "experience_quality_interaction": 12,
                "combined_experience_rating": 20.5,
                "proposal_quality_score_x_freelancer_rating": 11.2,
                "proposal_quality_score_x_freelancer_experience": 12,
                "client_rating_x_freelancer_rating": 7.0,
                "client_rating_category_encoded": 1,
                "experience_level_encoded": 2,
            },
        ]
    )

    batch_results = engine.batch_recommendations(sample_projects)
    print(batch_results.to_string(index=False))

    # Get decision matrix
    print("\n" + "=" * 60)
    print("DECISION MATRIX")
    print("=" * 60)
    for level, details in engine.get_decision_matrix().items():
        print(f"\n📊 {level}:")
        for key, value in details.items():
            print(f"   • {key}: {value}")

    print("\n" + "=" * 60)
    print("✅ Phase 4 - Recommendation Engine Completed Successfully!")
    print("=" * 60)
